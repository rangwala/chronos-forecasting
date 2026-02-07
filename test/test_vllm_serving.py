# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mock vLLM serving tests for Chronos-2.

Tests the full vLLM serving lifecycle (registration, engine creation, request
handling, batching, async serving, plugin discovery) using mock vLLM infrastructure
with real Chronos-2 model inference. This allows testing the serving path on
machines without vLLM/CUDA installed.
"""

import asyncio
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from chronos import BaseChronosPipeline
from chronos.chronos2.model import Chronos2Output
from chronos.chronos2.vllm import (
    Chronos2ForVLLM,
    Chronos2VLLMConfig,
    ForecastOutput,
    format_for_json,
    postprocess_chronos2_output,
    preprocess_for_chronos2,
)
from chronos.chronos2.vllm.register import (
    Chronos2VLLMPlugin,
    register_with_vllm,
)


DUMMY_MODEL_PATH = Path(__file__).parent / "dummy-chronos2-model"


# =====================================================================
# Mock vLLM Infrastructure
# =====================================================================


@dataclass
class MockPoolingOutput:
    """Simulates vllm.PoolingOutput for encoder-only model results."""

    data: torch.Tensor  # quantile_preds for one request: (1, num_quantiles, pred_len)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockRequestOutput:
    """Simulates vllm.RequestOutput."""

    request_id: str
    finished: bool
    outputs: List[MockPoolingOutput]
    prompt: Optional[Any] = None
    metrics: Optional[Dict[str, float]] = None


@dataclass
class MockSamplingParams:
    """Simulates vllm.SamplingParams, adapted for Chronos-2 forecasting."""

    prediction_length: int = 24
    quantile_levels: Optional[List[float]] = None
    cross_learning: bool = False


@dataclass
class MockEngineArgs:
    """Simulates vllm.EngineArgs."""

    model: str = ""
    dtype: str = "float32"
    device: str = "cpu"
    max_batch_size: int = 32
    task: str = "embed"


class MockModelRegistry:
    """Simulates vllm.ModelRegistry."""

    def __init__(self):
        self._registry: Dict[str, str] = {}

    def register_model(self, name: str, model_class_path: str):
        self._registry[name] = model_class_path

    def is_registered(self, name: str) -> bool:
        return name in self._registry

    def resolve_model_cls(self, architectures: List[str]) -> Optional[type]:
        """Resolve a model class by dynamically importing the registered path."""
        for arch in architectures:
            if arch in self._registry:
                module_path, class_name = self._registry[arch].split(":")
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
        return None


class MockLLMEngine:
    """
    Simulates vllm.LLMEngine for Chronos-2 serving.

    Holds a real Chronos2ForVLLM instance. add_request() queues requests;
    step() processes one batch with real inference and returns MockRequestOutput.
    """

    def __init__(self, model: Chronos2ForVLLM, max_batch_size: int = 32):
        self.model = model
        self.max_batch_size = max_batch_size
        self._pending_requests: Dict[str, dict] = {}
        self._aborted: set = set()

    @classmethod
    def from_engine_args(
        cls, engine_args: MockEngineArgs, registry: MockModelRegistry
    ):
        model_cls = registry.resolve_model_cls(
            ["Chronos2Model", "Chronos2ForForecasting"]
        )
        if model_cls is None:
            raise ValueError("Model architecture not found in registry")
        model = model_cls.from_pretrained(
            engine_args.model, device_map=engine_args.device
        )
        return cls(model=model, max_batch_size=engine_args.max_batch_size)

    def add_request(
        self, request_id: str, prompt: Any, params: MockSamplingParams
    ) -> None:
        if request_id in self._pending_requests:
            raise ValueError(f"Duplicate request_id: {request_id}")
        self._pending_requests[request_id] = {
            "prompt": prompt,
            "params": params,
        }

    def abort_request(self, request_id: str) -> None:
        self._aborted.add(request_id)
        self._pending_requests.pop(request_id, None)

    def step(self) -> List[MockRequestOutput]:
        outputs = []
        active = {
            rid: data
            for rid, data in self._pending_requests.items()
            if rid not in self._aborted
        }
        batch_ids = list(active.keys())[: self.max_batch_size]
        if not batch_ids:
            return []

        # Group by prediction_length
        groups: Dict[int, List[str]] = {}
        for rid in batch_ids:
            pl = active[rid]["params"].prediction_length
            groups.setdefault(pl, []).append(rid)

        for pred_len, rids in groups.items():
            contexts = []
            for rid in rids:
                ctx = active[rid]["prompt"]
                if isinstance(ctx, torch.Tensor) and ctx.dim() == 1:
                    contexts.append(ctx)
                elif isinstance(ctx, torch.Tensor) and ctx.dim() == 2:
                    # Each row is a separate series; add individually
                    for row in ctx:
                        contexts.append(row)
                else:
                    contexts.append(
                        torch.tensor(ctx, dtype=torch.float32)
                        if not isinstance(ctx, torch.Tensor)
                        else ctx
                    )

            preprocessed = preprocess_for_chronos2(
                contexts,
                prediction_length=pred_len,
                output_patch_size=self.model.output_patch_size,
                max_context_length=self.model.context_length,
            )

            with torch.no_grad():
                raw_output = self.model.forward(
                    preprocessed.context,
                    context_mask=preprocessed.context_mask,
                    group_ids=preprocessed.group_ids,
                    num_output_patches=preprocessed.num_output_patches,
                )

            preds = raw_output.quantile_preds[..., :pred_len]

            for i, rid in enumerate(rids):
                pooling_output = MockPoolingOutput(
                    data=preds[i : i + 1],
                    metadata={
                        "prediction_length": pred_len,
                        "quantile_levels": list(self.model.quantiles),
                    },
                )
                outputs.append(
                    MockRequestOutput(
                        request_id=rid,
                        finished=True,
                        outputs=[pooling_output],
                        prompt=active[rid]["prompt"],
                    )
                )

            for rid in rids:
                del self._pending_requests[rid]

        return outputs

    def has_pending_requests(self) -> bool:
        return len(self._pending_requests) > 0

    def get_num_pending_requests(self) -> int:
        return len(self._pending_requests)


class MockAsyncLLMEngine:
    """Simulates vllm.AsyncLLMEngine wrapping MockLLMEngine."""

    def __init__(self, engine: MockLLMEngine):
        self.engine = engine

    @classmethod
    async def from_engine_args(
        cls, engine_args: MockEngineArgs, registry: MockModelRegistry
    ):
        engine = MockLLMEngine.from_engine_args(engine_args, registry)
        return cls(engine=engine)

    async def add_request(
        self, request_id: str, prompt: Any, params: MockSamplingParams
    ):
        self.engine.add_request(request_id, prompt, params)

    async def abort_request(self, request_id: str):
        self.engine.abort_request(request_id)

    async def step(self) -> List[MockRequestOutput]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.engine.step)

    async def generate(
        self, request_id: str, prompt: Any, params: MockSamplingParams
    ) -> AsyncIterator[MockRequestOutput]:
        await self.add_request(request_id, prompt, params)
        results = await self.step()
        for result in results:
            if result.request_id == request_id:
                yield result


def _run_async(coro):
    """Run an async coroutine synchronously for testing."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def vllm_wrapper() -> Chronos2ForVLLM:
    return Chronos2ForVLLM.from_pretrained(DUMMY_MODEL_PATH, device_map="cpu")


@pytest.fixture
def mock_registry() -> MockModelRegistry:
    registry = MockModelRegistry()
    registry.register_model(
        "Chronos2Model",
        "chronos.chronos2.vllm.wrapper:Chronos2ForVLLM",
    )
    registry.register_model(
        "Chronos2ForForecasting",
        "chronos.chronos2.vllm.wrapper:Chronos2ForVLLM",
    )
    return registry


@pytest.fixture
def mock_engine(vllm_wrapper) -> MockLLMEngine:
    return MockLLMEngine(model=vllm_wrapper, max_batch_size=32)


@pytest.fixture
def mock_async_engine(mock_engine) -> MockAsyncLLMEngine:
    return MockAsyncLLMEngine(engine=mock_engine)


# =====================================================================
# Tests
# =====================================================================


class TestMockModelRegistry:
    def test_register_model_stores_entry(self):
        registry = MockModelRegistry()
        registry.register_model("TestArch", "some.module:SomeClass")
        assert registry.is_registered("TestArch")

    def test_register_multiple_architectures(self, mock_registry):
        assert mock_registry.is_registered("Chronos2Model")
        assert mock_registry.is_registered("Chronos2ForForecasting")

    def test_resolve_model_cls_returns_real_class(self, mock_registry):
        cls = mock_registry.resolve_model_cls(["Chronos2Model"])
        assert cls is Chronos2ForVLLM

    def test_resolve_unknown_returns_none(self, mock_registry):
        assert mock_registry.resolve_model_cls(["UnknownArch"]) is None

    def test_register_with_vllm_calls_registry(self):
        mock_vllm = MagicMock()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            result = register_with_vllm()
            assert result is True
            mock_vllm.ModelRegistry.register_model.assert_called_once_with(
                "Chronos2ForForecasting",
                "chronos.chronos2.vllm.wrapper:Chronos2ForVLLM",
            )


class TestEngineModelLoading:
    def test_engine_from_engine_args(self, mock_registry):
        args = MockEngineArgs(model=str(DUMMY_MODEL_PATH), device="cpu")
        engine = MockLLMEngine.from_engine_args(args, mock_registry)
        assert isinstance(engine.model, Chronos2ForVLLM)

    def test_engine_model_properties(self, mock_registry):
        args = MockEngineArgs(model=str(DUMMY_MODEL_PATH), device="cpu")
        engine = MockLLMEngine.from_engine_args(args, mock_registry)
        assert engine.model.is_pooling_model is True
        assert engine.model.supports_pp is False
        assert engine.model.is_attention_free is False

    def test_unregistered_architecture_raises(self):
        empty_registry = MockModelRegistry()
        args = MockEngineArgs(model=str(DUMMY_MODEL_PATH), device="cpu")
        with pytest.raises(ValueError, match="not found in registry"):
            MockLLMEngine.from_engine_args(args, empty_registry)

    def test_nonexistent_model_path_raises(self, mock_registry):
        args = MockEngineArgs(model="/nonexistent/path", device="cpu")
        with pytest.raises(Exception):
            MockLLMEngine.from_engine_args(args, mock_registry)


class TestRequestLifecycle:
    def test_single_request_lifecycle(self, mock_engine):
        context = torch.randn(100)
        mock_engine.add_request("req-1", context, MockSamplingParams(prediction_length=24))
        results = mock_engine.step()
        assert len(results) == 1
        assert results[0].request_id == "req-1"
        assert results[0].finished is True
        assert results[0].outputs[0].data.shape == (1, mock_engine.model.num_quantiles, 24)

    def test_output_no_nan_or_inf(self, mock_engine):
        context = torch.randn(100)
        mock_engine.add_request("req-1", context, MockSamplingParams())
        results = mock_engine.step()
        data = results[0].outputs[0].data
        assert not torch.isnan(data).any()
        assert not torch.isinf(data).any()

    def test_non_multiple_prediction_length(self, mock_engine):
        context = torch.randn(100)
        mock_engine.add_request("req-1", context, MockSamplingParams(prediction_length=17))
        results = mock_engine.step()
        assert results[0].outputs[0].data.shape[-1] == 17

    def test_quantile_ordering(self, mock_engine):
        context = torch.randn(100)
        mock_engine.add_request("req-1", context, MockSamplingParams())
        results = mock_engine.step()
        preds = results[0].outputs[0].data.squeeze(0)  # (num_quantiles, pred_len)
        for i in range(preds.shape[0] - 1):
            assert preds[i].mean() <= preds[i + 1].mean() + 1e-3

    def test_step_empty_returns_empty(self, mock_engine):
        assert mock_engine.step() == []

    def test_duplicate_request_id_raises(self, mock_engine):
        context = torch.randn(100)
        mock_engine.add_request("req-1", context, MockSamplingParams())
        with pytest.raises(ValueError, match="Duplicate request_id"):
            mock_engine.add_request("req-1", context, MockSamplingParams())

    def test_prompt_preserved_in_output(self, mock_engine):
        context = torch.randn(64)
        mock_engine.add_request("req-1", context, MockSamplingParams())
        results = mock_engine.step()
        assert torch.equal(results[0].prompt, context)


class TestBatchingBehavior:
    def test_multiple_requests_batched(self, mock_engine):
        for i in range(4):
            mock_engine.add_request(f"req-{i}", torch.randn(80), MockSamplingParams())
        results = mock_engine.step()
        assert len(results) == 4

    def test_different_prediction_lengths_grouped(self, mock_engine):
        mock_engine.add_request("a", torch.randn(80), MockSamplingParams(prediction_length=16))
        mock_engine.add_request("b", torch.randn(80), MockSamplingParams(prediction_length=16))
        mock_engine.add_request("c", torch.randn(80), MockSamplingParams(prediction_length=32))
        results = mock_engine.step()
        assert len(results) == 3
        result_map = {r.request_id: r for r in results}
        assert result_map["a"].outputs[0].data.shape[-1] == 16
        assert result_map["c"].outputs[0].data.shape[-1] == 32

    def test_max_batch_size_respected(self, vllm_wrapper):
        engine = MockLLMEngine(model=vllm_wrapper, max_batch_size=2)
        for i in range(5):
            engine.add_request(f"req-{i}", torch.randn(64), MockSamplingParams())
        results = engine.step()
        assert len(results) == 2
        assert engine.has_pending_requests()
        assert engine.get_num_pending_requests() == 3

    def test_variable_length_contexts(self, mock_engine):
        mock_engine.add_request("a", torch.randn(50), MockSamplingParams(prediction_length=16))
        mock_engine.add_request("b", torch.randn(100), MockSamplingParams(prediction_length=16))
        mock_engine.add_request("c", torch.randn(75), MockSamplingParams(prediction_length=16))
        results = mock_engine.step()
        assert len(results) == 3
        for r in results:
            assert r.outputs[0].data.shape[-1] == 16

    def test_abort_removes_from_batch(self, mock_engine):
        mock_engine.add_request("a", torch.randn(64), MockSamplingParams())
        mock_engine.add_request("b", torch.randn(64), MockSamplingParams())
        mock_engine.add_request("c", torch.randn(64), MockSamplingParams())
        mock_engine.abort_request("b")
        results = mock_engine.step()
        ids = {r.request_id for r in results}
        assert "b" not in ids
        assert len(results) == 2


class TestAsyncServingFlow:
    def test_async_add_and_step(self, mock_async_engine):
        async def _test():
            await mock_async_engine.add_request(
                "req-1", torch.randn(100), MockSamplingParams()
            )
            results = await mock_async_engine.step()
            assert len(results) == 1
            assert results[0].finished is True

        _run_async(_test())

    def test_async_generate_yields_result(self, mock_async_engine):
        async def _test():
            results = []
            async for r in mock_async_engine.generate(
                "req-1", torch.randn(80), MockSamplingParams()
            ):
                results.append(r)
            assert len(results) == 1
            assert results[0].request_id == "req-1"
            assert results[0].finished is True

        _run_async(_test())

    def test_async_multiple_concurrent(self, mock_async_engine):
        async def _test():
            for i in range(3):
                await mock_async_engine.add_request(
                    f"req-{i}", torch.randn(64), MockSamplingParams()
                )
            results = await mock_async_engine.step()
            assert len(results) == 3

        _run_async(_test())

    def test_async_abort(self, mock_async_engine):
        async def _test():
            await mock_async_engine.add_request(
                "req-1", torch.randn(64), MockSamplingParams()
            )
            await mock_async_engine.abort_request("req-1")
            results = await mock_async_engine.step()
            assert len(results) == 0

        _run_async(_test())

    def test_async_matches_sync(self, mock_engine, mock_async_engine):
        torch.manual_seed(42)
        context = torch.randn(80)

        # Sync
        mock_engine.add_request("sync", context.clone(), MockSamplingParams())
        sync_results = mock_engine.step()

        # Async
        async def _test():
            await mock_async_engine.add_request(
                "async", context.clone(), MockSamplingParams()
            )
            return await mock_async_engine.step()

        async_results = _run_async(_test())

        torch.testing.assert_close(
            sync_results[0].outputs[0].data,
            async_results[0].outputs[0].data,
        )


class TestPluginInterface:
    def test_plugin_architectures(self):
        archs = Chronos2VLLMPlugin.get_supported_architectures()
        assert "Chronos2Model" in archs
        assert "Chronos2ForForecasting" in archs

    def test_plugin_load_model(self):
        model = Chronos2VLLMPlugin.load_model(str(DUMMY_MODEL_PATH), device_map="cpu")
        assert isinstance(model, Chronos2ForVLLM)

    def test_plugin_register_with_mock(self):
        mock_vllm = MagicMock()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            result = Chronos2VLLMPlugin.register()
            assert result is True
            assert mock_vllm.ModelRegistry.register_model.call_count == 2

    def test_setup_integration_with_mock(self):
        mock_vllm = MagicMock()
        mock_vllm.__version__ = "0.6.0"
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            from chronos.chronos2.vllm.register import setup_vllm_integration

            status = setup_vllm_integration()
            assert status["vllm_available"] is True
            assert status["vllm_version"] == "0.6.0"
            assert status["registered"] is True
            assert isinstance(status["architectures"], list)


class TestErrorHandling:
    def test_empty_context(self, mock_engine):
        mock_engine.add_request("req-1", torch.tensor([]), MockSamplingParams())
        with pytest.raises(Exception):
            mock_engine.step()

    def test_nan_only_context(self, mock_engine):
        context = torch.full((64,), float("nan"))
        mock_engine.add_request("req-1", context, MockSamplingParams())
        # Should not crash; model handles NaN via masking
        results = mock_engine.step()
        assert len(results) == 1

    def test_abort_nonexistent_is_noop(self, mock_engine):
        mock_engine.abort_request("nonexistent")  # Should not raise

    def test_very_long_context_truncated(self, mock_engine):
        context = torch.randn(20000)
        mock_engine.add_request("req-1", context, MockSamplingParams(prediction_length=16))
        results = mock_engine.step()
        assert len(results) == 1
        assert results[0].outputs[0].data.shape[-1] == 16

    def test_3d_context_raises(self, mock_engine):
        context = torch.randn(2, 3, 100)
        mock_engine.add_request("req-1", context, MockSamplingParams())
        with pytest.raises(Exception):
            mock_engine.step()

    def test_zero_prediction_length(self, mock_engine):
        mock_engine.add_request(
            "req-1", torch.randn(64), MockSamplingParams(prediction_length=0)
        )
        with pytest.raises(Exception):
            mock_engine.step()

    def test_negative_prediction_length(self, mock_engine):
        mock_engine.add_request(
            "req-1", torch.randn(64), MockSamplingParams(prediction_length=-1)
        )
        with pytest.raises(Exception):
            mock_engine.step()


class TestEndToEndServingPipeline:
    def test_full_serving_flow(self, mock_registry):
        """Plugin register -> engine create -> request -> inference -> JSON output."""
        args = MockEngineArgs(model=str(DUMMY_MODEL_PATH), device="cpu")
        engine = MockLLMEngine.from_engine_args(args, mock_registry)

        context = torch.randn(100)
        engine.add_request("req-1", context, MockSamplingParams(prediction_length=24))
        results = engine.step()

        assert len(results) == 1
        data = results[0].outputs[0].data

        # Run through postprocessing
        output = Chronos2Output(quantile_preds=data)
        forecast = postprocess_chronos2_output(
            output,
            quantile_levels=results[0].outputs[0].metadata["quantile_levels"],
            prediction_length=24,
        )
        json_out = format_for_json(forecast)

        assert "predictions" in json_out
        assert "quantiles" in json_out
        assert json_out["prediction_length"] == 24
        assert json_out["batch_size"] == 1

    def test_serving_matches_direct_wrapper(self, mock_engine, vllm_wrapper):
        """Engine output should match direct forward_forecast()."""
        torch.manual_seed(42)
        context = torch.randn(80)

        # Via engine
        mock_engine.add_request("req-1", context.clone(), MockSamplingParams(prediction_length=16))
        engine_results = mock_engine.step()
        engine_preds = engine_results[0].outputs[0].data

        # Direct
        direct_output = vllm_wrapper.forward_forecast(
            context.clone().unsqueeze(0), prediction_length=16
        )
        direct_preds = direct_output.quantile_preds

        torch.testing.assert_close(engine_preds, direct_preds, rtol=1e-5, atol=1e-5)

    def test_same_input_deterministic(self, mock_engine):
        context = torch.randn(64)
        mock_engine.add_request("a", context.clone(), MockSamplingParams())
        mock_engine.add_request("b", context.clone(), MockSamplingParams())
        results = mock_engine.step()
        result_map = {r.request_id: r for r in results}
        torch.testing.assert_close(
            result_map["a"].outputs[0].data,
            result_map["b"].outputs[0].data,
        )

    def test_serving_with_cross_learning_context(self, mock_engine):
        """Cross-learning is handled at the preprocessing level."""
        for i in range(4):
            mock_engine.add_request(
                f"req-{i}",
                torch.randn(80),
                MockSamplingParams(prediction_length=16),
            )
        results = mock_engine.step()
        assert len(results) == 4
        for r in results:
            assert r.outputs[0].data.shape[-1] == 16


class TestOutputFormatConversion:
    def test_pooling_output_to_forecast_output(self, mock_engine):
        mock_engine.add_request("req-1", torch.randn(64), MockSamplingParams())
        results = mock_engine.step()
        data = results[0].outputs[0].data
        metadata = results[0].outputs[0].metadata

        output = Chronos2Output(quantile_preds=data)
        forecast = postprocess_chronos2_output(
            output,
            quantile_levels=metadata["quantile_levels"],
            prediction_length=metadata["prediction_length"],
        )
        assert isinstance(forecast, ForecastOutput)
        assert forecast.prediction_length == 24

    def test_forecast_output_to_json(self, mock_engine):
        mock_engine.add_request("req-1", torch.randn(64), MockSamplingParams())
        results = mock_engine.step()
        data = results[0].outputs[0].data
        metadata = results[0].outputs[0].metadata

        output = Chronos2Output(quantile_preds=data)
        forecast = postprocess_chronos2_output(
            output,
            quantile_levels=metadata["quantile_levels"],
            prediction_length=24,
        )
        json_out = format_for_json(forecast)

        assert isinstance(json_out["predictions"], list)
        assert len(json_out["predictions"]) == 1
        assert len(json_out["predictions"][0]) == 24

    def test_batch_outputs_split_correctly(self, mock_engine):
        torch.manual_seed(0)
        contexts = [torch.randn(64) for _ in range(3)]
        for i, ctx in enumerate(contexts):
            mock_engine.add_request(f"req-{i}", ctx, MockSamplingParams())
        results = mock_engine.step()
        assert len(results) == 3
        # Different inputs should produce different outputs
        tensors = [r.outputs[0].data for r in results]
        assert not torch.equal(tensors[0], tensors[1])
        assert not torch.equal(tensors[1], tensors[2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
