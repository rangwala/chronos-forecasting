# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Native vLLM integration tests for Chronos-2.

Exercises Chronos2ForForecasting (native.py) through a real vLLM engine on GPU,
verifying model registration, weight loading, forward pass correctness, numerical
equivalence with the Chronos2ForVLLM wrapper, and bfloat16 support.

Requirements
------------
- NVIDIA GPU with CUDA
- vLLM >= 0.6.0: pip install vllm
- chronos-forecasting: pip install -e .

Run
---
    pytest test/test_vllm_native.py -v                     # all phases
    pytest test/test_vllm_native.py -v -m "not slow"       # skip throughput
    pytest test/test_vllm_native.py::TestWeightLoading -v  # single phase

Note: all tests are automatically skipped if CUDA is unavailable or vLLM is
not installed. No manual skip flags needed.
"""

import math
import time
from pathlib import Path
from typing import List, Optional

import pytest
import torch

# ---------------------------------------------------------------------------
# Guard: skip entire module when CUDA or vLLM are absent
# ---------------------------------------------------------------------------

def _prerequisites_missing() -> Optional[str]:
    if not torch.cuda.is_available():
        return "CUDA not available"
    try:
        import vllm  # noqa: F401
    except ImportError:
        return "vLLM not installed — run: pip install vllm"
    return None


_SKIP_REASON = _prerequisites_missing()
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")


# ---------------------------------------------------------------------------
# Constants (from test/dummy-chronos2-model/config.json)
# ---------------------------------------------------------------------------

DUMMY_MODEL_PATH = Path(__file__).parent / "dummy-chronos2-model"

NUM_QUANTILES = 21       # len(chronos_config["quantiles"])
OUTPUT_PATCH_SIZE = 16   # chronos_config["output_patch_size"]
CONTEXT_LENGTH = 8192    # chronos_config["context_length"]
DEFAULT_PRED_LEN = 24


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nop(prediction_length: int, patch_size: int = OUTPUT_PATCH_SIZE) -> int:
    """Number of output patches required to cover prediction_length."""
    return math.ceil(prediction_length / patch_size)


def _decode_embedding(
    flat: torch.Tensor,
    num_quantiles: int = NUM_QUANTILES,
    prediction_length: int = DEFAULT_PRED_LEN,
    patch_size: int = OUTPUT_PATCH_SIZE,
) -> torch.Tensor:
    """
    Decode the flat vLLM pooling-output embedding back to quantile predictions.

    The native forward() returns qp.reshape(1, -1) where qp has shape
    (1, num_quantiles, nop * patch_size). We reshape then truncate here.

    Returns shape: (1, num_quantiles, prediction_length)
    """
    nop = _nop(prediction_length, patch_size)
    full_len = nop * patch_size
    flat_cpu = flat.reshape(1, num_quantiles, full_len).cpu()
    return flat_cpu[..., :prediction_length]


def _make_mm_input(context: torch.Tensor, prediction_length: int) -> dict:
    """Build the multi-modal input dict that Chronos2Processor expects."""
    nop = _nop(prediction_length)
    return {
        "prompt": "",
        "multi_modal_data": {
            "context": context,
            "num_output_patches": torch.tensor([nop], dtype=torch.long),
        },
    }


def _engine_request(
    engine,
    request_id: str,
    context: torch.Tensor,
    prediction_length: int = DEFAULT_PRED_LEN,
) -> torch.Tensor:
    """
    Submit one request to the vLLM engine, run one step, return flat embedding.
    Handles the PoolingParams import across vLLM versions.
    """
    try:
        from vllm.sampling_params import PoolingParams
    except ImportError:
        from vllm import PoolingParams  # older vLLM versions

    engine.add_request(
        request_id=request_id,
        inputs=_make_mm_input(context, prediction_length),
        params=PoolingParams(),
    )
    outputs = engine.step()
    assert len(outputs) >= 1
    # Find our request (engine may batch others)
    matched = [o for o in outputs if o.request_id == request_id]
    assert len(matched) == 1 and matched[0].finished
    return matched[0].outputs.embedding


# ---------------------------------------------------------------------------
# Fixtures (module-scoped so the GPU engine is created once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    """
    Real vLLM LLMEngine loaded with Chronos2ForForecasting via native.py.
    Uses float32 and 50% GPU memory to leave headroom for other tests.
    """
    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.llm_engine import LLMEngine
    from chronos.chronos2.vllm.register import register_with_vllm

    register_with_vllm(
        model_name="Chronos2ForForecasting",
        model_class_path="chronos.chronos2.vllm.native:Chronos2ForForecasting",
    )

    args = EngineArgs(
        model=str(DUMMY_MODEL_PATH),
        task="embed",
        dtype="float32",
        gpu_memory_utilization=0.5,
        max_model_len=CONTEXT_LENGTH,
    )
    eng = LLMEngine.from_engine_args(args)
    yield eng
    del eng
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def wrapper():
    """Chronos2ForVLLM on GPU (float32) for equivalence comparisons."""
    from chronos.chronos2.vllm.wrapper import Chronos2ForVLLM

    m = Chronos2ForVLLM.from_pretrained(
        str(DUMMY_MODEL_PATH), device_map="cuda", torch_dtype=torch.float32
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def native(engine):
    """
    The Chronos2ForForecasting instance embedded inside the vLLM engine.
    Walks common attribute paths across vLLM versions.
    """
    from chronos.chronos2.vllm.native import Chronos2ForForecasting

    candidates = [
        "model_executor.driver_worker.model_runner.model",
        "model_executor.driver_worker.model_runner.model.model",
        "model_executor.worker.model_runner.model",
    ]
    for path in candidates:
        obj = engine
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, Chronos2ForForecasting):
                return obj
        except AttributeError:
            continue
    pytest.skip("Cannot extract native model from engine — vLLM API path may differ by version")


@pytest.fixture(scope="class")
def bf16_engine():
    """Separate engine using bfloat16 (Ampere+ only)."""
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bfloat16 requires Ampere GPU (compute capability >= 8.0)")

    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.llm_engine import LLMEngine
    from chronos.chronos2.vllm.register import register_with_vllm

    register_with_vllm()
    args = EngineArgs(
        model=str(DUMMY_MODEL_PATH),
        task="embed",
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        max_model_len=CONTEXT_LENGTH,
    )
    eng = LLMEngine.from_engine_args(args)
    yield eng
    del eng
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Phase 1 — Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    """Verify the native class registers correctly with vLLM's ModelRegistry."""

    def test_register_returns_true(self):
        from chronos.chronos2.vllm.register import register_with_vllm
        result = register_with_vllm(
            model_name="Chronos2ForForecasting",
            model_class_path="chronos.chronos2.vllm.native:Chronos2ForForecasting",
        )
        assert result is True

    def test_architecture_in_registry(self):
        from vllm import ModelRegistry
        from chronos.chronos2.vllm.register import register_with_vllm

        register_with_vllm()
        assert ModelRegistry.is_registered("Chronos2ForForecasting")

    def test_native_class_importable(self):
        from chronos.chronos2.vllm.native import Chronos2ForForecasting  # noqa
        assert Chronos2ForForecasting is not None

    def test_native_exported_from_package(self):
        import chronos.chronos2.vllm as vllm_pkg
        assert hasattr(vllm_pkg, "Chronos2ForForecasting")

    def test_register_is_idempotent(self):
        """Calling register_with_vllm() twice does not raise."""
        from chronos.chronos2.vllm.register import register_with_vllm
        assert register_with_vllm() is True
        assert register_with_vllm() is True


# ---------------------------------------------------------------------------
# Phase 2 — Engine initialisation
# ---------------------------------------------------------------------------

class TestEngineInit:
    """Verify the engine loads correctly and native model has expected properties."""

    def test_engine_loads_native_class(self, native):
        from chronos.chronos2.vllm.native import Chronos2ForForecasting
        assert isinstance(native, Chronos2ForForecasting)

    def test_model_parameters_on_cuda(self, native):
        param = next(native.parameters())
        assert param.is_cuda, "Model parameters must be on GPU"

    def test_pooler_is_identity(self, native):
        from vllm.model_executor.layers.pooler import IdentityPooler
        assert isinstance(native.pooler, IdentityPooler)

    def test_config_num_quantiles(self, native):
        assert native.num_quantiles == NUM_QUANTILES

    def test_config_output_patch_size(self, native):
        assert native.output_patch_size == OUTPUT_PATCH_SIZE

    def test_config_use_reg_token(self, native):
        assert native.use_reg_token is True

    def test_config_use_arcsinh(self, native):
        assert native.use_arcsinh is True

    def test_config_context_length(self, native):
        assert native.context_length == CONTEXT_LENGTH


# ---------------------------------------------------------------------------
# Phase 3 — Weight loading
# ---------------------------------------------------------------------------

class TestWeightLoading:
    """
    Verify that load_weights() correctly remaps HF checkpoint names to the
    native layer structure and that the loaded values match the HF model.

    Key remappings tested:
      encoder.block[i].layer[2].mlp.wi  →  encoder.block[i].ff.wi
      encoder.block[i].layer[0].self_attention.{q,k,v}  →  block[i].time_attn.self_attention.qkv
      encoder.block[i].layer[1].self_attention.{q,k,v}  →  block[i].group_attn.self_attention.qkv
    """

    def test_named_parameters_non_empty(self, native):
        assert len(dict(native.named_parameters())) > 0

    def test_all_encoder_weights_finite(self, native):
        for name, param in native.encoder.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite weight: {name}"

    def test_qkv_weight_shape(self, native):
        """
        QKVParallelLinear weight for time-attention block 0.
        Config: d_model=6, d_kv=4, num_heads=4 → shape (3*4*4, 6) = (48, 6).
        """
        qkv = native.encoder.block[0].time_attn.self_attention.qkv.weight
        assert qkv.shape == (3 * 4 * 4, 6), f"Unexpected QKV shape: {qkv.shape}"

    def test_shared_embedding_matches_hf(self, native, wrapper):
        """shared (vocab embedding) weight matches between native and HF wrapper."""
        native_w = native.shared.weight.detach().cpu().float()
        hf_w = wrapper.model.shared.weight.detach().cpu().float()
        torch.testing.assert_close(native_w, hf_w, rtol=1e-5, atol=1e-5)

    def test_ff_wi_weight_matches_hf(self, native, wrapper):
        """
        Feedforward wi of block 0:
          HF path:    encoder.block[0].layer[2].mlp.wi.weight
          Native path: encoder.block[0].ff.wi.weight
        """
        native_wi = native.encoder.block[0].ff.wi.weight.detach().cpu().float()
        hf_wi = wrapper.model.encoder.block[0].layer[2].mlp.wi.weight.detach().cpu().float()
        torch.testing.assert_close(native_wi, hf_wi, rtol=1e-5, atol=1e-5)

    def test_ff_wo_weight_matches_hf(self, native, wrapper):
        """encoder.block[0].layer[2].mlp.wo  →  encoder.block[0].ff.wo"""
        native_wo = native.encoder.block[0].ff.wo.weight.detach().cpu().float()
        hf_wo = wrapper.model.encoder.block[0].layer[2].mlp.wo.weight.detach().cpu().float()
        torch.testing.assert_close(native_wo, hf_wo, rtol=1e-5, atol=1e-5)

    def test_time_attn_layer_norm_matches_hf(self, native, wrapper):
        """
        Time-attention layer norm of block 0:
          HF path:    encoder.block[0].layer[0].layer_norm.weight
          Native path: encoder.block[0].time_attn.layer_norm.weight
        """
        native_ln = native.encoder.block[0].time_attn.layer_norm.weight.detach().cpu().float()
        hf_ln = wrapper.model.encoder.block[0].layer[0].layer_norm.weight.detach().cpu().float()
        torch.testing.assert_close(native_ln, hf_ln, rtol=1e-5, atol=1e-5)

    def test_final_layer_norm_matches_hf(self, native, wrapper):
        """encoder.final_layer_norm weights match."""
        native_flnw = native.encoder.final_layer_norm.weight.detach().cpu().float()
        hf_flnw = wrapper.model.encoder.final_layer_norm.weight.detach().cpu().float()
        torch.testing.assert_close(native_flnw, hf_flnw, rtol=1e-5, atol=1e-5)

    def test_no_rope_buffers_in_params(self, native):
        """RoPE inv_freq buffers must NOT appear as trainable parameters."""
        param_names = [n for n, _ in native.named_parameters()]
        assert not any("rope_embed.inv_freq" in n for n in param_names)


# ---------------------------------------------------------------------------
# Phase 4 — Single series forward pass
# ---------------------------------------------------------------------------

class TestSingleSeriesForward:
    """Verify basic correctness of a single-series forward through the vLLM engine."""

    def test_output_shape(self, engine):
        context = torch.randn(100)
        flat = _engine_request(engine, "ss-shape", context, DEFAULT_PRED_LEN)
        nop = _nop(DEFAULT_PRED_LEN)
        expected_numel = NUM_QUANTILES * nop * OUTPUT_PATCH_SIZE
        assert flat.numel() == expected_numel, (
            f"Expected {expected_numel} elements, got {flat.numel()}"
        )

    def test_decoded_shape(self, engine):
        context = torch.randn(100)
        flat = _engine_request(engine, "ss-decoded", context, DEFAULT_PRED_LEN)
        preds = _decode_embedding(flat, prediction_length=DEFAULT_PRED_LEN)
        assert preds.shape == (1, NUM_QUANTILES, DEFAULT_PRED_LEN)

    def test_output_all_finite(self, engine):
        context = torch.randn(100)
        flat = _engine_request(engine, "ss-finite", context, DEFAULT_PRED_LEN)
        assert torch.isfinite(flat).all(), "Output contains NaN or Inf"

    def test_quantiles_monotonically_ordered(self, engine):
        """Lower quantile index → lower or equal predicted value at each time step."""
        context = torch.randn(100)
        flat = _engine_request(engine, "ss-order", context, DEFAULT_PRED_LEN)
        preds = _decode_embedding(flat, prediction_length=DEFAULT_PRED_LEN)
        # preds: (1, 21, 24) — dim 1 is quantile level, ascending
        for q in range(NUM_QUANTILES - 1):
            violations = (preds[0, q] > preds[0, q + 1] + 1e-3).sum().item()
            assert violations == 0, (
                f"Quantile ordering violated at q={q} vs q={q+1}: {violations} steps"
            )

    def test_deterministic_output(self, engine):
        """Same context submitted twice produces identical outputs."""
        torch.manual_seed(0)
        context = torch.randn(80)
        flat1 = _engine_request(engine, "ss-det-1", context.clone(), DEFAULT_PRED_LEN)
        flat2 = _engine_request(engine, "ss-det-2", context.clone(), DEFAULT_PRED_LEN)
        torch.testing.assert_close(flat1, flat2)

    def test_different_inputs_differ(self, engine):
        """Different contexts produce different outputs."""
        flat1 = _engine_request(engine, "ss-diff-1", torch.randn(80), DEFAULT_PRED_LEN)
        flat2 = _engine_request(engine, "ss-diff-2", torch.randn(80), DEFAULT_PRED_LEN)
        assert not torch.allclose(flat1, flat2), "Different inputs produced identical output"


# ---------------------------------------------------------------------------
# Phase 5 — Batch and variable-length contexts
# ---------------------------------------------------------------------------

class TestBatchForward:
    """Multiple requests at different context lengths and prediction horizons."""

    def test_four_requests_in_one_step(self, engine):
        try:
            from vllm.sampling_params import PoolingParams
        except ImportError:
            from vllm import PoolingParams

        for i in range(4):
            ctx = torch.randn(60 + i * 20)  # lengths: 60, 80, 100, 120
            engine.add_request(
                request_id=f"batch4-{i}",
                inputs=_make_mm_input(ctx, DEFAULT_PRED_LEN),
                params=PoolingParams(),
            )
        outputs = engine.step()
        matched = [o for o in outputs if o.request_id.startswith("batch4-")]
        assert len(matched) == 4
        for out in matched:
            assert out.finished
            assert torch.isfinite(out.outputs.embedding).all()

    def test_non_multiple_of_patch_size(self, engine):
        """pred_len=17 (not a multiple of 16) → correct decoded shape."""
        pred_len = 17
        flat = _engine_request(engine, "batch-nm", torch.randn(100), pred_len)
        preds = _decode_embedding(flat, prediction_length=pred_len)
        assert preds.shape == (1, NUM_QUANTILES, pred_len)

    def test_minimal_context_one_patch(self, engine):
        """Context of exactly one patch length (16 timesteps) runs without error."""
        ctx = torch.randn(OUTPUT_PATCH_SIZE)
        flat = _engine_request(engine, "batch-min", ctx, DEFAULT_PRED_LEN)
        assert torch.isfinite(flat).all()

    def test_prediction_length_one(self, engine):
        flat = _engine_request(engine, "batch-pl1", torch.randn(50), 1)
        preds = _decode_embedding(flat, prediction_length=1)
        assert preds.shape == (1, NUM_QUANTILES, 1)
        assert torch.isfinite(preds).all()

    def test_large_prediction_length(self, engine):
        """pred_len=512 runs and returns valid output."""
        pred_len = 512
        flat = _engine_request(engine, "batch-lpl", torch.randn(200), pred_len)
        preds = _decode_embedding(flat, prediction_length=pred_len)
        assert preds.shape == (1, NUM_QUANTILES, pred_len)
        assert torch.isfinite(preds).all()


# ---------------------------------------------------------------------------
# Phase 6 — Numerical equivalence: native vs Chronos2ForVLLM wrapper
# ---------------------------------------------------------------------------

class TestNumericalEquivalence:
    """
    Native Chronos2ForForecasting and the Chronos2ForVLLM wrapper share the same
    checkpoint. Their outputs should agree within tolerance caused by:
      - QKVParallelLinear (fused) vs. separate HF q/k/v linear layers
      - GPU float32 operation order differences

    Using rtol=1e-3, atol=1e-3 (tighter than bfloat16 tolerance).
    """

    RTOL = 1e-3
    ATOL = 1e-3

    def _wrapper_preds(self, wrapper, context: torch.Tensor, pred_len: int) -> torch.Tensor:
        """Run wrapper model, return (1, num_quantiles, pred_len) CPU float32 tensor."""
        ctx = context.unsqueeze(0).cuda().float()
        with torch.no_grad():
            out = wrapper.forward_forecast(ctx, prediction_length=pred_len)
        return out.quantile_preds.cpu().float()

    def test_single_series_close(self, engine, wrapper):
        torch.manual_seed(42)
        context = torch.randn(100)

        flat = _engine_request(engine, "equiv-single", context, DEFAULT_PRED_LEN)
        native_preds = _decode_embedding(flat.float(), prediction_length=DEFAULT_PRED_LEN)
        wrapper_preds = self._wrapper_preds(wrapper, context, DEFAULT_PRED_LEN)

        torch.testing.assert_close(
            native_preds, wrapper_preds, rtol=self.RTOL, atol=self.ATOL,
        )

    def test_median_quantile_matches(self, engine, wrapper):
        """0.5 quantile (index 10 of 21) matches between native and wrapper."""
        torch.manual_seed(99)
        context = torch.randn(64)

        flat = _engine_request(engine, "equiv-median", context, DEFAULT_PRED_LEN)
        native_median = _decode_embedding(flat.float(), prediction_length=DEFAULT_PRED_LEN)[0, 10, :]
        wrapper_median = self._wrapper_preds(wrapper, context, DEFAULT_PRED_LEN)[0, 10, :]

        torch.testing.assert_close(native_median, wrapper_median, rtol=self.RTOL, atol=self.ATOL)

    def test_extreme_quantiles_match(self, engine, wrapper):
        """q=0.01 (index 0) and q=0.99 (index 20) both match."""
        torch.manual_seed(7)
        context = torch.randn(80)

        flat = _engine_request(engine, "equiv-extreme", context, DEFAULT_PRED_LEN)
        native_preds = _decode_embedding(flat.float(), prediction_length=DEFAULT_PRED_LEN)
        wrapper_preds = self._wrapper_preds(wrapper, context, DEFAULT_PRED_LEN)

        torch.testing.assert_close(
            native_preds[0, 0, :], wrapper_preds[0, 0, :], rtol=self.RTOL, atol=self.ATOL
        )
        torch.testing.assert_close(
            native_preds[0, -1, :], wrapper_preds[0, -1, :], rtol=self.RTOL, atol=self.ATOL
        )

    def test_nan_context_equivalence(self, engine, wrapper):
        """Gap of NaN values in context is handled identically by both."""
        torch.manual_seed(13)
        context = torch.randn(80)
        context[25:35] = float("nan")

        flat = _engine_request(engine, "equiv-nan", context, DEFAULT_PRED_LEN)
        native_preds = _decode_embedding(flat.float(), prediction_length=DEFAULT_PRED_LEN)
        wrapper_preds = self._wrapper_preds(wrapper, context, DEFAULT_PRED_LEN)

        assert torch.isfinite(native_preds).all(), "Native output has NaN/Inf for gapped input"
        torch.testing.assert_close(
            native_preds, wrapper_preds, rtol=self.RTOL, atol=self.ATOL,
        )

    def test_short_context_equivalence(self, engine, wrapper):
        """Very short context (32 timesteps) matches."""
        torch.manual_seed(21)
        context = torch.randn(32)

        flat = _engine_request(engine, "equiv-short", context, DEFAULT_PRED_LEN)
        native_preds = _decode_embedding(flat.float(), prediction_length=DEFAULT_PRED_LEN)
        wrapper_preds = self._wrapper_preds(wrapper, context, DEFAULT_PRED_LEN)

        torch.testing.assert_close(
            native_preds, wrapper_preds, rtol=self.RTOL, atol=self.ATOL,
        )


# ---------------------------------------------------------------------------
# Phase 7 — Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_all_nan_context_does_not_crash(self, engine):
        """All-NaN context: model should not crash (output may be degenerate)."""
        context = torch.full((64,), float("nan"))
        flat = _engine_request(engine, "edge-allnan", context, DEFAULT_PRED_LEN)
        # We do not assert finite here — all-NaN is degenerate — just no crash
        assert flat is not None

    def test_context_exceeds_context_length(self, engine):
        """Context longer than context_length (8192) is silently truncated."""
        context = torch.randn(10_000)
        flat = _engine_request(engine, "edge-long", context, DEFAULT_PRED_LEN)
        preds = _decode_embedding(flat, prediction_length=DEFAULT_PRED_LEN)
        assert preds.shape == (1, NUM_QUANTILES, DEFAULT_PRED_LEN)
        assert torch.isfinite(preds).all()

    def test_constant_series(self, engine):
        """All-constant series (zero variance) runs without division-by-zero."""
        context = torch.ones(100) * 42.0
        flat = _engine_request(engine, "edge-const", context, DEFAULT_PRED_LEN)
        assert torch.isfinite(flat).all()

    def test_very_large_values(self, engine):
        """Context with very large values is handled (instance norm should scale)."""
        context = torch.randn(100) * 1e6
        flat = _engine_request(engine, "edge-large", context, DEFAULT_PRED_LEN)
        assert torch.isfinite(flat).all()

    def test_negative_series(self, engine):
        """All-negative context runs correctly."""
        context = -torch.abs(torch.randn(100))
        flat = _engine_request(engine, "edge-neg", context, DEFAULT_PRED_LEN)
        assert torch.isfinite(flat).all()


# ---------------------------------------------------------------------------
# Phase 8 — bfloat16
# ---------------------------------------------------------------------------

class TestBfloat16:
    """
    bfloat16 tests auto-skip on pre-Ampere GPUs (compute capability < 8.0).
    Requires a separate bf16 engine fixture.
    """

    def test_bf16_forward_runs(self, bf16_engine):
        context = torch.randn(100)
        flat = _engine_request(bf16_engine, "bf16-fwd", context, DEFAULT_PRED_LEN)
        assert torch.isfinite(flat).all()

    def test_bf16_decoded_shape(self, bf16_engine):
        context = torch.randn(80)
        flat = _engine_request(bf16_engine, "bf16-shape", context, DEFAULT_PRED_LEN)
        preds = _decode_embedding(flat.float(), prediction_length=DEFAULT_PRED_LEN)
        assert preds.shape == (1, NUM_QUANTILES, DEFAULT_PRED_LEN)

    def test_bf16_quantile_ordering(self, bf16_engine):
        context = torch.randn(80)
        flat = _engine_request(bf16_engine, "bf16-order", context, DEFAULT_PRED_LEN)
        preds = _decode_embedding(flat.float(), prediction_length=DEFAULT_PRED_LEN)
        for q in range(NUM_QUANTILES - 1):
            assert (preds[0, q] <= preds[0, q + 1] + 1e-2).all()

    def test_bf16_close_to_fp32(self, engine, bf16_engine):
        """bfloat16 and float32 results are within bfloat16's natural tolerance (~1%)."""
        torch.manual_seed(55)
        context = torch.randn(100)

        flat_fp32 = _engine_request(engine, "bf16-cmp-fp32", context.clone(), DEFAULT_PRED_LEN)
        flat_bf16 = _engine_request(bf16_engine, "bf16-cmp-bf16", context.clone(), DEFAULT_PRED_LEN)

        preds_fp32 = _decode_embedding(flat_fp32.float(), prediction_length=DEFAULT_PRED_LEN)
        preds_bf16 = _decode_embedding(flat_bf16.float(), prediction_length=DEFAULT_PRED_LEN)

        torch.testing.assert_close(preds_fp32, preds_bf16, rtol=0.02, atol=0.1)


# ---------------------------------------------------------------------------
# Phase 9 — Throughput benchmark
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestThroughput:
    """
    Performance benchmarks. Skipped by default; enable with:
        pytest test/test_vllm_native.py -m slow -v
    """

    WARMUP = 3
    MEASURE = 10
    CONTEXT_LEN = 100

    def _batch_step(self, engine, batch_size: int):
        try:
            from vllm.sampling_params import PoolingParams
        except ImportError:
            from vllm import PoolingParams

        for i in range(batch_size):
            engine.add_request(
                request_id=f"tput-{i}",
                inputs=_make_mm_input(torch.randn(self.CONTEXT_LEN), DEFAULT_PRED_LEN),
                params=PoolingParams(),
            )
        outputs = engine.step()
        assert len(outputs) == batch_size

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_requests_per_second(self, engine, batch_size):
        # Warmup
        for _ in range(self.WARMUP):
            self._batch_step(engine, batch_size)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(self.MEASURE):
            self._batch_step(engine, batch_size)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total = batch_size * self.MEASURE
        rps = total / elapsed
        ms_per = elapsed / total * 1000

        print(f"\n[Throughput] bs={batch_size:>2}: {rps:7.1f} req/s  ({ms_per:.1f} ms/req)")

        # Very conservative lower bound: any GPU should beat 1 req/s
        assert rps >= 1.0, f"Throughput too low: {rps:.2f} req/s"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
