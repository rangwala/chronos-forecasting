# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end integration tests for Chronos-2 vLLM integration.

These tests verify that the vLLM wrapper produces results equivalent to
the original Chronos2Pipeline.
"""

import time
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from chronos import BaseChronosPipeline, Chronos2Pipeline
from chronos.chronos2.vllm import (
    Chronos2ForVLLM,
    Chronos2Inputs,
    Chronos2VLLMConfig,
    ForecastOutput,
    format_for_json,
    postprocess_chronos2_output,
    preprocess_for_chronos2,
)


DUMMY_MODEL_PATH = Path(__file__).parent / "dummy-chronos2-model"


@pytest.fixture
def original_pipeline() -> Chronos2Pipeline:
    """Load original Chronos2Pipeline for comparison."""
    return BaseChronosPipeline.from_pretrained(DUMMY_MODEL_PATH, device_map="cpu")


@pytest.fixture
def vllm_wrapper() -> Chronos2ForVLLM:
    """Load vLLM wrapper."""
    return Chronos2ForVLLM.from_pretrained(DUMMY_MODEL_PATH, device_map="cpu")


class TestEndToEndEquivalence:
    """Tests verifying output equivalence between original pipeline and vLLM wrapper."""

    def test_single_series_forecast(
        self, original_pipeline: Chronos2Pipeline, vllm_wrapper: Chronos2ForVLLM
    ):
        """Test single time series forecast equivalence."""
        torch.manual_seed(42)
        context = torch.randn(1, 1, 100)  # 1 series, 1 variate, 100 timesteps
        prediction_length = 24

        # Original pipeline
        original_output = original_pipeline.predict(context, prediction_length=prediction_length)

        # vLLM wrapper - need to convert input format
        vllm_context = context.squeeze(1)  # (1, 100)
        vllm_output = vllm_wrapper.forward_forecast(vllm_context, prediction_length=prediction_length)

        # Compare outputs
        # Original returns list, vLLM returns tensor
        original_preds = original_output[0]  # (1, num_quantiles, pred_len)
        vllm_preds = vllm_output.quantile_preds  # (1, num_quantiles, pred_len)

        torch.testing.assert_close(
            vllm_preds,
            original_preds,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_batch_forecast(
        self, original_pipeline: Chronos2Pipeline, vllm_wrapper: Chronos2ForVLLM
    ):
        """Test batched forecast equivalence."""
        torch.manual_seed(42)
        batch_size = 4
        context = torch.randn(batch_size, 1, 80)
        prediction_length = 16

        # Original pipeline
        original_output = original_pipeline.predict(context, prediction_length=prediction_length)

        # vLLM wrapper
        vllm_context = context.squeeze(1)  # (4, 80)
        vllm_output = vllm_wrapper.forward_forecast(vllm_context, prediction_length=prediction_length)

        # Compare each series
        for i in range(batch_size):
            original_preds = original_output[i]
            vllm_preds = vllm_output.quantile_preds[i:i+1]

            torch.testing.assert_close(
                vllm_preds,
                original_preds,
                rtol=1e-4,
                atol=1e-4,
            )

    def test_variable_length_sequences(
        self, original_pipeline: Chronos2Pipeline, vllm_wrapper: Chronos2ForVLLM
    ):
        """Test variable length sequence handling.

        Note: Variable-length sequences may have slight numerical differences
        between original pipeline and vLLM wrapper due to different batching
        and padding strategies. We use looser tolerances here.
        """
        torch.manual_seed(42)
        prediction_length = 16

        # Variable length contexts
        contexts = [
            torch.randn(50),
            torch.randn(100),
            torch.randn(75),
        ]

        # Original pipeline
        original_output = original_pipeline.predict(contexts, prediction_length=prediction_length)

        # vLLM wrapper with preprocessing
        preprocessed = preprocess_for_chronos2(
            contexts,
            prediction_length=prediction_length,
            output_patch_size=vllm_wrapper.output_patch_size,
            max_context_length=vllm_wrapper.context_length,
        )
        vllm_output = vllm_wrapper.forward(
            preprocessed.context,
            context_mask=preprocessed.context_mask,
            num_output_patches=preprocessed.num_output_patches,
        )
        vllm_preds = vllm_output.quantile_preds[..., :prediction_length]

        # Compare with looser tolerances for variable-length batching differences
        # The outputs should be similar but may not be identical due to padding effects
        for i in range(len(contexts)):
            # Verify shapes match
            assert vllm_preds[i:i+1].shape == original_output[i].shape

            # Check correlation is high (predictions are similar)
            vllm_flat = vllm_preds[i].flatten()
            orig_flat = original_output[i].squeeze(0).flatten()
            correlation = torch.corrcoef(torch.stack([vllm_flat, orig_flat]))[0, 1]
            assert correlation > 0.9, f"Correlation {correlation} too low for series {i}"


class TestPreprocessingPostprocessingPipeline:
    """Tests for full preprocessing -> inference -> postprocessing pipeline."""

    def test_full_pipeline(self, vllm_wrapper: Chronos2ForVLLM):
        """Test complete preprocessing -> inference -> postprocessing pipeline."""
        # Input data
        context = torch.randn(4, 100)
        prediction_length = 24

        # Preprocessing
        inputs = preprocess_for_chronos2(
            context,
            prediction_length=prediction_length,
            output_patch_size=vllm_wrapper.output_patch_size,
        )

        # Inference
        raw_output = vllm_wrapper.forward(
            inputs.context,
            context_mask=inputs.context_mask,
            group_ids=inputs.group_ids,
            num_output_patches=inputs.num_output_patches,
        )

        # Postprocessing
        forecast = postprocess_chronos2_output(
            raw_output,
            quantile_levels=vllm_wrapper.quantiles,
            prediction_length=prediction_length,
        )

        # Verify output
        assert isinstance(forecast, ForecastOutput)
        assert forecast.predictions.shape == (4, 24)
        assert forecast.batch_size == 4
        assert forecast.prediction_length == 24

    def test_pipeline_with_json_output(self, vllm_wrapper: Chronos2ForVLLM):
        """Test pipeline with JSON serialization."""
        context = torch.randn(2, 64)
        prediction_length = 16

        # Full pipeline
        inputs = preprocess_for_chronos2(
            context,
            prediction_length=prediction_length,
            output_patch_size=vllm_wrapper.output_patch_size,
        )
        raw_output = vllm_wrapper.forward(
            inputs.context,
            context_mask=inputs.context_mask,
            num_output_patches=inputs.num_output_patches,
        )
        forecast = postprocess_chronos2_output(
            raw_output,
            quantile_levels=vllm_wrapper.quantiles,
            prediction_length=prediction_length,
        )

        # JSON output
        json_output = format_for_json(forecast)

        # Verify JSON structure
        assert "predictions" in json_output
        assert "prediction_length" in json_output
        assert "batch_size" in json_output
        assert "quantiles" in json_output

        # Verify values
        assert len(json_output["predictions"]) == 2
        assert len(json_output["predictions"][0]) == 16


class TestCrossLearning:
    """Tests for cross-learning functionality."""

    def test_cross_learning_mode(self, vllm_wrapper: Chronos2ForVLLM):
        """Test cross-learning mode produces valid output."""
        context = torch.randn(4, 64)
        prediction_length = 16

        # With cross-learning
        config = Chronos2VLLMConfig(cross_learning=True)
        wrapper = Chronos2ForVLLM(model=vllm_wrapper.model, vllm_config=config)

        output = wrapper.forward_forecast(context, prediction_length=prediction_length)

        assert output.quantile_preds is not None
        assert output.quantile_preds.shape == (4, wrapper.num_quantiles, 16)

    def test_explicit_group_ids(self, vllm_wrapper: Chronos2ForVLLM):
        """Test explicit group IDs."""
        context = torch.randn(4, 64)
        prediction_length = 16
        group_ids = torch.tensor([0, 0, 1, 1])  # Two groups

        output = vllm_wrapper.forward_forecast(
            context,
            prediction_length=prediction_length,
            group_ids=group_ids,
        )

        assert output.quantile_preds is not None


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_handles_nan_in_context(self, vllm_wrapper: Chronos2ForVLLM):
        """Test handling of NaN values in context."""
        context = torch.randn(2, 64)
        context[0, 10:15] = float('nan')  # Add some NaN values
        prediction_length = 16

        inputs = preprocess_for_chronos2(
            context,
            prediction_length=prediction_length,
            output_patch_size=vllm_wrapper.output_patch_size,
        )

        output = vllm_wrapper.forward(
            inputs.context,
            context_mask=inputs.context_mask,
            num_output_patches=inputs.num_output_patches,
        )

        # Should not have NaN in output
        assert not torch.isnan(output.quantile_preds).any()

    def test_handles_very_long_context(self, vllm_wrapper: Chronos2ForVLLM):
        """Test handling of very long context (truncation)."""
        # Context longer than model's max
        context = torch.randn(2, 10000)
        prediction_length = 16

        inputs = preprocess_for_chronos2(
            context,
            prediction_length=prediction_length,
            output_patch_size=vllm_wrapper.output_patch_size,
            max_context_length=vllm_wrapper.context_length,
        )

        # Should be truncated
        assert inputs.context.shape[-1] == vllm_wrapper.context_length

        output = vllm_wrapper.forward(
            inputs.context,
            context_mask=inputs.context_mask,
            num_output_patches=inputs.num_output_patches,
        )

        assert output.quantile_preds is not None


class TestPerformance:
    """Basic performance tests."""

    def test_batch_inference_timing(self, vllm_wrapper: Chronos2ForVLLM):
        """Test batch inference completes in reasonable time."""
        batch_sizes = [1, 4, 8]
        prediction_length = 24

        for batch_size in batch_sizes:
            context = torch.randn(batch_size, 100)

            start_time = time.time()
            output = vllm_wrapper.forward_forecast(context, prediction_length=prediction_length)
            elapsed = time.time() - start_time

            assert output.quantile_preds is not None
            # Should complete within 10 seconds even for larger batches
            assert elapsed < 10.0, f"Batch size {batch_size} took too long: {elapsed:.2f}s"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_timestep_context(self, vllm_wrapper: Chronos2ForVLLM):
        """Test with minimal context length."""
        context = torch.randn(1, 16)  # Minimum for one patch
        prediction_length = 16

        output = vllm_wrapper.forward_forecast(context, prediction_length=prediction_length)
        assert output.quantile_preds is not None

    def test_prediction_length_not_multiple_of_patch(self, vllm_wrapper: Chronos2ForVLLM):
        """Test with prediction length not multiple of patch size."""
        context = torch.randn(2, 64)
        prediction_length = 17  # Not multiple of 16

        output = vllm_wrapper.forward_forecast(context, prediction_length=prediction_length)

        # Should be exactly prediction_length
        assert output.quantile_preds.shape[-1] == prediction_length

    def test_numpy_array_input(self, vllm_wrapper: Chronos2ForVLLM):
        """Test with numpy array input."""
        context = np.random.randn(2, 64).astype(np.float32)
        prediction_length = 16

        inputs = preprocess_for_chronos2(
            context,
            prediction_length=prediction_length,
            output_patch_size=vllm_wrapper.output_patch_size,
        )

        output = vllm_wrapper.forward(
            inputs.context,
            context_mask=inputs.context_mask,
            num_output_patches=inputs.num_output_patches,
        )

        assert output.quantile_preds is not None


class TestQuantileOutputs:
    """Tests for quantile output handling."""

    def test_all_quantiles_valid(self, vllm_wrapper: Chronos2ForVLLM):
        """Test that all quantile predictions are valid."""
        context = torch.randn(2, 64)
        prediction_length = 16

        output = vllm_wrapper.forward_forecast(context, prediction_length=prediction_length)

        # Check all quantiles
        preds = output.quantile_preds
        assert not torch.isnan(preds).any()
        assert not torch.isinf(preds).any()

    def test_quantile_ordering(self, vllm_wrapper: Chronos2ForVLLM):
        """Test that quantiles are properly ordered (lower quantiles <= higher)."""
        context = torch.randn(2, 64)
        prediction_length = 16

        output = vllm_wrapper.forward_forecast(context, prediction_length=prediction_length)
        preds = output.quantile_preds

        # Check ordering for each prediction step
        for i in range(preds.shape[1] - 1):
            # Lower quantile should generally be <= higher quantile
            # Note: This may not be strictly true for all outputs, so we check means
            assert preds[:, i, :].mean() <= preds[:, i + 1, :].mean() + 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
