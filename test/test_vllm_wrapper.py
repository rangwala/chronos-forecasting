# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Chronos-2 vLLM wrapper.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from chronos import BaseChronosPipeline
from chronos.chronos2.model import Chronos2Model
from chronos.chronos2.vllm import Chronos2ForVLLM, Chronos2VLLMConfig


DUMMY_MODEL_PATH = Path(__file__).parent / "dummy-chronos2-model"


@pytest.fixture
def dummy_model() -> Chronos2Model:
    """Load the dummy Chronos-2 model for testing."""
    pipeline = BaseChronosPipeline.from_pretrained(DUMMY_MODEL_PATH, device_map="cpu")
    return pipeline.model


@pytest.fixture
def vllm_wrapper(dummy_model: Chronos2Model) -> Chronos2ForVLLM:
    """Create vLLM wrapper from dummy model."""
    return Chronos2ForVLLM(model=dummy_model)


@pytest.fixture
def vllm_wrapper_from_pretrained() -> Chronos2ForVLLM:
    """Load vLLM wrapper directly from pretrained path."""
    return Chronos2ForVLLM.from_pretrained(DUMMY_MODEL_PATH, device_map="cpu")


class TestChronos2ForVLLMLoading:
    """Tests for model loading and initialization."""

    def test_wrapper_initializes_from_model(self, dummy_model: Chronos2Model):
        """Test that wrapper can be initialized from an existing model."""
        wrapper = Chronos2ForVLLM(model=dummy_model)
        assert wrapper.model is dummy_model
        assert wrapper.vllm_config is not None

    def test_wrapper_loads_from_pretrained(self, vllm_wrapper_from_pretrained: Chronos2ForVLLM):
        """Test that wrapper can load from pretrained path."""
        wrapper = vllm_wrapper_from_pretrained
        assert wrapper.model is not None
        assert isinstance(wrapper.model, Chronos2Model)

    def test_wrapper_accepts_custom_config(self, dummy_model: Chronos2Model):
        """Test that wrapper accepts custom vLLM config."""
        config = Chronos2VLLMConfig(
            prediction_length=48,
            batch_size=64,
            cross_learning=True,
        )
        wrapper = Chronos2ForVLLM(model=dummy_model, vllm_config=config)
        assert wrapper.vllm_config.prediction_length == 48
        assert wrapper.vllm_config.batch_size == 64
        assert wrapper.vllm_config.cross_learning is True


class TestChronos2ForVLLMProperties:
    """Tests for vLLM-required properties."""

    def test_is_pooling_model_true(self, vllm_wrapper: Chronos2ForVLLM):
        """Test that is_pooling_model returns True (encoder-only)."""
        assert vllm_wrapper.is_pooling_model is True

    def test_supports_pp_false(self, vllm_wrapper: Chronos2ForVLLM):
        """Test that pipeline parallelism is not supported."""
        assert vllm_wrapper.supports_pp is False

    def test_is_attention_free_false(self, vllm_wrapper: Chronos2ForVLLM):
        """Test that model has attention layers."""
        assert vllm_wrapper.is_attention_free is False

    def test_config_accessible(self, vllm_wrapper: Chronos2ForVLLM):
        """Test that model config is accessible."""
        assert vllm_wrapper.config is not None
        assert hasattr(vllm_wrapper.config, "d_model")

    def test_chronos_config_accessible(self, vllm_wrapper: Chronos2ForVLLM):
        """Test that Chronos config is accessible."""
        assert vllm_wrapper.chronos_config is not None
        assert hasattr(vllm_wrapper.chronos_config, "context_length")
        assert hasattr(vllm_wrapper.chronos_config, "quantiles")

    def test_num_quantiles(self, vllm_wrapper: Chronos2ForVLLM):
        """Test that num_quantiles matches config."""
        with open(DUMMY_MODEL_PATH / "config.json") as f:
            config = json.load(f)
        expected_quantiles = len(config["chronos_config"]["quantiles"])
        assert vllm_wrapper.num_quantiles == expected_quantiles

    def test_context_length(self, vllm_wrapper: Chronos2ForVLLM):
        """Test that context_length is accessible."""
        assert vllm_wrapper.context_length > 0
        assert vllm_wrapper.context_length == 8192  # From dummy model config


class TestChronos2ForVLLMForward:
    """Tests for forward pass equivalence with original model."""

    def test_forward_runs_without_error(self, vllm_wrapper: Chronos2ForVLLM):
        """Test that forward pass executes without error."""
        context = torch.randn(2, 32)  # 2 series, 32 timesteps
        output = vllm_wrapper.forward(context, num_output_patches=1)
        assert output is not None
        assert output.quantile_preds is not None

    def test_forward_output_shape(self, vllm_wrapper: Chronos2ForVLLM):
        """Test that forward pass produces correct output shape."""
        batch_size = 4
        context_length = 64
        num_output_patches = 2
        context = torch.randn(batch_size, context_length)

        output = vllm_wrapper.forward(context, num_output_patches=num_output_patches)

        expected_pred_length = num_output_patches * vllm_wrapper.output_patch_size
        assert output.quantile_preds.shape == (
            batch_size,
            vllm_wrapper.num_quantiles,
            expected_pred_length,
        )

    def test_forward_matches_original_model(
        self, vllm_wrapper: Chronos2ForVLLM, dummy_model: Chronos2Model
    ):
        """Test that wrapper forward matches original model forward."""
        torch.manual_seed(42)
        context = torch.randn(3, 48)
        num_output_patches = 1

        # Run through wrapper
        wrapper_output = vllm_wrapper.forward(context, num_output_patches=num_output_patches)

        # Run through original model
        original_output = dummy_model.forward(context, num_output_patches=num_output_patches)

        # Outputs should be identical
        torch.testing.assert_close(
            wrapper_output.quantile_preds,
            original_output.quantile_preds,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_forward_with_mask(self, vllm_wrapper: Chronos2ForVLLM):
        """Test forward pass with context mask."""
        context = torch.randn(2, 32)
        mask = torch.ones_like(context)
        mask[:, -5:] = 0  # Mask last 5 timesteps

        output = vllm_wrapper.forward(context, context_mask=mask, num_output_patches=1)
        assert output.quantile_preds is not None

    def test_forward_with_group_ids(self, vllm_wrapper: Chronos2ForVLLM):
        """Test forward pass with group IDs for cross-learning."""
        context = torch.randn(4, 32)
        group_ids = torch.tensor([0, 0, 1, 1])  # Two groups of 2

        output = vllm_wrapper.forward(
            context, group_ids=group_ids, num_output_patches=1
        )
        assert output.quantile_preds is not None


class TestChronos2ForVLLMForecast:
    """Tests for the simplified forecasting interface."""

    def test_forward_forecast_basic(self, vllm_wrapper: Chronos2ForVLLM):
        """Test basic forward_forecast call."""
        context = torch.randn(2, 64)
        prediction_length = 24

        output = vllm_wrapper.forward_forecast(context, prediction_length=prediction_length)

        assert output.quantile_preds is not None
        assert output.quantile_preds.shape[-1] == prediction_length

    def test_forward_forecast_truncates_to_exact_length(self, vllm_wrapper: Chronos2ForVLLM):
        """Test that predictions are truncated to exact prediction_length."""
        context = torch.randn(2, 64)
        prediction_length = 17  # Not a multiple of patch size

        output = vllm_wrapper.forward_forecast(context, prediction_length=prediction_length)

        # Should be exactly prediction_length, not rounded up to patch size
        assert output.quantile_preds.shape[-1] == prediction_length

    def test_forward_forecast_cross_learning(self, vllm_wrapper: Chronos2ForVLLM):
        """Test cross-learning mode."""
        # Create wrapper with cross_learning enabled
        config = Chronos2VLLMConfig(cross_learning=True)
        wrapper = Chronos2ForVLLM(model=vllm_wrapper.model, vllm_config=config)

        context = torch.randn(4, 64)
        output = wrapper.forward_forecast(context, prediction_length=16)

        assert output.quantile_preds is not None

    def test_get_quantile_predictions(self, vllm_wrapper: Chronos2ForVLLM):
        """Test dictionary-based quantile predictions."""
        context = torch.randn(2, 64)
        prediction_length = 16

        result = vllm_wrapper.get_quantile_predictions(
            context, prediction_length=prediction_length
        )

        assert isinstance(result, dict)
        assert "median" in result
        assert result["median"].shape == (2, prediction_length)

    def test_get_quantile_predictions_filtered(self, vllm_wrapper: Chronos2ForVLLM):
        """Test filtering specific quantile levels."""
        context = torch.randn(2, 64)
        prediction_length = 16
        requested_quantiles = [0.1, 0.5, 0.9]

        result = vllm_wrapper.get_quantile_predictions(
            context,
            prediction_length=prediction_length,
            quantile_levels=requested_quantiles,
        )

        # Should only contain requested quantiles (plus median)
        assert "q0.10" in result
        assert "q0.50" in result
        assert "q0.90" in result
        assert "median" in result


class TestChronos2ForVLLMNumOutputPatches:
    """Tests for output patch computation."""

    def test_compute_num_output_patches_exact(self, vllm_wrapper: Chronos2ForVLLM):
        """Test patch computation for exact multiple of patch size."""
        patch_size = vllm_wrapper.output_patch_size
        prediction_length = patch_size * 3

        num_patches = vllm_wrapper._compute_num_output_patches(prediction_length)
        assert num_patches == 3

    def test_compute_num_output_patches_rounds_up(self, vllm_wrapper: Chronos2ForVLLM):
        """Test patch computation rounds up for non-exact multiples."""
        patch_size = vllm_wrapper.output_patch_size
        prediction_length = patch_size * 2 + 1  # Needs 3 patches

        num_patches = vllm_wrapper._compute_num_output_patches(prediction_length)
        assert num_patches == 3


class TestChronos2ForVLLMRepr:
    """Tests for string representation."""

    def test_repr(self, vllm_wrapper: Chronos2ForVLLM):
        """Test that repr returns informative string."""
        repr_str = repr(vllm_wrapper)
        assert "Chronos2ForVLLM" in repr_str
        assert "context_length" in repr_str
        assert "num_quantiles" in repr_str
        assert "is_pooling_model" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
