# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Chronos-2 vLLM postprocessing.
"""

import numpy as np
import pytest
import torch

from chronos.chronos2.model import Chronos2Output
from chronos.chronos2.vllm.postprocessing import (
    ForecastOutput,
    compute_prediction_intervals,
    extract_median,
    extract_point_prediction,
    format_for_json,
    postprocess_chronos2_output,
    quantiles_to_dict,
    samples_from_quantiles,
    to_numpy,
    truncate_predictions,
)


# Standard quantile levels used by Chronos-2
STANDARD_QUANTILES = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99
]


@pytest.fixture
def sample_quantile_preds() -> torch.Tensor:
    """Create sample quantile predictions."""
    batch_size = 4
    num_quantiles = len(STANDARD_QUANTILES)
    prediction_length = 32
    return torch.randn(batch_size, num_quantiles, prediction_length)


@pytest.fixture
def sample_chronos2_output(sample_quantile_preds) -> Chronos2Output:
    """Create sample Chronos2Output."""
    return Chronos2Output(
        loss=None,
        quantile_preds=sample_quantile_preds,
        enc_time_self_attn_weights=None,
        enc_group_self_attn_weights=None,
    )


class TestTruncatePredictions:
    """Tests for truncate_predictions function."""

    def test_truncates_correctly(self, sample_quantile_preds):
        """Test that predictions are truncated to exact length."""
        result = truncate_predictions(sample_quantile_preds, prediction_length=16)
        assert result.shape[-1] == 16

    def test_preserves_batch_and_quantiles(self, sample_quantile_preds):
        """Test that batch and quantile dimensions are preserved."""
        result = truncate_predictions(sample_quantile_preds, prediction_length=16)
        assert result.shape[0] == sample_quantile_preds.shape[0]
        assert result.shape[1] == sample_quantile_preds.shape[1]

    def test_no_truncation_when_shorter(self, sample_quantile_preds):
        """Test behavior when requested length >= actual length."""
        result = truncate_predictions(sample_quantile_preds, prediction_length=100)
        assert result.shape[-1] == 32  # Original length


class TestExtractMedian:
    """Tests for extract_median function."""

    def test_extracts_correct_quantile(self, sample_quantile_preds):
        """Test that correct quantile is extracted."""
        result = extract_median(sample_quantile_preds, STANDARD_QUANTILES)
        median_idx = STANDARD_QUANTILES.index(0.5)
        expected = sample_quantile_preds[:, median_idx, :]
        torch.testing.assert_close(result, expected)

    def test_output_shape(self, sample_quantile_preds):
        """Test output shape is (batch_size, prediction_length)."""
        result = extract_median(sample_quantile_preds, STANDARD_QUANTILES)
        assert result.shape == (4, 32)

    def test_raises_if_no_median(self, sample_quantile_preds):
        """Test error when 0.5 not in quantile levels."""
        quantiles_without_median = [0.1, 0.25, 0.75, 0.9]
        with pytest.raises(ValueError, match="Median.*not found"):
            extract_median(sample_quantile_preds[:, :4, :], quantiles_without_median)


class TestExtractPointPrediction:
    """Tests for extract_point_prediction function."""

    def test_median_method(self, sample_quantile_preds):
        """Test median extraction method."""
        result = extract_point_prediction(
            sample_quantile_preds, STANDARD_QUANTILES, method="median"
        )
        expected = extract_median(sample_quantile_preds, STANDARD_QUANTILES)
        torch.testing.assert_close(result, expected)

    def test_mean_method(self, sample_quantile_preds):
        """Test mean extraction method."""
        result = extract_point_prediction(
            sample_quantile_preds, STANDARD_QUANTILES, method="mean"
        )
        expected = sample_quantile_preds.mean(dim=1)
        torch.testing.assert_close(result, expected)

    def test_invalid_method_raises(self, sample_quantile_preds):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown point prediction method"):
            extract_point_prediction(
                sample_quantile_preds, STANDARD_QUANTILES, method="invalid"
            )


class TestQuantilesToDict:
    """Tests for quantiles_to_dict function."""

    def test_includes_all_quantiles(self, sample_quantile_preds):
        """Test that all quantiles are included."""
        result = quantiles_to_dict(sample_quantile_preds, STANDARD_QUANTILES)
        for q in STANDARD_QUANTILES:
            key = f"q{q:.2f}"
            assert key in result

    def test_includes_aliases(self, sample_quantile_preds):
        """Test that common aliases are included."""
        result = quantiles_to_dict(sample_quantile_preds, STANDARD_QUANTILES)
        assert "median" in result
        assert "lower_80" in result  # q0.10
        assert "upper_80" in result  # q0.90
        assert "lower_90" in result  # q0.05
        assert "upper_90" in result  # q0.95
        assert "lower_50" in result  # q0.25
        assert "upper_50" in result  # q0.75

    def test_selected_quantiles(self, sample_quantile_preds):
        """Test filtering to selected quantiles."""
        selected = [0.1, 0.5, 0.9]
        result = quantiles_to_dict(
            sample_quantile_preds,
            STANDARD_QUANTILES,
            include_all=False,
            selected_quantiles=selected,
        )
        assert "q0.10" in result
        assert "q0.50" in result
        assert "q0.90" in result
        assert "q0.30" not in result  # Not selected


class TestToNumpy:
    """Tests for to_numpy function."""

    def test_converts_tensor(self):
        """Test tensor to numpy conversion."""
        tensor = torch.randn(4, 16)
        result = to_numpy(tensor)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 16)

    def test_handles_gpu_tensor(self):
        """Test handles tensors with gradients."""
        tensor = torch.randn(4, 16, requires_grad=True)
        result = to_numpy(tensor)
        assert isinstance(result, np.ndarray)


class TestPostprocessChronos2Output:
    """Tests for postprocess_chronos2_output function."""

    def test_basic_postprocessing(self, sample_chronos2_output):
        """Test basic postprocessing flow."""
        result = postprocess_chronos2_output(
            sample_chronos2_output,
            quantile_levels=STANDARD_QUANTILES,
            prediction_length=24,
        )
        assert isinstance(result, ForecastOutput)
        assert result.predictions.shape == (4, 24)
        assert result.prediction_length == 24
        assert result.batch_size == 4

    def test_truncates_predictions(self, sample_chronos2_output):
        """Test that predictions are truncated."""
        result = postprocess_chronos2_output(
            sample_chronos2_output,
            quantile_levels=STANDARD_QUANTILES,
            prediction_length=16,
        )
        assert result.predictions.shape[-1] == 16
        for v in result.quantiles.values():
            assert v.shape[-1] == 16

    def test_quantiles_dict_populated(self, sample_chronos2_output):
        """Test that quantiles dictionary is populated."""
        result = postprocess_chronos2_output(
            sample_chronos2_output,
            quantile_levels=STANDARD_QUANTILES,
            prediction_length=24,
        )
        assert len(result.quantiles) > 0
        assert "median" in result.quantiles

    def test_raises_on_none_preds(self):
        """Test error when quantile_preds is None."""
        output = Chronos2Output(loss=None, quantile_preds=None)
        with pytest.raises(ValueError, match="does not contain quantile predictions"):
            postprocess_chronos2_output(output, STANDARD_QUANTILES, prediction_length=24)


class TestFormatForJson:
    """Tests for format_for_json function."""

    def test_basic_format(self, sample_chronos2_output):
        """Test basic JSON formatting."""
        forecast = postprocess_chronos2_output(
            sample_chronos2_output,
            quantile_levels=STANDARD_QUANTILES,
            prediction_length=24,
        )
        result = format_for_json(forecast)

        assert "predictions" in result
        assert "prediction_length" in result
        assert "batch_size" in result
        assert result["prediction_length"] == 24
        assert result["batch_size"] == 4

    def test_as_list_format(self, sample_chronos2_output):
        """Test that as_list produces nested lists."""
        forecast = postprocess_chronos2_output(
            sample_chronos2_output,
            quantile_levels=STANDARD_QUANTILES,
            prediction_length=24,
        )
        result = format_for_json(forecast, as_list=True)

        assert isinstance(result["predictions"], list)
        assert len(result["predictions"]) == 4
        assert len(result["predictions"][0]) == 24

    def test_exclude_quantiles(self, sample_chronos2_output):
        """Test excluding quantiles from output."""
        forecast = postprocess_chronos2_output(
            sample_chronos2_output,
            quantile_levels=STANDARD_QUANTILES,
            prediction_length=24,
        )
        result = format_for_json(forecast, include_quantiles=False)

        assert "quantiles" not in result


class TestSamplesFromQuantiles:
    """Tests for samples_from_quantiles function."""

    def test_output_shape(self, sample_quantile_preds):
        """Test output shape is correct."""
        num_samples = 50
        result = samples_from_quantiles(
            sample_quantile_preds,
            STANDARD_QUANTILES,
            num_samples=num_samples,
        )
        assert result.shape == (4, num_samples, 32)

    def test_samples_within_range(self, sample_quantile_preds):
        """Test that samples are roughly within quantile range."""
        # Create ordered quantile predictions
        batch_size, num_quantiles, pred_len = 2, len(STANDARD_QUANTILES), 16
        ordered_preds = torch.zeros(batch_size, num_quantiles, pred_len)
        for i, q in enumerate(STANDARD_QUANTILES):
            ordered_preds[:, i, :] = q * 10  # Values from 0.1 to 9.9

        result = samples_from_quantiles(
            ordered_preds, STANDARD_QUANTILES, num_samples=100
        )

        # Samples should be roughly between min and max quantile values
        assert result.min() >= 0.0  # Close to q0.01 * 10
        assert result.max() <= 10.0  # Close to q0.99 * 10


class TestComputePredictionIntervals:
    """Tests for compute_prediction_intervals function."""

    def test_90_percent_interval(self, sample_quantile_preds):
        """Test 90% prediction interval."""
        lower, upper = compute_prediction_intervals(
            sample_quantile_preds,
            STANDARD_QUANTILES,
            coverage=0.9,
        )
        # For 90% coverage: lower should be q0.05, upper should be q0.95
        assert lower.shape == (4, 32)
        assert upper.shape == (4, 32)

    def test_80_percent_interval(self, sample_quantile_preds):
        """Test 80% prediction interval."""
        lower, upper = compute_prediction_intervals(
            sample_quantile_preds,
            STANDARD_QUANTILES,
            coverage=0.8,
        )
        # For 80% coverage: lower should be q0.10, upper should be q0.90
        assert lower.shape == (4, 32)
        assert upper.shape == (4, 32)


class TestForecastOutput:
    """Tests for ForecastOutput dataclass."""

    def test_creation(self):
        """Test dataclass creation."""
        output = ForecastOutput(
            predictions=torch.randn(4, 24),
            quantiles={"median": torch.randn(4, 24)},
            prediction_length=24,
            batch_size=4,
            model_quantiles=STANDARD_QUANTILES,
        )
        assert output.prediction_length == 24
        assert output.batch_size == 4
        assert len(output.model_quantiles) == 21


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
