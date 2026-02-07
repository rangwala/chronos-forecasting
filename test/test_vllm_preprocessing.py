# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Chronos-2 vLLM preprocessing.
"""

import numpy as np
import pytest
import torch

from chronos.chronos2.vllm.preprocessing import (
    Chronos2Inputs,
    compute_num_output_patches,
    prepare_batch,
    prepare_context,
    prepare_future_covariates,
    prepare_group_ids,
    preprocess_for_chronos2,
    to_tensor,
    validate_context,
)


class TestToTensor:
    """Tests for to_tensor function."""

    def test_torch_tensor_passthrough(self):
        """Test that torch tensors are passed through."""
        t = torch.randn(10)
        result = to_tensor(t)
        assert torch.is_tensor(result)
        assert result.dtype == torch.float32

    def test_numpy_array_conversion(self):
        """Test numpy array conversion."""
        arr = np.random.randn(10)
        result = to_tensor(arr)
        assert torch.is_tensor(result)
        np.testing.assert_allclose(result.numpy(), arr, rtol=1e-6)

    def test_list_conversion(self):
        """Test list conversion."""
        lst = [1.0, 2.0, 3.0]
        result = to_tensor(lst)
        assert torch.is_tensor(result)
        assert result.tolist() == lst

    def test_dtype_specification(self):
        """Test dtype specification."""
        arr = np.array([1, 2, 3])
        result = to_tensor(arr, dtype=torch.float64)
        assert result.dtype == torch.float64


class TestValidateContext:
    """Tests for validate_context function."""

    def test_1d_valid(self):
        """Test 1D tensor is valid."""
        context = torch.randn(100)
        validate_context(context)  # Should not raise

    def test_2d_valid(self):
        """Test 2D tensor is valid."""
        context = torch.randn(4, 100)
        validate_context(context)  # Should not raise

    def test_3d_invalid(self):
        """Test 3D tensor raises error."""
        context = torch.randn(4, 100, 2)
        with pytest.raises(ValueError, match="1D.*or 2D"):
            validate_context(context)


class TestPrepareContext:
    """Tests for prepare_context function."""

    def test_adds_batch_dimension(self):
        """Test that 1D input gets batch dimension added."""
        context = torch.randn(100)
        result, mask = prepare_context(context)
        assert result.shape == (1, 100)
        assert mask.shape == (1, 100)

    def test_keeps_batch_dimension(self):
        """Test that 2D input keeps its shape."""
        context = torch.randn(4, 100)
        result, mask = prepare_context(context)
        assert result.shape == (4, 100)
        assert mask.shape == (4, 100)

    def test_truncates_to_max_length(self):
        """Test truncation to max context length."""
        context = torch.randn(4, 1000)
        result, mask = prepare_context(context, max_context_length=500)
        assert result.shape == (4, 500)

    def test_creates_mask_from_nan(self):
        """Test mask creation from NaN values."""
        context = torch.tensor([[1.0, float('nan'), 3.0, float('nan')]])
        result, mask = prepare_context(context)
        expected_mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        torch.testing.assert_close(mask, expected_mask)

    def test_replaces_nan_with_zero(self):
        """Test NaN values are replaced with zero."""
        context = torch.tensor([[1.0, float('nan'), 3.0]])
        result, _ = prepare_context(context)
        assert not torch.isnan(result).any()
        assert result[0, 1] == 0.0

    def test_moves_to_device(self):
        """Test device placement."""
        context = torch.randn(4, 100)
        result, mask = prepare_context(context, device=torch.device('cpu'))
        assert result.device == torch.device('cpu')


class TestPrepareBatch:
    """Tests for prepare_batch function."""

    def test_variable_length_sequences(self):
        """Test batching of variable length sequences."""
        contexts = [
            torch.randn(50),
            torch.randn(100),
            torch.randn(75),
        ]
        result, mask = prepare_batch(contexts)
        assert result.shape == (3, 100)  # Max length
        assert mask.shape == (3, 100)

    def test_left_padding(self):
        """Test that sequences are left-padded."""
        contexts = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0]),
        ]
        result, mask = prepare_batch(contexts)
        # First sequence: [1, 2, 3]
        # Second sequence: [0, 4, 5] (left-padded)
        assert result[0, 0] == 1.0
        assert result[1, 0] == 0.0  # Padding
        assert mask[1, 0] == 0.0  # Padding masked

    def test_truncates_long_sequences(self):
        """Test truncation of sequences longer than max."""
        contexts = [
            torch.randn(1000),
            torch.randn(500),
        ]
        result, mask = prepare_batch(contexts, max_context_length=200)
        assert result.shape == (2, 200)

    def test_handles_numpy_arrays(self):
        """Test handling of numpy arrays in list."""
        contexts = [
            np.random.randn(50),
            np.random.randn(75),
        ]
        result, mask = prepare_batch(contexts)
        assert result.shape == (2, 75)


class TestPrepareFutureCovariates:
    """Tests for prepare_future_covariates function."""

    def test_none_input(self):
        """Test None input returns None."""
        result, mask = prepare_future_covariates(None, batch_size=4, prediction_length=24)
        assert result is None
        assert mask is None

    def test_adds_batch_dimension(self):
        """Test batch dimension is added for 1D input."""
        future = torch.randn(24)
        result, mask = prepare_future_covariates(future, batch_size=1, prediction_length=24)
        assert result.shape == (1, 24)

    def test_validates_batch_size(self):
        """Test batch size validation."""
        future = torch.randn(2, 24)
        with pytest.raises(ValueError, match="batch size"):
            prepare_future_covariates(future, batch_size=4, prediction_length=24)

    def test_truncates_to_prediction_length(self):
        """Test truncation to prediction length."""
        future = torch.randn(4, 100)
        result, _ = prepare_future_covariates(future, batch_size=4, prediction_length=24)
        assert result.shape == (4, 24)


class TestPrepareGroupIds:
    """Tests for prepare_group_ids function."""

    def test_none_returns_none_without_cross_learning(self):
        """Test None input without cross_learning returns None."""
        result = prepare_group_ids(None, batch_size=4, cross_learning=False)
        assert result is None

    def test_cross_learning_creates_single_group(self):
        """Test cross_learning creates all-zeros group IDs."""
        result = prepare_group_ids(None, batch_size=4, cross_learning=True)
        assert result is not None
        assert result.shape == (4,)
        assert (result == 0).all()

    def test_validates_length(self):
        """Test group_ids length validation."""
        group_ids = torch.tensor([0, 0, 1])
        with pytest.raises(ValueError, match="doesn't match batch size"):
            prepare_group_ids(group_ids, batch_size=4)

    def test_converts_to_long_dtype(self):
        """Test conversion to long dtype."""
        group_ids = [0, 0, 1, 1]
        result = prepare_group_ids(group_ids, batch_size=4)
        assert result.dtype == torch.long


class TestComputeNumOutputPatches:
    """Tests for compute_num_output_patches function."""

    def test_exact_multiple(self):
        """Test exact multiple of patch size."""
        assert compute_num_output_patches(32, 16) == 2
        assert compute_num_output_patches(48, 16) == 3

    def test_rounds_up(self):
        """Test rounding up for non-multiples."""
        assert compute_num_output_patches(17, 16) == 2
        assert compute_num_output_patches(33, 16) == 3
        assert compute_num_output_patches(1, 16) == 1


class TestPreprocessForChronos2:
    """Tests for full preprocessing pipeline."""

    def test_basic_preprocessing(self):
        """Test basic preprocessing flow."""
        context = torch.randn(4, 100)
        result = preprocess_for_chronos2(
            context,
            prediction_length=24,
            output_patch_size=16,
        )
        assert isinstance(result, Chronos2Inputs)
        assert result.context.shape == (4, 100)
        assert result.context_mask.shape == (4, 100)
        assert result.num_output_patches == 2  # ceil(24/16)

    def test_variable_length_list(self):
        """Test preprocessing list of variable-length sequences."""
        contexts = [
            torch.randn(50),
            torch.randn(100),
            torch.randn(75),
        ]
        result = preprocess_for_chronos2(
            contexts,
            prediction_length=24,
            output_patch_size=16,
        )
        assert result.context.shape == (3, 100)

    def test_with_max_context_length(self):
        """Test max context length truncation."""
        context = torch.randn(4, 1000)
        result = preprocess_for_chronos2(
            context,
            prediction_length=24,
            output_patch_size=16,
            max_context_length=500,
        )
        assert result.context.shape == (4, 500)

    def test_with_future_covariates(self):
        """Test including future covariates."""
        context = torch.randn(4, 100)
        future = torch.randn(4, 24)
        result = preprocess_for_chronos2(
            context,
            prediction_length=24,
            output_patch_size=16,
            future_covariates=future,
        )
        assert result.future_covariates is not None
        assert result.future_covariates.shape == (4, 24)
        assert result.future_covariates_mask is not None

    def test_with_cross_learning(self):
        """Test cross-learning mode."""
        context = torch.randn(4, 100)
        result = preprocess_for_chronos2(
            context,
            prediction_length=24,
            output_patch_size=16,
            cross_learning=True,
        )
        assert result.group_ids is not None
        assert (result.group_ids == 0).all()

    def test_with_explicit_group_ids(self):
        """Test explicit group IDs."""
        context = torch.randn(4, 100)
        group_ids = torch.tensor([0, 0, 1, 1])
        result = preprocess_for_chronos2(
            context,
            prediction_length=24,
            output_patch_size=16,
            group_ids=group_ids,
        )
        torch.testing.assert_close(result.group_ids, group_ids)

    def test_with_numpy_input(self):
        """Test numpy array input."""
        context = np.random.randn(4, 100)
        result = preprocess_for_chronos2(
            context,
            prediction_length=24,
            output_patch_size=16,
        )
        assert torch.is_tensor(result.context)
        assert result.context.shape == (4, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
