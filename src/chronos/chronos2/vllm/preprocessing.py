# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Input preprocessing for Chronos-2 vLLM integration.

This module handles the conversion of various input formats into the tensor format
expected by the Chronos-2 model.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


TensorLike = Union[torch.Tensor, np.ndarray, List[float]]


@dataclass
class Chronos2Inputs:
    """
    Preprocessed inputs for Chronos-2 model.

    Attributes
    ----------
    context : torch.Tensor
        Historical time series of shape (batch_size, context_length)
    context_mask : torch.Tensor, optional
        Binary mask indicating valid observations
    group_ids : torch.Tensor, optional
        Group IDs for cross-learning
    future_covariates : torch.Tensor, optional
        Known future values
    future_covariates_mask : torch.Tensor, optional
        Mask for future covariates
    num_output_patches : int
        Number of output patches to generate
    """
    context: torch.Tensor
    context_mask: Optional[torch.Tensor] = None
    group_ids: Optional[torch.Tensor] = None
    future_covariates: Optional[torch.Tensor] = None
    future_covariates_mask: Optional[torch.Tensor] = None
    num_output_patches: int = 1


def to_tensor(data: TensorLike, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert input data to torch tensor.

    Parameters
    ----------
    data : TensorLike
        Input data (tensor, numpy array, or list)
    dtype : torch.dtype
        Target data type

    Returns
    -------
    torch.Tensor
        Converted tensor
    """
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype=dtype)
    else:
        return torch.tensor(data, dtype=dtype)


def validate_context(context: torch.Tensor) -> None:
    """
    Validate context tensor format.

    Parameters
    ----------
    context : torch.Tensor
        Context tensor to validate

    Raises
    ------
    ValueError
        If context has invalid shape
    """
    if context.ndim == 1:
        pass  # Single time series, will be unsqueezed
    elif context.ndim == 2:
        pass  # Batch of time series
    else:
        raise ValueError(
            f"context must be 1D (single series) or 2D (batch), got shape {context.shape}"
        )


def prepare_context(
    context: TensorLike,
    max_context_length: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare context tensor and mask for model input.

    Parameters
    ----------
    context : TensorLike
        Historical time series data
    max_context_length : int, optional
        Maximum context length (truncates if exceeded)
    device : torch.device, optional
        Target device
    dtype : torch.dtype
        Target data type

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (context tensor, mask tensor)
    """
    context = to_tensor(context, dtype=dtype)
    validate_context(context)

    # Ensure batch dimension
    if context.ndim == 1:
        context = context.unsqueeze(0)

    # Move to device
    if device is not None:
        context = context.to(device)

    # Truncate if needed
    if max_context_length is not None and context.shape[-1] > max_context_length:
        context = context[..., -max_context_length:]

    # Create mask from non-NaN values
    mask = ~torch.isnan(context)
    mask = mask.to(dtype=dtype)

    # Replace NaN with 0 for computation (mask handles them)
    context = torch.nan_to_num(context, nan=0.0)

    return context, mask


def prepare_batch(
    contexts: Sequence[TensorLike],
    max_context_length: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare a batch of time series with variable lengths.

    Parameters
    ----------
    contexts : Sequence[TensorLike]
        List of time series (can have different lengths)
    max_context_length : int, optional
        Maximum context length
    device : torch.device, optional
        Target device
    dtype : torch.dtype
        Target data type

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (padded context tensor, mask tensor)
    """
    # Convert all to tensors
    tensors = [to_tensor(c, dtype=dtype) for c in contexts]

    # Ensure 1D
    tensors = [t.squeeze() if t.ndim > 1 else t for t in tensors]

    # Find max length
    lengths = [t.shape[0] for t in tensors]
    batch_length = max(lengths)

    # Apply max_context_length limit
    if max_context_length is not None:
        batch_length = min(batch_length, max_context_length)

    # Pad sequences (left-padding to align end points)
    batch_size = len(tensors)
    context = torch.zeros(batch_size, batch_length, dtype=dtype)
    mask = torch.zeros(batch_size, batch_length, dtype=dtype)

    for i, t in enumerate(tensors):
        # Truncate if needed
        if t.shape[0] > batch_length:
            t = t[-batch_length:]

        # Left-pad
        start_idx = batch_length - t.shape[0]
        context[i, start_idx:] = torch.nan_to_num(t, nan=0.0)
        mask[i, start_idx:] = ~torch.isnan(t)

    if device is not None:
        context = context.to(device)
        mask = mask.to(device)

    return context, mask


def prepare_future_covariates(
    future_covariates: Optional[TensorLike],
    batch_size: int,
    prediction_length: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Prepare future covariates tensor and mask.

    Parameters
    ----------
    future_covariates : TensorLike, optional
        Known future values
    batch_size : int
        Expected batch size
    prediction_length : int
        Expected prediction length
    device : torch.device, optional
        Target device
    dtype : torch.dtype
        Target data type

    Returns
    -------
    Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
        (future covariates tensor, mask tensor)
    """
    if future_covariates is None:
        return None, None

    future_covariates = to_tensor(future_covariates, dtype=dtype)

    # Ensure batch dimension
    if future_covariates.ndim == 1:
        future_covariates = future_covariates.unsqueeze(0)

    # Validate shape
    if future_covariates.shape[0] != batch_size:
        raise ValueError(
            f"future_covariates batch size {future_covariates.shape[0]} "
            f"doesn't match context batch size {batch_size}"
        )

    # Truncate if longer than prediction_length
    if future_covariates.shape[-1] > prediction_length:
        future_covariates = future_covariates[..., :prediction_length]

    if device is not None:
        future_covariates = future_covariates.to(device)

    # Create mask
    mask = ~torch.isnan(future_covariates)
    mask = mask.to(dtype=dtype)

    # Replace NaN with 0
    future_covariates = torch.nan_to_num(future_covariates, nan=0.0)

    return future_covariates, mask


def prepare_group_ids(
    group_ids: Optional[TensorLike],
    batch_size: int,
    cross_learning: bool = False,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """
    Prepare group IDs for cross-learning.

    Parameters
    ----------
    group_ids : TensorLike, optional
        Group IDs for each series
    batch_size : int
        Expected batch size
    cross_learning : bool
        If True and group_ids is None, all series share one group
    device : torch.device, optional
        Target device

    Returns
    -------
    torch.Tensor, optional
        Group IDs tensor
    """
    if group_ids is not None:
        group_ids = to_tensor(group_ids, dtype=torch.long)
        if group_ids.shape[0] != batch_size:
            raise ValueError(
                f"group_ids length {group_ids.shape[0]} doesn't match batch size {batch_size}"
            )
    elif cross_learning:
        # All series in same group for cross-learning
        group_ids = torch.zeros(batch_size, dtype=torch.long)

    if group_ids is not None and device is not None:
        group_ids = group_ids.to(device)

    return group_ids


def compute_num_output_patches(
    prediction_length: int,
    output_patch_size: int,
) -> int:
    """
    Compute the number of output patches needed.

    Parameters
    ----------
    prediction_length : int
        Desired prediction horizon
    output_patch_size : int
        Size of each output patch

    Returns
    -------
    int
        Number of output patches (rounded up)
    """
    return (prediction_length + output_patch_size - 1) // output_patch_size


def preprocess_for_chronos2(
    context: Union[TensorLike, Sequence[TensorLike]],
    prediction_length: int,
    output_patch_size: int,
    max_context_length: Optional[int] = None,
    context_mask: Optional[TensorLike] = None,
    group_ids: Optional[TensorLike] = None,
    future_covariates: Optional[TensorLike] = None,
    future_covariates_mask: Optional[TensorLike] = None,
    cross_learning: bool = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Chronos2Inputs:
    """
    Full preprocessing pipeline for Chronos-2 inputs.

    This function handles all input preprocessing including:
    - Converting various input formats to tensors
    - Batching variable-length sequences
    - Creating masks
    - Preparing group IDs for cross-learning
    - Computing output patches

    Parameters
    ----------
    context : Union[TensorLike, Sequence[TensorLike]]
        Historical time series (single, batch, or list of variable-length)
    prediction_length : int
        Desired prediction horizon
    output_patch_size : int
        Size of each output patch (from model config)
    max_context_length : int, optional
        Maximum context length
    context_mask : TensorLike, optional
        Pre-computed context mask
    group_ids : TensorLike, optional
        Group IDs for cross-learning
    future_covariates : TensorLike, optional
        Known future values
    future_covariates_mask : TensorLike, optional
        Pre-computed future covariates mask
    cross_learning : bool
        Enable cross-learning across batch
    device : torch.device, optional
        Target device
    dtype : torch.dtype
        Target data type

    Returns
    -------
    Chronos2Inputs
        Preprocessed inputs ready for the model
    """
    # Handle list of variable-length sequences
    if isinstance(context, (list, tuple)) and not isinstance(context[0], (int, float)):
        # Check if it's a list of tensors/arrays
        if hasattr(context[0], '__len__') or hasattr(context[0], 'shape'):
            context, auto_mask = prepare_batch(
                context,
                max_context_length=max_context_length,
                device=device,
                dtype=dtype,
            )
            if context_mask is None:
                context_mask = auto_mask
    else:
        context, auto_mask = prepare_context(
            context,
            max_context_length=max_context_length,
            device=device,
            dtype=dtype,
        )
        if context_mask is None:
            context_mask = auto_mask

    batch_size = context.shape[0]

    # Process context_mask if provided separately
    if context_mask is not None and not isinstance(context_mask, torch.Tensor):
        context_mask = to_tensor(context_mask, dtype=dtype)
        if device is not None:
            context_mask = context_mask.to(device)

    # Prepare future covariates
    future_cov_tensor = None
    future_cov_mask = None
    if future_covariates is not None:
        future_cov_tensor, auto_future_mask = prepare_future_covariates(
            future_covariates,
            batch_size=batch_size,
            prediction_length=prediction_length,
            device=device,
            dtype=dtype,
        )
        if future_covariates_mask is None:
            future_cov_mask = auto_future_mask
        else:
            future_cov_mask = to_tensor(future_covariates_mask, dtype=dtype)
            if device is not None:
                future_cov_mask = future_cov_mask.to(device)

    # Prepare group IDs
    group_ids_tensor = prepare_group_ids(
        group_ids,
        batch_size=batch_size,
        cross_learning=cross_learning,
        device=device,
    )

    # Compute output patches
    num_output_patches = compute_num_output_patches(prediction_length, output_patch_size)

    return Chronos2Inputs(
        context=context,
        context_mask=context_mask,
        group_ids=group_ids_tensor,
        future_covariates=future_cov_tensor,
        future_covariates_mask=future_cov_mask,
        num_output_patches=num_output_patches,
    )
