# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Output postprocessing for Chronos-2 vLLM integration.

This module handles the conversion of Chronos-2 model outputs into various
formats suitable for serving and downstream applications.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from chronos.chronos2.model import Chronos2Output


@dataclass
class ForecastOutput:
    """
    Structured forecast output for serving.

    Attributes
    ----------
    predictions : torch.Tensor
        Point predictions (median by default), shape (batch_size, prediction_length)
    quantiles : Dict[str, torch.Tensor]
        Dictionary mapping quantile labels to prediction tensors
    prediction_length : int
        Length of the prediction horizon
    batch_size : int
        Number of time series in the batch
    model_quantiles : List[float]
        List of quantile levels from the model
    """
    predictions: torch.Tensor
    quantiles: Dict[str, torch.Tensor]
    prediction_length: int
    batch_size: int
    model_quantiles: List[float] = field(default_factory=list)


def truncate_predictions(
    quantile_preds: torch.Tensor,
    prediction_length: int,
) -> torch.Tensor:
    """
    Truncate predictions to exact prediction length.

    Parameters
    ----------
    quantile_preds : torch.Tensor
        Quantile predictions of shape (batch_size, num_quantiles, full_length)
    prediction_length : int
        Desired prediction length

    Returns
    -------
    torch.Tensor
        Truncated predictions of shape (batch_size, num_quantiles, prediction_length)
    """
    return quantile_preds[..., :prediction_length]


def extract_median(
    quantile_preds: torch.Tensor,
    quantile_levels: List[float],
) -> torch.Tensor:
    """
    Extract median (0.5 quantile) from predictions.

    Parameters
    ----------
    quantile_preds : torch.Tensor
        Quantile predictions of shape (batch_size, num_quantiles, prediction_length)
    quantile_levels : List[float]
        List of quantile levels corresponding to second dimension

    Returns
    -------
    torch.Tensor
        Median predictions of shape (batch_size, prediction_length)

    Raises
    ------
    ValueError
        If 0.5 quantile is not in quantile_levels
    """
    if 0.5 not in quantile_levels:
        raise ValueError(
            f"Median (0.5) not found in quantile levels: {quantile_levels}. "
            "Cannot extract median predictions."
        )
    median_idx = quantile_levels.index(0.5)
    return quantile_preds[:, median_idx, :]


def extract_point_prediction(
    quantile_preds: torch.Tensor,
    quantile_levels: List[float],
    method: str = "median",
) -> torch.Tensor:
    """
    Extract point predictions from quantile forecasts.

    Parameters
    ----------
    quantile_preds : torch.Tensor
        Quantile predictions of shape (batch_size, num_quantiles, prediction_length)
    quantile_levels : List[float]
        List of quantile levels
    method : str
        Method for extracting point prediction:
        - "median": Use 0.5 quantile (default)
        - "mean": Average across all quantiles (approximate mean)

    Returns
    -------
    torch.Tensor
        Point predictions of shape (batch_size, prediction_length)
    """
    if method == "median":
        return extract_median(quantile_preds, quantile_levels)
    elif method == "mean":
        return quantile_preds.mean(dim=1)
    else:
        raise ValueError(f"Unknown point prediction method: {method}")


def quantiles_to_dict(
    quantile_preds: torch.Tensor,
    quantile_levels: List[float],
    include_all: bool = True,
    selected_quantiles: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert quantile predictions to dictionary format.

    Parameters
    ----------
    quantile_preds : torch.Tensor
        Quantile predictions of shape (batch_size, num_quantiles, prediction_length)
    quantile_levels : List[float]
        List of quantile levels
    include_all : bool
        If True, include all quantiles; otherwise use selected_quantiles
    selected_quantiles : List[float], optional
        Specific quantiles to include

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary mapping quantile names to tensors
    """
    result = {}
    levels_to_include = quantile_levels if include_all else (selected_quantiles or [])

    for i, q in enumerate(quantile_levels):
        if q in levels_to_include:
            key = f"q{q:.2f}"
            result[key] = quantile_preds[:, i, :]

    # Add common aliases
    if 0.5 in quantile_levels:
        median_idx = quantile_levels.index(0.5)
        result["median"] = quantile_preds[:, median_idx, :]

    if 0.1 in quantile_levels and 0.9 in quantile_levels:
        result["lower_80"] = quantile_preds[:, quantile_levels.index(0.1), :]
        result["upper_80"] = quantile_preds[:, quantile_levels.index(0.9), :]

    if 0.05 in quantile_levels and 0.95 in quantile_levels:
        result["lower_90"] = quantile_preds[:, quantile_levels.index(0.05), :]
        result["upper_90"] = quantile_preds[:, quantile_levels.index(0.95), :]

    if 0.25 in quantile_levels and 0.75 in quantile_levels:
        result["lower_50"] = quantile_preds[:, quantile_levels.index(0.25), :]
        result["upper_50"] = quantile_preds[:, quantile_levels.index(0.75), :]

    return result


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy array.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor

    Returns
    -------
    np.ndarray
        Numpy array
    """
    return tensor.detach().cpu().numpy()


def postprocess_chronos2_output(
    output: Chronos2Output,
    quantile_levels: List[float],
    prediction_length: int,
    point_prediction_method: str = "median",
    include_all_quantiles: bool = True,
    selected_quantiles: Optional[List[float]] = None,
) -> ForecastOutput:
    """
    Full postprocessing pipeline for Chronos-2 outputs.

    Parameters
    ----------
    output : Chronos2Output
        Raw model output
    quantile_levels : List[float]
        Quantile levels from model config
    prediction_length : int
        Desired prediction length (for truncation)
    point_prediction_method : str
        Method for point predictions ("median" or "mean")
    include_all_quantiles : bool
        Whether to include all quantiles in output
    selected_quantiles : List[float], optional
        Specific quantiles to include

    Returns
    -------
    ForecastOutput
        Processed forecast output
    """
    quantile_preds = output.quantile_preds
    if quantile_preds is None:
        raise ValueError("Model output does not contain quantile predictions")

    # Truncate to exact prediction length
    quantile_preds = truncate_predictions(quantile_preds, prediction_length)

    batch_size = quantile_preds.shape[0]

    # Extract point predictions
    point_preds = extract_point_prediction(
        quantile_preds, quantile_levels, method=point_prediction_method
    )

    # Convert to dictionary format
    quantile_dict = quantiles_to_dict(
        quantile_preds,
        quantile_levels,
        include_all=include_all_quantiles,
        selected_quantiles=selected_quantiles,
    )

    return ForecastOutput(
        predictions=point_preds,
        quantiles=quantile_dict,
        prediction_length=prediction_length,
        batch_size=batch_size,
        model_quantiles=quantile_levels,
    )


def format_for_json(
    output: ForecastOutput,
    include_quantiles: bool = True,
    as_list: bool = True,
) -> Dict[str, Any]:
    """
    Format forecast output for JSON serialization.

    Parameters
    ----------
    output : ForecastOutput
        Forecast output to format
    include_quantiles : bool
        Whether to include quantile predictions
    as_list : bool
        If True, convert tensors to nested lists; otherwise numpy arrays

    Returns
    -------
    Dict[str, Any]
        JSON-serializable dictionary
    """
    def convert(tensor: torch.Tensor):
        arr = to_numpy(tensor)
        return arr.tolist() if as_list else arr

    result = {
        "predictions": convert(output.predictions),
        "prediction_length": output.prediction_length,
        "batch_size": output.batch_size,
    }

    if include_quantiles:
        result["quantiles"] = {
            k: convert(v) for k, v in output.quantiles.items()
        }
        result["quantile_levels"] = output.model_quantiles

    return result


def samples_from_quantiles(
    quantile_preds: torch.Tensor,
    quantile_levels: List[float],
    num_samples: int = 100,
) -> torch.Tensor:
    """
    Generate approximate samples from quantile predictions.

    Uses linear interpolation between quantiles to generate samples.
    This is useful for downstream tasks that expect sample-based forecasts.

    Parameters
    ----------
    quantile_preds : torch.Tensor
        Quantile predictions of shape (batch_size, num_quantiles, prediction_length)
    quantile_levels : List[float]
        Quantile levels
    num_samples : int
        Number of samples to generate

    Returns
    -------
    torch.Tensor
        Samples of shape (batch_size, num_samples, prediction_length)
    """
    batch_size, num_quantiles, pred_len = quantile_preds.shape
    device = quantile_preds.device
    dtype = quantile_preds.dtype

    # Generate uniform random values for sampling
    uniform_samples = torch.rand(batch_size, num_samples, device=device, dtype=dtype)

    # Convert quantile levels to tensor
    q_levels = torch.tensor(quantile_levels, device=device, dtype=dtype)

    # For each sample, interpolate between quantiles
    samples = torch.zeros(batch_size, num_samples, pred_len, device=device, dtype=dtype)

    for b in range(batch_size):
        for s in range(num_samples):
            u = uniform_samples[b, s]
            # Find which quantile interval this sample falls into
            idx = torch.searchsorted(q_levels, u)

            if idx == 0:
                # Below lowest quantile - extrapolate
                samples[b, s, :] = quantile_preds[b, 0, :]
            elif idx >= num_quantiles:
                # Above highest quantile - extrapolate
                samples[b, s, :] = quantile_preds[b, -1, :]
            else:
                # Interpolate between quantiles
                q_low = q_levels[idx - 1]
                q_high = q_levels[idx]
                weight = (u - q_low) / (q_high - q_low)
                samples[b, s, :] = (
                    (1 - weight) * quantile_preds[b, idx - 1, :]
                    + weight * quantile_preds[b, idx, :]
                )

    return samples


def compute_prediction_intervals(
    quantile_preds: torch.Tensor,
    quantile_levels: List[float],
    coverage: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute prediction intervals at specified coverage level.

    Parameters
    ----------
    quantile_preds : torch.Tensor
        Quantile predictions
    quantile_levels : List[float]
        Quantile levels
    coverage : float
        Desired coverage (e.g., 0.9 for 90% interval)

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (lower_bound, upper_bound) tensors
    """
    alpha = (1 - coverage) / 2
    lower_q = alpha
    upper_q = 1 - alpha

    # Find closest available quantiles
    q_tensor = torch.tensor(quantile_levels)
    lower_idx = (q_tensor - lower_q).abs().argmin().item()
    upper_idx = (q_tensor - upper_q).abs().argmin().item()

    lower_bound = quantile_preds[:, lower_idx, :]
    upper_bound = quantile_preds[:, upper_idx, :]

    return lower_bound, upper_bound
