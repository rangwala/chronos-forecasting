# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM-compatible wrapper for Chronos-2 models.

This module provides a wrapper class that exposes Chronos-2 models with vLLM-compatible
interfaces, enabling deployment via vLLM's serving infrastructure.

Note: Due to architectural differences (GroupSelfAttention, bidirectional attention,
patch-based inputs), this wrapper uses vLLM's Transformers backend rather than
native vLLM attention kernels.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from chronos.chronos2.config import Chronos2CoreConfig, Chronos2ForecastingConfig
from chronos.chronos2.model import Chronos2Model, Chronos2Output


@dataclass
class Chronos2VLLMConfig:
    """Configuration for Chronos-2 vLLM wrapper.

    Attributes
    ----------
    prediction_length : int
        Default prediction length for forecasting
    batch_size : int
        Maximum batch size for inference
    quantile_levels : List[float], optional
        Which quantile levels to return. If None, returns all model quantiles.
    cross_learning : bool
        Whether to enable cross-learning (information sharing across batch).
        Default is False for independent forecasting.
    """
    prediction_length: int = 24
    batch_size: int = 32
    quantile_levels: Optional[List[float]] = None
    cross_learning: bool = False


class Chronos2ForVLLM(nn.Module):
    """
    vLLM-compatible wrapper for Chronos-2 time series forecasting models.

    This wrapper adapts Chronos-2 models for use with vLLM's serving infrastructure.
    It handles the translation between vLLM's expected interfaces and Chronos-2's
    unique architecture (patch-based inputs, quantile outputs, GroupSelfAttention).

    Architecture Notes
    ------------------
    - Chronos-2 is an encoder-only model (no autoregressive decoding)
    - Uses GroupSelfAttention which attends over the batch dimension
    - Inputs are continuous time series patches, not discrete tokens
    - Outputs are quantile predictions, not next-token logits

    vLLM Compatibility
    ------------------
    - Treated as a "pooling" model (encoder-only)
    - Does not use KV-cache (bidirectional attention)
    - Custom input/output processing required

    Example
    -------
    >>> from chronos.chronos2.vllm import Chronos2ForVLLM
    >>> model = Chronos2ForVLLM.from_pretrained("amazon/chronos-2")
    >>> context = torch.randn(4, 100)  # 4 time series, 100 timesteps each
    >>> output = model.forward_forecast(context, prediction_length=24)
    >>> print(output.quantile_preds.shape)  # (4, 21, 24)
    """

    # vLLM model properties
    supports_pp: bool = False  # Pipeline parallelism not supported (GroupSelfAttention)
    is_attention_free: bool = False  # Has attention layers

    def __init__(
        self,
        model: Chronos2Model,
        vllm_config: Optional[Chronos2VLLMConfig] = None,
    ):
        """
        Initialize the vLLM wrapper.

        Parameters
        ----------
        model : Chronos2Model
            The underlying Chronos-2 model
        vllm_config : Chronos2VLLMConfig, optional
            Configuration for vLLM-specific behavior
        """
        super().__init__()
        self.model = model
        self.vllm_config = vllm_config or Chronos2VLLMConfig()

        # Cache model properties for quick access
        self._model_config = model.config
        self._chronos_config = model.chronos_config

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        vllm_config: Optional[Chronos2VLLMConfig] = None,
        device_map: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "Chronos2ForVLLM":
        """
        Load a pretrained Chronos-2 model with vLLM wrapper.

        Parameters
        ----------
        model_name_or_path : str
            HuggingFace model ID or local path
        vllm_config : Chronos2VLLMConfig, optional
            Configuration for vLLM-specific behavior
        device_map : str, optional
            Device placement strategy
        torch_dtype : torch.dtype, optional
            Data type for model weights
        **kwargs
            Additional arguments passed to model loading

        Returns
        -------
        Chronos2ForVLLM
            The wrapped model ready for vLLM serving
        """
        from transformers import AutoConfig

        # Load config first to determine model type
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)

        # Load the Chronos-2 model
        model = Chronos2Model.from_pretrained(
            model_name_or_path,
            config=config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        return cls(model=model, vllm_config=vllm_config)

    @property
    def is_pooling_model(self) -> bool:
        """vLLM property: indicates this is an encoder-only pooling model."""
        return True

    @property
    def config(self) -> Chronos2CoreConfig:
        """Return the underlying model config."""
        return self._model_config

    @property
    def chronos_config(self) -> Chronos2ForecastingConfig:
        """Return the Chronos-specific forecasting config."""
        return self._chronos_config

    @property
    def device(self) -> torch.device:
        """Return the model's device."""
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the model's dtype."""
        return self.model.dtype

    @property
    def num_quantiles(self) -> int:
        """Return the number of quantile levels predicted by the model."""
        return len(self._chronos_config.quantiles)

    @property
    def quantiles(self) -> List[float]:
        """Return the quantile levels predicted by the model."""
        return self._chronos_config.quantiles

    @property
    def context_length(self) -> int:
        """Return the maximum context length supported by the model."""
        return self._chronos_config.context_length

    @property
    def output_patch_size(self) -> int:
        """Return the output patch size."""
        return self._chronos_config.output_patch_size

    def _compute_num_output_patches(self, prediction_length: int) -> int:
        """Compute the number of output patches needed for a given prediction length."""
        return (prediction_length + self.output_patch_size - 1) // self.output_patch_size

    def forward(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        future_covariates: Optional[torch.Tensor] = None,
        future_covariates_mask: Optional[torch.Tensor] = None,
        num_output_patches: int = 1,
        **kwargs,
    ) -> Chronos2Output:
        """
        Forward pass through the Chronos-2 model.

        This is a thin wrapper around Chronos2Model.forward() that maintains
        compatibility with the original interface.

        Parameters
        ----------
        context : torch.Tensor
            Input tensor of shape (batch_size, context_length)
        context_mask : torch.Tensor, optional
            Binary mask of shape (batch_size, context_length)
        group_ids : torch.Tensor, optional
            Group IDs of shape (batch_size,) for cross-learning
        future_covariates : torch.Tensor, optional
            Future covariates of shape (batch_size, future_length)
        future_covariates_mask : torch.Tensor, optional
            Mask for future covariates
        num_output_patches : int
            Number of output patches to generate
        **kwargs
            Additional arguments passed to the model

        Returns
        -------
        Chronos2Output
            Model output containing quantile predictions
        """
        return self.model.forward(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=num_output_patches,
            **kwargs,
        )

    def forward_forecast(
        self,
        context: torch.Tensor,
        prediction_length: int,
        context_mask: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        future_covariates: Optional[torch.Tensor] = None,
        future_covariates_mask: Optional[torch.Tensor] = None,
    ) -> Chronos2Output:
        """
        Simplified forecasting interface for vLLM serving.

        This method provides a cleaner interface for common forecasting tasks,
        automatically computing the required number of output patches.

        Parameters
        ----------
        context : torch.Tensor
            Historical time series of shape (batch_size, context_length)
        prediction_length : int
            Number of future timesteps to predict
        context_mask : torch.Tensor, optional
            Binary mask indicating valid observations
        group_ids : torch.Tensor, optional
            Group IDs for cross-learning. If None and cross_learning is enabled
            in vllm_config, all series are treated as one group.
        future_covariates : torch.Tensor, optional
            Known future values (e.g., calendar features)
        future_covariates_mask : torch.Tensor, optional
            Mask for future covariates

        Returns
        -------
        Chronos2Output
            Output containing:
            - quantile_preds: (batch_size, num_quantiles, prediction_length)
        """
        batch_size = context.shape[0]
        num_output_patches = self._compute_num_output_patches(prediction_length)

        # Handle cross-learning configuration
        if group_ids is None and self.vllm_config.cross_learning:
            # All series in the same group for cross-learning
            group_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        output = self.forward(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            num_output_patches=num_output_patches,
        )

        # Truncate predictions to exact prediction_length
        if output.quantile_preds is not None:
            output.quantile_preds = output.quantile_preds[..., :prediction_length]

        return output

    def get_quantile_predictions(
        self,
        context: torch.Tensor,
        prediction_length: int,
        quantile_levels: Optional[List[float]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Get quantile predictions in a dictionary format suitable for serving.

        Parameters
        ----------
        context : torch.Tensor
            Historical time series
        prediction_length : int
            Number of future timesteps to predict
        quantile_levels : List[float], optional
            Specific quantile levels to return. If None, returns all.
        **kwargs
            Additional arguments passed to forward_forecast

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary mapping quantile level strings to prediction tensors
        """
        output = self.forward_forecast(
            context=context,
            prediction_length=prediction_length,
            **kwargs,
        )

        result = {}
        model_quantiles = self.quantiles

        # Filter to requested quantile levels if specified
        if quantile_levels is None:
            quantile_levels = self.vllm_config.quantile_levels or model_quantiles

        for i, q in enumerate(model_quantiles):
            if q in quantile_levels:
                result[f"q{q:.2f}"] = output.quantile_preds[:, i, :]

        # Always include median (0.5) if available
        if 0.5 in model_quantiles:
            median_idx = model_quantiles.index(0.5)
            result["median"] = output.quantile_preds[:, median_idx, :]

        return result

    def __repr__(self) -> str:
        return (
            f"Chronos2ForVLLM(\n"
            f"  context_length={self.context_length},\n"
            f"  num_quantiles={self.num_quantiles},\n"
            f"  output_patch_size={self.output_patch_size},\n"
            f"  is_pooling_model={self.is_pooling_model}\n"
            f")"
        )
