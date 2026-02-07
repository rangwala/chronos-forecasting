# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM integration for Chronos-2 models.

This module provides vLLM-compatible wrappers for Chronos-2 time series forecasting models.
"""

from .wrapper import Chronos2ForVLLM, Chronos2VLLMConfig
from .preprocessing import Chronos2Inputs, preprocess_for_chronos2
from .postprocessing import ForecastOutput, postprocess_chronos2_output, format_for_json
from .register import (
    Chronos2VLLMPlugin,
    is_vllm_available,
    register_with_vllm,
    setup_vllm_integration,
)

__all__ = [
    # Wrapper
    "Chronos2ForVLLM",
    "Chronos2VLLMConfig",
    # Preprocessing
    "Chronos2Inputs",
    "preprocess_for_chronos2",
    # Postprocessing
    "ForecastOutput",
    "postprocess_chronos2_output",
    "format_for_json",
    # Registration
    "Chronos2VLLMPlugin",
    "is_vllm_available",
    "register_with_vllm",
    "setup_vllm_integration",
]
