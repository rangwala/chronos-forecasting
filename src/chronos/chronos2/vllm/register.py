# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM model registration for Chronos-2.

This module provides utilities for registering Chronos-2 models with vLLM's
model registry, enabling seamless integration with vLLM's serving infrastructure.
"""

import logging
from typing import Any, Callable, Dict, Optional, Type

logger = logging.getLogger(__name__)


# Registry for Chronos-2 model variants
_CHRONOS2_MODEL_REGISTRY: Dict[str, Type] = {}


def register_chronos2_model(name: str) -> Callable:
    """
    Decorator to register a Chronos-2 model variant.

    Parameters
    ----------
    name : str
        Name of the model variant

    Returns
    -------
    Callable
        Decorator function

    Example
    -------
    >>> @register_chronos2_model("Chronos2ForForecasting")
    ... class MyChronos2Model:
    ...     pass
    """
    def decorator(cls: Type) -> Type:
        _CHRONOS2_MODEL_REGISTRY[name] = cls
        logger.debug(f"Registered Chronos-2 model: {name}")
        return cls
    return decorator


def get_chronos2_model(name: str) -> Optional[Type]:
    """
    Get a registered Chronos-2 model by name.

    Parameters
    ----------
    name : str
        Name of the model

    Returns
    -------
    Type or None
        The model class, or None if not found
    """
    return _CHRONOS2_MODEL_REGISTRY.get(name)


def list_chronos2_models() -> list[str]:
    """
    List all registered Chronos-2 models.

    Returns
    -------
    list[str]
        List of model names
    """
    return list(_CHRONOS2_MODEL_REGISTRY.keys())


def register_with_vllm(
    model_name: str = "Chronos2ForForecasting",
    model_class_path: str = "chronos.chronos2.vllm.wrapper:Chronos2ForVLLM",
) -> bool:
    """
    Register Chronos-2 model with vLLM's model registry.

    This function attempts to register the Chronos-2 model with vLLM's
    ModelRegistry, enabling it to be loaded via vLLM's standard model
    loading mechanisms.

    Parameters
    ----------
    model_name : str
        Architecture name to register
    model_class_path : str
        Import path for the model class

    Returns
    -------
    bool
        True if registration succeeded, False otherwise

    Notes
    -----
    This function requires vLLM to be installed. If vLLM is not available,
    it will log a warning and return False.

    Example
    -------
    >>> from chronos.chronos2.vllm.register import register_with_vllm
    >>> register_with_vllm()
    True
    """
    try:
        from vllm import ModelRegistry
        ModelRegistry.register_model(model_name, model_class_path)
        logger.info(f"Successfully registered {model_name} with vLLM")
        return True
    except ImportError:
        logger.warning(
            "vLLM is not installed. Cannot register Chronos-2 with vLLM. "
            "Install vLLM with: pip install vllm"
        )
        return False
    except Exception as e:
        logger.error(f"Failed to register Chronos-2 with vLLM: {e}")
        return False


def is_vllm_available() -> bool:
    """
    Check if vLLM is available.

    Returns
    -------
    bool
        True if vLLM can be imported
    """
    try:
        import vllm
        return True
    except ImportError:
        return False


def get_vllm_version() -> Optional[str]:
    """
    Get the installed vLLM version.

    Returns
    -------
    str or None
        Version string, or None if vLLM is not installed
    """
    try:
        import vllm
        return vllm.__version__
    except (ImportError, AttributeError):
        return None


class Chronos2VLLMPlugin:
    """
    vLLM plugin for Chronos-2 models.

    This class provides a plugin interface for vLLM, allowing Chronos-2
    models to be loaded and served via vLLM's infrastructure.

    Attributes
    ----------
    model_architectures : list[str]
        List of supported architecture names
    """

    model_architectures = ["Chronos2Model", "Chronos2ForForecasting"]

    @classmethod
    def get_supported_architectures(cls) -> list[str]:
        """Get list of supported architectures."""
        return cls.model_architectures

    @classmethod
    def is_architecture_supported(cls, architecture: str) -> bool:
        """Check if an architecture is supported."""
        return architecture in cls.model_architectures

    @classmethod
    def load_model(
        cls,
        model_path: str,
        device_map: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        **kwargs,
    ):
        """
        Load a Chronos-2 model.

        Parameters
        ----------
        model_path : str
            Path to the model (HuggingFace ID or local path)
        device_map : str, optional
            Device placement strategy
        torch_dtype : Any, optional
            Data type for model weights
        **kwargs
            Additional arguments

        Returns
        -------
        Chronos2ForVLLM
            Loaded model wrapped for vLLM
        """
        from chronos.chronos2.vllm.wrapper import Chronos2ForVLLM

        return Chronos2ForVLLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs,
        )

    @classmethod
    def register(cls) -> bool:
        """
        Register all supported architectures with vLLM.

        Returns
        -------
        bool
            True if all registrations succeeded
        """
        success = True
        for arch in cls.model_architectures:
            if not register_with_vllm(
                model_name=arch,
                model_class_path="chronos.chronos2.vllm.wrapper:Chronos2ForVLLM",
            ):
                success = False
        return success


# Auto-register common model variant
@register_chronos2_model("Chronos2ForForecasting")
class _Chronos2ForForecastingProxy:
    """Proxy class for registration."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from chronos.chronos2.vllm.wrapper import Chronos2ForVLLM
        return Chronos2ForVLLM.from_pretrained(*args, **kwargs)


def setup_vllm_integration() -> Dict[str, Any]:
    """
    Setup vLLM integration for Chronos-2.

    This function performs all necessary setup for using Chronos-2 with vLLM,
    including checking vLLM availability, registering models, and returning
    integration status.

    Returns
    -------
    Dict[str, Any]
        Integration status containing:
        - vllm_available: bool
        - vllm_version: str or None
        - registered: bool
        - architectures: list[str]

    Example
    -------
    >>> status = setup_vllm_integration()
    >>> if status["registered"]:
    ...     print("Chronos-2 is ready for vLLM serving")
    """
    status = {
        "vllm_available": is_vllm_available(),
        "vllm_version": get_vllm_version(),
        "registered": False,
        "architectures": Chronos2VLLMPlugin.get_supported_architectures(),
    }

    if status["vllm_available"]:
        status["registered"] = Chronos2VLLMPlugin.register()

    return status
