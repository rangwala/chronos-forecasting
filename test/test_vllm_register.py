# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Chronos-2 vLLM registration.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chronos.chronos2.vllm.register import (
    Chronos2VLLMPlugin,
    _CHRONOS2_MODEL_REGISTRY,
    get_chronos2_model,
    get_vllm_version,
    is_vllm_available,
    list_chronos2_models,
    register_chronos2_model,
    register_with_vllm,
    setup_vllm_integration,
)


DUMMY_MODEL_PATH = Path(__file__).parent / "dummy-chronos2-model"


class TestRegisterChronos2Model:
    """Tests for register_chronos2_model decorator."""

    def test_registers_model(self):
        """Test that decorator registers model."""
        @register_chronos2_model("TestModel")
        class TestModel:
            pass

        assert "TestModel" in _CHRONOS2_MODEL_REGISTRY
        assert _CHRONOS2_MODEL_REGISTRY["TestModel"] is TestModel

    def test_returns_original_class(self):
        """Test that decorator returns the original class."""
        @register_chronos2_model("TestModel2")
        class TestModel2:
            pass

        assert TestModel2.__name__ == "TestModel2"


class TestGetChronos2Model:
    """Tests for get_chronos2_model function."""

    def test_gets_registered_model(self):
        """Test getting a registered model."""
        result = get_chronos2_model("Chronos2ForForecasting")
        assert result is not None

    def test_returns_none_for_unknown(self):
        """Test returns None for unknown model."""
        result = get_chronos2_model("NonExistentModel")
        assert result is None


class TestListChronos2Models:
    """Tests for list_chronos2_models function."""

    def test_returns_list(self):
        """Test returns a list."""
        result = list_chronos2_models()
        assert isinstance(result, list)

    def test_includes_registered_models(self):
        """Test includes registered models."""
        result = list_chronos2_models()
        assert "Chronos2ForForecasting" in result


class TestIsVllmAvailable:
    """Tests for is_vllm_available function."""

    def test_returns_bool(self):
        """Test returns boolean."""
        result = is_vllm_available()
        assert isinstance(result, bool)

    @patch.dict("sys.modules", {"vllm": MagicMock()})
    def test_returns_true_when_installed(self):
        """Test returns True when vLLM is importable."""
        # Clear any cached import state
        import sys
        if "chronos.chronos2.vllm.register" in sys.modules:
            # Re-import after patching
            import importlib
            from chronos.chronos2.vllm import register
            importlib.reload(register)
            assert register.is_vllm_available() is True


class TestGetVllmVersion:
    """Tests for get_vllm_version function."""

    def test_returns_string_or_none(self):
        """Test returns string or None."""
        result = get_vllm_version()
        assert result is None or isinstance(result, str)

    @patch.dict("sys.modules", {"vllm": MagicMock(__version__="0.5.0")})
    def test_returns_version_when_installed(self):
        """Test returns version string when vLLM is installed."""
        import sys
        if "chronos.chronos2.vllm.register" in sys.modules:
            import importlib
            from chronos.chronos2.vllm import register
            importlib.reload(register)
            result = register.get_vllm_version()
            assert result == "0.5.0"


class TestRegisterWithVllm:
    """Tests for register_with_vllm function."""

    def test_returns_false_without_vllm(self):
        """Test returns False when vLLM is not installed."""
        # This test assumes vLLM is not installed in the test environment
        if not is_vllm_available():
            result = register_with_vllm()
            assert result is False

    def test_attempts_registration(self):
        """Test attempts to register with vLLM when available."""
        # Create mock vllm module
        mock_vllm = MagicMock()
        mock_registry = MagicMock()
        mock_vllm.ModelRegistry = mock_registry

        with patch.dict("sys.modules", {"vllm": mock_vllm}):
            # Reload the register module to pick up the mock
            import importlib
            from chronos.chronos2.vllm import register
            importlib.reload(register)

            result = register.register_with_vllm("TestArch", "test.path:TestClass")

            # Should attempt registration
            mock_registry.register_model.assert_called_once_with(
                "TestArch", "test.path:TestClass"
            )
            assert result is True


class TestChronos2VLLMPlugin:
    """Tests for Chronos2VLLMPlugin class."""

    def test_model_architectures(self):
        """Test model architectures list."""
        assert "Chronos2Model" in Chronos2VLLMPlugin.model_architectures
        assert "Chronos2ForForecasting" in Chronos2VLLMPlugin.model_architectures

    def test_get_supported_architectures(self):
        """Test get_supported_architectures method."""
        result = Chronos2VLLMPlugin.get_supported_architectures()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_is_architecture_supported_true(self):
        """Test is_architecture_supported returns True for supported."""
        assert Chronos2VLLMPlugin.is_architecture_supported("Chronos2Model") is True

    def test_is_architecture_supported_false(self):
        """Test is_architecture_supported returns False for unsupported."""
        assert Chronos2VLLMPlugin.is_architecture_supported("UnknownArch") is False

    def test_load_model(self):
        """Test load_model method."""
        model = Chronos2VLLMPlugin.load_model(str(DUMMY_MODEL_PATH), device_map="cpu")
        assert model is not None
        assert hasattr(model, "forward")


class TestSetupVllmIntegration:
    """Tests for setup_vllm_integration function."""

    def test_returns_dict(self):
        """Test returns a dictionary."""
        result = setup_vllm_integration()
        assert isinstance(result, dict)

    def test_contains_expected_keys(self):
        """Test contains expected keys."""
        result = setup_vllm_integration()
        assert "vllm_available" in result
        assert "vllm_version" in result
        assert "registered" in result
        assert "architectures" in result

    def test_vllm_available_is_bool(self):
        """Test vllm_available is boolean."""
        result = setup_vllm_integration()
        assert isinstance(result["vllm_available"], bool)

    def test_architectures_is_list(self):
        """Test architectures is a list."""
        result = setup_vllm_integration()
        assert isinstance(result["architectures"], list)


class TestModuleImports:
    """Tests for module-level imports from __init__.py."""

    def test_import_wrapper(self):
        """Test importing Chronos2ForVLLM from package."""
        from chronos.chronos2.vllm import Chronos2ForVLLM
        assert Chronos2ForVLLM is not None

    def test_import_config(self):
        """Test importing Chronos2VLLMConfig from package."""
        from chronos.chronos2.vllm import Chronos2VLLMConfig
        assert Chronos2VLLMConfig is not None

    def test_import_preprocessing(self):
        """Test importing preprocessing from package."""
        from chronos.chronos2.vllm import Chronos2Inputs, preprocess_for_chronos2
        assert Chronos2Inputs is not None
        assert preprocess_for_chronos2 is not None

    def test_import_postprocessing(self):
        """Test importing postprocessing from package."""
        from chronos.chronos2.vllm import ForecastOutput, postprocess_chronos2_output
        assert ForecastOutput is not None
        assert postprocess_chronos2_output is not None

    def test_import_registration(self):
        """Test importing registration from package."""
        from chronos.chronos2.vllm import (
            Chronos2VLLMPlugin,
            is_vllm_available,
            register_with_vllm,
            setup_vllm_integration,
        )
        assert Chronos2VLLMPlugin is not None
        assert is_vllm_available is not None
        assert register_with_vllm is not None
        assert setup_vllm_integration is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
