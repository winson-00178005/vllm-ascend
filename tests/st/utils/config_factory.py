#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""Configuration factory for ST tests.

Creates VllmConfig, ModelConfig, etc. with fallback for environments without vllm.
"""

from typing import Any, Dict, Optional

try:
    from vllm.config import VllmConfig, ModelConfig, ParallelConfig
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

    class VllmConfig:
        """Fallback VllmConfig when vllm is not available."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class ModelConfig:
        """Fallback ModelConfig when vllm is not available."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class ParallelConfig:
        """Fallback ParallelConfig when vllm is not available."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


def create_vllm_config(**kwargs) -> VllmConfig:
    """Create VllmConfig with default values.

    Args:
        **kwargs: Configuration parameters

    Returns:
        VllmConfig instance
    """
    defaults = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_model_len": 2048,
        "dtype": "float16",
        "tensor_parallel_size": 1,
    }
    config_kwargs = {**defaults, **kwargs}
    return VllmConfig(**config_kwargs)


def create_model_config(**kwargs) -> ModelConfig:
    """Create ModelConfig with default values.

    Args:
        **kwargs: Configuration parameters

    Returns:
        ModelConfig instance
    """
    defaults = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "dtype": "float16",
        "max_model_len": 2048,
    }
    config_kwargs = {**defaults, **kwargs}
    return ModelConfig(**config_kwargs)


def create_parallel_config(**kwargs) -> ParallelConfig:
    """Create ParallelConfig with default values.

    Args:
        **kwargs: Configuration parameters

    Returns:
        ParallelConfig instance
    """
    defaults = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }
    config_kwargs = {**defaults, **kwargs}
    return ParallelConfig(**config_kwargs)


def create_ascend_config(**kwargs) -> Any:
    """Create AscendConfig (plugin-specific config).

    Args:
        **kwargs: Configuration parameters

    Returns:
        AscendConfig instance
    """
    try:
        from vllm_ascend.ascend_config import AscendConfig
        return AscendConfig(**kwargs)
    except ImportError:
        config = type('AscendConfig', (), {})()
        for key, value in kwargs.items():
            setattr(config, key, value)
        return config


def validate_config(config_obj: Any) -> bool:
    """Validate configuration object.

    Args:
        config_obj: Configuration object to validate

    Returns:
        bool: True if configuration is valid
    """
    required_attrs = ['model', 'dtype', 'max_model_len']
    for attr in required_attrs:
        if not hasattr(config_obj, attr):
            return False
    return True


def get_default_config() -> Dict[str, Any]:
    """Get default test configuration.

    Returns:
        dict with default configuration
    """
    return {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_model_len": 2048,
        "dtype": "float16",
        "batch_size": [1, 16],
        "tensor_parallel_size": 1,
    }