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

"""ModelPool - Model cache pool for avoiding repeated model loading.

Environment Layer: CPU Mock / NPU Real
Model Layer: ModelPool manages model instances
Test Case Layer: Tests use cached models
"""

import hashlib
from typing import Any, Dict, Optional
from unittest.mock import MagicMock
from pathlib import Path
import yaml


class ModelPool:
    """Model cache pool with cache strategy."""
    
    _instance: Optional['ModelPool'] = None
    _models: Dict[str, Any] = {}
    _stats: Dict[str, Any] = {'hits': 0, 'misses': 0}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_or_create(cls, model_name: str, config: Dict[str, Any], exec_mode: str = "cpu_mock") -> Any:
        """Get or create model instance with cache strategy.
        
        Args:
            model_name: Model name
            config: Model configuration
            exec_mode: Execution mode (cpu_mock / npu_real)
            
        Returns:
            Model instance (Mock or real model)
        """
        cache_key = cls._create_cache_key(model_name, config, exec_mode)
        
        if cache_key in cls._models:
            cls._stats['hits'] += 1
            return cls._models[cache_key]
        
        cls._stats['misses'] += 1
        model = cls._load_model(model_name, config, exec_mode)
        cls._models[cache_key] = model
        return model
    
    @classmethod
    def _create_cache_key(cls, model_name: str, config: Dict[str, Any], exec_mode: str) -> str:
        """Create cache key considering model name, quantization, distributed config."""
        config_str = yaml.dump(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{model_name}_{exec_mode}_{config_hash}"
    
    @classmethod
    def _load_model(cls, model_name: str, config: Dict[str, Any], exec_mode: str) -> Any:
        """Load model - CPU Mock creates Mock model, NPU Real loads real model."""
        if exec_mode == "cpu_mock":
            mock_model = MagicMock()
            mock_model.model_name = model_name
            mock_model.config = config
            mock_model.forward = MagicMock(return_value=MagicMock())
            return mock_model
        else:
            # NPU Real mode - placeholder for future implementation
            raise NotImplementedError("NPU Real model loading requires real hardware")
    
    @classmethod
    def release(cls, model_name: str) -> bool:
        """Manually release specified model."""
        keys_to_remove = [k for k in cls._models if k.startswith(model_name)]
        for key in keys_to_remove:
            del cls._models[key]
        return len(keys_to_remove) > 0
    
    @classmethod
    def clear(cls):
        """Clear all models, called when module ends."""
        cls._models.clear()
        cls._stats = {'hits': 0, 'misses': 0}
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get statistics: model count, cache hit rate."""
        total_requests = cls._stats['hits'] + cls._stats['misses']
        hit_rate = cls._stats['hits'] / total_requests if total_requests > 0 else 0
        return {
            'model_count': len(cls._models),
            'hits': cls._stats['hits'],
            'misses': cls._stats['misses'],
            'hit_rate': hit_rate,
        }


class ConfigLoader:
    """Configuration manager for loading YAML config files."""
    
    _configs: Dict[str, Any] = {}
    _config_dir: Path = Path(__file__).parent.parent / "configs"
    
    @classmethod
    def load(cls, config_file: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if config_file in cls._configs:
            return cls._configs[config_file]
        
        config_path = cls._config_dir / config_file
        if not config_path.exists():
            return {}
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        cls._configs[config_file] = config
        return config
    
    @classmethod
    def load_model_config(cls, model_name: str, config_file: str = "default.yaml") -> Dict[str, Any]:
        """Load specific model configuration."""
        config = cls.load(config_file)
        models = config.get('models', {})
        return models.get(model_name, models.get('default_model', {}))
    
    @classmethod
    def load_test_config(cls, test_name: str, config_file: str = "default.yaml") -> Dict[str, Any]:
        """Load specific test configuration."""
        config = cls.load(config_file)
        tests = config.get('tests', {})
        return tests.get(test_name, tests.get('default_test', {}))
    
    @classmethod
    def clear_cache(cls):
        """Clear configuration cache."""
        cls._configs.clear()