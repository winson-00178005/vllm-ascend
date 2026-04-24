# ST Framework - Scene-based System Test Framework
#
# Core components
from tests.st.framework.scene_manager import SceneManager, SceneInfo
from tests.st.framework.model_config import ModelConfig, ModelInfo
from tests.st.framework.decorators import require_scene, require_model, require_scene_and_model
from tests.st.framework.fixtures import *
from tests.st.framework.compat import scene_aware_parametrize, get_models_for_current_scene

# Extensions (based on PR #8557)
from tests.st.framework.extensions import (
    vllm_version_is,
    vllm_version_above,
    vllm_version_below,
    PerformanceMonitor,
    PrecisionComparator,
    require_hardware,
    require_npu_device,
    TestReport,
)

__all__ = [
    # Core
    "SceneManager",
    "SceneInfo",
    "ModelConfig",
    "ModelInfo",
    "require_scene",
    "require_model",
    "require_scene_and_model",
    "scene_aware_parametrize",
    "get_models_for_current_scene",
    # Extensions
    "vllm_version_is",
    "vllm_version_above",
    "vllm_version_below",
    "PerformanceMonitor",
    "PrecisionComparator",
    "require_hardware",
    "require_npu_device",
    "TestReport",
]
