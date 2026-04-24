"""
ST测试框架核心组件 - Environment First + Model Centric + Parallel

三层架构:
1. Environment Layer: EnvironmentManager, EnvironmentInfo
2. Model Layer: ModelRegistry, ModelInstance
3. Test Layer: TestRegistry, TestInfo, model_test decorator
"""
from .scene_manager import SceneManager, Scene
from .model_loader import ModelLoader, ModelConfig
from .decorators import (
    require_scene,
    require_model,
    require_device,
    require_tags,
)
from .device_types import (
    RunnerDeviceType,
    DeviceType310P,
    DeviceType910B,
    DeviceTypeCPU,
    RunnerKey,
    DEFAULT_RUNNER_KEY,
)
from .npu_test import (
    npu_test,
    NpuTestInfo,
    get_npu_test_info,
    is_npu_test,
)
from .environment_manager import (
    EnvironmentManager,
    EnvironmentInfo,
)
from .model_registry import (
    ModelRegistry,
    ModelInstance,
)
from .test_registry import (
    TestRegistry,
    TestInfo,
    model_test,
    get_model_test_info,
    is_model_test,
)

__all__ = [
    # Scene Management (Legacy)
    "SceneManager",
    "Scene",
    
    # Model Loading (Legacy)
    "ModelLoader",
    "ModelConfig",
    
    # Decorators (Legacy)
    "require_scene",
    "require_model",
    "require_device",
    "require_tags",
    
    # Device Types (New)
    "RunnerDeviceType",
    "DeviceType310P",
    "DeviceType910B",
    "DeviceTypeCPU",
    "RunnerKey",
    "DEFAULT_RUNNER_KEY",
    
    # NPU Test Decorator (New)
    "npu_test",
    "NpuTestInfo",
    "get_npu_test_info",
    "is_npu_test",
    
    # Environment Management (New - Layer 1)
    "EnvironmentManager",
    "EnvironmentInfo",
    
    # Model Registry (New - Layer 2)
    "ModelRegistry",
    "ModelInstance",
    
    # Test Registry (New - Layer 3)
    "TestRegistry",
    "TestInfo",
    "model_test",
    "get_model_test_info",
    "is_model_test",
]