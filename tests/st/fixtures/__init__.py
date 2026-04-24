"""
pytest fixtures模块 - Environment First + Model Centric + Parallel

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

提供pytest fixtures用于ST测试框架：
- Legacy fixtures: scene_manager, current_scene, vllm_runner
- New session fixtures: environment, loaded_models, model_runner
"""

# Legacy fixtures (function level)
from .scene_fixtures import (
    scene_manager,
    model_loader,
    current_scene,
    loaded_model,
)

from .model_fixtures import (
    model_config,
    vllm_runner,
    remote_server,
)

# New session-level fixtures (Environment First + Model Centric)
from .session_fixtures import (
    environment,
    loaded_models,
    model_runner,
    model_name,
    environment_info,
)

__all__ = [
    # Legacy (function level)
    "scene_manager",
    "model_loader",
    "current_scene",
    "loaded_model",
    "vllm_runner",
    "remote_server",
    
    # New (session level)
    "environment",
    "loaded_models",
    "model_runner",
    "model_name",
    "environment_info",
]