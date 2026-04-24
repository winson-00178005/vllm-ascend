#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#

"""
ST Framework 装饰器功能验证脚本

运行方式:
    python tests/st/test_framework_validation.py

无需 NPU 硬件，仅验证框架组件功能
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from tests.st.framework.scene_manager import SceneManager, SceneInfo
from tests.st.framework.model_config import ModelConfig, ModelInfo
from tests.st.framework.decorators import require_scene, require_model, require_scene_and_model
from tests.st.framework.compat import scene_aware_parametrize, get_models_for_current_scene
