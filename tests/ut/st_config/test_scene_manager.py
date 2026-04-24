#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

"""
ST 框架单元测试 - 验证 SceneManager 和 ModelManager 功能
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tests.e2e.st_config.framework.model import AscendModelInfo, ModelManager
from tests.e2e.st_config.framework.scene import SceneManager


@pytest.fixture(autouse=True)
def reset_managers():
    """每个测试后重置管理器，避免状态污染"""
    yield
    SceneManager.reset_instance()
    ModelManager.reset_instance()


class TestSceneManager:
    """测试场景管理器"""

    def test_load_config_success(self):
        """测试成功加载场景配置"""
        mgr = SceneManager()
        assert "single_card" in mgr.all_scenes
        assert "multi_card_tp2" in mgr.all_scenes
        assert "ascend_310p" in mgr.all_scenes

    def test_get_config_returns_correct_values(self):
        """测试获取场景配置返回正确值"""
        mgr = SceneManager()
        config = mgr.get_scene_config("single_card")
        assert config["tp_size"] == 1
        assert config["dp_size"] == 1

    def test_is_valid_scene(self):
        """测试场景有效性检查"""
        mgr = SceneManager()
        assert mgr.is_valid_scene("single_card") is True
        assert mgr.is_valid_scene("invalid_scene") is False

    def test_cli_scene_priority(self):
        """测试命令行参数优先级最高"""
        mgr = SceneManager(cli_scene="multi_card_tp2")
        assert mgr.current == "multi_card_tp2"

    def test_env_var_priority(self):
        """测试环境变量优先级（高于自动探测）"""
        with patch.dict(os.environ, {"ST_SCENE": "multi_card_tp4"}):
            SceneManager.reset_instance()
            mgr = SceneManager()
            assert mgr.current == "multi_card_tp4"

    def test_auto_detect_fallback(self):
        """测试自动探测失败时返回默认值"""
        with patch.object(SceneManager, "_detect_via_npu_smi", return_value=1):
            with patch.object(SceneManager, "_detect_hardware", return_value="910b"):
                with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "", "ST_SCENE": ""}, clear=False):
                    # 清除 ST_SCENE 避免干扰
                    os.environ.pop("ST_SCENE", None)
                    os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
                    SceneManager.reset_instance()
                    mgr = SceneManager()
                    assert mgr.current == "single_card"

    def test_apply_env_vars(self):
        """测试应用环境变量"""
        mgr = SceneManager()
        mgr.apply_env_vars("single_card")
        assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "0"

    def test_get_config_invalid_scene_raises_error(self):
        """测试获取无效场景配置时抛出异常"""
        mgr = SceneManager()
        with pytest.raises(ValueError, match="未知场景"):
            mgr.get_scene_config("invalid_scene")


class TestModelManager:
    """测试模型管理器"""

    def test_load_config_success(self):
        """测试成功加载模型配置"""
        mgr = ModelManager()
        assert "Qwen3-8B-Base" in mgr.all_models
        assert "DeepSeek-V2-Lite" in mgr.all_models

    def test_get_models_for_scene(self):
        """测试获取场景支持的模型列表"""
        mgr = ModelManager()
        models = mgr.get_models_for_scene("single_card")
        assert "Qwen3-8B-Base" in models
        assert "DeepSeek-V2-Lite" not in models  # DeepSeek 不支持 single_card

    def test_is_supported(self):
        """测试模型场景支持检查"""
        mgr = ModelManager()
        assert mgr.is_supported("Qwen3-8B-Base", "single_card") is True
        assert mgr.is_supported("DeepSeek-V2-Lite", "single_card") is False
        assert mgr.is_supported("NonExistentModel", "single_card") is False

    def test_get_info_returns_correct_data(self):
        """测试获取模型信息返回正确数据"""
        mgr = ModelManager()
        info = mgr.get_info("Qwen3-8B-Base")
        assert info is not None
        assert info.model_id == "Qwen/Qwen3-8B-Base"
        assert info.dtype == "bfloat16"
        assert info.min_gpu_gb == 16
        assert "single_card" in info.supported_scenes

    def test_model_info_dataclass(self):
        """测试 AscendModelInfo 数据类"""
        info = AscendModelInfo(
            name="TestModel",
            model_id="test/model",
            architecture="TestForCausalLM",
            dtype="float16",
            supported_scenes=["single_card"],
            enable_test=True,
            is_moe=False,
            min_gpu_gb=8,
        )
        assert info.name == "TestModel"
        assert info.is_moe is False
        assert info.min_gpu_gb == 8

    def test_disabled_model_not_in_scene_models(self):
        """测试禁用的模型不在场景模型列表中"""
        # 注意：当前配置中所有模型都是 enable_test: true
        # 这里验证逻辑正确性
        mgr = ModelManager()
        models = mgr.get_models_for_scene("single_card")
        # 验证返回的模型都是 enable_test 为 True 的
        for model_name in models:
            info = mgr.get_info(model_name)
            assert info.enable_test is True


class TestConfigValidation:
    """测试配置文件格式"""

    def test_scenes_yaml_has_scenes_key(self):
        """测试 scenes.yaml 包含 scenes 键"""
        config_path = Path(__file__).parent.parent / "st_config" / "scenes.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert "scenes" in data
        assert isinstance(data["scenes"], dict)

    def test_models_yaml_has_models_key(self):
        """测试 models.yaml 包含 models 键"""
        config_path = Path(__file__).parent.parent / "st_config" / "models.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert "models" in data
        assert isinstance(data["models"], dict)

    def test_scene_config_has_required_fields(self):
        """测试场景配置包含必需字段"""
        config_path = Path(__file__).parent.parent / "st_config" / "scenes.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        for scene_name, scene_config in data["scenes"].items():
            assert "description" in scene_config, f"场景 {scene_name} 缺少 description"
            assert "tp_size" in scene_config, f"场景 {scene_name} 缺少 tp_size"
            assert "dp_size" in scene_config, f"场景 {scene_name} 缺少 dp_size"

    def test_model_config_has_required_fields(self):
        """测试模型配置包含必需字段"""
        config_path = Path(__file__).parent.parent / "st_config" / "models.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        for model_name, model_config in data["models"].items():
            assert "model_id" in model_config, f"模型 {model_name} 缺少 model_id"
            assert "supported_scenes" in model_config, f"模型 {model_name} 缺少 supported_scenes"
            assert isinstance(model_config["supported_scenes"], list)
