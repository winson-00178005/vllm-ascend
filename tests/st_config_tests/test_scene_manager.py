"""
ST 框架单元测试 - 独立运行，不依赖 vllm_ascend
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
        assert "DeepSeek-V2-Lite" not in models

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


class TestConfigValidation:
    """测试配置文件格式"""

    def test_scenes_yaml_has_scenes_key(self):
        """测试 scenes.yaml 包含 scenes 键"""
        config_path = Path(__file__).parent.parent / "e2e" / "st_config" / "scenes.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert "scenes" in data
        assert isinstance(data["scenes"], dict)

    def test_models_yaml_has_models_key(self):
        """测试 models.yaml 包含 models 键"""
        config_path = Path(__file__).parent.parent / "e2e" / "st_config" / "models.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert "models" in data
        assert isinstance(data["models"], dict)

    def test_scene_config_has_required_fields(self):
        """测试场景配置包含必需字段（使用 get_scene_config 解析继承）"""
        mgr = SceneManager()

        for scene_name in mgr.all_scenes:
            config = mgr.get_scene_config(scene_name)
            # 解析继承后应该包含 tp_size 和 dp_size
            assert "tp_size" in config, f"场景 {scene_name} 解析后缺少 tp_size"
            assert "dp_size" in config, f"场景 {scene_name} 解析后缺少 dp_size"
            assert "description" in config, f"场景 {scene_name} 缺少 description"

    def test_model_config_has_required_fields(self):
        """测试模型配置包含必需字段"""
        config_path = Path(__file__).parent.parent / "e2e" / "st_config" / "models.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        for model_name, model_config in data["models"].items():
            assert "model_id" in model_config, f"模型 {model_name} 缺少 model_id"
            assert "supported_scenes" in model_config, f"模型 {model_name} 缺少 supported_scenes"
            assert isinstance(model_config["supported_scenes"], list)


class TestSceneInheritance:
    """测试场景继承功能（Phase 2）"""

    def test_base_scene_inheritance(self):
        """测试基础场景继承"""
        mgr = SceneManager()
        # single_card_w8a8 继承 single_card
        config = mgr.get_scene_config("single_card_w8a8")
        assert config["tp_size"] == 1  # 继承自 single_card
        assert config["dp_size"] == 1  # 继承自 single_card
        assert config["quantization"] == "w8a8"  # 自有配置

    def test_env_vars_merge(self):
        """测试环境变量合并"""
        mgr = SceneManager()
        config = mgr.get_scene_config("single_card_w8a8")
        # 应该包含基础场景和自有环境变量
        assert "ASCEND_RT_VISIBLE_DEVICES" in config["env_vars"]
        assert "VLLM_TEST_FORCE_LOAD_FORMAT" in config["env_vars"]

    def test_features_list(self):
        """测试特性列表"""
        mgr = SceneManager()
        assert mgr.has_feature("quantization", "single_card_w8a8")
        assert not mgr.has_feature("quantization", "single_card")
        assert mgr.has_feature("graph_mode", "single_card_graph")

    def test_quantization_config(self):
        """测试量化配置获取"""
        mgr = SceneManager()
        assert mgr.get_quantization_config("single_card_w8a8") == "w8a8"
        assert mgr.get_quantization_config("single_card") is None

    def test_graph_mode_config(self):
        """测试图模式配置获取"""
        mgr = SceneManager()
        assert mgr.get_graph_mode("single_card_graph") == "acl"
        assert mgr.get_graph_mode("single_card") is None

    def test_combined_scene(self):
        """测试组合场景（量化 + 图模式）"""
        mgr = SceneManager()
        config = mgr.get_scene_config("single_card_w8a8_graph")
        assert config["quantization"] == "w8a8"
        assert config["graph_mode"] == "acl"
        assert "quantization" in config.get("features", [])
        assert "graph_mode" in config.get("features", [])
