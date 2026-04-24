"""
示例测试用例 - 演示ST测试框架的使用方式

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

包含:
- 使用装饰器的测试示例
- 使用fixtures的测试示例
- 兼容现有测试的示例
"""
import pytest
import sys
from pathlib import Path

# 确保framework模块可以导入
st_path = Path(__file__).parent.parent
if str(st_path) not in sys.path:
    sys.path.insert(0, str(st_path))

from framework.decorators import require_scene, require_model, require_device, require_tags

# ============================================================================
# 示例1: 使用装饰器的测试
# ============================================================================

@require_scene("SINGLE_CARD_310P|MULTI_CARD_310P_2")
def test_with_decorator_scene():
    """
    示例：使用装饰器标记测试需要的场景
    
    只在SINGLE_CARD_310P或MULTI_CARD_310P_2场景下运行
    其他场景会自动跳过
    """
    assert True, "装饰器测试通过"

@require_scene("SINGLE_CARD_310P")
@require_model("qwen-7b")
def test_with_decorator_scene_model():
    """
    示例：使用装饰器标记测试需要的场景和模型
    
    只在SINGLE_CARD_310P场景且qwen-7b模型下运行
    """
    assert True, "场景和模型装饰器测试通过"

@require_device("310P")
def test_with_decorator_device():
    """
    示例：使用装饰器标记测试需要的设备类型
    
    只在310P设备下运行
    """
    assert True, "设备装饰器测试通过"

@require_tags("single")
def test_with_decorator_tags():
    """
    示例：使用装饰器标记测试需要的场景标签
    
    只在包含'single'标签的场景下运行
    """
    assert True, "标签装饰器测试通过"

# ============================================================================
# 示例2: 使用fixtures的测试
# ============================================================================

class TestWithFixtures:
    """使用fixtures的测试类示例"""
    
    def test_scene_manager_fixture(self, scene_manager):
        """
        示例：使用scene_manager fixture
        
        fixture提供场景管理器实例
        """
        scenes = scene_manager.get_all_scenes()
        assert len(scenes) > 0, "至少应该有一个场景"
    
    def test_model_loader_fixture(self, model_loader):
        """
        示例：使用model_loader fixture
        
        fixture提供模型加载器实例
        """
        models = model_loader.get_all_models()
        assert len(models) > 0, "至少应该有一个模型"
    
    def test_current_scene_fixture(self, current_scene):
        """
        示例：使用current_scene fixture
        
        fixture提供当前场景对象
        """
        assert current_scene is not None, "当前场景应该存在"
        assert current_scene.name, "场景名称应该存在"
        assert current_scene.card_count > 0, "卡数应该大于0"
    
    def test_loaded_model_fixture(self, loaded_model):
        """
        示例：使用loaded_model fixture
        
        fixture提供当前加载的模型信息
        """
        assert loaded_model is not None, "应该有模型加载"
        assert 'name' in loaded_model, "模型信息应包含名称"
        assert 'config' in loaded_model, "模型信息应包含配置"
        assert 'model' in loaded_model, "模型信息应包含实例"

# ============================================================================
# 示例3: 使用pytest标记的测试
# ============================================================================

@pytest.mark.scene("MULTI_CARD_310P_4")
def test_with_pytest_marker_scene():
    """
    示例：使用pytest.mark.scene标记
    
    使用pytest原生标记方式标记场景要求
    """
    assert True, "pytest标记测试通过"

@pytest.mark.scene("SINGLE_CARD_310P")
@pytest.mark.model("llama-7b")
def test_with_pytest_markers():
    """
    示例：使用pytest原生标记
    
    可以组合多个标记
    """
    assert True, "组合pytest标记测试通过"

# ============================================================================
# 示例4: 参数化测试
# ============================================================================

@pytest.mark.parametrize("scene_name", ["SINGLE_CARD_310P", "MULTI_CARD_310P_2"])
def test_parametrized_scene(scene_name, current_scene):
    """
    示例：参数化测试
    
    注意：参数化测试需要检查当前场景是否匹配参数
    """
    if current_scene.name != scene_name:
        pytest.skip(f"场景不匹配: 当前是 {current_scene.name}, 需要的是 {scene_name}")
    assert current_scene.card_count > 0

# ============================================================================
# 示例5: 快速验证测试（用于验证框架功能）
# ============================================================================

def test_framework_quick_verify():
    """
    快速验证框架基本功能
    
    用于确认框架各组件工作正常
    """
    # 验证场景管理器
    from framework.scene_manager import SceneManager
    scene_manager = SceneManager()
    assert len(scene_manager.get_all_scenes()) >= 9, "应该有9个场景"
    
    # 验证模型加载器
    from framework.model_loader import ModelLoader
    model_loader = ModelLoader()
    assert len(model_loader.get_all_models()) >= 8, "应该有8个模型"
    
    # 验证场景检测器
    from framework.scene_detector import SceneDetector
    test_path = "tests/e2e/310p/singlecard/test_xxx.py"
    scene_name, scene = SceneDetector.detect_from_path(test_path)
    assert scene_name == "SINGLE_CARD_310P", "应该检测到SINGLE_CARD_310P场景"
    
    # 验证装饰器
    from framework.decorators import require_scene
    assert callable(require_scene), "require_scene应该是可调用的"

def test_config_files_loaded():
    """
    验证配置文件加载
    
    确保YAML配置文件正确加载
    """
    from framework.scene_manager import SceneManager
    from framework.model_loader import ModelLoader
    
    scene_manager = SceneManager()
    model_loader = ModelLoader()
    
    # 检查场景配置
    scene = scene_manager.get_scene("SINGLE_CARD_310P")
    assert scene is not None, "SINGLE_CARD_310P场景应该存在"
    assert scene.device_type == "310P", "设备类型应该是310P"
    
    # 检查模型配置
    model = model_loader.get_model_config("qwen-7b")
    assert model is not None, "qwen-7b模型应该存在"
    assert "310P" in model.supported_devices, "应该支持310P设备"