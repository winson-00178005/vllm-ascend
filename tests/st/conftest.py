"""
pytest配置文件 - ST测试框架入口

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

提供pytest命令行选项、配置、标记注册等功能
"""
import pytest
import logging
import sys
from pathlib import Path

# 确保框架模块可以导入
st_path = Path(__file__).parent
if str(st_path) not in sys.path:
    sys.path.insert(0, str(st_path))

logger = logging.getLogger(__name__)

# ============================================================================
# 命令行选项
# ============================================================================

def pytest_addoption(parser):
    """
    添加pytest命令行选项
    
    支持的选项:
    --scene: 指定测试场景
    --model: 指定测试模型
    --list-scenes: 列出所有可用场景
    --list-models: 列出所有可用模型
    --use-legacy-mode: 使用兼容模式（不使用新框架）
    """
    group = parser.getgroup('st-framework', 'ST测试框架选项')
    
    # 场景选项
    group.addoption(
        "--scene",
        action="store",
        default=None,
        metavar="SCENE_NAME",
        help="指定测试场景，例如: --scene=SINGLE_CARD_310P"
    )
    
    # 模型选项
    group.addoption(
        "--model",
        action="store",
        default=None,
        metavar="MODEL_NAME",
        help="指定测试模型，例如: --model=qwen-7b"
    )
    
    # 列表选项
    group.addoption(
        "--list-scenes",
        action="store_true",
        default=False,
        help="列出所有可用场景并退出"
    )
    
    group.addoption(
        "--list-models",
        action="store_true",
        default=False,
        help="列出所有可用模型并退出"
    )
    
    # 兼容模式选项
    group.addoption(
        "--use-legacy-mode",
        action="store_true",
        default=False,
        help="使用兼容模式运行测试（不使用新框架功能）"
    )
    
    # 自动检测选项
    group.addoption(
        "--auto-detect-scene",
        action="store_true",
        default=True,
        help="自动检测当前场景（默认开启）"
    )

# ============================================================================
# pytest配置
# ============================================================================

def pytest_configure(config):
    """
    pytest配置钩子
    
    功能:
    - 注册自定义标记
    - 配置日志
    - 加载插件
    """
    # 注册自定义标记
    config.addinivalue_line(
        "markers", "scene(name): 标记测试需要的场景，例如: @pytest.mark.scene('SINGLE_CARD_310P')"
    )
    config.addinivalue_line(
        "markers", "model(name): 标记测试需要的模型，例如: @pytest.mark.model('qwen-7b')"
    )
    config.addinivalue_line(
        "markers", "device(name): 标记测试需要的设备类型，例如: @pytest.mark.device('310P')"
    )
    config.addinivalue_line(
        "markers", "tags(name): 标记测试需要场景包含的标签，例如: @pytest.mark.tags('single')"
    )
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 加载兼容插件（如果不是兼容模式）
    if not config.getoption("--use-legacy-mode"):
        try:
            from pytest_plugins import STFrameworkPlugin
            config.pluginmanager.register(STFrameworkPlugin(), 'st_framework_plugin')
            logger.info("ST Framework Plugin registered")
        except ImportError as e:
            logger.warning(f"Failed to load ST Framework Plugin: {e}")

# ============================================================================
# 测试收集钩子
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    测试收集钩子 - 处理列表选项和测试过滤
    
    功能:
    - 处理--list-scenes和--list-models选项
    - 根据场景标记过滤测试（如果使用兼容模式）
    """
    # 处理--list-scenes选项
    if config.getoption("--list-scenes"):
        _list_scenes()
        pytest.exit("列出场景后退出")
    
    # 处理--list-models选项
    if config.getoption("--list-models"):
        _list_models()
        pytest.exit("列出模型后退出")
    
    # 如果使用兼容模式，进行测试过滤
    if config.getoption("--use-legacy-mode"):
        _filter_tests_by_scene(config, items)

def _list_scenes():
    """
    列出所有可用场景
    
    输出格式:
    - 场景名称
    - 场景描述
    - 设备类型
    - 卡数
    - 支持的模型
    """
    try:
        from framework.scene_manager import SceneManager
        scene_manager = SceneManager()
        
        print("\n" + "=" * 80)
        print("可用场景列表")
        print("=" * 80)
        
        for scene in scene_manager.get_all_scenes():
            print(f"\n场景名称: {scene.name}")
            print(f"  描述: {scene.description}")
            print(f"  设备类型: {scene.device_type}")
            print(f"  卡数: {scene.card_count}")
            if scene.node_count > 1:
                print(f"  节点数: {scene.node_count}")
            print(f"  支持的模型: {', '.join(scene.models)}")
            print(f"  标签: {', '.join(scene.tags)}")
        
        print("\n" + "=" * 80)
        print(f"总计: {len(scene_manager.get_all_scenes())} 个场景")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Failed to list scenes: {e}")
        print(f"\n错误: 无法加载场景配置 - {e}")

def _list_models():
    """
    列出所有可用模型
    
    输出格式:
    - 模型名称
    - 模型路径
    - 支持的设备
    - 最小内存要求
    - 是否需要多卡
    """
    try:
        from framework.model_loader import ModelLoader
        model_loader = ModelLoader()
        
        print("\n" + "=" * 80)
        print("可用模型列表")
        print("=" * 80)
        
        for model in model_loader.get_all_models():
            print(f"\n模型名称: {model.name}")
            print(f"  路径: {model.model_path}")
            print(f"  支持设备: {', '.join(model.supported_devices)}")
            print(f"  最小内存: {model.min_memory}")
            print(f"  最大批次: {model.max_batch_size}")
            if model.requires_multi_card:
                print(f"  需要多卡: 是")
            print(f"  标签: {', '.join(model.tags)}")
        
        print("\n" + "=" * 80)
        print(f"总计: {len(model_loader.get_all_models())} 个模型")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        print(f"\n错误: 无法加载模型配置 - {e}")

def _filter_tests_by_scene(config, items):
    """
    根据场景过滤测试（兼容模式）
    
    Args:
        config: pytest配置对象
        items: 测试项列表
    """
    scene_name = config.getoption("--scene")
    
    if not scene_name:
        return
    
    from framework.scene_detector import SceneDetector
    
    selected = []
    deselected = []
    
    for item in items:
        # 检查测试是否标记了场景
        scene_marker = item.get_closest_marker("scene")
        
        if scene_marker:
            # 测试有场景标记
            required_scene = scene_marker.args[0] if scene_marker.args else scene_marker.kwargs.get('name')
            if scene_name == required_scene or scene_name in required_scene.split('|'):
                selected.append(item)
            else:
                deselected.append(item)
        else:
            # 测试没有场景标记，尝试从路径检测
            test_path = str(Path(item.fspath))
            detected_scene, _ = SceneDetector.detect_from_path(test_path)
            
            if detected_scene == scene_name:
                selected.append(item)
            else:
                deselected.append(item)
    
    # 更新测试列表
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected

# ============================================================================
# 测试报告钩子
# ============================================================================

def pytest_runtest_makereport(item, call):
    """
    测试报告钩子 - 添加场景和模型信息到测试报告
    
    Args:
        item: 测试项
        call: 测试调用信息
    """
    if call.when == "call":
        try:
            if call.excinfo is None and hasattr(call, '_result') and call._result is not None:
                # 添加场景信息
                scene_marker = item.get_closest_marker("scene")
                if scene_marker:
                    scene_name = scene_marker.args[0] if scene_marker.args else scene_marker.kwargs.get('name')
                    call._result._st_scene = scene_name
                
                # 添加模型信息
                model_marker = item.get_closest_marker("model")
                if model_marker:
                    model_name = model_marker.args[0] if model_marker.args else model_marker.kwargs.get('name')
                    call._result._st_model = model_name
        except (AttributeError, TypeError):
            pass

# ============================================================================
# 命令行帮助
# ============================================================================

def show_st_help():
    """
    显示ST框架帮助信息（非pytest hook，手动调用）
    """
    print("\nST测试框架使用帮助:")
    print("\n命令行选项:")
    print("  --scene=SCENE_NAME      指定测试场景")
    print("  --model=MODEL_NAME      指定测试模型")
    print("  --list-scenes           列出所有可用场景")
    print("  --list-models           列出所有可用模型")
    print("  --use-legacy-mode       使用兼容模式")
    
    print("\n示例:")
    print("  pytest tests/st/ --list-scenes")
    print("  pytest tests/st/ --scene=SINGLE_CARD_310P")
    print("  pytest tests/st/ --scene=MULTI_CARD_310P_2 --model=qwen-7b")
    print("  pytest tests/e2e/ --scene=SINGLE_CARD_310P --use-legacy-mode")
    
    print("\n装饰器使用:")
    print("  @pytest.mark.scene('SINGLE_CARD_310P')")
    print("  @pytest.mark.model('qwen-7b')")
    print("  @pytest.mark.device('310P')")
    print("  @pytest.mark.tags('single')")

# ============================================================================
# 导入fixtures
# ============================================================================

# 确保fixtures模块被pytest发现
pytest_plugins = ["fixtures.scene_fixtures", "fixtures.model_fixtures"]