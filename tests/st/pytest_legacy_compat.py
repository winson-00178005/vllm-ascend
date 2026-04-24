"""
pytest兼容插件 - 为tests/e2e/提供ST框架兼容性

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

功能:
- 自动为tests/e2e/下的测试添加场景标记
- 兼容现有conftest.py中的Runner类
- 支持在tests/e2e/中使用ST框架功能

用法:
    pytest tests/e2e/ -p st_legacy_compat --scene=SINGLE_CARD_310P
    pytest tests/e2e/310p/singlecard/ --list-scenes

无需修改tests/e2e/中的任何测试文件
"""
import pytest
from pathlib import Path
import sys
import logging

logger = logging.getLogger(__name__)

ST_FRAMEWORK_PATH = Path(__file__).parent


class LegacyCompatPlugin:
    """
    兼容插件 - 为tests/e2e/提供ST框架功能
    
    功能:
    - 从测试路径自动检测场景
    - 包装VllmRunner等类
    - 支持场景过滤
    """
    
    @pytest.hookimpl(tryfirst=True)
    def pytest_configure(self, config):
        """
        pytest配置钩子
        
        功能:
        - 添加ST框架路径到sys.path
        - 包装Runner类
        - 注册ST框架标记
        """
        self._setup_path()
        self._register_markers(config)
        self._wrap_runners(config)
    
    def _setup_path(self):
        """添加ST框架路径到sys.path"""
        st_path = str(ST_FRAMEWORK_PATH)
        if st_path not in sys.path:
            sys.path.insert(0, st_path)
        
        tests_path = str(ST_FRAMEWORK_PATH.parent)
        if tests_path not in sys.path:
            sys.path.insert(0, tests_path)
        
        logger.info(f"ST framework path added: {st_path}")
    
    def _register_markers(self, config):
        """注册ST框架标记"""
        config.addinivalue_line(
            "markers", "scene(name): 标记测试需要的场景"
        )
        config.addinivalue_line(
            "markers", "model(name): 标记测试需要的模型"
        )
        config.addinivalue_line(
            "markers", "device(name): 标记测试需要的设备类型"
        )
        config.addinivalue_line(
            "markers", "tags(name): 标记测试需要的场景标签"
        )
        config.addinivalue_line(
            "markers", "auto_scene: 自动检测场景的测试"
        )
    
    def _wrap_runners(self, config):
        """包装tests/e2e/conftest中的Runner类"""
        try:
            import tests.e2e.conftest as e2e_conftest
        except ImportError:
            try:
                import e2e.conftest as e2e_conftest
            except ImportError:
                logger.warning("tests.e2e.conftest not found")
                return
        
        from framework.legacy_adapter import auto_patch_conftest
        auto_patch_conftest(e2e_conftest)
        logger.info("Runner classes wrapped for compatibility")
    
    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, config, items):
        """
        测试收集钩子
        
        功能:
        - 处理--list-scenes和--list-models
        - 自动添加场景标记
        - 根据场景过滤测试
        """
        self._handle_list_options(config)
        self._auto_add_scene_markers(items)
        self._filter_by_scene(config, items)
    
    def _handle_list_options(self, config):
        """处理列表选项"""
        if config.getoption("--list-scenes", default=False):
            self._list_scenes()
            pytest.exit("列出场景后退出", 0)
        
        if config.getoption("--list-models", default=False):
            self._list_models()
            pytest.exit("列出模型后退出", 0)
    
    def _list_scenes(self):
        """列出所有可用场景"""
        try:
            from framework.scene_manager import SceneManager
            scene_manager = SceneManager()
            
            print("\n" + "=" * 60)
            print("可用场景列表 (ST Framework)")
            print("=" * 60)
            
            for scene in scene_manager.get_all_scenes():
                print(f"\n{scene.name}:")
                print(f"  设备: {scene.device_type}")
                print(f"  卡数: {scene.card_count}")
                print(f"  描述: {scene.description}")
            
            print("\n" + "=" * 60)
        except Exception as e:
            print(f"错误: {e}")
    
    def _list_models(self):
        """列出所有可用模型"""
        try:
            from framework.model_loader import ModelLoader
            model_loader = ModelLoader()
            
            print("\n" + "=" * 60)
            print("可用模型列表 (ST Framework)")
            print("=" * 60)
            
            for model in model_loader.get_all_models():
                print(f"\n{model.name}:")
                print(f"  路径: {model.model_path}")
                print(f"  设备: {', '.join(model.supported_devices)}")
            
            print("\n" + "=" * 60)
        except Exception as e:
            print(f"错误: {e}")
    
    def _auto_add_scene_markers(self, items):
        """自动为tests/e2e/下的测试添加场景标记"""
        from framework.scene_detector import SceneDetector
        
        for item in items:
            if item.get_closest_marker("scene"):
                continue
            
            test_path = str(Path(item.fspath))
            if "tests/e2e/" not in test_path.replace("\\", "/"):
                continue
            
            scene_name, scene_obj = SceneDetector.detect_from_path(test_path)
            
            if scene_name:
                item.add_marker(pytest.mark.scene(scene_name))
                item.add_marker(pytest.mark.auto_scene)
                item._st_auto_scene = scene_name
                logger.debug(f"Auto-added scene marker: {item.nodeid} -> {scene_name}")
    
    def _filter_by_scene(self, config, items):
        """根据--scene选项过滤测试"""
        scene_filter = config.getoption("--scene", default=None)
        
        if not scene_filter:
            return
        
        from framework.scene_manager import SceneManager
        scene_manager = SceneManager()
        scene_manager.set_current_scene(scene_filter)
        
        selected = []
        deselected = []
        
        for item in items:
            scene_marker = item.get_closest_marker("scene")
            
            if scene_marker:
                required_scene = scene_marker.args[0] if scene_marker.args else None
                if required_scene and self._scene_matches(required_scene, scene_filter):
                    selected.append(item)
                else:
                    deselected.append(item)
            else:
                auto_scene = getattr(item, '_st_auto_scene', None)
                if auto_scene and self._scene_matches(auto_scene, scene_filter):
                    selected.append(item)
                else:
                    deselected.append(item)
        
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = selected
        
        logger.info(f"Filtered: {len(selected)} selected, {len(deselected)} deselected")
    
    def _scene_matches(self, required_scene, filter_scene):
        """检查场景是否匹配"""
        if '|' in required_scene:
            return filter_scene in required_scene.split('|')
        return required_scene == filter_scene
    
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_setup(self, item):
        """测试setup钩子 - 检查场景"""
        yield
        
        scene_marker = item.get_closest_marker("scene")
        auto_scene = getattr(item, '_st_auto_scene', None)
        
        if scene_marker or auto_scene:
            from framework.scene_manager import SceneManager
            scene_manager = SceneManager()
            current_scene = scene_manager.get_current_scene()
            
            required_scene = None
            if scene_marker:
                required_scene = scene_marker.args[0] if scene_marker.args else None
            elif auto_scene:
                required_scene = auto_scene
            
            if current_scene and required_scene:
                if not self._scene_matches(required_scene, current_scene.name):
                    pytest.skip(
                        f"Scene mismatch: requires '{required_scene}', "
                        f"current is '{current_scene.name}'"
                    )


def pytest_addoption(parser):
    """添加命令行选项"""
    group = parser.getgroup('st-legacy-compat', 'ST框架兼容选项')
    
    group.addoption(
        "--scene",
        action="store",
        default=None,
        metavar="SCENE",
        help="指定测试场景"
    )
    
    group.addoption(
        "--model",
        action="store",
        default=None,
        metavar="MODEL",
        help="指定测试模型"
    )
    
    group.addoption(
        "--list-scenes",
        action="store_true",
        default=False,
        help="列出所有可用场景"
    )
    
    group.addoption(
        "--list-models",
        action="store_true",
        default=False,
        help="列出所有可用模型"
    )


def pytest_configure(config):
    """注册插件"""
    if config.pluginmanager.has_plugin('st_legacy_compat'):
        return
    
    plugin = LegacyCompatPlugin()
    config.pluginmanager.register(plugin, 'st_legacy_compat')