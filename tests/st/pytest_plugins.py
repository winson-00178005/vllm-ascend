"""
pytest插件 - ST测试框架pytest插件实现

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

提供pytest插件钩子实现，包括测试过滤、场景检查、报告增强等功能
"""
import pytest
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class STFrameworkPlugin:
    """
    ST测试框架pytest插件
    
    功能:
    - 自动设置场景（如果未指定）
    - 测试过滤（根据装饰器）
    - 测试报告增强
    - 兼容性包装
    """
    
    def pytest_configure(self, config):
        """
        pytest配置钩子
        
        功能:
        - 包装tests/e2e/conftest中的Runner类（兼容性）
        """
        # 如果不是兼容模式，包装现有Runner类
        if not config.getoption("--use-legacy-mode"):
            self._wrap_legacy_runners()
    
    def _wrap_legacy_runners(self):
        """
        包装tests/e2e/conftest中的Runner类
        
        使现有Runner类自动适配新框架
        """
        try:
            import sys
            from pathlib import Path
            
            # 确保tests路径在sys.path中
            tests_path = Path(__file__).parent.parent
            if str(tests_path) not in sys.path:
                sys.path.insert(0, str(tests_path))
            
            # 尝试导入并包装
            try:
                import tests.e2e.conftest as e2e_conftest
            except ImportError:
                try:
                    import e2e.conftest as e2e_conftest
                except ImportError:
                    logger.warning("tests.e2e.conftest not found, skipping Runner wrapping")
                    return
            
            from framework.legacy_adapter import auto_patch_conftest
            auto_patch_conftest(e2e_conftest)
            
            logger.info("Legacy Runner classes wrapped for compatibility")
            
        except Exception as e:
            logger.warning(f"Failed to wrap legacy runners: {e}")
    
    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, config, items):
        """
        测试收集钩子 - 为测试添加自动检测的场景标记
        
        Args:
            config: pytest配置
            items: 测试项列表
        """
        from framework.scene_detector import SceneDetector
        
        # 获取当前场景（如果已设置）
        from framework.scene_manager import SceneManager
        scene_manager = SceneManager()
        current_scene = scene_manager.get_current_scene()
        
        # 如果用户指定了场景，设置当前场景
        scene_name = config.getoption("--scene")
        if scene_name:
            scene_manager.set_current_scene(scene_name)
            current_scene = scene_manager.get_current_scene()
        
        # 为没有场景标记的测试添加自动检测标记
        for item in items:
            # 检查是否有场景标记
            has_scene_marker = item.get_closest_marker("scene")
            
            if not has_scene_marker:
                # 从测试文件路径自动检测场景
                test_path = str(Path(item.fspath))
                detected_scene, scene_obj = SceneDetector.detect_from_path(test_path)
                
                if detected_scene:
                    # 添加自动检测的场景信息
                    item._auto_detected_scene = detected_scene
                    logger.debug(f"Auto-detected scene for {item.nodeid}: {detected_scene}")
    
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_setup(self, item):
        """
        测试setup钩子 - 检查场景匹配
        
        Args:
            item: 测试项
        """
        yield
        
        # 检查自动检测的场景是否匹配当前场景
        auto_scene = getattr(item, '_auto_detected_scene', None)
        
        if auto_scene:
            from framework.scene_manager import SceneManager
            scene_manager = SceneManager()
            current_scene = scene_manager.get_current_scene()
            
            if current_scene and auto_scene != current_scene.name:
                pytest.skip(
                    f"Auto-detected scene '{auto_scene}' doesn't match "
                    f"current scene '{current_scene.name}'"
                )
    
    def pytest_runtest_makereport(self, item, call):
        """
        测试报告钩子 - 添加场景和模型信息
        
        Args:
            item: 测试项
            call: 测试调用信息
        """
        if call.when == "call":
            try:
                if call.excinfo is None and hasattr(call, '_result') and call._result is not None:
                    # 添加自动检测的场景信息
                    auto_scene = getattr(item, '_auto_detected_scene', None)
                    if auto_scene:
                        call._result._st_auto_scene = auto_scene
                    
                    # 添加装饰器标记的场景信息
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
    
    @pytest.hookimpl(trylast=True)
    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        """
        测试总结钩子 - 显示场景和模型信息
        
        Args:
            terminalreporter: 终端报告器
            exitstatus: 退出状态
            config: pytest配置
        """
        # 显示场景信息
        from framework.scene_manager import SceneManager
        scene_manager = SceneManager()
        current_scene = scene_manager.get_current_scene()
        
        if current_scene:
            terminalreporter.write_sep("=", "ST测试框架信息")
            terminalreporter.write_line(f"当前场景: {current_scene.name}")
            terminalreporter.write_line(f"设备类型: {current_scene.device_type}")
            terminalreporter.write_line(f"卡数: {current_scene.card_count}")
            terminalreporter.write_sep("=", "")
        
        # 统计跳过的测试
        skipped = terminalreporter.stats.get('skipped', [])
        if skipped:
            scene_skipped = sum(1 for report in skipped if hasattr(report, '_st_auto_scene'))
            if scene_skipped > 0:
                terminalreporter.write_line(f"因场景不匹配跳过的测试: {scene_skipped}")

# 注册插件（在conftest.py中调用）
def pytest_configure(config):
    """插件注册入口"""
    if not config.getoption("--use-legacy-mode"):
        config.pluginmanager.register(STFrameworkPlugin(), 'st_framework_plugin')