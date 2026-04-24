#!/usr/bin/env python3
"""
ST测试框架核心功能验证脚本

运行此脚本可以快速验证核心框架是否正常工作:
    python tests/st/verify_framework.py
"""

import sys
import os

# 确保可以导入框架模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def verify_scene_manager():
    """验证场景管理器"""
    print("=" * 60)
    print("验证场景管理器")
    print("=" * 60)
    
    try:
        from tests.st.framework.scene_manager import SceneManager
        
        # 获取场景管理器实例
        scene_manager = SceneManager()
        print(f"✅ 场景管理器初始化成功")
        
        # 获取所有场景
        scenes = scene_manager.get_all_scenes()
        print(f"✅ 加载场景数量: {len(scenes)}")
        
        # 列出场景
        for scene in scenes:
            print(f"  - {scene.name}: {scene.description} ({scene.device_type}, {scene.card_count}卡)")
        
        # 设置当前场景
        scene_manager.set_current_scene("SINGLE_CARD_310P")
        current_scene = scene_manager.get_current_scene()
        print(f"✅ 设置当前场景: {current_scene.name}")
        
        # 场景匹配测试
        test_scenes = "SINGLE_CARD_310P|MULTI_CARD_310P_2"
        matches = current_scene.matches(test_scenes)
        print(f"✅ 场景匹配测试: {test_scenes} -> {matches}")
        
        # 按标签查询
        scenes_by_tag = scene_manager.get_scenes_by_tag("single")
        print(f"✅ 按标签查询 (single): {len(scenes_by_tag)}个场景")
        
        return True
        
    except Exception as e:
        print(f"❌ 场景管理器验证失败: {e}")
        return False

def verify_model_loader():
    """验证模型加载器"""
    print("\n" + "=" * 60)
    print("验证模型加载器")
    print("=" * 60)
    
    try:
        from tests.st.framework.model_loader import ModelLoader
        
        # 获取模型加载器实例
        model_loader = ModelLoader()
        print(f"✅ 模型加载器初始化成功")
        
        # 获取所有模型
        models = model_loader.get_all_models()
        print(f"✅ 加载模型数量: {len(models)}")
        
        # 列出模型
        for model in models:
            print(f"  - {model.name}: {model.model_path}")
        
        # 获取特定模型配置
        config = model_loader.get_model_config("qwen-7b")
        if config:
            print(f"✅ 获取模型配置: {config.name}")
            print(f"  路径: {config.model_path}")
            print(f"  支持设备: {', '.join(config.supported_devices)}")
            print(f"  最小内存: {config.min_memory}")
        
        # 按设备查询
        models_by_device = model_loader.get_models_by_device("310P")
        print(f"✅ 按设备查询 (310P): {len(models_by_device)}个模型")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载器验证失败: {e}")
        return False

def verify_decorators():
    """验证装饰器"""
    print("\n" + "=" * 60)
    print("验证测试装饰器")
    print("=" * 60)
    
    try:
        from tests.st.framework.decorators import require_scene, require_model, require_device
        
        print(f"✅ require_scene装饰器可用")
        print(f"✅ require_model装饰器可用")
        print(f"✅ require_device装饰器可用")
        
        # 创建测试函数
        @require_scene("SINGLE_CARD_310P")
        def test_function():
            pass
        
        print(f"✅ 装饰器应用成功: test_function._required_scenes = {test_function._required_scenes}")
        
        return True
        
    except Exception as e:
        print(f"❌ 装饰器验证失败: {e}")
        return False

def verify_scene_detector():
    """验证场景检测器"""
    print("\n" + "=" * 60)
    print("验证场景检测器")
    print("=" * 60)
    
    try:
        from tests.st.framework.scene_detector import SceneDetector
        
        # 测试路径检测
        test_paths = [
            "tests/e2e/310p/singlecard/test_xxx.py",
            "tests/e2e/multicard/2-cards/test_yyy.py",
            "tests/e2e/nightly/single_node/test_zzz.py"
        ]
        
        for path in test_paths:
            scene_name, scene = SceneDetector.detect_from_path(path)
            print(f"✅ 路径检测: {path} -> {scene_name}")
        
        # 测试代码检测
        test_code = """
        with VllmRunner("Qwen/Qwen3-8B", tensor_parallel_size=2, enforce_eager=True):
            pass
        """
        model_name, params = SceneDetector.detect_from_code(test_code)
        print(f"✅ 代码检测: model={model_name}, tp={params.get('tensor_parallel_size')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 场景检测器验证失败: {e}")
        return False

def verify_config_files():
    """验证配置文件"""
    print("\n" + "=" * 60)
    print("验证配置文件")
    print("=" * 60)
    
    try:
        import yaml
        from pathlib import Path
        
        config_dir = Path(__file__).parent / "config"
        
        # 验证场景配置
        scenes_file = config_dir / "scenes.yaml"
        with open(scenes_file) as f:
            scenes_config = yaml.safe_load(f)
        print(f"✅ scenes.yaml: {len(scenes_config['scenes'])}个场景")
        
        # 验证模型配置
        models_file = config_dir / "models.yaml"
        with open(models_file) as f:
            models_config = yaml.safe_load(f)
        print(f"✅ models.yaml: {len(models_config['models'])}个模型")
        
        # 验证环境配置
        env_file = config_dir / "environments.yaml"
        with open(env_file) as f:
            env_config = yaml.safe_load(f)
        print(f"✅ environments.yaml: 配置正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件验证失败: {e}")
        return False

def main():
    """主验证流程"""
    print("\n" + "🎉" * 20)
    print("vllm-ascend ST测试框架核心功能验证")
    print("🎉" * 20 + "\n")
    
    results = []
    
    # 验证各个组件
    results.append(("配置文件", verify_config_files()))
    results.append(("场景管理器", verify_scene_manager()))
    results.append(("模型加载器", verify_model_loader()))
    results.append(("测试装饰器", verify_decorators()))
    results.append(("场景检测器", verify_scene_detector()))
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    total = len(results)
    passed = sum(1 for _, result in results if result)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有验证通过！核心框架正常工作！")
        print("\n下一步: 开始阶段2 - 创建pytest组件")
        return 0
    else:
        print("\n❌ 验证失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)