"""
ST测试框架自检脚本 - 验证框架功能与设计要求是否相符

运行方式:
    python tests/st/self_check.py

验证内容:
    1. 场景配置完整性（9场景）
    2. 模型配置完整性（8模型）
    3. 装饰器功能正确性
    4. 场景自动检测功能
    5. 兼容性适配功能
    6. 文件完整性
"""
import sys
import os
from pathlib import Path

ST_PATH = Path(__file__).parent
sys.path.insert(0, str(ST_PATH))

results = []

def check(name, condition, details=None):
    """记录检查结果"""
    status = "PASS" if condition else "FAIL"
    result = f"[{status}] {name}"
    if details:
        result += f": {details}"
    results.append((name, condition, details))
    print(result)
    return condition

print("=" * 70)
print("ST测试框架自检报告")
print("=" * 70)
print()

# ============================================================================
# 1. 场景配置完整性验证
# ============================================================================
print("[1] 场景配置完整性验证")
print("-" * 70)

try:
    from framework.scene_manager import SceneManager
    sm = SceneManager()
    
    check("场景数量", len(sm.get_all_scenes()) == 9, f"共{len(sm.get_all_scenes())}个场景")
    
    check("310P单卡场景", sm.get_scene("SINGLE_CARD_310P") is not None)
    check("310P 2卡场景", sm.get_scene("MULTI_CARD_310P_2") is not None)
    check("310P 4卡场景", sm.get_scene("MULTI_CARD_310P_4") is not None)
    
    check("910B单卡场景", sm.get_scene("SINGLE_CARD_910B") is not None)
    check("910B 2卡场景", sm.get_scene("MULTI_CARD_910B_2") is not None)
    
    check("单机多卡场景", sm.get_scene("SINGLE_NODE_MULTI_CARD") is not None)
    check("多机多卡场景", sm.get_scene("MULTI_NODE_MULTI_CARD") is not None)
    
    check("Nightly场景", sm.get_scene("NIGHTLY_SINGLE_NODE") is not None)
    check("Weekly场景", sm.get_scene("WEEKLY_SINGLE_NODE") is not None)
    
    scene_310p = sm.get_scene("SINGLE_CARD_310P")
    if scene_310p:
        check("场景属性完整", 
              scene_310p.description and 
              scene_310p.device_type == "310P" and 
              scene_310p.card_count == 1 and 
              len(scene_310p.models) > 0 and 
              len(scene_310p.tags) > 0,
              f"device={scene_310p.device_type}, cards={scene_310p.card_count}")
    
    check("按标签查询", len(sm.get_scenes_by_tag("single")) >= 2)
    check("按设备查询", len(sm.get_scenes_by_device("310P")) >= 6)
    
except Exception as e:
    check("场景管理器加载", False, str(e))

print()

# ============================================================================
# 2. 模型配置完整性验证
# ============================================================================
print("[2] 模型配置完整性验证")
print("-" * 70)

try:
    from framework.model_loader import ModelLoader
    ml = ModelLoader()
    
    check("模型数量", len(ml.get_all_models()) == 8, f"共{len(ml.get_all_models())}个模型")
    
    check("qwen-7b模型", ml.get_model_config("qwen-7b") is not None)
    check("qwen2-7b模型", ml.get_model_config("qwen2-7b") is not None)
    check("qwen3-8b模型", ml.get_model_config("qwen3-8b") is not None)
    check("qwen3-30b-moe模型", ml.get_model_config("qwen3-30b-moe") is not None)
    check("qwen-72b模型", ml.get_model_config("qwen-72b") is not None)
    
    check("llama-7b模型", ml.get_model_config("llama-7b") is not None)
    check("llama-13b模型", ml.get_model_config("llama-13b") is not None)
    
    check("deepseek-67b模型", ml.get_model_config("deepseek-67b") is not None)
    
    model_qwen = ml.get_model_config("qwen-7b")
    if model_qwen:
        check("模型属性完整",
              model_qwen.model_path and 
              len(model_qwen.supported_devices) > 0 and 
              model_qwen.min_memory and 
              model_qwen.max_batch_size > 0,
              f"path={model_qwen.model_path}, devices={model_qwen.supported_devices}")
    
    check("按设备查询模型", len(ml.get_models_by_device("310P")) >= 6)
    
    model_72b = ml.get_model_config("qwen-72b")
    if model_72b:
        check("多卡模型标记", model_72b.requires_multi_card == True)
    
except Exception as e:
    check("模型加载器加载", False, str(e))

print()

# ============================================================================
# 3. 装饰器功能验证
# ============================================================================
print("[3] 装饰器功能验证")
print("-" * 70)

try:
    from framework.decorators import require_scene, require_model, require_device, require_tags
    from framework.scene_manager import SceneManager
    
    sm = SceneManager()
    sm.set_current_scene("SINGLE_CARD_310P")
    
    @require_scene("SINGLE_CARD_310P")
    def test_match_scene():
        return "passed"
    
    result = test_match_scene()
    check("require_scene匹配", result == "passed")
    
    @require_scene("SINGLE_CARD_310P|MULTI_CARD_310P_2")
    def test_multi_scene():
        return "passed"
    
    result = test_multi_scene()
    check("require_scene多场景", result == "passed", "SINGLE_CARD_310P匹配")
    
    @require_model("qwen-7b")
    def test_model():
        return "passed"
    
    result = test_model()
    check("require_model功能", result == "passed")
    
    @require_device("310P")
    def test_device():
        return "passed"
    
    result = test_device()
    check("require_device功能", result == "passed")
    
    @require_tags("single")
    def test_tags():
        return "passed"
    
    result = test_tags()
    check("require_tags功能", result == "passed")
    
except Exception as e:
    check("装饰器加载", False, str(e))

print()

# ============================================================================
# 4. 场景自动检测验证
# ============================================================================
print("[4] 场景自动检测验证")
print("-" * 70)

try:
    from framework.scene_detector import SceneDetector
    
    test_paths = [
        ("tests/e2e/310p/singlecard/test_xxx.py", "SINGLE_CARD_310P"),
        ("tests/e2e/multicard/2-cards/test_xxx.py", "MULTI_CARD_310P_2"),
        ("tests/e2e/multicard/4-cards/test_xxx.py", "MULTI_CARD_310P_4"),
        ("tests/e2e/nightly/test_xxx.py", "NIGHTLY_SINGLE_NODE"),
        ("tests/e2e/weekly/test_xxx.py", "WEEKLY_SINGLE_NODE"),
    ]
    
    for path, expected in test_paths:
        scene_name, _ = SceneDetector.detect_from_path(path)
        check(f"路径检测: {path}", scene_name == expected, f"期望={expected}, 结果={scene_name}")
    
    scene_info = SceneDetector.detect_all("tests/e2e/310p/singlecard/test.py", "")
    check("综合检测功能", "scene_name" in scene_info and "scene" in scene_info and scene_info["scene_name"] == "SINGLE_CARD_310P")
    
except Exception as e:
    check("场景检测器加载", False, str(e))

print()

# ============================================================================
# 5. 兼容性适配验证
# ============================================================================
print("[5] 兼容性适配验证")
print("-" * 70)

try:
    from framework.legacy_adapter import LegacyTestWrapper, adapt_legacy_test
    
    check("LegacyTestWrapper类", LegacyTestWrapper is not None)
    check("wrap_vllm_runner方法", hasattr(LegacyTestWrapper, 'wrap_vllm_runner'))
    check("wrap_remote_server方法", hasattr(LegacyTestWrapper, 'wrap_remote_server'))
    check("adapt_legacy_test装饰器", callable(adapt_legacy_test))
    
    from framework.legacy_adapter import auto_patch_conftest
    check("auto_patch_conftest函数", callable(auto_patch_conftest))
    
except Exception as e:
    check("兼容适配器加载", False, str(e))

print()

# ============================================================================
# 6. 文件完整性验证
# ============================================================================
print("[6] 文件完整性验证")
print("-" * 70)

required_files = [
    "config/scenes.yaml",
    "config/models.yaml",
    "config/environments.yaml",
    "framework/__init__.py",
    "framework/scene_manager.py",
    "framework/model_loader.py",
    "framework/decorators.py",
    "framework/scene_detector.py",
    "framework/legacy_adapter.py",
    "fixtures/__init__.py",
    "fixtures/scene_fixtures.py",
    "fixtures/model_fixtures.py",
    "conftest.py",
    "pytest_plugins.py",
    "pytest_legacy_compat.py",
    "testcases/__init__.py",
    "testcases/test_examples.py",
]

for file in required_files:
    path = ST_PATH / file
    check(f"文件存在: {file}", path.exists())

print()

# ============================================================================
# 7. 汇总报告
# ============================================================================
print("=" * 70)
print("自检汇总报告")
print("=" * 70)

passed = sum(1 for _, cond, _ in results if cond)
failed = sum(1 for _, cond, _ in results if not cond)
total = len(results)

print(f"总检查项: {total}")
print(f"通过: {passed} ({passed/total*100:.1f}%)")
print(f"失败: {failed} ({failed/total*100:.1f}%)")
print()

if failed > 0:
    print("失败项详情:")
    for name, cond, details in results:
        if not cond:
            print(f"  - {name}: {details}")
    print()

print("=" * 70)
if failed == 0:
    print("所有检查项通过，框架功能与设计要求相符")
else:
    print(f"有{failed}项检查失败，需要修复")
print("=" * 70)

sys.exit(0 if failed == 0 else 1)