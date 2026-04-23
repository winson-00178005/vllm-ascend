#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#

"""
ST Framework 装饰器功能验证脚本

运行方式:
    python tests/e2e/st/test_framework_validation.py

无需 NPU 硬件，仅验证框架组件功能
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from tests.e2e.st.framework.scene_manager import SceneManager, SceneInfo
from tests.e2e.st.framework.model_config import ModelConfig, ModelInfo
from tests.e2e.st.framework.decorators import require_scene, require_model, require_scene_and_model
from tests.e2e.st.framework.compat import scene_aware_parametrize, get_models_for_current_scene


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results = []

    def add_pass(self, name, detail=""):
        self.passed += 1
        self.results.append(f"  [PASS] {name}" + (f" - {detail}" if detail else ""))
        print(f"  [PASS] {name}" + (f" - {detail}" if detail else ""))

    def add_fail(self, name, detail=""):
        self.failed += 1
        self.results.append(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))

    def add_skip(self, name, detail=""):
        self.skipped += 1
        self.results.append(f"  [SKIP] {name}" + (f" - {detail}" if detail else ""))
        print(f"  [SKIP] {name}" + (f" - {detail}" if detail else ""))

    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'=' * 60}")
        print(f"Total: {total} | Passed: {self.passed} | Failed: {self.failed} | Skipped: {self.skipped}")
        print(f"{'=' * 60}")
        return self.failed == 0


def setup_singleton(scene_id, model_name):
    """设置单例为指定场景和模型"""
    SceneManager.reset_instance()
    ModelConfig.reset_instance()

    scene_mgr = SceneManager.get_instance()
    scene_mgr.set_current_scene(scene_id)

    model_cfg = ModelConfig.get_instance()
    model_cfg.current_model = model_name

    return scene_mgr, model_cfg


class TestDecoratorBehavior:
    """测试装饰器在实际调用时的行为"""

    def __init__(self, result: TestResult):
        self.result = result

    def test_require_scene_match(self):
        """测试 require_scene 场景匹配时正常执行"""
        scene_mgr, model_cfg = setup_singleton("SINGLECARD", "Qwen/Qwen3-8B")

        @require_scene("SINGLECARD")
        def dummy_test():
            return "executed"

        try:
            result = dummy_test()
            if result == "executed":
                self.result.add_pass("require_scene: 场景匹配时正常执行", f"返回: {result}")
            else:
                self.result.add_fail("require_scene: 场景匹配时正常执行", f"结果异常: {result}")
        except Exception as e:
            self.result.add_fail("require_scene: 场景匹配时正常执行", str(e))

    def test_require_scene_mismatch(self):
        """测试 require_scene 场景不匹配时抛出 Skip"""
        import pytest
        scene_mgr, model_cfg = setup_singleton("SINGLECARD", "Qwen/Qwen3-8B")

        @require_scene("MULTICARD_4Cards")
        def dummy_test():
            return "executed"

        try:
            result = dummy_test()
            self.result.add_fail("require_scene: 场景不匹配时应跳过", f"未跳过，返回: {result}")
        except pytest.skip.Exception as e:
            self.result.add_pass("require_scene: 场景不匹配时跳过", f"正确抛出 Skip")
        except Exception as e:
            self.result.add_fail("require_scene: 场景不匹配时跳过", f"异常类型错误: {type(e)}")

    def test_require_model_match(self):
        """测试 require_model 模型匹配时正常执行"""
        scene_mgr, model_cfg = setup_singleton("SINGLECARD", "Qwen/Qwen3-8B")

        @require_model("Qwen/Qwen3-8B")
        def dummy_test():
            return "executed"

        try:
            result = dummy_test()
            if result == "executed":
                self.result.add_pass("require_model: 模型匹配时正常执行", f"返回: {result}")
            else:
                self.result.add_fail("require_model: 模型匹配时正常执行", f"结果异常: {result}")
        except Exception as e:
            self.result.add_fail("require_model: 模型匹配时正常执行", str(e))

    def test_require_model_mismatch(self):
        """测试 require_model 模型不匹配时抛出 Skip"""
        import pytest
        scene_mgr, model_cfg = setup_singleton("SINGLECARD", "Qwen/Qwen3-8B")

        @require_model("Qwen/Qwen3-30B-A3B")
        def dummy_test():
            return "executed"

        try:
            result = dummy_test()
            self.result.add_fail("require_model: 模型不匹配时应跳过", f"未跳过，返回: {result}")
        except pytest.skip.Exception:
            self.result.add_pass("require_model: 模型不匹配时跳过")
        except Exception as e:
            self.result.add_fail("require_model: 模型不匹配时跳过", f"异常类型错误: {type(e)}")

    def test_require_scene_and_model_match(self):
        """测试 require_scene_and_model 条件匹配时正常执行"""
        scene_mgr, model_cfg = setup_singleton("SINGLECARD", "Qwen/Qwen3-8B")

        @require_scene_and_model("SINGLECARD", "Qwen/Qwen3-8B")
        def dummy_test():
            return "executed"

        try:
            result = dummy_test()
            if result == "executed":
                self.result.add_pass("require_scene_and_model: 条件匹配时正常执行")
            else:
                self.result.add_fail("require_scene_and_model: 条件匹配时正常执行", f"结果异常: {result}")
        except Exception as e:
            self.result.add_fail("require_scene_and_model: 条件匹配时正常执行", str(e))

    def test_require_scene_and_model_scene_mismatch(self):
        """测试 require_scene_and_model 场景不匹配时跳过"""
        import pytest
        scene_mgr, model_cfg = setup_singleton("SINGLECARD", "Qwen/Qwen3-8B")

        @require_scene_and_model("MULTICARD_4Cards", "Qwen/Qwen3-8B")
        def dummy_test():
            return "executed"

        try:
            result = dummy_test()
            self.result.add_fail("require_scene_and_model: 场景不匹配时应跳过", f"未跳过")
        except pytest.skip.Exception:
            self.result.add_pass("require_scene_and_model: 场景不匹配时跳过")
        except Exception as e:
            self.result.add_fail("require_scene_and_model: 场景不匹配时跳过", f"异常类型错误: {type(e)}")

    def test_require_scene_and_model_model_mismatch(self):
        """测试 require_scene_and_model 模型不匹配时跳过"""
        import pytest
        scene_mgr, model_cfg = setup_singleton("SINGLECARD", "Qwen/Qwen3-8B")

        @require_scene_and_model("SINGLECARD", "Qwen/Qwen3-30B-A3B")
        def dummy_test():
            return "executed"

        try:
            result = dummy_test()
            self.result.add_fail("require_scene_and_model: 模型不匹配时应跳过", f"未跳过")
        except pytest.skip.Exception:
            self.result.add_pass("require_scene_and_model: 模型不匹配时跳过")
        except Exception as e:
            self.result.add_fail("require_scene_and_model: 模型不匹配时跳过", f"异常类型错误: {type(e)}")


class TestCompatLayer:
    """测试兼容层功能"""

    def __init__(self, result: TestResult):
        self.result = result

    def test_get_models_for_current_scene_singlecard(self):
        """测试 get_models_for_current_scene 在单卡场景下过滤"""
        setup_singleton("SINGLECARD", None)

        models = get_models_for_current_scene([
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-30B-A3B",
            "Qwen/Qwen3-235B-A22B"
        ])

        if "Qwen/Qwen3-8B" in models:
            self.result.add_pass("get_models_for_current_scene: 正确保留支持的模型", f"Qwen3-8B 在列表中")
        else:
            self.result.add_fail("get_models_for_current_scene: Qwen3-8B 应在列表中", f"实际: {models}")

        if "Qwen/Qwen3-235B-A22B" not in models:
            self.result.add_pass("get_models_for_current_scene: 正确过滤不支持的模型", f"Qwen3-235B 不在列表中")
        else:
            self.result.add_fail("get_models_for_current_scene: Qwen3-235B 不应在单卡列表中", f"实际: {models}")

    def test_get_models_for_current_scene_4cards(self):
        """测试 get_models_for_current_scene 在4卡场景下"""
        setup_singleton("MULTICARD_4Cards", None)

        models = get_models_for_current_scene([
            "Qwen/Qwen3-8B",           # 不支持4卡
            "Qwen/Qwen3-30B-A3B",      # 支持4卡
            "Qwen/Qwen3-235B-A22B"     # 支持4卡
        ])

        if "Qwen/Qwen3-8B" not in models:
            self.result.add_pass("get_models_for_current_scene: 4卡正确过滤Qwen3-8B", f"Qwen3-8B不在列表中")
        else:
            self.result.add_fail("get_models_for_current_scene: 4卡不应包含Qwen3-8B", f"实际: {models}")

        if "Qwen/Qwen3-30B-A3B" in models and "Qwen/Qwen3-235B-A22B" in models:
            self.result.add_pass("get_models_for_current_scene: 4卡正确保留30B和235B", f"数量: {len(models)}")
        else:
            self.result.add_fail("get_models_for_current_scene: 4卡应保留30B和235B", f"实际: {models}")

    def test_get_models_for_current_scene_310p(self):
        """测试 get_models_for_current_scene 在310P场景下"""
        setup_singleton("310P_SINGLECARD", None)

        models = get_models_for_current_scene([
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-30B-A3B"
        ])

        if "Qwen/Qwen3-8B" in models and len(models) == 1:
            self.result.add_pass("get_models_for_current_scene: 310P场景正确过滤", f"仅 Qwen3-8B 在列表中")
        else:
            self.result.add_fail("get_models_for_current_scene: 310P场景过滤异常", f"实际: {models}")


class TestSceneManagerBehavior:
    """测试场景管理器行为"""

    def __init__(self, result: TestResult):
        self.result = result

    def test_env_switch_2cards(self):
        """测试切换到2卡场景时环境变量"""
        import os

        original = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)

        SceneManager.reset_instance()
        scene_mgr = SceneManager.get_instance()
        scene_mgr.set_current_scene("MULTICARD_2Cards")

        env_val = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if env_val == "0,1":
            self.result.add_pass("MULTICARD_2Cards 环境变量设置正确", f"ASCEND_RT_VISIBLE_DEVICES={env_val}")
        else:
            self.result.add_fail("MULTICARD_2Cards 环境变量设置错误", f"期望 0,1, 实际: {env_val}")

        SceneManager.reset_instance()

    def test_env_switch_4cards(self):
        """测试切换到4卡场景时环境变量"""
        import os

        os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)

        SceneManager.reset_instance()
        scene_mgr = SceneManager.get_instance()
        scene_mgr.set_current_scene("MULTICARD_4Cards")

        env_val = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if env_val == "0,1,2,3":
            self.result.add_pass("MULTICARD_4Cards 环境变量设置正确", f"ASCEND_RT_VISIBLE_DEVICES={env_val}")
        else:
            self.result.add_fail("MULTICARD_4Cards 环境变量设置错误", f"期望 0,1,2,3, 实际: {env_val}")

        SceneManager.reset_instance()

    def test_env_restore(self):
        """测试切换回单卡后环境变量恢复"""
        import os

        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "7"
        SceneManager.reset_instance()

        scene_mgr = SceneManager.get_instance()
        scene_mgr.set_current_scene("MULTICARD_2Cards")
        scene_mgr.set_current_scene("SINGLECARD")

        env_val = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if env_val == "7":
            self.result.add_pass("环境变量正确恢复", f"ASCEND_RT_VISIBLE_DEVICES={env_val}")
        else:
            self.result.add_fail("环境变量恢复错误", f"期望 7, 实际: {env_val}")

        SceneManager.reset_instance()


def main():
    print("=" * 60)
    print("ST Framework 装饰器功能验证")
    print("=" * 60)

    result = TestResult()

    print("\n[1] 装饰器行为测试")
    print("-" * 40)
    decorator_tests = TestDecoratorBehavior(result)
    decorator_tests.test_require_scene_match()
    decorator_tests.test_require_scene_mismatch()
    decorator_tests.test_require_model_match()
    decorator_tests.test_require_model_mismatch()
    decorator_tests.test_require_scene_and_model_match()
    decorator_tests.test_require_scene_and_model_scene_mismatch()
    decorator_tests.test_require_scene_and_model_model_mismatch()

    print("\n[2] 兼容层功能测试")
    print("-" * 40)
    compat_tests = TestCompatLayer(result)
    compat_tests.test_get_models_for_current_scene_singlecard()
    compat_tests.test_get_models_for_current_scene_4cards()
    compat_tests.test_get_models_for_current_scene_310p()

    print("\n[3] 场景管理器行为测试")
    print("-" * 40)
    scene_tests = TestSceneManagerBehavior(result)
    scene_tests.test_env_switch_2cards()
    scene_tests.test_env_switch_4cards()
    scene_tests.test_env_restore()

    print("\n")
    success = result.summary()

    if success:
        print("\n[SUCCESS] 所有验证通过!")
        return 0
    else:
        print("\n[FAILED] 部分验证失败，请检查!")
        return 1


if __name__ == "__main__":
    exit(main())
