"""
新架构示例测试 - Environment First + Model Centric + Parallel

使用 @model_test 装饰器和 session fixtures
"""
import pytest
from framework.test_registry import model_test
from framework.npu_test import npu_test
from framework.device_types import RunnerDeviceType

EXAMPLE_PROMPTS = ["Hello, my name is"]
MAX_TOKENS = 5


# ============================================================================
# 新架构测试 - @model_test + session fixtures
# ============================================================================

@model_test(models=["qwen-7b", "qwen3-8b"])
def test_inference_with_shared_model(model_runner, model_name):
    """
    使用共享模型实例的推理测试
    
    模型在 session 开始时加载，所有测试共享
    """
    outputs = model_runner.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)
    assert len(outputs) == len(EXAMPLE_PROMPTS)
    print(f"Test passed for model: {model_name}")


@model_test(models=["qwen-7b", "qwen2-7b", "qwen3-8b"])
@pytest.mark.parametrize("max_tokens", [5, 10, 20])
def test_inference_parametrized(model_runner, max_tokens):
    """
    参数化测试 - 共享模型实例
    
    不同参数组合使用同一模型
    """
    outputs = model_runner.generate_greedy(EXAMPLE_PROMPTS, max_tokens)
    assert len(outputs) == len(EXAMPLE_PROMPTS)


@model_test(models=["llama-7b", "llama-13b"])
def test_llama_inference(model_runner, model_config):
    """
    LLaMA模型推理测试
    
    不支持的模型会自动跳过
    """
    outputs = model_runner.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)
    assert len(outputs) == len(EXAMPLE_PROMPTS)
    assert "310p" in model_config.supported_devices


@model_test(models=["all"])
def test_all_models(model_runner):
    """
    支持所有模型的测试
    
    在所有已加载的模型上运行
    """
    outputs = model_runner.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)
    assert len(outputs) == len(EXAMPLE_PROMPTS)


# ============================================================================
# 环境验证测试
# ============================================================================

def test_environment_info(environment):
    """验证环境信息"""
    assert environment is not None
    assert environment.npu_type in [RunnerDeviceType._310P, RunnerDeviceType.A2]
    assert environment.num_npus >= 1
    print(f"Environment: {environment.label} ({environment.npu_type.display_name} x{environment.num_npus})")


def test_loaded_models_count(loaded_models):
    """验证已加载的模型数量"""
    from framework.model_registry import ModelRegistry
    
    loaded = ModelRegistry.get_loaded_models()
    assert len(loaded) > 0
    print(f"Loaded models: {loaded}")


# ============================================================================
# 组合测试 - @npu_test + @model_test
# ============================================================================

@npu_test(num_npus=1, npu_type=RunnerDeviceType._310P)
@model_test(models=["qwen-7b"])
def test_qwen_single_card(model_runner):
    """
    组合测试 - NPU + 模型
    
    需要 1 张 310P 和 qwen-7b 模型
    """
    outputs = model_runner.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)
    assert len(outputs) == len(EXAMPLE_PROMPTS)


@npu_test(num_npus=2, npu_type=RunnerDeviceType._310P)
@model_test(models=["qwen3-30b-moe"])
def test_moe_tp2(model_runner):
    """
    组合测试 - 多卡 NPU + MoE 模型
    
    需要 2 张 310P 和 qwen3-30b-moe 模型
    """
    outputs = model_runner.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)
    assert len(outputs) == len(EXAMPLE_PROMPTS)