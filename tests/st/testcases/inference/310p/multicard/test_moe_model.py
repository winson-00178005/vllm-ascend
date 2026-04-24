"""
310P MoE模型推理测试

使用 @npu_test 装饰器标记 NPU 资源需求
"""
from framework.npu_test import npu_test
from framework.device_types import RunnerDeviceType

EXAMPLE_PROMPTS = ["Hello, my name is"]
MAX_TOKENS = 5


@npu_test(num_npus=2, npu_type=RunnerDeviceType._310P)
def test_qwen3_moe_tp2_fp16():
    """测试Qwen3-30B-MoE TP2 FP16推理 - 需要2张310P"""
    from tests.e2e.conftest import VllmRunner
    
    with VllmRunner(
        "Qwen/Qwen3-30B-A3B",
        tensor_parallel_size=2,
        enforce_eager=True,
        dtype="float16",
        max_model_len=8192,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)
        assert len(outputs) == len(EXAMPLE_PROMPTS)


@npu_test(num_npus=8, npu_type=RunnerDeviceType._310P)
def test_deepseek_67b_tp8_fp16():
    """测试DeepSeek-67B TP8 FP16推理 - 需要8张310P"""
    from tests.e2e.conftest import VllmRunner
    
    with VllmRunner(
        "deepseek-ai/deepseek-llm-67b-chat",
        tensor_parallel_size=8,
        enforce_eager=True,
        dtype="float16",
        max_model_len=4096,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)
        assert len(outputs) == len(EXAMPLE_PROMPTS)


@npu_test(num_npus=8, npu_type=RunnerDeviceType.A2)
def test_qwen72b_distributed_910b():
    """测试Qwen-72B分布式推理 - 需要8张910B"""
    from tests.e2e.conftest import VllmRunner
    
    with VllmRunner(
        "Qwen/Qwen2.5-72B",
        tensor_parallel_size=8,
        enforce_eager=True,
        dtype="float16",
        max_model_len=4096,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)
        assert len(outputs) == len(EXAMPLE_PROMPTS)