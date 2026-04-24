"""
910B单卡Dense模型推理测试

使用 @npu_test 装饰器标记 NPU 资源需求
支持 CI 路由和运行时跳过
"""
from framework.npu_test import npu_test
from framework.device_types import RunnerDeviceType

EXAMPLE_PROMPTS = ["Hello, my name is"]
MAX_TOKENS = 5


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_qwen_dense_910b_fp16():
    """测试Qwen-7B在910B上的FP16推理 - 需要1张910B"""
    from tests.e2e.conftest import VllmRunner
    
    with VllmRunner(
        "Qwen/Qwen-7B",
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="float16",
        max_model_len=8192,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)
        assert len(outputs) == len(EXAMPLE_PROMPTS)


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_qwen2_dense_910b_fp16():
    """测试Qwen2-7B在910B上的FP16推理 - 需要1张910B"""
    from tests.e2e.conftest import VllmRunner
    
    with VllmRunner(
        "Qwen/Qwen2-7B",
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="float16",
        max_model_len=8192,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)
        assert len(outputs) == len(EXAMPLE_PROMPTS)


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_qwen3_dense_910b_fp16():
    """测试Qwen3-8B在910B上的FP16推理 - 需要1张910B"""
    from tests.e2e.conftest import VllmRunner
    
    with VllmRunner(
        "Qwen/Qwen3-8B",
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="float16",
        max_model_len=16384,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)
        assert len(outputs) == len(EXAMPLE_PROMPTS)