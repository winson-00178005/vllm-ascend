"""
310P单卡Dense模型推理测试

使用 @npu_test 装饰器标记 NPU 资源需求
支持 CI 路由和运行时跳过
"""
from framework.npu_test import npu_test
from framework.device_types import RunnerDeviceType

EXAMPLE_PROMPTS = ["Hello, my name is"]
MAX_TOKENS = 5


@npu_test(num_npus=1, npu_type=RunnerDeviceType._310P)
def test_qwen3_dense_tp1_fp16():
    """测试Qwen3-8B FP16推理 - 需要1张310P"""
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


@npu_test(num_npus=1, npu_type=RunnerDeviceType._310P)
def test_qwen3_dense_tp1_w8a8():
    """测试Qwen3-8B W8A8量化推理 - 需要1张310P"""
    from tests.e2e.conftest import VllmRunner
    
    with VllmRunner(
        "vllm-ascend/Qwen3-8B-W8A8",
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="float16",
        quantization="ascend",
        max_model_len=16384,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)
        assert len(outputs) == len(EXAMPLE_PROMPTS)


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
def test_qwen_dense_910b_fp16():
    """测试Qwen-7B FP16推理 - 需要1张910B"""
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


@npu_test(num_npus=1, npu_type=RunnerDeviceType._310P)
def test_llama2_dense_tp1_fp16():
    """测试LLaMA-2-7B FP16推理 - 需要1张310P"""
    from tests.e2e.conftest import VllmRunner
    
    with VllmRunner(
        "meta-llama/Llama-2-7b-hf",
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="float16",
        max_model_len=4096,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)
        assert len(outputs) == len(EXAMPLE_PROMPTS)