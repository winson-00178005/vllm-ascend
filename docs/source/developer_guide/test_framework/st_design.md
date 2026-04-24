# ST 测试框架设计方案

> **状态**: 设计中  
> **创建日期**: 2026-04-23  
> **最后更新**: 2026-04-23

## 一、设计目标

基于 pytest 构建场景驱动的 ST（System Test）测试框架，核心借鉴 MindIE-LLM `tests/dlt/st` 的设计思想，并融合 vLLM 社区测试的最佳实践：

- **场景配置化**：通过 YAML 配置文件定义运行场景（单卡/多卡/310P 等）
- **模型配置化**：每个场景关联支持的模型列表，模型配置集中管理
- **用例按需执行**：测试用例声明所需的场景和模型，不匹配则自动跳过（skip）
- **兼容现有 e2e**：现有 `tests/e2e/` 用例可快速迁移，保持零修改即可运行
- **社区模式对齐**：参考 vLLM 社区测试组织方式（模型注册、CI 环境变量、参数化矩阵、分布式测试装饰器等）

## 二、目录结构

```
tests/e2e/
├── conftest.py                 # 改造：注入场景能力（原有内容保留）
├── st_config/                  # 新增：ST 框架配置与核心
│   ├── scenes.yaml             # 场景定义
│   ├── models.yaml             # 模型定义与场景映射
│   ├── ci_envs.py              # CI 环境变量集中管理（参考 vllm/tests/ci_envs.py）
│   ├── test_utils.py           # 测试工具函数（参考 vllm/tests/utils.py）
│   ├── model_utils.py          # 模型比较工具（复用现有 e2e/model_utils.py）
│   └── framework/
│       ├── __init__.py
│       ├── scene.py            # 场景管理器（自动探测+命令行）
│       ├── model.py            # 模型管理器（参考 vllm/tests/models/registry.py）
│       └── fixtures.py         # fixture 工厂
├── singlecard/
│   ├── conftest.py             # 可覆盖 session fixture 行为
│   └── test_*.py               # 现有用例不动
├── multicard/
│   ├── conftest.py
│   └── test_*.py
├── models/                     # 精度测试，保持不动
└── pd_disaggreate/             # PD解耦测试，保持不动
```

## 三、场景配置

### 3.1 `st_config/scenes.yaml`

```yaml
scenes:
  single_card:
    description: "单卡推理"
    env_vars:
      ASCEND_RT_VISIBLE_DEVICES: "0"
    tp_size: 1
    dp_size: 1

  multi_card_tp2:
    description: "双卡TP"
    env_vars:
      ASCEND_RT_VISIBLE_DEVICES: "0,1"
    tp_size: 2
    dp_size: 1

  multi_card_tp4:
    description: "四卡TP"
    env_vars:
      ASCEND_RT_VISIBLE_DEVICES: "0,1,2,3"
    tp_size: 4
    dp_size: 1

  multi_card_dp2:
    description: "双卡DP"
    env_vars:
      ASCEND_RT_VISIBLE_DEVICES: "0,1"
    tp_size: 1
    dp_size: 2

  ascend_310p:
    description: "310P硬件"
    env_vars:
      ASCEND_RT_VISIBLE_DEVICES: "0"
    tp_size: 1
    dp_size: 1
    hardware: "310p"
```

### 3.2 `st_config/models.yaml`

参考 vLLM 社区的 `ModelInfo` 数据类模式，模型配置包含更丰富的元数据：

```yaml
models:
  Qwen3-8B-Base:
    model_id: "Qwen/Qwen3-8B-Base"
    architecture: "Qwen3ForCausalLM"
    dtype: "bfloat16"
    max_model_len: 4096
    supported_scenes: ["single_card", "multi_card_tp2", "multi_card_tp4"]
    enable_test: true
    hf_ppl: 8.5
    min_gpu_gb: 16

  DeepSeek-V2-Lite:
    model_id: "deepseek-ai/DeepSeek-V2-Lite"
    architecture: "DeepseekV2ForCausalLM"
    dtype: "bfloat16"
    max_model_len: 4096
    supported_scenes: ["multi_card_tp2", "multi_card_tp4", "multi_card_dp2"]
    enable_test: true
    min_gpu_gb: 32

  Qwen2.5-VL-7B-Instruct:
    model_id: "Qwen/Qwen2.5-VL-7B-Instruct"
    architecture: "Qwen2_5_VLForConditionalGeneration"
    dtype: "bfloat16"
    max_model_len: 4096
    supported_scenes: ["single_card", "multi_card_tp2"]
    enable_test: true
    is_multimodal: true
    min_gpu_gb: 24

  Qwen3-30B-A3B:
    model_id: "Qwen/Qwen3-30B-A3B"
    architecture: "Qwen3MoeForCausalLM"
    dtype: "bfloat16"
    max_model_len: 4096
    supported_scenes: ["multi_card_tp4", "multi_card_dp2"]
    enable_test: true
    is_moe: true
    min_gpu_gb: 64
```

## 四、核心设计

### 4.1 场景加载方式（三级优先级）

优先级：**命令行参数** > **环境变量** > **自动探测**

```bash
# 方式1：自动探测（默认）
pytest tests/e2e/

# 方式2：命令行强制指定（覆盖自动探测）
pytest tests/e2e/ --scene single_card

# 方式3：环境变量指定
ST_SCENE=multi_card_tp4 pytest tests/e2e/
```

**自动探测逻辑** (`framework/scene.py`)：
1. 检测 NPU 卡数（`ASCEND_RT_VISIBLE_DEVICES` 或 `npu-smi info`）
2. 检测硬件型号（310P / 910B）
3. 匹配场景配置

```python
import os
import yaml
from pathlib import Path

class SceneManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, cli_scene: str | None = None):
        self._scenes = self._load_config()
        priority = cli_scene or os.environ.get("ST_SCENE") or self._auto_detect()
        self._current = priority

    def _load_config(self) -> dict:
        config_path = Path(__file__).parent.parent / "scenes.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)["scenes"]

    def _auto_detect(self) -> str:
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        card_count = len(visible.split(",")) if visible else self._detect_via_npu_smi()
        hardware = self._detect_hardware()
        if hardware == "310p":
            return "ascend_310p"
        if card_count == 1:
            return "single_card"
        if card_count == 2:
            return "multi_card_tp2"
        if card_count >= 4:
            return "multi_card_tp4"
        return "single_card"

    def _detect_via_npu_smi(self) -> int:
        try:
            out = os.popen("npu-smi info -l 2>/dev/null").read()
            return max(1, out.count("HBM"))
        except Exception:
            return 1

    def _detect_hardware(self) -> str:
        try:
            out = os.popen("npu-smi info -t device-info 2>/dev/null").read()
            if "310P" in out:
                return "310p"
        except Exception:
            pass
        return "910b"

    @property
    def current(self) -> str:
        return self._current

    def get_config(self) -> dict:
        return self._scenes.get(self._current, {})
```

### 4.2 模型管理器 (`framework/model.py`)

参考 vLLM 社区的 `ModelInfo` 数据类模式：

```python
import yaml
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class AscendModelInfo:
    """模型信息数据类（参考 vllm/tests/models/utils.py 的 ModelInfo）"""
    name: str
    model_id: str
    architecture: str = ""
    dtype: str = "auto"
    max_model_len: int | None = None
    supported_scenes: list[str] = field(default_factory=list)
    enable_test: bool = True
    is_multimodal: bool = False
    is_moe: bool = False
    min_gpu_gb: int = 0
    hf_ppl: float | None = None


class ModelManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._models: dict[str, AscendModelInfo] = {}
        self._load_config()

    def _load_config(self) -> None:
        config_path = Path(__file__).parent.parent / "models.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)["models"]
        
        for name, cfg in data.items():
            self._models[name] = AscendModelInfo(
                name=name,
                model_id=cfg.get("model_id", name),
                architecture=cfg.get("architecture", ""),
                dtype=cfg.get("dtype", "auto"),
                max_model_len=cfg.get("max_model_len"),
                supported_scenes=cfg.get("supported_scenes", []),
                enable_test=cfg.get("enable_test", True),
                is_multimodal=cfg.get("is_multimodal", False),
                is_moe=cfg.get("is_moe", False),
                min_gpu_gb=cfg.get("min_gpu_gb", 0),
                hf_ppl=cfg.get("hf_ppl"),
            )

    def get_models_for_scene(self, scene: str) -> list[str]:
        return [
            name for name, info in self._models.items()
            if scene in info.supported_scenes and info.enable_test
        ]

    def is_supported(self, model: str, scene: str) -> bool:
        info = self._models.get(model)
        if not info:
            return False
        return scene in info.supported_scenes

    def get_info(self, model: str) -> AscendModelInfo | None:
        return self._models.get(model)

    def check_gpu_requirement(self, model: str) -> bool:
        """检查当前 GPU 是否满足模型要求（参考 vllm/tests/utils.py 的 large_gpu_mark）"""
        info = self._models.get(model)
        if not info or info.min_gpu_gb == 0:
            return True
        
        try:
            import torch_npu
            total_memory_gb = torch_npu.npu.get_device_properties(0).total_memory / (1024**3)
            return total_memory_gb >= info.min_gpu_gb
        except Exception:
            return True
```

### 4.3 CI 环境变量管理 (`st_config/ci_envs.py`)

参考 vLLM 社区的 `ci_envs.py` 模式：

```python
"""
CI 环境变量集中管理（参考 vllm/tests/ci_envs.py）

这些环境变量用于控制测试行为，如：
- 是否跳过某些模型测试
- 强制使用特定 dtype
- 控制 enforce_eager 等
"""

import os
from collections.abc import Callable
from typing import Any

environment_variables: dict[str, Callable[[], Any]] = {
    # 是否测试所有模型（默认只测试一个代表性模型）
    "ST_CI_NO_SKIP": lambda: bool(int(os.getenv("ST_CI_NO_SKIP", "0"))),
    
    # 强制使用特定 dtype
    "ST_CI_DTYPE": lambda: os.getenv("ST_CI_DTYPE", None),
    
    # 是否强制使用 eager 模式
    "ST_CI_ENFORCE_EAGER": lambda: os.getenv("ST_CI_ENFORCE_EAGER", None),
    
    # 目标测试套件（用于跳过不匹配的测试）
    "ST_CI_TARGET_SUITE": lambda: os.getenv("ST_CI_TARGET_SUITE", None),
}


def __getattr__(name: str):
    """延迟求值环境变量"""
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """检查环境变量是否显式设置"""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### 4.4 测试工具函数 (`st_config/test_utils.py`)

参考 vLLM 社区的 `utils.py` 模式：

```python
"""
测试工具函数（参考 vllm/tests/utils.py）
"""

import functools
import os
import signal
import tempfile
from collections.abc import Callable
from typing import ParamSpec

import pytest
import torch_npu

_P = ParamSpec("_P")


def fork_new_process_for_each_test(func: Callable[_P, None]) -> Callable[_P, None]:
    """装饰器：为每个测试函数 fork 新进程（参考 vllm/tests/utils.py）"""
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        from _pytest.outcomes import Skipped

        with tempfile.NamedTemporaryFile(
            delete=False,
            mode="w+b",
            prefix=f"st_test_{func.__name__}_{os.getpid()}_",
            suffix=".exc",
        ) as exc_file:
            exc_file_path = exc_file.name
            pid = os.fork()
            
            if pid == 0:
                os.setpgrp()
                try:
                    func(*args, **kwargs)
                except Skipped:
                    os._exit(0)
                except Exception:
                    import traceback
                    with open(exc_file_path, "w") as f:
                        f.write(traceback.format_exc())
                    os._exit(1)
                else:
                    os._exit(0)
            else:
                pgid = pid
                _pid, _exitcode = os.waitpid(pid, 0)
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pgid, signal.SIGTERM)
                
                if _exitcode != 0:
                    if os.path.exists(exc_file_path):
                        with open(exc_file_path) as f:
                            tb = f.read()
                        raise AssertionError(
                            f"Test {func.__name__} failed (exit code: {_exitcode}):\n{tb}"
                        )
                    raise AssertionError(
                        f"Test {func.__name__} failed (exit code: {_exitcode})"
                    )
    
    return wrapper


def multi_gpu_test(num_gpus: int):
    """装饰器：仅在满足 GPU 数量要求时执行测试（参考 vllm/tests/utils.py）"""
    marks = [
        pytest.mark.distributed(num_gpus=num_gpus),
        pytest.mark.skipif(
            torch_npu.npu.device_count() < num_gpus,
            reason=f"Need at least {num_gpus} GPUs to run the test.",
        ),
    ]
    
    def wrapper(f: Callable[_P, None]) -> Callable[_P, None]:
        func = fork_new_process_for_each_test()(f)
        for mark in reversed(marks):
            func = mark(func)
        return func
    
    return wrapper


def large_gpu_test(min_gb: int):
    """装饰器：跳过 GPU 内存不足的测试（参考 vllm/tests/utils.py）"""
    try:
        memory_gb = torch_npu.npu.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        memory_gb = 0
    
    return pytest.mark.skipif(
        memory_gb < min_gb,
        reason=f"Need at least {min_gb}GB GPU memory to run the test.",
    )


def get_vllm_extra_kwargs(model_info, vllm_extra_kwargs):
    """获取 vLLM 额外参数（参考 vllm/tests/models/utils.py）"""
    from .ci_envs import ST_CI_DTYPE, ST_CI_ENFORCE_EAGER, ST_CI_NO_SKIP
    
    vllm_extra_kwargs = vllm_extra_kwargs or {}
    
    if not ST_CI_NO_SKIP and not model_info.enable_test:
        pytest.skip("Skipping test.")
    
    vllm_extra_kwargs["dtype"] = ST_CI_DTYPE or model_info.dtype
    
    if ST_CI_ENFORCE_EAGER is not None:
        vllm_extra_kwargs["enforce_eager"] = ST_CI_ENFORCE_EAGER
    
    return vllm_extra_kwargs
```

### 4.5 Fixture 设计（三级生命周期）

#### Session 级复用

一个场景×模型组合只创建一次 LLM 实例，所有用例复用。

```python
@pytest.fixture(scope="session")
def vllm_runner_session(request):
    scene_mgr = SceneManager.get_instance()
    model_mgr = ModelManager.get_instance()
    scene_cfg = scene_mgr.get_config()
    
    for k, v in scene_cfg.get("env_vars", {}).items():
        os.environ[k] = v
    
    def _runner(model_name, **kwargs):
        model_info = model_mgr.get_info(model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {model_name}")
        
        from tests.e2e.conftest import VllmRunner
        return VllmRunner(
            model_name=model_info.model_id,
            dtype=kwargs.pop("dtype", model_info.dtype),
            max_model_len=kwargs.pop("max_model_len", model_info.max_model_len or 4096),
            tensor_parallel_size=kwargs.pop("tensor_parallel_size", scene_cfg.get("tp_size", 1)),
            **kwargs
        )
    
    yield _runner
```

#### Module 级创建

每个测试模块创建一次 LLM 实例。

```python
@pytest.fixture(scope="module")
def vllm_runner_module(request):
    scene_mgr = SceneManager.get_instance()
    model_mgr = ModelManager.get_instance()
    scene_cfg = scene_mgr.get_config()
    
    for k, v in scene_cfg.get("env_vars", {}).items():
        os.environ[k] = v
    
    model_name = getattr(request, "param", None)
    if not model_name:
        pytest.skip("未指定模型参数")
    
    model_info = model_mgr.get_info(model_name)
    if not model_info:
        pytest.skip(f"未知模型: {model_name}")
    
    from tests.e2e.conftest import VllmRunner
    runner = VllmRunner(
        model_name=model_info.model_id,
        dtype=model_info.dtype,
        tensor_parallel_size=scene_cfg.get("tp_size", 1),
    )
    
    yield runner
    runner.model = None
```

#### 参数化动态创建

通过 pytest 参数化动态创建，支持多模型遍历。

```python
@pytest.mark.parametrize("vllm_runner_module", 
                         ["Qwen3-8B-Base", "Qwen2.5-VL-7B-Instruct"],
                         indirect=True)
def test_multi_model(vllm_runner_module, example_prompts):
    outputs = vllm_runner_module.generate(example_prompts)
    assert len(outputs) > 0
```

### 4.6 顶层 conftest.py 改造

```python
# tests/e2e/conftest.py - 在原有内容基础上增加

import os
import pytest
from .st_config.framework.scene import SceneManager
from .st_config.framework.model import ModelManager
from .st_config.framework.fixtures import (
    create_vllm_runner_fixture,
    create_vllm_runner_module_fixture,
)

def pytest_addoption(parser):
    parser.addoption("--scene", default=None, help="指定运行场景")
    parser.addoption("--model", default=None, help="指定运行模型")
    parser.addoption("--st-ci-no-skip", action="store_true", default=False, 
                     help="运行所有模型测试，不跳过")

def pytest_configure(config):
    cli_scene = config.getoption("--scene")
    cli_model = config.getoption("--model")
    
    scene_mgr = SceneManager(cli_scene=cli_scene)
    model_mgr = ModelManager()
    
    config.scene_mgr = scene_mgr
    config.model_mgr = model_mgr
    config.cli_model = cli_model
    
    config.addinivalue_line("markers", "scenario(names): 指定需要哪些场景")
    config.addinivalue_line("markers", "model(names): 指定需要哪些模型")
    config.addinivalue_line("markers", "distributed(num_gpus): 标记分布式测试")
    config.addinivalue_line("markers", "large_gpu(min_gb): 标记需要大内存 GPU 的测试")

def pytest_generate_tests(metafunc):
    """参数化动态创建：场景 × 模型矩阵（参考 vllm/tests/models/conftest.py）"""
    scene_mgr = metafunc.config.scene_mgr
    model_mgr = metafunc.config.model_mgr
    cli_model = metafunc.config.cli_model
    
    if "vllm_runner_module" in metafunc.fixturenames:
        model_marker = metafunc.definition.get_closest_marker("model")
        if cli_model:
            models = [cli_model]
        elif model_marker:
            models = model_marker.args
        else:
            models = model_mgr.get_models_for_scene(scene_mgr.current)
        
        # 过滤：只保留当前场景支持的模型
        models = [m for m in models if model_mgr.is_supported(m, scene_mgr.current)]
        
        metafunc.parametrize("vllm_runner_module", models, indirect=True)

def pytest_collection_modifyitems(config, items):
    """修改测试集合（参考 vllm/tests/conftest.py）"""
    # 处理 --st-ci-no-skip 选项
    if config.getoption("--st-ci-no-skip"):
        return
    
    # 跳过不匹配的测试
    scene_mgr = config.scene_mgr
    model_mgr = config.model_mgr
    
    for item in items:
        scenario_marker = item.get_closest_marker("scenario")
        if scenario_marker:
            required = set(scenario_marker.args)
            if scene_mgr.current not in required:
                item.add_marker(pytest.mark.skip(
                    reason=f"场景 '{scene_mgr.current}' 不在 {required}"
                ))

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """用例执行前检查场景/模型匹配"""
    scene_mgr = item.config.scene_mgr
    model_mgr = item.config.model_mgr
    
    scenario_marker = item.get_closest_marker("scenario")
    if scenario_marker:
        required = set(scenario_marker.args)
        if scene_mgr.current not in required:
            pytest.skip(f"场景 '{scene_mgr.current}' 不在 {required}")
    
    model_marker = item.get_closest_marker("model")
    if model_marker:
        required = set(model_marker.args)
        current_model = getattr(item, "_model_name", None)
        if current_model and current_model not in required:
            pytest.skip(f"模型 '{current_model}' 不在 {required}")

# 注入 fixture
vllm_runner_session = create_vllm_runner_fixture()
vllm_runner_module = create_vllm_runner_module_fixture()

# 保留原有 fixtures（VllmRunner, HfRunner, example_prompts 等）
# ... 原有代码不动 ...
```

### 4.7 用例编写示例

参考 vLLM 社区的参数化模式：

```python
# tests/e2e/singlecard/test_offline_inference.py
# 现有用例无需修改，自动在匹配场景下执行

import pytest
from ..st_config.test_utils import multi_gpu_test, large_gpu_test
from ..st_config.model_utils import check_outputs_equal

# 模式1：简单参数化（参考 vllm/tests/basic_correctness/test_basic_correctness.py）
MODELS = ["Qwen3-8B-Base"]
BACKENDS = ["FLASH_ATTN"]

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("max_tokens", [5])
def test_models(
    hf_runner,
    vllm_runner_session,
    model: str,
    backend: str,
    max_tokens: int,
) -> None:
    example_prompts = ["Hello, my name is"]
    
    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    
    with vllm_runner_session(model) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    
    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


# 模式2：分布式测试（参考 vllm/tests/basic_correctness/test_basic_correctness.py）
@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model, distributed_executor_backend",
    [
        ("Qwen3-8B-Base", "ray"),
        ("Qwen3-8B-Base", "mp"),
    ],
)
def test_models_distributed(
    hf_runner,
    vllm_runner_session,
    example_prompts,
    model: str,
    distributed_executor_backend: str,
) -> None:
    max_tokens = 5
    
    with vllm_runner_session(
        model,
        tensor_parallel_size=2,
        distributed_executor_backend=distributed_executor_backend,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        
        with hf_runner(model) as hf_model:
            hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    
    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


# 模式3：场景+模型 marker 精细控制
@pytest.mark.scenario("single_card", "ascend_310p")
@pytest.mark.model("Qwen3-8B-Base")
def test_qwen3_singlecard(vllm_runner_module, example_prompts):
    outputs = vllm_runner_module.generate(example_prompts)
    assert all(len(o) > 0 for o in outputs)


# 模式4：大 GPU 测试（参考 vllm/tests/utils.py 的 large_gpu_test）
@large_gpu_test(min_gb=32)
def test_large_model(vllm_runner_session, example_prompts):
    with vllm_runner_session("Qwen3-30B-A3B") as vllm_model:
        outputs = vllm_model.generate(example_prompts, max_tokens=10)
        assert len(outputs) > 0


# 模式5：参数化遍历模型（参考 vllm/tests/models/conftest.py）
@pytest.mark.scenario("multi_card_tp2")
class TestMultiCardTP2:
    
    @pytest.mark.parametrize("vllm_runner_module", 
                             ["Qwen3-8B-Base", "Qwen2.5-VL-7B-Instruct"],
                             indirect=True)
    def test_generation(self, vllm_runner_module, example_prompts):
        outputs = vllm_runner_module.generate(example_prompts)
        assert len(outputs) > 0
```

## 五、执行流程

```
pytest tests/e2e/ --scene single_card --model Qwen3-8B-Base
│
├─ pytest_addoption() → 注册 --scene, --model, --st-ci-no-skip 参数
├─ pytest_configure() → SceneManager(cli_scene="single_card")
│   └─ 加载 scenes.yaml, models.yaml
│
├─ pytest_collection_modifyitems() → 过滤不匹配的测试
│   └─ 根据场景/模型 marker 跳过不匹配的用例
│
├─ pytest_generate_tests() → 生成场景 × 模型矩阵
│   └─ 过滤：只生成 single_card × Qwen3-8B-Base 的组合
│
├─ pytest_runtest_setup() → 每个用例执行前检查
│   ├─ 有 @pytest.mark.scenario → 检查是否匹配
│   ├─ 有 @pytest.mark.model → 检查是否匹配
│   └─ 无 marker → 默认执行
│
├─ fixture: vllm_runner_session/module → 按场景配置创建 LLM
└─ 执行用例
```

## 六、e2e 兼容方案

### 6.1 conftest 层次覆盖

不新建 `tests/st/` 目录，而是**直接在 `tests/e2e/` 下改造**，通过 conftest.py 的层次覆盖注入场景能力：

- **顶层 conftest.py**：场景加载 + 全局 fixtures + hook
- **子目录 conftest.py**：可覆盖 session fixture 行为
- **现有测试用例**：零修改即可运行

### 6.2 兼容性保证

| 现有 e2e 特性 | 兼容方式 |
|---|---|
| `VllmRunner` / `HfRunner` | 直接复用，不修改原类 |
| `example_prompts` fixture | 保留原有定义 |
| `fork_new_process_for_each_test` | 保留原有装饰器 |
| 现有 `conftest.py` fixtures | 全部保留，新增场景相关 fixtures |
| 现有 pytest markers | 全部保留，新增 `scenario` / `model` |
| `check_outputs_equal` / `check_logprobs_close` | 复用现有 model_utils.py |

### 6.3 测试矩阵模式（参考 vLLM 社区）

| 维度 | 如何组织 |
|---|---|
| **模型** | `MODELS = [...]` 列表 + `@pytest.mark.parametrize` |
| **TP/PP/DP** | 参数化大小传入 `VllmRunner` |
| **后端** | `distributed_executor_backend=["mp", "ray"]` |
| **量化** | `quantization="ascend"` + 环境变量 |
| **硬件** | `is_310p` 模拟 + `torch_npu` patches |
| **Graph 模式** | `torchair_graph_config.enabled` |
| **调度器** | `ascend_scheduler_config.enabled` |
| **模型配置** | YAML 文件 + `pytest_generate_tests` 动态加载 |

## 七、CI 渐进替换方案

### 7.1 第一阶段：新增 st job，与现有 e2e 并行

```yaml
# .github/workflows/vllm_ascend_test.yaml - 新增
st-singlecard:
  name: ST / Single Card
  runs-on: linux-aarch64-a2-1
  steps:
    - uses: actions/checkout@v4
    - name: Run ST tests
      run: |
        pytest tests/e2e/singlecard/ --scene single_card -v

st-multicard:
  name: ST / Multi Card
  runs-on: linux-aarch64-a2-2
  steps:
    - uses: actions/checkout@v4
    - name: Run ST tests
      run: |
        pytest tests/e2e/multicard/ --scene multi_card_tp2 -v
```

### 7.2 第二阶段：验证通过后，利用 matrix 策略按场景拆分

```yaml
# 最终形态
e2e:
  strategy:
    matrix:
      scene: [single_card, multi_card_tp2, multi_card_tp4, ascend_310p]
      include:
        - scene: single_card
          runner: linux-aarch64-a2-1
        - scene: multi_card_tp2
          runner: linux-aarch64-a2-2
        - scene: multi_card_tp4
          runner: linux-aarch64-a2-4
        - scene: ascend_310p
          runner: linux-aarch64-310p-1
  runs-on: ${{ matrix.runner }}
  steps:
    - run: pytest tests/e2e/ --scene ${{ matrix.scene }} -v
```

## 八、价值与收益分析

### 8.1 核心价值主张

当前 `tests/e2e/` 测试体系存在以下痛点：

| 痛点 | 现状 | 影响 |
|---|---|---|
| **场景硬编码** | 单卡/多卡/310P 测试分散在不同目录，用例与场景耦合 | 新增场景需复制用例或修改代码 |
| **模型选择困难** | 哪些模型该在哪些场景下跑，依赖开发者记忆 | 容易遗漏或重复执行 |
| **CI 配置冗余** | 每个场景/模型组合需要独立的 CI job 配置 | 维护成本高，扩展困难 |
| **资源浪费** | 不匹配的用例仍然被收集和执行，最后才 skip | CI 时间浪费在无效用例上 |
| **新成员上手慢** | 不清楚测试组织逻辑，不知道用例该放哪里 | 学习成本高，容易放错位置 |

ST 框架的核心价值：**将场景和模型的选择从代码中解耦到配置中**，实现"写一次用例，到处运行"。

### 8.2 用例写作增益

#### 8.2.1 零迁移成本

现有用例无需任何修改即可在新框架下运行：

```python
# 现有用例 - 完全不动
def test_models(vllm_runner, example_prompts):
    outputs = vllm_runner.generate_greedy(example_prompts, max_tokens=5)
    assert len(outputs) > 0
```

框架自动根据当前场景加载对应的模型，不匹配的用例在 collection 阶段就被过滤。

#### 8.2.2 精细控制只需加 marker

需要精细控制的用例，只需添加一行 marker：

```python
# 只在这两个场景下执行
@pytest.mark.scenario("single_card", "ascend_310p")
def test_310p_specific(vllm_runner, example_prompts):
    ...

# 只针对特定模型
@pytest.mark.model("Qwen3-8B-Base")
def test_qwen3_feature(vllm_runner):
    ...
```

#### 8.2.3 参数化矩阵一行搞定

参考 vLLM 社区的参数化模式，一行声明多维测试矩阵：

```python
# 生成 2(模型) × 2(后端) × 2(max_tokens) = 8 个测试组合
@pytest.mark.parametrize("model", ["Qwen3-8B-Base", "DeepSeek-V2-Lite"])
@pytest.mark.parametrize("backend", ["FLASH_ATTN", "ASCEND"])
@pytest.mark.parametrize("max_tokens", [5, 10])
def test_models(model, backend, max_tokens, vllm_runner):
    ...
```

#### 8.2.4 分布式测试装饰器

```python
# 自动跳过 GPU 数量不足的环境，自动 fork 进程隔离
@multi_gpu_test(num_gpus=2)
def test_distributed_inference(vllm_runner, example_prompts):
    ...

# 自动跳过 GPU 内存不足的环境
@large_gpu_test(min_gb=32)
def test_large_model(vllm_runner, example_prompts):
    ...
```

### 8.3 用例执行增益

#### 8.3.1 Collection 阶段过滤

框架在 `pytest_collection_modifyitems` 阶段就过滤掉不匹配的用例，避免无效用例进入执行流程：

```
# 当前模式（执行阶段才 skip）
collected 100 items
tests/e2e/singlecard/test_a.py::test_x SKIPPED (场景不匹配)    # 浪费了 collection 时间
tests/e2e/singlecard/test_b.py::test_y PASSED
...

# ST 框架模式（collection 阶段就过滤）
collected 20 items  # 只收集匹配的用例
tests/e2e/singlecard/test_b.py::test_y PASSED
...
```

#### 8.3.2 三级优先级场景选择

| 方式 | 使用场景 | 示例 |
|---|---|---|
| 自动探测 | 本地开发，无需指定 | `pytest tests/e2e/` |
| 环境变量 | CI 脚本统一管理 | `ST_SCENE=single_card pytest tests/e2e/` |
| 命令行参数 | 临时指定场景 | `pytest tests/e2e/ --scene multi_card_tp2` |

#### 8.3.3 三级 Fixture 生命周期

| 生命周期 | 适用场景 | 内存占用 | 执行速度 |
|---|---|---|---|
| Session 级 | 轻量用例，共享 LLM 实例 | 低 | 快 |
| Module 级 | 需要模块隔离的用例 | 中 | 中 |
| 参数化 | 需要遍历多个模型 | 高 | 慢但覆盖全 |

开发者可根据用例特点选择合适的 fixture，平衡速度和隔离性。

### 8.4 场景覆盖增益

#### 8.4.1 场景-模型映射可视化

通过 `models.yaml` 配置，场景和模型的关系一目了然：

```yaml
# 一眼看出 Qwen3-8B-Base 支持哪些场景
Qwen3-8B-Base:
  supported_scenes: ["single_card", "multi_card_tp2", "multi_card_tp4"]

# 一眼看出 single_card 场景支持哪些模型
# 通过 get_models_for_scene("single_card") 查询
```

#### 8.4.2 覆盖率矩阵

| 场景 ↓ \ 模型 → | Qwen3-8B | DeepSeek-V2 | Qwen2.5-VL | Qwen3-30B |
|---|---|---|---|---|
| single_card | ✅ | ❌ | ✅ | ❌ |
| multi_card_tp2 | ✅ | ✅ | ✅ | ❌ |
| multi_card_tp4 | ✅ | ✅ | ❌ | ✅ |
| multi_card_dp2 | ❌ | ✅ | ❌ | ✅ |
| ascend_310p | ✅ | ❌ | ❌ | ❌ |

框架自动根据矩阵执行测试，不产生无效执行。

#### 8.4.3 新增场景只需改配置

新增一个场景（如 `multi_card_tp8`）只需：

```yaml
# scenes.yaml 新增
multi_card_tp8:
  description: "八卡TP"
  env_vars:
    ASCEND_RT_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
  tp_size: 8
  dp_size: 1
```

然后在 `models.yaml` 中为需要支持该场景的模型添加 `multi_card_tp8` 即可。

**无需修改任何测试代码**。

### 8.5 效率增益分析（定量）

#### 8.5.1 CI 执行时间对比

| 指标 | 当前模式 | ST 框架 | 改善 |
|---|---|---|---|
| 用例收集时间 | 100 items（包含不匹配） | 20 items（只收集匹配） | **减少 80%** |
| 无效用例执行 | 80 items × 2s = 160s（skip） | 0 items | **节省 160s** |
| 场景切换开销 | 手动指定目录 | 自动探测/命令行 | **减少人工操作** |
| CI 配置维护 | 每个场景独立 job | matrix 策略 | **减少 60% 配置** |

#### 8.5.2 开发效率对比

| 任务 | 当前模式 | ST 框架 | 时间节省 |
|---|---|---|---|
| 新增场景测试 | 复制目录 + 修改用例 | 修改 YAML 配置 | **从 2h → 10min** |
| 新增模型测试 | 修改每个用例 | 修改 models.yaml | **从 1h → 5min** |
| 排查测试遗漏 | 逐个目录检查 | 查看配置矩阵 | **从 30min → 2min** |
| 新成员上手 | 理解目录结构 | 理解配置格式 | **从 2 天 → 半天** |

#### 8.5.3 场景覆盖效率

| 场景数量 | 当前模式用例数 | ST 框架用例数 | 说明 |
|---|---|---|---|
| 1 个场景 | N | N | 基准 |
| 5 个场景 | 5N（需复制） | N（配置驱动） | **减少 80% 用例代码** |
| 10 个场景 | 10N | N | **减少 90% 用例代码** |

#### 8.5.4 维护成本对比

| 维度 | 当前模式 | ST 框架 |
|---|---|---|
| 新增场景 | 需修改代码 | 只改配置 |
| 新增模型 | 需修改用例 | 只改配置 |
| 修复场景 bug | 多处修改 | 一处修改 |
| CI 配置变更 | 每个 job 单独改 | matrix 统一改 |
| 文档维护 | 多处更新 | 配置即文档 |

### 8.6 投资回报分析

| 投入 | 产出 |
|---|---|
| 8.5 人日开发 | 长期减少场景测试维护成本 |
| 一次性迁移成本 | 后续新增场景/模型成本降低 80%+ |
| 学习成本（半天） | 新成员上手时间从 2 天缩短到半天 |

**预计 3 个月内收回开发成本**，之后持续产生正向收益。

### 8.7 风险与缓解

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| 框架 bug 导致用例误跳过 | 测试覆盖不足 | 渐进替换，新旧并行验证 |
| 配置文件格式错误 | 场景加载失败 | 增加配置校验和单元测试 |
| 性能回归 | CI 时间增加 | 保留现有 e2e 作为基线对比 |
| 学习曲线 | 开发者不适应 | 提供迁移指南和示例 |

## 九、实施计划

| 阶段 | 任务 | 产出 | 预计工时 |
|---|---|---|---|
| **P0** | 创建 `st_config/` 目录结构、scenes.yaml、models.yaml | 配置文件就绪 | 0.5d |
| **P1** | 实现 SceneManager、ModelManager、AscendModelInfo | 框架核心 | 1d |
| **P2** | 实现 ci_envs.py、test_utils.py（参考 vLLM 社区） | CI 工具 | 1d |
| **P3** | 实现 fixture 工厂（session/module/参数化） | 三种生命周期 fixture | 1d |
| **P4** | 改造顶层 conftest.py（hooks + pytest_generate_tests） | pytest 集成 | 1d |
| **P5** | 迁移 2-3 个典型 e2e 用例验证框架 | 验证框架可用性 | 1d |
| **P6** | 批量迁移 + CI 集成（第一阶段） | CI 新增 st job | 2d |
| **P7** | CI matrix 策略优化（第二阶段） | 完全替换原有 e2e CI | 1d |

**总计**: 约 8.5 人日

## 十、参考设计

本框架设计参考了以下项目的组织模式：

- **MindIE-LLM** (`tests/dlt/st`): 场景驱动的 C++ gtest 框架，核心思想包括：
  - `scene.json` 配置化场景定义
  - `REQUIRE_SCENARIO` 宏控制用例执行
  - Environment 单例管理场景上下文
  - 工厂模式创建测试环境

- **vLLM 社区** (`tests/`): pytest 测试最佳实践，包括：
  - `conftest.py` 层次化 fixture（`tests/conftest.py`）
  - `VllmRunner` / `HfRunner` 封装模式
  - `ModelInfo` 数据类 + 模型注册表（`tests/models/utils.py`）
  - CI 环境变量集中管理（`tests/ci_envs.py`）
  - 测试工具函数（`tests/utils.py`）：`multi_gpu_test`、`large_gpu_test`、`fork_new_process_for_each_test`
  - 参数化矩阵模式（`tests/basic_correctness/test_basic_correctness.py`）
  - `pytest_generate_tests` 动态参数化（`tests/models/conftest.py`）
  - `pytest_collection_modifyitems` 测试集合过滤
  - 输出比较工具（`check_outputs_equal`、`check_logprobs_close`）

- **vllm-ascend e2e**: 现有 pytest 测试体系，包括：
  - `VllmRunner` / `HfRunner` 封装
  - `conftest.py` 层次化 fixture
  - `pytest.mark.parametrize` 参数化测试

## 十一、后续规划

### 11.1 Phase 2 确定实施

以下功能已明确需要实现，将在第一期框架验证通过后纳入 Phase 2 实施计划：

| 功能 | 说明 | 优先级 |
|---|---|---|
| **场景环境变量扩展** | 增加量化配置（W8A8/W4A8）、graph mode 等环境变量支持 | 高 |
| **场景组合** | 支持场景叠加，如 `single_card + quant_w8a8`、`multi_card_tp2 + graph_mode` | 高 |
| **模型注册表** | 建立类似 vLLM `HF_EXAMPLE_MODELS` 的模型注册表，支持自动发现 | 中 |
| **分布式测试扩展** | 支持 DP、EP、PP 组合测试模式 | 中 |
| **多模态测试支持** | 支持图像、视频、音频等多模态输入测试 | 中 |
| **量化测试支持** | 框架原生支持 W8A8、W4A8 等量化场景 | 中 |

### 11.2 待评估项

以下功能需要进一步调研需求和技术可行性，评估后决定是否纳入后续版本：

| 功能 | 待评估内容 | 评估方向 |
|---|---|---|
| **场景探测缓存** | npu-smi 调用频率和性能影响 | 测量缓存收益 vs 复杂度 |
| **远程场景探测** | 通过 API 获取集群信息的实际使用场景 | 是否有远程集群测试需求 |
| **测试报告场景聚合** | 场景维度的测试报告生成 | 报告受众和工具选型 |
