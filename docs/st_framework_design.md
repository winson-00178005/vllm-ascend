# vLLM Ascend ST 测试框架设计方案

## 一、设计目标

基于 pytest 建设场景化 ST（System Test）测试框架，实现：
1. **场景加载机制**：支持单卡/多卡/310P 等不同执行环境
2. **模型配置化**：通过配置文件管理支持的模型列表
3. **条件执行**：测试用例声明所需场景和模型，按条件自动跳过
4. **平滑迁移**：支持现有测试代码快速切换到新框架

---

## 二、现有测试代码模式分析

### 2.1 测试组织模式

| 模式类型 | 示例 | 占比 |
|----------|------|------|
| 函数式测试 | `def test_qwen3_w8a8_quant():` | ~60% |
| 类组织测试 | `class TestBasicInference:` | ~30% |
| 参数化测试 | `@pytest.mark.parametrize("model", [...])` | ~40% |

### 2.2 VllmRunner 使用模式

```python
# 模式1: 直接使用默认配置
with VllmRunner("Qwen/Qwen3-8B") as runner:
    runner.generate_greedy(prompts, max_tokens)

# 模式2: 指定 tensor_parallel_size
with VllmRunner("Qwen/Qwen3-30B-A3B", tensor_parallel_size=2) as runner:
    ...

# 模式3: 指定量化参数
with VllmRunner("vllm-ascend/Qwen3-0.6B-W8A8", quantization="ascend") as runner:
    ...

# 模式4: 特殊配置
with VllmRunner(model,
    max_model_len=8192,
    gpu_memory_utilization=0.7,
    cudagraph_capture_sizes=[1, 2, 4, 8],
) as runner:
    ...
```

### 2.3 环境变量配置模式

```python
# 常用环境变量配置
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
os.environ["HCCL_BUFFSIZE"] = "1024"

# 使用 patch 临时设置
@patch.dict(os.environ, {"HCCL_OP_EXPANSION_MODE": "AIV"})
def test_xxx():
    ...
```

### 2.4 Skip/Skipif 使用

```python
@pytest.mark.skip(reason="TODO: fix later")
@pytest.mark.skipif(vllm_version_is("0.19.0"), reason="...")
@pytest.mark.skipif(True, reason="Fix me, ...")
```

### 2.5 现有 Fixtures

```python
# vl_config fixture - 用于多模态测试
@pytest.fixture(params=PROMPT_CONFIGS.keys())
def vl_config(request):
    ...

# ilama_lora_files, llama32_lora_files - LoRA 测试
@pytest.fixture(scope="session")
def ilama_lora_files():
    ...

# wait_until_npu_memory_free - 内存管理
@pytest.fixture
@wait_until_npu_memory_free(target_free_percentage=0.95)
def test_xxx():
    ...
```

---

## 三、框架架构设计

### 3.1 目录结构

```
tests/e2e/
├── st/                              # 新建 ST 测试框架
│   ├── config/
│   │   ├── scene.yaml               # 场景配置
│   │   └── models.yaml              # 模型列表配置
│   ├── framework/
│   │   ├── __init__.py
│   │   ├── scene_manager.py         # 场景管理器
│   │   ├── model_config.py          # 模型配置加载器
│   │   ├── decorators.py            # 场景/模型筛选装饰器
│   │   ├── fixtures.py              # pytest fixtures
│   │   └── compat.py                # 兼容层 - 支持现有测试模式
│   └── testcases/
│       ├── __init__.py
│       ├── test_basic.py            # 示例测试
│       └── ...
├── conftest.py                      # 保留现有配置
```

### 3.2 核心组件

| 组件 | 职责 |
|------|------|
| `scene_manager.py` | 管理场景配置，提供场景切换和匹配功能 |
| `model_config.py` | 管理模型配置，支持模型-场景映射 |
| `decorators.py` | 提供 `@require_scene`, `@require_model` 等装饰器 |
| `fixtures.py` | 定义 pytest fixtures，处理框架初始化 |
| `compat.py` | **新增**：兼容层，支持现有测试模式无缝迁移 |

---

## 四、配置文件设计

### 4.1 场景配置 (`scene.yaml`)

```yaml
version: "v1.0"
scenarios:
  SINGLECARD:
    name: "单卡测试"
    hardware: "Atlas A2 Series"
    tp_size: 1
    dp_size: 1
    env:
      VLLM_WORKER_MULTIPROC_METHOD: "spawn"
    description: "单卡推理场景"

  MULTICARD_2Cards:
    name: "2卡测试"
    hardware: "Atlas A2 Series"
    tp_size: 2
    dp_size: 1
    env:
      VLLM_WORKER_MULTIPROC_METHOD: "spawn"
      ASCEND_RT_VISIBLE_DEVICES: "0,1"
      HCCL_BUFFSIZE: "1024"
    description: "2卡推理场景"

  MULTICARD_4Cards:
    name: "4卡测试"
    hardware: "Atlas A2 Series"
    tp_size: 4
    dp_size: 1
    env:
      VLLM_WORKER_MULTIPROC_METHOD: "spawn"
      ASCEND_RT_VISIBLE_DEVICES: "0,1,2,3"
      HCCL_BUFFSIZE: "1024"
    description: "4卡推理场景"

  310P_SINGLECARD:
    name: "310P单卡测试"
    hardware: "Ascend 310P"
    tp_size: 1
    dp_size: 1
    description: "310P单卡场景"
```

### 4.2 模型配置 (`models.yaml`)

```yaml
version: "v1.0"
model_groups:
  dense_models:
    - name: "Qwen/Qwen3-8B"
      supported_scenes: ["SINGLECARD", "MULTICARD_2Cards", "MULTICARD_4Cards"]
      max_model_len: 8192
      tp_size: [1, 2, 4]
      quantizations: ["fp16", "fp16"]  # [fp16, w8a8] 模型支持哪些量化

    - name: "Qwen/Qwen3-30B-A3B"
      supported_scenes: ["MULTICARD_2Cards", "MULTICARD_4Cards"]
      max_model_len: 8192
      tp_size: [2, 4]

    - name: "vllm-ascend/Qwen3-0.6B-W8A8"
      supported_scenes: ["SINGLECARD", "MULTICARD_2Cards", "MULTICARD_4Cards"]
      max_model_len: 8192
      tp_size: [1, 2, 4]
      quantizations: ["ascend"]
      enforce_eager: true

  moe_models:
    - name: "Qwen/Qwen3-235B-A22B"
      supported_scenes: ["MULTICARD_4Cards"]
      max_model_len: 4096
      tp_size: [4]
      enable_expert_parallel: true

    - name: "Qwen/Qwen3-30B-A3B"
      supported_scenes: ["MULTICARD_2Cards", "MULTICARD_4Cards"]
      max_model_len: 8192
      tp_size: [2, 4]
      enable_expert_parallel: true

  310p_models:
    - name: "Qwen/Qwen3-8B"
      supported_scenes: ["310P_SINGLECARD"]
      max_model_len: 4096
      tp_size: [1]

  vl_models:
    - name: "Qwen/Qwen3-VL-8B-Instruct"
      supported_scenes: ["SINGLECARD"]
      max_model_len: 8192
      tp_size: [1]
      is_multimodal: true
```

---

## 五、核心组件实现

### 5.1 场景管理器 (`scene_manager.py`)

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import yaml
import os

@dataclass
class SceneInfo:
    name: str
    hardware: str
    tp_size: int
    dp_size: int
    description: str
    env: Dict[str, str] = field(default_factory=dict)  # 新增：场景相关环境变量

class SceneManager:
    _instance: Optional["SceneManager"] = None

    def __init__(self, config_path: str):
        self.scenes: dict[str, SceneInfo] = {}
        self.current_scene: Optional[str] = None
        self._original_env: Dict[str, str] = {}  # 保存原始环境变量
        self._load_config(config_path)

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> "SceneManager":
        if cls._instance is None:
            path = config_path or cls._get_default_config_path()
            cls._instance = cls(path)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """重置单例 - 用于测试隔离"""
        if cls._instance is not None:
            cls._instance._restore_env()
        cls._instance = None

    def _load_config(self, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        for scene_id, info in config["scenarios"].items():
            self.scenes[scene_id] = SceneInfo(
                name=info["name"],
                hardware=info["hardware"],
                tp_size=info["tp_size"],
                dp_size=info["dp_size"],
                description=info.get("description", ""),
                env=info.get("env", {})
            )

    def set_current_scene(self, scene_id: str) -> bool:
        if scene_id not in self.scenes:
            return False

        self._restore_env()  # 先恢复之前的环境变量
        self.current_scene = scene_id
        self._apply_env()   # 应用新场景的环境变量
        return True

    def _apply_env(self):
        """应用当前场景的环境变量"""
        if self.current_scene is None:
            return
        scene = self.scenes[self.current_scene]
        self._original_env = {}
        for key, value in scene.env.items():
            self._original_env[key] = os.environ.get(key)
            os.environ[key] = value

    def _restore_env(self):
        """恢复原始环境变量"""
        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._original_env = {}

    def is_scene_match(self, required_scenes: List[str]) -> bool:
        if not required_scenes:
            return True
        if self.current_scene is None:
            return True
        return self.current_scene in required_scenes

    def get_scene_info(self, scene_id: Optional[str] = None) -> Optional[SceneInfo]:
        if scene_id is None:
            scene_id = self.current_scene
        return self.scenes.get(scene_id)
```

### 5.2 模型配置加载器 (`model_config.py`)

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import yaml

@dataclass
class ModelInfo:
    name: str
    supported_scenes: List[str]
    max_model_len: int
    tp_size: List[int]
    quantizations: List[str] = field(default_factory=list)
    enforce_eager: bool = False
    enable_expert_parallel: bool = False
    is_multimodal: bool = False
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

class ModelConfig:
    _instance: Optional["ModelConfig"] = None

    def __init__(self, config_path: str):
        self.model_groups: Dict[str, List[ModelInfo]] = {}
        self.all_models: List[ModelInfo] = []
        self.current_model: Optional[str] = None
        self._load_config(config_path)

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> "ModelConfig":
        if cls._instance is None:
            path = config_path or cls._get_default_config_path()
            cls._instance = cls(path)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def _load_config(self, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        for group_name, models in config["model_groups"].items():
            model_list = []
            for model in models:
                model_info = ModelInfo(
                    name=model["name"],
                    supported_scenes=model["supported_scenes"],
                    max_model_len=model["max_model_len"],
                    tp_size=model["tp_size"],
                    quantizations=model.get("quantizations", []),
                    enforce_eager=model.get("enforce_eager", False),
                    enable_expert_parallel=model.get("enable_expert_parallel", False),
                    is_multimodal=model.get("is_multimodal", False),
                    extra_kwargs=model.get("extra_kwargs", {})
                )
                model_list.append(model_info)
                self.all_models.append(model_info)
            self.model_groups[group_name] = model_list

    def get_models_for_scene(self, scene_id: str) -> List[ModelInfo]:
        return [m for m in self.all_models if scene_id in m.supported_scenes]

    def is_model_supported(self, model_name: str, scene_id: str) -> bool:
        for model in self.all_models:
            if model.name == model_name and scene_id in model.supported_scenes:
                return True
        return False

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        for model in self.all_models:
            if model.name == model_name:
                return model
        return None
```

### 5.3 兼容层 (`compat.py`) - 新增核心组件

这是支持现有测试平滑迁移的关键组件：

```python
"""
兼容层模块 - 支持现有测试代码无缝迁移到新框架

提供:
1. 兼容现有 @pytest.mark.parametrize 模式
2. 场景感知的模型列表 fixture
3. 自动跳过不支持的测试
"""

import pytest
from typing import List, Optional, Callable, Any
from functools import wraps

from tests.e2e.st.framework.scene_manager import SceneManager
from tests.e2e.st.framework.model_config import ModelConfig

class SceneAwareParametrize:
    """
    场景感知的 parametrize - 替代 pytest.mark.parametrize

    功能:
    1. 只生成当前场景支持的模型列表
    2. 自动跳过测试如果没有任何模型支持当前场景
    3. 兼容现有 @pytest.mark.parametrize 语法

    用法:
    @scene_aware_parametrize("model", ["Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B"])
    def test_model(model):
        ...
    """

    def __init__(self, param_name: str, model_list: List[str]):
        self.param_name = param_name
        self.model_list = model_list

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # 收集当前场景支持的模型
        scene_manager = SceneManager.get_instance()
        current_scene = scene_manager.current_scene

        supported_models = []
        model_config = ModelConfig.get_instance()

        for model in self.model_list:
            if model_config.is_model_supported(model, current_scene):
                supported_models.append(model)

        if not supported_models:
            pytest.skip(f"No models from {self.model_list} supported in scene {current_scene}")

        # 使用 pytest.mark.parametrize 生成测试
        return pytest.mark.parametrize(self.param_name, supported_models)(wrapper)

        return wrapper


def scene_aware_parametrize(param_name: str, model_list: List[str]):
    """
    便捷函数 - 创建场景感知的参数化测试

    示例:
    @scene_aware_parametrize("model", [
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-30B-A3B",
        "vllm-ascend/Qwen3-0.6B-W8A8"
    ])
    def test_inference(model):
        with VllmRunner(model) as runner:
            runner.generate_greedy(["Hello"], max_tokens=10)
    """
    return SceneAwareParametrize(param_name, model_list)


class ModelFromConfigParametrize:
    """
    从配置文件加载模型列表的参数化

    用法:
    # 使用配置文件中的模型组
    @model_from_config("model", group="dense_models")
    def test_inference(model):
        ...

    # 使用所有支持的模型
    @model_from_config("model")
    def test_inference(model):
        ...
    """

    def __init__(self, param_name: str, group: Optional[str] = None):
        self.param_name = param_name
        self.group = group

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        scene_manager = SceneManager.get_instance()
        current_scene = scene_manager.current_scene
        model_config = ModelConfig.get_instance()

        if self.group:
            models = model_config.model_groups.get(self.group, [])
        else:
            models = model_config.all_models

        # 过滤出当前场景支持的模型
        supported = [m for m in models if current_scene in m.supported_scenes]

        if not supported:
            pytest.skip(f"No models in group '{self.group}' supported for scene {current_scene}")

        model_names = [m.name for m in supported]
        return pytest.mark.parametrize(self.param_name, model_names)(wrapper)

        return wrapper


def model_from_config(param_name: str, group: Optional[str] = None):
    """便捷函数"""
    return ModelFromConfigParametrize(param_name, group)


def skip_if_model_not_supported(model: str, scene: Optional[str] = None):
    """
    跳过如果模型在当前场景不支持

    用法:
    def test_specific_model():
        skip_if_model_not_supported("Qwen/Qwen3-8B")
        # ... 测试代码 ...
    """
    scene_manager = SceneManager.get_instance()
    model_config = ModelConfig.get_instance()

    current_scene = scene or scene_manager.current_scene
    if not model_config.is_model_supported(model, current_scene):
        pytest.skip(f"Model {model} not supported in scene {current_scene}")


# 便捷的模型列表定义
def get_models_for_current_scene(model_names: List[str]) -> List[str]:
    """
    获取当前场景支持的模型列表

    用于现有测试迁移时的快速适配:

    MODELS = get_models_for_current_scene([
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-30B-A3B"
    ])

    @pytest.mark.parametrize("model", MODELS)
    def test_model(model):
        ...
    """
    scene_manager = SceneManager.get_instance()
    current_scene = scene_manager.current_scene
    model_config = ModelConfig.get_instance()

    supported = []
    for model in model_names:
        if model_config.is_model_supported(model, current_scene):
            supported.append(model)

    return supported
```

### 5.4 装饰器 (`decorators.py`)

```python
from typing import List, Union, Optional, Callable
import pytest
from functools import wraps

def require_scene(*scenes: str):
    """
    装饰器：要求特定场景才能执行测试

    支持函数式和类方法测试:
    @require_scene("SINGLECARD")
    def test_singlecard_only():
        ...

    @require_scene("SINGLECARD", "MULTICARD_2Cards")
    def test_multi_scene():
        ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _check_scene_match(list(scenes)):
                pytest.skip(f"Test requires scene {scenes}, current scene does not match")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_model(*models: str):
    """
    装饰器：要求特定模型才能执行测试

    示例:
    @require_model("Qwen/Qwen3-8B")
    def test_qwen8b_only():
        ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _check_model_match(list(models)):
                pytest.skip(f"Test requires model {models}, current model does not match")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_scene_and_model(scene: str, model: str):
    """
    装饰器：要求特定场景+模型组合

    示例:
    @require_scene_and_model("SINGLECARD", "Qwen/Qwen3-8B")
    def test_specific_combination():
        ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            scene_ok = _check_scene_match([scene])
            model_ok = _check_model_match([model])

            if not scene_ok:
                pytest.skip(f"Test requires scene {scene}")
            if not model_ok:
                pytest.skip(f"Test requires model {model}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


def _check_scene_match(scenes: List[str]) -> bool:
    manager = SceneManager.get_instance()
    return manager.is_scene_match(scenes)


def _check_model_match(models: List[str]) -> bool:
    config = ModelConfig.get_instance()
    current = getattr(config, 'current_model', None)
    if current is None:
        return True
    return current in models
```

### 5.5 Fixtures (`fixtures.py`)

```python
import pytest
import os
from typing import Optional

DEFAULT_SCENE = os.getenv("VLLM_TEST_SCENE", "SINGLECARD")
DEFAULT_MODEL = os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-8B")


@pytest.fixture(scope="session", autouse=True)
def setup_test_framework():
    """
    自动初始化测试框架

    通过环境变量配置:
    - VLLM_TEST_SCENE: 场景名称 (默认 SINGLECARD)
    - VLLM_TEST_MODEL: 模型名称 (默认 Qwen/Qwen3-8B)
    """
    from tests.e2e.st.framework.scene_manager import SceneManager
    from tests.e2e.st.framework.model_config import ModelConfig

    scene_manager = SceneManager.get_instance()
    model_config = ModelConfig.get_instance()

    scene_manager.set_current_scene(DEFAULT_SCENE)
    model_config.current_model = DEFAULT_MODEL

    yield

    SceneManager.reset_instance()
    ModelConfig.reset_instance()


@pytest.fixture
def scene_info():
    """获取当前场景信息"""
    manager = SceneManager.get_instance()
    return manager.get_scene_info()


@pytest.fixture
def current_model():
    """获取当前测试模型"""
    config = ModelConfig.get_instance()
    return config.current_model


@pytest.fixture
def scene_manager():
    """获取场景管理器实例"""
    return SceneManager.get_instance()


@pytest.fixture
def model_config():
    """获取模型配置实例"""
    return ModelConfig.get_instance()


@pytest.fixture
def vllm_runner_for_scene():
    """
    根据场景配置创建 VllmRunner

    自动应用当前场景的默认配置:
    - tensor_parallel_size
    - 环境变量等
    """
    from tests.e2e.conftest import VllmRunner

    def _create(model_name: str, **kwargs):
        manager = SceneManager.get_instance()
        scene = manager.get_scene_info()

        if scene:
            kwargs.setdefault("tensor_parallel_size", scene.tp_size)

        return VllmRunner(model_name, **kwargs)

    return _create


@pytest.fixture
def model_info():
    """
    获取指定模型的配置信息

    用法:
    def test_something(model_info):
        if model_info.enforce_eager:
            ...
    """
    def _get(model_name: str = None):
        config = ModelConfig.get_instance()
        name = model_name or config.current_model
        return config.get_model_info(name)
    return _get
```

---

## 六、现有测试迁移指南

### 6.1 迁移模式对比

| 现有模式 | 迁移后模式 | 迁移难度 |
|----------|------------|----------|
| 直接硬编码模型 | `@require_model(...)` | ⭐ |
| `@pytest.mark.parametrize("model", [..])` | `@scene_aware_parametrize("model", [..])` | ⭐⭐ |
| 手动环境变量设置 | 场景配置自动加载 | ⭐⭐ |
| 目录分离 (`singlecard/`, `multicard/`) | `@require_scene(...)` | ⭐⭐⭐ |

### 6.2 迁移示例

#### 示例1: 简单测试迁移

**迁移前:**
```python
# tests/e2e/singlecard/test_models.py
def test_minicpm():
    with VllmRunner("openbmb/MiniCPM-2B-sft-bf16", max_model_len=512) as runner:
        runner.generate_greedy(["Hello"], max_tokens=5)
```

**迁移后:**
```python
# tests/e2e/st/testcases/test_models.py
from tests.e2e.st.framework.decorators import require_scene

@require_scene("SINGLECARD")
def test_minicpm():
    with VllmRunner("openbmb/MiniCPM-2B-sft-bf16", max_model_len=512) as runner:
        runner.generate_greedy(["Hello"], max_tokens=5)
```

#### 示例2: 参数化测试迁移

**迁移前:**
```python
# tests/e2e/multicard/2-cards/test_qwen3_moe.py
@pytest.mark.parametrize("model", ["Qwen/Qwen3-30B-A3B", "vllm-ascend/Qwen3-30B-A3B-W8A8"])
def test_model(model):
    with VllmRunner(model, tensor_parallel_size=2) as runner:
        runner.generate_greedy(["Hello"], max_tokens=5)
```

**迁移后:**
```python
# tests/e2e/st/testcases/test_qwen3_moe.py
from tests.e2e.st.framework.compat import scene_aware_parametrize

@scene_aware_parametrize("model", [
    "Qwen/Qwen3-30B-A3B",
    "vllm-ascend/Qwen3-30B-A3B-W8A8"
])
def test_model(model):
    with VllmRunner(model, tensor_parallel_size=2) as runner:
        runner.generate_greedy(["Hello"], max_tokens=5)
```

#### 示例3: 多卡测试迁移

**迁移前:**
```python
# tests/e2e/multicard/2-cards/test_data_parallel.py
@patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"})
def test_dp2():
    ...
```

**迁移后:**
```python
# tests/e2e/st/testcases/test_data_parallel.py
from tests.e2e.st.framework.decorators import require_scene

@require_scene("MULTICARD_2Cards")
def test_dp2():
    # 场景配置已自动设置 ASCEND_RT_VISIBLE_DEVICES
    ...
```

### 6.3 渐进式迁移策略

```
Phase 1: 框架搭建
├── 创建目录结构
├── 实现核心组件
└── 添加 compat 兼容层

Phase 2: 新测试使用新框架
└── 新编写的测试使用 @require_scene 等装饰器

Phase 3: 现有测试迁移
├── 优先迁移 singlecard 测试 (改动最小)
├── 然后迁移 multicard 测试
└── 最后迁移 310p 测试

Phase 4: 目录整合 (可选)
├── 将 singlecard/, multicard/, 310p/ 测试迁移到 st/testcases/
└── 或保持现状，通过符号链接或 pytest.ini 配置
```

---

## 七、执行方式

### 7.1 命令行执行

```bash
# 使用默认配置执行 (SINGLECARD + Qwen/Qwen3-8B)
pytest -sv tests/e2e/st/

# 指定场景
export VLLM_TEST_SCENE="MULTICARD_2Cards"
export VLLM_TEST_MODEL="Qwen/Qwen3-30B-A3B"
pytest -sv tests/e2e/st/

# 执行特定测试
pytest -sv tests/e2e/st/testcases/test_basic.py

# 使用 pytest 内置过滤
pytest -sv tests/e2e/st/ -k "test_specific"

# 查看跳过的测试
pytest -sv tests/e2e/st/ -v --tb=no
```

### 7.2 CI 集成矩阵

```yaml
# .github/workflows/st-matrix-test.yml
jobs:
  st-test:
    strategy:
      fail-fast: false
      matrix:
        scene: [SINGLECARD, MULTICARD_2Cards, MULTICARD_4Cards, 310P_SINGLECARD]
        model:
          - Qwen/Qwen3-8B
          - Qwen/Qwen3-30B-A3B
          - vllm-ascend/Qwen3-0.6B-W8A8
    runs-on: ascend-runner
    steps:
      - name: Run ST Tests
        env:
          VLLM_TEST_SCENE: ${{ matrix.scene }}
          VLLM_TEST_MODEL: ${{ matrix.model }}
        run: |
          pytest -sv tests/e2e/st/ --tb=short -x
```

---

## 八、设计优化点

### 8.1 相比初版设计的优化

| 优化项 | 初版设计 | 优化后 | 原因 |
|--------|----------|--------|------|
| **兼容层** | 无 | `compat.py` | 支持现有 `@pytest.mark.parametrize` 模式 |
| **环境变量管理** | 基础 | 完整保存/恢复 | 多卡测试需要动态切换环境变量 |
| **场景配置** | 基础字段 | 支持 `env` 映射 | 多卡测试需要配置设备映射 |
| **模型配置** | 基础字段 | 支持 `quantizations`, `extra_kwargs` | 覆盖现有测试的各种配置组合 |
| **单例重置** | 无 | `reset_instance()` | 测试隔离需要 |

### 8.2 向后兼容性保证

1. **VllmRunner 直接使用** - 现有测试的 `with VllmRunner(...)` 模式保持不变
2. **Fixtures 独立** - 新框架的 fixtures 不影响现有 `vl_config` 等
3. **渐进式迁移** - 可以逐个测试迁移，不需要一次性全部改完
4. **装饰器可选** - 现有测试不迁移也能正常运行

---

## 九、目录结构最终版

```
tests/e2e/
├── st/                              # ST 测试框架根目录
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── scene.yaml               # 场景配置
│   │   └── models.yaml              # 模型列表配置
│   ├── framework/
│   │   ├── __init__.py
│   │   ├── scene_manager.py         # 场景管理器
│   │   ├── model_config.py         # 模型配置加载器
│   │   ├── decorators.py           # 场景/模型筛选装饰器
│   │   ├── fixtures.py              # pytest fixtures
│   │   └── compat.py               # 兼容层 (关键!)
│   └── testcases/
│       ├── __init__.py
│       └── ...                      # 迁移/新建的测试
│
├── conftest.py                      # 现有配置 (保留)
├── singlecard/                      # 现有测试 (可逐步迁移)
├── multicard/                       # 现有测试 (可逐步迁移)
└── 310p/                            # 现有测试 (可逐步迁移)
```

---

## 十、后续扩展建议

1. **动态场景切换**：支持在同一个测试会话中动态切换场景
2. **覆盖率收集**：集成 coverage.py 自动收集测试覆盖率
3. **测试报告增强**：生成场景-模型矩阵报告
4. **资源管理**：基于场景自动管理 NPU 资源分配
5. **超时控制**：不同场景配置不同的测试超时时间
