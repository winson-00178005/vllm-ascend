# vllm-ascend ST测试框架

场景化测试框架，支持场景自动检测、模型配置管理、测试过滤等功能。

## 快速开始

```bash
# 列出可用场景
pytest tests/st/ --list-scenes

# 列出可用模型
pytest tests/st/ --list-models

# 运行指定场景的测试
pytest tests/st/ --scene=SINGLE_CARD_310P

# 运行指定场景和模型的测试
pytest tests/st/ --scene=MULTI_CARD_310P_2 --model=qwen-7b

# 兼容模式运行现有E2E测试
pytest tests/e2e/ -p st_legacy_compat --scene=SINGLE_CARD_310P
```

## 项目结构

```
tests/st/
├── config/
│   ├── scenes.yaml           # 9个场景配置
│   ├── models.yaml           # 8个模型配置
│   └── environments.yaml     # 环境配置
│
├── framework/                # 核心框架
│   ├── scene_manager.py      # 场景管理（单例）
│   ├── model_loader.py       # 模型加载（单例）
│   ├── decorators.py         # 测试装饰器
│   ├── scene_detector.py     # 路径检测
│   └── legacy_adapter.py     # 兼容适配
│
├── fixtures/                 # pytest fixtures
│   ├── scene_fixtures.py     # scene_manager, current_scene
│   └── model_fixtures.py     # model_config, vllm_runner
│
├── testcases/                # 测试用例
│   └── test_examples.py      # 示例测试
│
├── conftest.py               # pytest配置入口
├── pytest_plugins.py         # pytest插件
└── pytest_legacy_compat.py   # E2E兼容插件
```

## 使用方式

### 1. 装饰器标记

```python
from framework.decorators import require_scene, require_model, require_device, require_tags

@require_scene("SINGLE_CARD_310P")
def test_single_card():
    pass

@require_scene("SINGLE_CARD_310P|MULTI_CARD_310P_2")
def test_multi_scene():
    pass

@require_scene("SINGLE_CARD_310P")
@require_model("qwen-7b")
def test_with_model():
    pass

@require_device("310P")
def test_310p_only():
    pass

@require_tags("single")
def test_single_tag():
    pass
```

### 2. pytest标记

```python
import pytest

@pytest.mark.scene("SINGLE_CARD_310P")
@pytest.mark.model("qwen-7b")
def test_with_markers():
    pass
```

### 3. fixtures（推荐方式）

```python
def test_with_fixtures(scene_manager, current_scene, model_loader):
    scenes = scene_manager.get_all_scenes()
    assert current_scene.card_count > 0
    models = model_loader.get_all_models()
```

### 4. 实际测试用例示例（最佳实践）

使用`current_scene` fixture自动获取场景参数：

```python
from framework.decorators import require_scene, require_model

EXAMPLE_PROMPTS = ["Hello, my name is"]
MAX_TOKENS = 5

@require_scene("MULTI_CARD_310P_2")
def test_qwen3_dense_tp2_fp16(current_scene):
    """测试Qwen3-8B TP2 FP16推理"""
    from tests.e2e.conftest import VllmRunner
    
    # 自动从场景获取tensor_parallel_size
    with VllmRunner(
        "Qwen/Qwen3-8B",
        tensor_parallel_size=current_scene.card_count,  # 自动获取
        enforce_eager=True,
        dtype="float16",
        max_model_len=16384,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(EXAMPLE_PROMPTS, MAX_TOKENS)
        assert len(outputs) == len(EXAMPLE_PROMPTS)
        # 验证场景属性
        assert current_scene.card_count == 2
        assert current_scene.device_type == "310P"
```

**关键改进点**：
- ✅ 使用`current_scene.card_count`替代硬编码`tensor_parallel_size`
- ✅ 使用`current_scene.device_type`验证设备类型
- ✅ 使用`@require_scene`装饰器进行场景过滤

### 4. E2E兼容模式

无需修改tests/e2e/中的任何文件：

```bash
# 使用兼容插件运行E2E测试
pytest tests/e2e/ -p st_legacy_compat --scene=SINGLE_CARD_310P

# 列出场景（兼容模式）
pytest tests/e2e/ -p st_legacy_compat --list-scenes

# 路径自动检测场景
pytest tests/e2e/310p/singlecard/ -p st_legacy_compat
# 自动检测为 SINGLE_CARD_310P
```

## 场景定义

| 场景名称 | 设备 | 卡数 | 描述 |
|---------|------|------|------|
| SINGLE_CARD_310P | 310P | 1 | 310P单卡 |
| MULTI_CARD_310P_2 | 310P | 2 | 310P双卡 |
| MULTI_CARD_310P_4 | 310P | 4 | 310P四卡 |
| SINGLE_CARD_910B | 910B | 1 | 910B单卡 |
| MULTI_CARD_910B_2 | 910B | 2 | 910B双卡 |
| SINGLE_NODE_MULTI_CARD | 310P | 8 | 单节点多卡 |
| MULTI_NODE_MULTI_CARD | 310P | 16 | 多节点多卡 |
| NIGHTLY_SINGLE_NODE | 310P | 8 | 每日测试 |
| WEEKLY_SINGLE_NODE | 310P | 8 | 每周测试 |

## 模型定义

| 模型名称 | 路径 | 支持设备 |
|---------|------|---------|
| qwen-7b | Qwen/Qwen-7B | 310P, 910B |
| qwen2-7b | Qwen/Qwen2-7B | 310P, 910B |
| qwen3-8b | Qwen/Qwen3-8B | 310P, 910B |
| qwen3-30b-moe | Qwen/Qwen3-30B-A3B | 310P, 910B |
| qwen-72b | Qwen/Qwen2.5-72B | 310P, 910B |
| llama-7b | meta-llama/Llama-2-7b-hf | 310P |
| llama-13b | meta-llama/Llama-2-13b-hf | 310P |
| deepseek-67b | deepseek-ai/deepseek-llm-67b-chat | 310P, 910B |

## 场景自动检测

测试路径 → 场景映射：

```
tests/e2e/310p/singlecard/     → SINGLE_CARD_310P
tests/e2e/310p/multicard/      → MULTI_CARD_310P_*
tests/e2e/multicard/2-cards/   → MULTI_CARD_310P_2
tests/e2e/multicard/4-cards/   → MULTI_CARD_310P_4
tests/e2e/nightly/             → NIGHTLY_SINGLE_NODE
tests/e2e/weekly/              → WEEKLY_SINGLE_NODE
```

## API参考

### SceneManager

```python
from framework.scene_manager import SceneManager

sm = SceneManager()  # 单例

sm.set_current_scene("SINGLE_CARD_310P")
scene = sm.get_current_scene()
scenes = sm.get_all_scenes()
scenes = sm.get_scenes_by_tag("single")
scenes = sm.get_scenes_by_device("310P")
```

### ModelLoader

```python
from framework.model_loader import ModelLoader

ml = ModelLoader()  # 单例

config = ml.get_model_config("qwen-7b")
models = ml.get_all_models()
models = ml.get_models_by_device("310P")
ml.load_model("qwen-7b")
ml.unload_model("qwen-7b")
```

### SceneDetector

```python
from framework.scene_detector import SceneDetector

scene_name, scene = SceneDetector.detect_from_path(test_file_path)
model_name, params = SceneDetector.detect_from_code(test_code)
scene_info = SceneDetector.detect_all(test_file_path, test_code)
```

## 命令行选项

| 选项 | 说明 |
|------|------|
| `--scene=NAME` | 指定测试场景 |
| `--model=NAME` | 指定测试模型 |
| `--list-scenes` | 列出可用场景 |
| `--list-models` | 列出可用模型 |
| `--use-legacy-mode` | 兼容模式运行 |
| `--auto-detect-scene` | 自动检测场景（默认开启） |

## 进度

- **Phase 1**: 核心框架 ✅ (1108行)
- **Phase 2**: pytest组件 ✅ (817行)
- **Phase 3**: 兼容插件 ✅ (新增226行)

**总代码**: 2940行

## 设计文档

- `docs/st_test_framework_design.md` - 详细设计
- `docs/st_framework_compatibility_design.md` - 兼容性设计
- `BENEFIT_ANALYSIS.md` - ROI分析