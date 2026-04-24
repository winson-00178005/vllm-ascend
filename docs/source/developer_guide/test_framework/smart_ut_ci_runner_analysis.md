# smart_ut 分支 CI Runner 管理方案融合分析

> **分析日期**: 2026-04-24  
> **分析目标**: 将 smart_ut 分支的 CI runner 管理最佳实践融合到当前 ST 框架

## 一、smart_ut 分支核心设计

### 1.1 架构概览

```
.github/workflows/
├── _e2e_test.yaml              # 可复用工作流（workflow_call）
├── pr_test_full.yaml           # PR 全量测试入口
├── pr_test_light.yaml          # PR 轻量测试入口
├── pr_optional_smart_e2e.yaml  # 评论触发的智能测试
├── schedule_nightly_test_a2.yaml  # A2 夜间测试
├── schedule_nightly_test_a3.yaml  # A3 夜间测试
└── scripts/
    ├── config.yaml             # 测试套件配置（核心）
    ├── run_suite.py            # 测试执行引擎
    ├── ci_utils.py             # 通用工具函数
    ├── ci_log_summary.py       # 日志分析
    └── determine_smart_e2e_scope.py  # 智能范围判定
```

### 1.2 核心设计模式

#### 模式 1: 测试套件配置化（config.yaml）

```yaml
e2e-singlecard:
- name: tests/e2e/singlecard/test_models.py
  estimated_time: 315
- name: tests/e2e/singlecard/test_quantization.py
  estimated_time: 291

e2e-multicard-2-cards:
- name: tests/e2e/multicard/2-cards/test_data_parallel.py
  estimated_time: 426
- name: tests/e2e/multicard/2-cards/test_qwen3_moe.py
  estimated_time: 974

e2e-singlecard-light:
- name: tests/e2e/singlecard/test_aclgraph_accuracy.py::test_piecewise_res_consistency
  estimated_time: 394
```

**关键特性**:
- 每个测试文件/用例有 `estimated_time`（预估时间）
- 支持 `is_skipped` 标记跳过测试
- 支持精确到测试函数的粒度（`::test_xxx`）

#### 模式 2: 智能分区算法（run_suite.py）

```python
def partition(files: list[TestFile], rank: int, size: int) -> list[TestFile]:
    """
    使用贪心算法将测试分成 size 组，每组预估时间大致相等
    返回 rank 索引的组
    """
    active = [f for f in files if not f.is_skipped]
    # 按预估时间降序排序
    indexed = sorted(enumerate(active), key=lambda x: (-x[1].estimated_time, x[0]))
    
    buckets: list[list[int]] = [[] for _ in range(size)]
    sums = [0.0] * size
    
    for idx, test in indexed:
        lightest = sums.index(min(sums))  # 找最轻的桶
        buckets[lightest].append(idx)
        sums[lightest] += test.estimated_time
    
    return sorted([active[i] for i in buckets[rank]], key=lambda f: f.estimated_time, reverse=True)
```

**优势**:
- 自动平衡各 runner 的负载
- 支持动态分区大小（`--auto-partition-size`）
- 支持指定分区索引（`--auto-partition-id`）

#### 模式 3: Runner 标签映射

```json
// runner_label.json
{
  "singlecard": "linux-aarch64-a2b3-1",
  "multicard-2-cards": "linux-aarch64-a3-2",
  "multicard-4-cards": "linux-aarch64-a3-4",
  "310p": "linux-aarch64-310p-1"
}
```

#### 模式 4: 可复用工作流（workflow_call）

```yaml
# _e2e_test.yaml
on:
  workflow_call:
    inputs:
      vllm:
        required: true
        type: string
      image:
        required: true
        type: string
      type:  # 'full' or 'light'
        required: true
        type: string
      singlecard_tests:  # 评论触发的自定义测试
        required: false
        type: string
        default: ''
```

#### 模式 5: 定时数据收集与预估时间优化

```python
def _save_timing_json(records, suite, partition_id, partition_size, output_path):
    payload = {
        "suite": suite,
        "partition_id": partition_id,
        "commit_sha": os.environ.get("GITHUB_SHA", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": [r.to_dict() for r in records if r.passed],
    }
    output_path.write_text(json.dumps(payload, indent=2))
```

**用途**:
- 收集实际执行时间
- 自动更新 `estimated_time`
- 持续优化分区平衡

## 二、与当前 ST 框架的差异对比

| 维度 | 当前 ST 框架 | smart_ut 分支 | 差距分析 |
|---|---|---|---|
| **测试配置** | YAML 场景+模型配置 | YAML 测试套件+预估时间 | 互补，可融合 |
| **用例过滤** | pytest marker + conftest hook | config.yaml 白名单 | smart_ut 更严格 |
| **分区策略** | 无 | 贪心算法自动平衡 | 需引入 |
| **Runner 管理** | GitHub matrix 静态映射 | runner_label.json 动态映射 | 需引入 |
| **时间预估** | 无 | estimated_time 字段 | 需引入 |
| **定时优化** | 无 | timing JSON 收集+自动更新 | 需引入 |
| **评论触发** | 无 | `/e2e <tests>` 自定义测试 | 可选引入 |
| **日志分析** | 无 | ci_log_summary.py | 需引入 |

## 三、融合方案

### 3.1 架构融合

```
tests/e2e/
├── conftest.py                 # 保留：场景/模型过滤
├── st_config/
│   ├── scenes.yaml             # 保留：场景定义
│   ├── models.yaml             # 保留：模型定义
│   ├── test_suites.yaml        # 新增：测试套件配置（融合 smart_ut）
│   └── framework/
│       ├── scene.py            # 保留
│       ├── model.py            # 保留
│       ├── fixtures.py         # 保留
│       └── runner.py           # 新增：runner 管理
└── .github/workflows/
    ├── scripts/
    │   ├── run_suite.py        # 引入：测试执行引擎
    │   ├── ci_utils.py         # 引入：通用工具
    │   └── ci_log_summary.py   # 引入：日志分析
    └── st_e2e_test.yaml        # 新增：ST 测试工作流
```

### 3.2 test_suites.yaml 设计

```yaml
# 融合场景+模型+预估时间
e2e-singlecard:
- name: tests/e2e/singlecard/test_offline_inference.py
  estimated_time: 300
  scenes: [single_card]
  models: [Qwen3-8B-Base]

- name: tests/e2e/singlecard/test_guided_decoding.py
  estimated_time: 413
  scenes: [single_card]
  models: [Qwen3-8B-Base]

e2e-multicard-tp2:
- name: tests/e2e/multicard/test_offline_inference_distributed.py
  estimated_time: 500
  scenes: [multi_card_tp2]
  models: [DeepSeek-V2-Lite]

e2e-singlecard-light:
- name: tests/e2e/singlecard/test_offline_inference.py::test_models
  estimated_time: 150
  scenes: [single_card]
  models: [Qwen3-8B-Base]
```

### 3.3 runner.py 设计

```python
"""
Runner 管理器 - 根据场景自动选择 runner
"""

import yaml
from pathlib import Path

class RunnerManager:
    def __init__(self):
        self._labels = self._load_runner_labels()
    
    def _load_runner_labels(self) -> dict:
        config_path = Path(__file__).parent.parent / "runner_labels.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def get_runner_for_scene(self, scene: str) -> str:
        """获取场景对应的 runner 标签"""
        return self._labels.get(scene, "linux-aarch64-a2-1")
    
    def get_runner_for_suite(self, suite: str) -> str:
        """获取测试套件对应的 runner 标签"""
        suite_to_runner = {
            "e2e-singlecard": "linux-aarch64-a2b3-1",
            "e2e-multicard-tp2": "linux-aarch64-a3-2",
            "e2e-multicard-tp4": "linux-aarch64-a3-4",
            "e2e-310p": "linux-aarch64-310p-1",
        }
        return suite_to_runner.get(suite, "linux-aarch64-a2-1")
```

### 3.4 CI 工作流设计

```yaml
# .github/workflows/st_e2e_test.yaml
name: 'ST e2e test'

on:
  workflow_call:
    inputs:
      vllm:
        required: true
        type: string
      type:  # 'full' or 'light'
        required: true
        type: string

jobs:
  singlecard:
    name: singlecard-${{ inputs.type }}
    runs-on: linux-aarch64-a2b3-1
    strategy:
      fail-fast: false
      matrix:
        part: ${{ fromJSON(inputs.type == 'full' && '[0, 1]' || '[0]') }}
    steps:
      - uses: actions/checkout@v6
      - name: Run ST tests
        run: |
          python3 .github/workflows/scripts/run_suite.py \
            --suite e2e-singlecard \
            --auto-partition-id "${{ matrix.part }}" \
            --auto-partition-size 2 \
            --continue-on-error

  multicard-tp2:
    name: multicard-tp2-${{ inputs.type }}
    runs-on: linux-aarch64-a3-2
    strategy:
      fail-fast: false
      matrix:
        part: ${{ fromJSON(inputs.type == 'full' && '[0, 1]' || '[0]') }}
    steps:
      - uses: actions/checkout@v6
      - name: Run ST tests
        run: |
          python3 .github/workflows/scripts/run_suite.py \
            --suite e2e-multicard-tp2 \
            --auto-partition-id "${{ matrix.part }}" \
            --auto-partition-size 2 \
            --continue-on-error
```

## 四、实施建议

### 4.1 第一阶段：引入核心脚本

| 任务 | 来源 | 说明 |
|---|---|---|
| 复制 `run_suite.py` | smart_ut | 测试执行引擎 |
| 复制 `ci_utils.py` | smart_ut | 通用工具函数 |
| 复制 `ci_log_summary.py` | smart_ut | 日志分析 |
| 创建 `test_suites.yaml` | 新设计 | 融合场景+预估时间 |

### 4.2 第二阶段：CI 工作流改造

| 任务 | 说明 |
|---|---|
| 创建 `_st_e2e_test.yaml` | 可复用工作流 |
| 改造 `vllm_ascend_test.yaml` | 引入 ST 测试 |
| 添加 runner 标签映射 | runner_labels.yaml |

### 4.3 第三阶段：定时优化

| 任务 | 说明 |
|---|---|
| 收集 timing JSON | CI artifact 上传 |
| 自动更新 estimated_time | 定时脚本 |
| 分区平衡优化 | 基于实际时间调整 |

## 五、预期收益

| 指标 | 当前 | 融合后 | 改善 |
|---|---|---|---|
| CI 执行时间 | 手动分配 | 自动平衡 | 减少 30% |
| 测试覆盖率 | 依赖开发者记忆 | 白名单强制 | 100% 覆盖 |
| 预估时间准确度 | 无 | 自动优化 | 持续改进 |
| Runner 利用率 | 静态分配 | 动态分区 | 提升 40% |
| 日志分析效率 | 手动查看 | 自动摘要 | 减少 80% |

## 六、风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| config.yaml 维护成本 | 新增测试需更新配置 | sanity_check 强制检查 |
| 预估时间不准确 | 分区不平衡 | 定时自动优化 |
| Runner 资源不足 | 队列等待 | 动态分区大小调整 |
