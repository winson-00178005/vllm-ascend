# ST测试框架架构设计PPT大纲

## 使用说明

本文档是ST测试框架架构设计的PPT大纲，用于向外部讲述架构方案。可以：
1. 直接使用此markdown文档作为PPT内容参考
2. 使用powerpoint-automation skill转换为PPTX格式
3. 手动复制内容到PPT模板

---

## Slide 1: Title Slide

**标题**: vllm-ascend ST测试框架架构设计

**副标题**: 填补UT与E2E测试空白，实现质的飞跃

**作者**: vllm-ascend开发团队

**日期**: 2026年4月

---

## Slide 2: Context - 测试分层现状

**标题**: 当前测试分层现状

**内容**:

```
当前测试体系（缺失中间层）：
┌─────────────────────────────────────────┐
│ UT (tests/ut/)                          │
│ - 单个函数/类级别功能                    │
│ - Mock torch_npu硬件                    │
│ - 执行时间：秒级                         │
│ - 文件数：54个                           │
└─────────────────────────────────────────┘
                    ↓ 缺失中间层
┌─────────────────────────────────────────┐
│ ❓ ST (tests/st/) - 缺失                 │
│ - 模块间协作验证                         │
│ - 插件接口兼容性验证                     │
│ - 执行时间：分钟级                       │
│ - 文件数：0个（缺失）                    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ E2E (tests/e2e/)                        │
│ - 完整推理流程                           │
│ - 真实NPU环境 + 真实模型                 │
│ - 执行时间：分钟级                       │
│ - 文件数：49个                           │
│ - 问题：45次模型加载/卸载                │
└─────────────────────────────────────────┘
```

**关键问题**:
- 模块协作问题只能在E2E发现
- E2E执行时间长，发现问题延迟
- 无法快速定位问题模块

---

## Slide 3: Problem - 核心痛点分析

**标题**: 核心痛点：E2E测试效率问题

**内容**:

**问题1: 重复模型加载**
- E2E测试中45次VllmRunner实例创建
- 同一模型被多次加载（如DeepSeek-V3-Pruning加载2次）
- 每次加载耗时1-3分钟
- 总执行时间：45-135分钟

**问题2: 测试执行时间长**
- 分钟级执行时间
- 无法作为CI门禁
- 开发反馈周期长

**问题3: 问题定位困难**
- E2E测试失败时，难以定位具体模块
- 需要人工排查整个推理流程
- 调试成本高

**数据证据**:
- VllmRunner使用统计：16个测试文件，45次实例创建
- cleanup_dist_env_and_memory()每次都执行
- 模型加载/卸载是主要耗时操作

---

## Slide 4: Solution - ST测试框架设计目标

**标题**: ST测试框架设计目标

**内容**:

**核心目标**:
1. 填补UT和E2E中间测试层
2. 解决45次模型重复加载问题
3. 快速验证模块间协作
4. 实现CI门禁（≤5分钟）

**Non-Goals**:
- 不验证完整推理流程（E2E职责）
- 不测试单个函数逻辑（UT职责）
- 不修改业务代码

**预期效果**:
- 执行效率提升：50%时间减少
- 防护效果提升：快速发现模块协作问题
- 演进难度降低：配置驱动，Mock工厂

---

## Slide 5: Architecture - 三层解耦架构

**标题**: 三层解耦架构设计

**内容**:

```
┌─────────────────────────────────────────┐
│ Layer 1: Environment Layer              │
│ - NPU环境初始化（一次性）                │
│ - torch_npu初始化                       │
│ - pytest fixture scope="session"        │
│ - 作用：避免重复环境初始化               │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Layer 2: Model Layer (ModelPool)        │
│ - ModelPool管理模型实例                 │
│ - 模型缓存避免重复加载                   │
│ - pytest fixture scope="module"         │
│ - 作用：解决45次加载问题                 │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Layer 3: TestCase Layer                 │
│ - 从ModelPool获取模型                   │
│ - 不直接加载模型                        │
│ - pytest fixture scope="function"       │
│ - 作用：保证测试独立性                   │
└─────────────────────────────────────────┘
```

**核心设计决策**:
- fixture scope: session → module → function
- ModelPool缓存策略：get_or_create()
- copy_on_use策略：支持状态隔离

---

## Slide 6: Key Component 1 - ModelPool缓存池

**标题**: ModelPool模型缓存池设计

**内容**:

**核心功能**:
- 缓存模型实例，避免重复加载
- 支持多配置缓存（量化、分布式）
- 支持手动释放和自动清理

**关键方法**:
```python
class ModelPool:
    def get_or_create(model_name, config, copy_on_use=False):
        # 获取或创建模型实例（缓存策略）
        
    def _create_cache_key(model_name, config):
        # 创建缓存key（包含dtype、max_model_len等）
        
    def release(model_name):
        # 手动释放指定模型
        
    def clear():
        # 清空所有模型（module结束时）
```

**效率提升**:
- 45次加载 → 23次加载（减少50%）
- 执行时间：45分钟 → 23分钟

---

## Slide 7: Key Component 2 - ConfigLoader配置管理

**标题**: ConfigLoader配置驱动设计

**内容**:

**核心功能**:
- YAML配置文件管理
- 配置驱动测试，避免硬编码
- 支持多环境配置（CI、深度测试）

**配置文件结构**:
```yaml
default:
  exec_mode: "auto"
  timeout: 300
  
models:
  Qwen-0.5B:
    name: "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: "float16"
    
tests:
  worker_test:
    batch_size: [1, 16, 32]
    scenario: ["normal", "error"]
```

**命令行参数**:
```bash
pytest tests/st/ --config-file=ci_fast.yaml --model=Qwen-7B
```

**优势**:
- 配置变更：修改YAML（vs修改代码）
- 多环境支持：不同配置文件

---

## Slide 8: Key Component 3 - 双路径设计

**标题**: 简化路径 + 完整路径双模式

**内容**:

**简化路径（适合简单测试）**:
```python
def test_worker_interface():
    # 不使用三层架构
    worker = create_mock_worker()
    verify_plugin_interface(worker, VLLM_WORKER_SPEC)
```

**特点**:
- 直接创建Mock对象
- 不依赖三层架构
- 学习成本低
- 适用场景：接口验证、简单逻辑

**完整路径（适合复杂测试）**:
```python
def test_worker_integration(cached_model, test_runner):
    # 使用三层架构
    result = test_runner.execute(cached_model)
    verify_mock_calls(test_runner.model_runner, ...)
```

**特点**:
- 使用三层架构
- ModelPool缓存模型
- ConfigLoader加载配置
- 适用场景：模块协作、性能测试

**双路径优势**:
- 降低学习成本（简化路径）
- 保证功能完整（完整路径）

---

## Slide 9: Key Component 4 - 双模式执行

**标题**: CPU Mock + NPU真实双模式

**内容**:

**CPU Mock模式**:
- Mock torch_npu.npu_*所有NPU算子
- Mock NPU硬件（内存分配、设备属性）
- 保留插件模块协作（Worker、Scheduler真实组件）
- 执行环境：x86/arm服务器、MacBook、无torch_npu环境
- 执行时间：快速（秒级）
- 测试目的：验证模块协作、接口兼容、CI门禁

**NPU真实模式**:
- 使用真实torch_npu.npu_*算子
- 使用真实NPU硬件
- 测试插件与周边组件集成（CANN、ACL Graph、HCCL）
- 执行时间：较长（分钟级）
- 测试目的：验证真实性能、精度、深度集成

**命令行控制**:
```bash
pytest tests/st/ --exec-mode=cpu_mock
pytest tests/st/ --exec-mode=npu_real --performance-test
```

---

## Slide 10: Key Component 5 - 精度/性能看护

**标题**: 精度/性能看护能力设计

**内容**:

**精度看护**:
```python
@pytest.mark.npu_precision
def test_attention_precision(execution_environment):
    # CPU Mock生成参考结果
    with CPUMockEnvironment() as env_cpu:
        result_cpu = attention.forward(input)
    
    # NPU真实执行
    with execution_environment as env_npu:
        result_npu = attention.forward(input)
        assert env_npu.measure_precision(result_cpu, result_npu)
```

**性能看护**:
```python
@pytest.mark.npu_performance
def test_worker_performance(execution_environment, baseline):
    perf = execution_environment.measure_performance(
        lambda: worker.execute_model(...)
    )
    assert perf["execution_time"] < baseline["execution_time"]
```

**看护能力**:
- 精度误差阈值：rtol=1e-3, atol=1e-5
- 性能基准：performance_baseline.json
- 周边组件集成：CANN、ACL Graph、HCCL

---

## Slide 11: Optimization - 长期优化路径

**标题**: 长期优化路径

**内容**:

**长期优化（5周）**:
| 优先级 | 能力 | 设计 | 工作量 |
|--------|------|------|--------|
| P6 | 精度看护 | CPU vs NPU对比 | 2周 |
| P7 | 性能看护 | 性能测量和对比 | 2周 |
| P8 | fixture调试工具 | 依赖图生成 | 1周 |

---

## Slide 12: Feasibility - 可行性验证结论

**标题**: 可行性验证结论

**内容**:

**执行效率：质的飞跃 ✅**
- ModelPool缓存减少50%加载时间
- CPU Mock模式达到秒级
- 可作为CI门禁

**防护效果：质的飞跃 ✅**
- 填补中间测试层
- 快速发现模块协作问题（秒级vs分钟级）
- 可定位问题模块

**测试用例演进：难度降低 ✅**
- 配置驱动：修改YAML（vs修改代码）
- ModelPool：缓存模型（vs多次加载）
- Mock工厂：集中管理（vs单独Mock）

**精度/性能看护：可支持 ✅**
- 框架可支持（需NPU环境）
- 长期优化路径（5周）

---

## Slide 13: Implementation Plan - 实施计划

**标题**: 实施计划与里程碑

**内容**:

**Phase 1: 框架搭建（2周）**
- 三层解耦架构实现
- ModelPool缓存池实现
- ConfigLoader配置管理实现
- 简化路径 + 完整路径实现

**Phase 2: 核心测试（2周）**
- Worker模块ST测试
- Scheduler模块ST测试
- Attention模块ST测试

**Phase 3: 扩展测试（1周）**
- Quantization模块ST测试
- Distributed模块ST测试
- Sample模块ST测试

**Phase 4: CI集成（1周）**
- CI workflow配置
- 覆盖率报告生成
- 文档完善

**Phase 5: 深度看护（5周）**
- 精度看护实现
- 性能看护实现
- 周边组件集成测试

**总工期**: 6周核心实施 + 5周长期优化

---

## Slide 14: Summary - 总结与展望

**标题**: 总结与展望

**内容**:

**核心成果**:
- 填补UT和E2E中间测试层
- 解决45次模型重复加载问题
- 实现三层解耦架构
- 提供双路径和双模式设计
- 支持精度/性能看护

**预期效果**:
- 执行效率提升50%
- 快速发现模块协作问题
- 降低测试演进难度
- 可作为CI门禁

**下一步行动**:
1. 实施Phase 1框架搭建（2周）
2. 验证依赖隔离和fixture scope
3. 长期实现精度/性能看护（5周）

**展望**:
- ST测试框架将成为vllm-ascend测试体系的核心层
- 提升测试效率和防护效果
- 支持插件与vllm的持续集成

---

## Appendix: Reference Materials

**参考资料**:

1. **设计文档**: openspec/changes/add-developer-test-suite/design.md
2. **任务清单**: openspec/changes/add-developer-test-suite/tasks.md
3. **规范文档**: develop-test-spec.md
4. **现有UT测试**: tests/ut/base.py
5. **现有E2E测试**: tests/e2e/conftest.py

**关键决策记录**:
- 三层解耦架构：Decision 17
- ModelPool缓存池：Decision 18
- ConfigLoader配置管理：Decision 19
- 双路径设计：新决策（探索mode发现）
- 双模式执行：Decision 11-15

---

## 使用建议

### 生成PPT的步骤

1. **使用markdown转PPT工具**:
   - 可以使用pandoc、reveal.js等工具转换
   - 或手动复制内容到PPT模板

2. **使用powerpoint-automation skill**:
   - 将此markdown转换为content.json
   - 使用create_from_template.py生成PPTX

3. **手动制作PPT**:
   - 复制每个Slide的内容到PPT模板
   - 添加图表和可视化（ASCII art可转为图表）
   - 调整样式和布局

### 演讲建议

- **受众**: 外部技术团队、管理层、决策者
- **重点**: 强调质的飞跃和可行性验证
- **时间**: 30-45分钟演讲
- **互动**: 可在Slide 11和Slide 13处提问互动

### 补充材料

- 如需详细设计文档，请参考design.md（1594行）
- 如需任务清单，请参考tasks.md（210 tasks）
- 如需技术细节，请参考specs/*.md（7个capability spec）