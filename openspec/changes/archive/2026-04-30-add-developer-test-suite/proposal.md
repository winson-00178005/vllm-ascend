## 为什么需要

当前vllm-ascend代码仓缺乏系统级的开发者测试（ST）框架。现有测试结构包含单元测试（tests/ut/）和端到端测试（tests/e2e/），但缺少关注模块间配合关系的集成测试层。根据开发者测试规范，ST测试能够：
- 验证模块间接口和数据流正确性
- 提前拦截模块集成问题（成本远低于生产环境发现）
- 为CI门禁提供快速反馈（≤5分钟）
- 保护重构时的模块交互稳定性

现在建立ST框架可以填补测试金字塔的中间层，完善测试分层策略，提升代码质量保障能力。

## 变更内容

**新增内容：**
- 创建系统集成测试（ST）框架，位于tests/st/目录
- 建立ST测试基类、fixtures、配置管理基础设施
- 为核心模块生成首批ST测试用例，验证模块间交互
- 集成pytest框架，支持ST用例执行和覆盖率统计
- 添加ST测试运行脚本和CI集成配置

**测试分层：**
- UT（单元测试）：tests/ut/ - 验证单个函数/类行为
- ST（系统集成测试）：tests/st/ - 验证模块间配合关系
- E2E（端到端测试）：tests/e2e/ - 验证完整业务流程

## 能力

### 新增能力

- `st-test-framework`: ST测试框架基础设施，包括测试基类、fixtures、配置管理、Mock工具、测试数据生成器等
- `st-worker-integration`: Worker模块集成测试，验证worker_v1与model_runner_v1、npu_input_batch等模块的协作关系
- `st-attention-integration`: Attention模块集成测试，验证attention_v1与attention_mask、mla_v1等组件的交互正确性
- `st-scheduler-integration`: Scheduler模块集成测试，验证scheduler与schedule_config、worker模块的协作流程
- `st-quantization-integration`: Quantization模块集成测试，验证量化配置、量化器与模型加载的集成正确性
- `st-distributed-integration`: Distributed模块集成测试，验证分布式通信、tensor parallel等模块间的协作
- `st-sample-integration`: Sample模块集成测试，验证sampler与rejection_sampler在工作流中的协作关系

### 修改的能力

无现有capability需要修改。本次变更仅新增ST测试框架和测试用例，不修改现有功能代码。

## 影响

**新增目录结构：**
- tests/st/ - ST测试根目录
- tests/st/base.py - ST测试基类
- tests/st/conftest.py - pytest配置和fixtures
- tests/st/utils/ - 测试工具和Mock辅助类
- tests/st/fixtures/ - 测试数据和配置fixtures
- tests/st/configs/ - 配置文件目录（default.yaml、ci_fast.yaml、npu_deep.yaml）
- tests/st/scripts/ - 辅助脚本目录（coverage_trend.py、config_validate.py）
- tests/st/coverage_history/ - 覆盖率历史数据目录
- tests/st/worker/ - Worker模块ST测试
- tests/st/attention/ - Attention模块ST测试
- tests/st/core/ - Scheduler模块ST测试
- tests/st/quantization/ - Quantization模块ST测试
- tests/st/distributed/ - Distributed模块ST测试
- tests/st/sample/ - Sample模块ST测试
- tests/st/integration/ - 周边组件集成测试（可选）

**依赖影响：**
- pytest相关依赖已存在于requirements-dev.txt（pytest>=6.0, pytest-mock, pytest-cov, pytest-asyncio）
- ST测试依赖vllm和torch_npu环境（vllm-ascend是vllm的插件，源代码强依赖vllm和torch_npu）
- ST测试必须在CANN container执行（CPU Mock模式也需要torch_npu安装）
- 需要更新测试运行脚本，支持ST测试单独执行和联合执行
- CI配置文件需要调整以支持ST测试门禁

**CI/CD集成影响：**
- 修改.github/workflows/vllm_ascend_test.yaml，添加st_tracker和st-cpu-mock job
- 修改codecov.yml，添加st_tests flags和分级门禁
- ST测试集成到现有workflow（lint → ut → st-cpu-mock → e2e）
- ST覆盖率与UT覆盖率合并上传Codecov
- ST测试执行时间门禁≤5分钟

**无API变更：** 本次变更仅新增测试代码，不修改任何公共API或业务代码行为。