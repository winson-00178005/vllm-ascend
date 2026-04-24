# ST 测试框架实施日志

> 开始日期：2026-04-23

## 实施记录

### P0: 创建 st_config/ 目录结构、配置文件

- **开始时间**: 2026-04-23
- **完成时间**: 2026-04-23
- **状态**: 已完成
- **内容**:
  - 创建 `tests/e2e/st_config/` 目录
  - 创建 `tests/e2e/st_config/framework/` 目录
  - 创建 `scenes.yaml` 场景配置（5 个场景：single_card, multi_card_tp2, multi_card_tp4, multi_card_dp2, ascend_310p）
  - 创建 `models.yaml` 模型配置（4 个模型：Qwen3-8B-Base, DeepSeek-V2-Lite, Qwen2.5-VL-7B-Instruct, Qwen3-30B-A3B）
  - 创建 `__init__.py` 文件

### P1: 实现 SceneManager、ModelManager、AscendModelInfo

- **开始时间**: 2026-04-23
- **完成时间**: 2026-04-23
- **状态**: 已完成
- **内容**:
  - 实现 `framework/scene.py` - 场景管理器
    - 单例模式，支持 reset_instance() 用于测试
    - 三级优先级：命令行 > 环境变量 > 自动探测
    - 自动探测：通过 npu-smi 检测卡数和硬件型号
    - 提供 get_config(), get_scene_config(), is_valid_scene(), apply_env_vars() 方法
  - 实现 `framework/model.py` - 模型管理器 + AscendModelInfo 数据类
    - AscendModelInfo 数据类（参考 vLLM ModelInfo）
    - 单例模式，支持 reset_instance() 用于测试
    - 提供 get_models_for_scene(), is_supported(), get_info(), check_gpu_requirement() 方法

### P2: 实现 ci_envs.py、test_utils.py

- **开始时间**: 2026-04-23
- **完成时间**: 2026-04-23
- **状态**: 已完成
- **内容**:
  - 实现 `ci_envs.py` - CI 环境变量集中管理
    - ST_CI_NO_SKIP: 是否测试所有模型
    - ST_CI_DTYPE: 强制使用特定 dtype
    - ST_CI_ENFORCE_EAGER: 是否强制使用 eager 模式
    - ST_CI_TARGET_SUITE: 目标测试套件
    - 支持延迟求值和 is_set() 检查
  - 实现 `test_utils.py` - 测试工具函数
    - fork_new_process_for_each_test: 进程隔离装饰器
    - multi_gpu_test: 多 GPU 测试装饰器
    - large_gpu_test: 大 GPU 内存测试装饰器
    - get_vllm_extra_kwargs: 根据 CI 环境变量调整 vLLM 参数

### P3: 实现 fixture 工厂（session/module/参数化）

- **开始时间**: 2026-04-23
- **完成时间**: 2026-04-23
- **状态**: 已完成
- **内容**:
  - 实现 `framework/fixtures.py` - fixture 工厂
    - create_vllm_runner_session_fixture(): session 级 fixture，整个会话共享 LLM 实例
    - create_vllm_runner_module_fixture(): module 级 fixture，每个模块创建 LLM 实例
    - create_hf_runner_session_fixture(): session 级 HF runner fixture
  - 所有 fixture 自动从场景配置和模型配置获取参数
  - 支持延迟导入，避免在没有 vLLM 的环境中报错

### P4: 改造顶层 conftest.py（hooks + pytest_generate_tests）

- **开始时间**: 2026-04-23
- **完成时间**: 2026-04-23
- **状态**: 已完成
- **内容**:
  - 改造 `tests/e2e/conftest.py`，在文件末尾添加 ST 框架扩展
  - 添加 `pytest_addoption` 注册 --scene, --model, --st-ci-no-skip 参数
  - 添加 `pytest_configure` 初始化 SceneManager 和 ModelManager，注册 marker
  - 添加 `pytest_generate_tests` 实现动态参数化（场景 × 模型矩阵）
  - 添加 `pytest_collection_modifyitems` 在 collection 阶段过滤不匹配用例
  - 添加 `pytest_runtest_setup` 在执行阶段检查场景/模型匹配
  - 注入 fixtures: vllm_runner_session, vllm_runner_module, hf_runner_session
  - **原有内容完全保留**，确保现有用例零修改即可运行

### P5: 迁移 2-3 个典型 e2e 用例验证框架

- **开始时间**: 2026-04-23
- **完成时间**: 2026-04-23
- **状态**: 已完成
- **内容**:
  - 创建 `tests/e2e/st_validation/` 目录存放验证用例
  - 创建 `test_st_framework.py` 验证用例，展示以下功能：
    - 场景过滤（`@pytest.mark.scenario`）
    - 模型 marker（`@pytest.mark.model`）
    - session 级 fixture（`vllm_runner_session`）
    - module 级 fixture 参数化（`vllm_runner_module`）
    - 多 GPU 装饰器（`@multi_gpu_test`）
    - 大 GPU 内存装饰器（`@large_gpu_test`）
  - 验证用例设计为占位测试，不需要实际 GPU 资源即可验证框架逻辑

### P6: CI 集成（第一阶段）

- **开始时间**: 2026-04-23
- **完成时间**: 2026-04-23
- **状态**: 已完成
- **内容**:
  - 在 `.github/workflows/vllm_ascend_test.yaml` 中新增 `st-e2e` job
  - 与现有 `e2e` job 并行运行
  - 使用 `linux-aarch64-a2-1` runner（单卡环境）
  - 传递 `--scene single_card` 参数验证场景功能
  - 运行 `tests/e2e/st_validation/` 验证用例

### P7: CI matrix 策略优化（第二阶段）

- **开始时间**: 2026-04-23
- **完成时间**: 2026-04-23
- **状态**: 已完成
- **内容**:
  - 在 CI workflow 中新增 `st-e2e-multi-scene` job（默认禁用，需手动启用）
  - 使用 matrix 策略按场景拆分：
    - `single_card` → `linux-aarch64-a2-1`
    - `multi_card_tp2` → `linux-aarch64-a2-2`
  - 预留 `multi_card_tp4` 和 `ascend_310p` 场景配置（注释状态）
  - 每个场景自动传递 `--scene ${{ matrix.scene }}` 参数

## 实施总结

- **总耗时**: 约 0.5 天（实际执行时间）
- **完成阶段**: P0-P7 全部完成
- **产出文件**:
  - `tests/e2e/st_config/` - ST 框架核心代码
    - `scenes.yaml` - 场景配置
    - `models.yaml` - 模型配置
    - `ci_envs.py` - CI 环境变量管理
    - `test_utils.py` - 测试工具函数
    - `framework/scene.py` - 场景管理器
    - `framework/model.py` - 模型管理器
    - `framework/fixtures.py` - Fixture 工厂
  - `tests/e2e/conftest.py` - 改造完成（原有内容保留）
  - `tests/e2e/st_validation/` - 验证用例
  - `.github/workflows/vllm_ascend_test.yaml` - CI 集成完成

## 后续工作记录

### 框架自测（2026-04-23）

- **状态**: 已完成
- **内容**:
  - 创建 `tests/st_config_tests/test_scene_manager.py` 独立测试文件
  - 22 个单元测试全部通过（0.25s）：
    - SceneManager: 7 个测试（配置加载、优先级、环境变量、异常处理）
    - ModelManager: 5 个测试（配置加载、场景过滤、模型信息）
    - ConfigValidation: 4 个测试（YAML 格式校验、必需字段检查）
    - SceneInheritance: 6 个测试（Phase 2 场景继承、环境变量合并、特性列表、量化配置、图模式、组合场景）
- **结果**: ✅ 22 passed in 0.25s

### 用例迁移（2026-04-23）

- **状态**: 已完成
- **已完成**:
  - `tests/e2e/singlecard/test_offline_inference.py` - 添加 `pytestmark = pytest.mark.scenario("single_card")`
  - `tests/e2e/singlecard/test_offline_inference_310p.py` - 添加 `pytestmark = pytest.mark.scenario("ascend_310p")`
  - `tests/e2e/multicard/test_offline_inference_distributed.py` - 添加 `pytestmark = pytest.mark.scenario("multi_card_tp2")`
  - `tests/e2e/multicard/test_data_parallel.py` - 添加 `pytestmark = pytest.mark.scenario("multi_card_dp2")`
  - `tests/e2e/multicard/test_pipeline_parallel.py` - 添加 `pytestmark = pytest.mark.scenario("multi_card_tp2")`
  - `tests/e2e/multicard/test_torchair_graph_mode.py` - 添加 `pytestmark = pytest.mark.scenario("multi_card_tp2")`
  - `tests/e2e/multicard/test_expert_parallel.py` - 添加 `pytestmark = pytest.mark.scenario("multi_card_tp2")`
  - `tests/e2e/multicard/test_ilama_lora_tp2.py` - 添加 `pytestmark = pytest.mark.scenario("multi_card_tp2")`
  - `tests/e2e/multicard/test_offline_inference_310p.py` - 添加 `pytestmark = pytest.mark.scenario("ascend_310p")`
  - `tests/e2e/multicard/test_qwen3_moe.py` - 添加 `pytestmark = pytest.mark.scenario("multi_card_tp2")`
  - `tests/e2e/multicard/test_prefix_caching.py` - 添加 `pytestmark = pytest.mark.scenario("multi_card_tp2")`
  - `tests/e2e/multicard/test_pyhccl_distributed.py` - 添加 `pytestmark = pytest.mark.scenario("multi_card_tp2")`
  - `tests/e2e/multicard/test_external_launcher.py` - 添加 `pytestmark = pytest.mark.scenario("multi_card_tp2")`
  - `tests/e2e/multicard/test_fused_moe_allgather_ep.py` - 添加 `pytestmark = pytest.mark.scenario("multi_card_tp2")`
  - `tests/e2e/multicard/test_dynamic_npugraph_batchsize.py` - 添加 `pytestmark = pytest.mark.scenario("multi_card_tp2")`
- **迁移统计**:
  - singlecard: 2 个文件
  - multicard: 13 个文件
  - 总计: 15 个文件

### smart_ut 分支 CI Runner 管理方案融合分析（2026-04-24）

- **状态**: 已完成
- **分析内容**:
  - 分析 `others/vllm-ascend` 仓库 `smart_ut` 分支的 CI runner 管理方案
  - 识别 5 个核心设计模式：
    1. 测试套件配置化（config.yaml）
    2. 智能分区算法（贪心算法自动平衡）
    3. Runner 标签映射（runner_label.json）
    4. 可复用工作流（workflow_call）
    5. 定时数据收集与预估时间优化
  - 输出融合分析文档：`docs/source/developer_guide/test_framework/smart_ut_ci_runner_analysis.md`
- **融合建议**:
  - 第一阶段：引入 `run_suite.py`、`ci_utils.py`、`ci_log_summary.py`
  - 第二阶段：创建 `_st_e2e_test.yaml` 可复用工作流
  - 第三阶段：实施定时优化（timing JSON 收集+自动更新 estimated_time）

### smart_ut CI Runner 管理方案实施（2026-04-24）

- **状态**: 已完成
- **实施内容**:
  - **核心脚本引入**:
    - 复制 `ci_utils.py` - 通用工具函数（TestFile, TestRecord, run_tests）
    - 复制 `run_suite.py` - 测试执行引擎（支持分区、定时数据收集）
    - 复制 `ci_log_summary.py` - 日志分析工具
    - 修改 `run_suite.py` 支持 ST 框架配置路径（`test_suites.yaml`）
  - **配置文件创建**:
    - 创建 `test_suites.yaml` - 融合场景+模型+预估时间的测试套件配置
    - 创建 `runner_labels.yaml` - 场景/测试套件到 Runner 的映射
  - **CI 工作流创建**:
    - 创建 `_st_e2e_test.yaml` - 可复用工作流（workflow_call）
    - 改造 `vllm_ascend_test.yaml` - 引入 ST e2e full/light 测试 job
- **新增文件**:
  - `.github/workflows/scripts/ci_utils.py`
  - `.github/workflows/scripts/run_suite.py`
  - `.github/workflows/scripts/ci_log_summary.py`
  - `tests/e2e/st_config/test_suites.yaml`
  - `tests/e2e/st_config/runner_labels.yaml`
  - `.github/workflows/_st_e2e_test.yaml`

### 下一步计划

1. ~~框架自测~~ ✅ 已完成
2. ~~配置校验~~ ✅ 已完成
3. ~~Phase 2 功能~~ ✅ 已完成
4. ~~用例迁移~~ ✅ 已完成
5. ~~smart_ut CI Runner 分析~~ ✅ 已完成
6. 引入 smart_ut 核心脚本（run_suite.py 等）
7. 创建 test_suites.yaml 融合配置
8. 实施 CI 工作流改造
9. 实际运行验证（需要 NPU 硬件环境）
