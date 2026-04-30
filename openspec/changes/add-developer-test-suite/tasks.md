## 1. ST测试框架基础设施搭建

- [x] 1.1 创建tests/st/目录结构（包括base.py、conftest.py、utils/、fixtures/、runner_factory.py、performance/、precision/、integration/）
- [x] 1.2 实现双基类设计（TestSTBase继承unittest.TestCase + PytestSTBase使用fixture autouse）
- [x] 1.3 实现setUp/tearDown资源管理（setUp初始化Mock环境、tearDown清理Mock状态）
- [x] 1.4 实现层级化fixtures（全局conftest.py使用session scope、模块级conftest.py使用参数化fixture）
- [x] 1.5 实现pytest_addoption命令行参数（参考tests/e2e/models/conftest.py模式）
- [x] 1.6 实现命令行参数fixture（exec_mode、enable_mock、performance_test、precision_test、npu_device）
- [x] 1.7 实现环境检测机制env_detector.py（自动检测torch_npu、NPU可用性、is_310p）
- [x] 1.8 实现选择性Mock工厂env_factory.py（CPUMockEnvironment、NPURealEnvironment类）
- [x] 1.9 实现CPU Mock环境类（Mock torch_npu.npu_*、torch.npu.*、is_310p）
- [x] 1.10 实现NPU真实环境类（验证NPU可用、性能测量measure_performance、精度测量measure_precision）
- [x] 1.11 实现Mock工具类mock_utils.py（含Mock工厂函数：create_mock_worker、create_mock_distribution_env等）
- [x] 1.12 实现Mock验证函数verify_mock_calls（验证调用次数、参数、顺序）
- [x] 1.13 实现spec限制Mock对象（提供create_spec_mock函数，避免过度Mock）
- [x] 1.14 实现数据生成器data_generator.py（含预生成tensor、配置模板、全局常量定义）
- [x] 1.15 实现配置工厂config_factory.py（创建VllmConfig、ModelConfig、ParallelConfig等）
- [x] 1.16 实现配置工厂隔离vllm依赖（try-except ImportError、Fallback配置类、环境检测HAS_VLLM）
- [x] 1.17 定义vllm接口spec常量（VLLM_WORKER_SPEC、VLLM_ATTENTION_BACKEND_SPEC、VLLM_SCHEDULER_SPEC等）
- [x] 1.18 实现插件接口兼容性验证函数verify_plugin_interface_compatibility（检查插件组件实现vllm接口）
- [x] 1.19 实现STRunner上下文管理器（__enter__/__exit__自动管理模块资源）
- [x] 1.20 配置pytest运行脚本（支持单模块执行、全量执行、覆盖率统计、双模式执行）
- [x] 1.21 配置pytest marker定义（cpu_mock、npu_real、npu_performance、npu_precision）
- [x] 1.22 添加pytest依赖到requirements-dev.txt（pytest-mock、pytest-cov、pytest-xdist、pytest-rerunfailures）
- [x] 1.23 编写ST测试编写规范文档（强制Docstring模板、参数化指南、异常测试模式）
- [x] 1.24 编写插件测试职责说明文档（明确ST测试聚焦插件内部协作、不测试vllm核心逻辑）
- [x] 1.25 编写pytest命令行参数使用文档（详细说明--exec-mode、--enable-mock、--performance-test等参数）
- [x] 1.26 编写双模式执行策略文档（CPU Mock vs NPU真实模式使用场景、CI配置指南）

## 2. Worker模块集成测试（插件内部协作 + 双模式）

- [x] 2.1 创建tests/st/worker/目录和conftest.py（模块级fixtures：worker_runner fixture、worker_config参数化fixture、execution_environment fixture）
- [x] 2.2 使用Mock工厂函数创建Worker和ModelRunner Mock对象（create_mock_worker、create_mock_model_runner含spec限制）
- [x] 2.3 实现NPUWorker与AscendModelRunner集成测试（插件内部协作、STRunner管理资源）
- [x] 2.4 验证NPUWorker实现了vllm.Worker接口（使用verify_plugin_interface_compatibility检查VLLM_WORKER_SPEC）
- [x] 2.5 测试NPUWorker替换逻辑正确性（初始化ModelRunner、调度协作、接口兼容）
- [x] 2.6 参数化多维度测试（batch_size=[1,16,32]、scenario=["normal","error"]、dtype=["float16","bfloat16"])
- [x] 2.7 参数化双模式测试（exec_mode=["cpu_mock", "npu_real"]、验证不同模式执行结果）
- [x] 2.8 使用execution_environment fixture动态切换CPU Mock和NPU真实环境
- [x] 2.9 强制添加Docstring（包含"验证"、"场景"、"预期结果"、"执行模式"四个部分）
- [x] 2.10 实现异常场景测试（with assertRaises验证错误传递、assert "error_msg" in str(cm.exception))
- [x] 2.11 使用Mock验证策略（verify_mock_calls验证调用次数和参数）
- [x] 2.12 实现NPUWorker与NpuInputBatch集成测试（数据流、格式转换、插件内部协作）
- [x] 2.13 实现NPUWorker多批次并发处理集成测试（并发调度、资源冲突处理）
- [x] 2.14 实现NPUWorker与MTProposer集成测试（初始化、推测token生成）
- [x] 2.15 吸纳现有UT优秀模式（参考tests/ut/worker/test_worker_v1.py的设计风格）
- [x] 2.16 不测试vllm.Worker核心逻辑（明确ST测试聚焦插件内部协作，文档说明职责边界）
- [x] 2.17 标记双模式测试用例（@pytest.mark.cpu_mock、@pytest.mark.npu_real）

## 3. Scheduler模块集成测试（插件内部协作 + 双模式）

- [x] 3.1 创建tests/st/core/目录和conftest.py（模块级fixtures：scheduler_runner fixture、execution_environment fixture）
- [ ] 3.2 使用Mock工厂函数创建Scheduler和Worker Mock对象（含spec限制）
- [ ] 3.3 实现AscendScheduler与AscendSchedulerConfig集成测试（插件内部协作、STRunner管理资源）
- [ ] 3.4 验证AscendScheduler实现了vllm.Scheduler接口（使用verify_plugin_interface_compatibility检查VLLM_SCHEDULER_SPEC）
- [ ] 3.5 测试AscendScheduler替换逻辑正确性（调度协作、接口兼容）
- [ ] 3.6 实现AscendScheduler与NPUWorker集成测试（参数化调度场景、Mock验证调用次数、插件内部协作）
- [ ] 3.7 参数化双模式测试（exec_mode=["cpu_mock", "npu_real"]）
- [ ] 3.8 实现AscendScheduler多批次调度集成测试（优先级调度、批次抢占、Docstring强制）
- [ ] 3.9 实现AscendScheduler连续解码调度集成测试（prefill/decode阶段调度、异常场景验证）
- [ ] 3.10 吸纳现有UT优秀模式（setUp初始化、tearDown清理）
- [ ] 3.11 不测试vllm.Scheduler核心逻辑（明确ST测试聚焦插件内部协作）
- [ ] 3.12 标记双模式测试用例（@pytest.mark.cpu_mock、@pytest.mark.npu_real）

## 4. Attention模块集成测试（插件内部协作 + 双模式 + 精度测试）

- [ ] 4.1 创建tests/st/attention/目录和conftest.py（模块级fixtures、execution_environment fixture）
- [ ] 4.2 使用Mock工厂函数创建Attention各组件Mock（create_mock_attention、create_mock_attention_mask含spec）
- [ ] 4.3 实现AscendAttention与AscendAttentionMask集成测试（插件内部协作、多层Mock、参数化mask类型）
- [ ] 4.4 验证AscendAttention实现了vllm.AttentionBackend接口（使用verify_plugin_interface_compatibility检查VLLM_ATTENTION_BACKEND_SPEC）
- [ ] 4.5 测试AscendAttention替换逻辑正确性（接口兼容、组件协作）
- [ ] 4.6 实现AscendAttention与AscendMLA集成测试（KV cache压缩、解压缩、Mock验证策略、插件内部协作）
- [ ] 4.7 实现AscendAttention不同序列长度集成测试（参数化seq_len、setUp/tearDown管理tensor资源）
- [ ] 4.8 实现AscendAttention与Torchair集成测试（图模式编译、异常场景验证）
- [ ] 4.9 参数化双模式测试（exec_mode=["cpu_mock", "npu_real"]）
- [ ] 4.10 实现NPU真实模式精度测试（对比CPU Mock和NPU结果、验证数值精度）
- [ ] 4.11 吸纳tests/ut/attention/test_attention_v1.py优秀模式（详细Docstring、多层@patch）
- [ ] 4.12 不测试vllm.Attention核心逻辑（明确ST测试聚焦插件内部协作）
- [ ] 4.13 标记双模式测试用例（@pytest.mark.cpu_mock、@pytest.mark.npu_real）
- [ ] 4.14 标记精度测试用例（@pytest.mark.npu_precision）

## 5. Quantization模块集成测试（插件内部协作 + 双模式 + 精度测试）

- [ ] 5.1 创建tests/st/quantization/目录和conftest.py（模块级fixtures、execution_environment fixture）
- [ ] 5.2 使用Mock工厂函数创建量化相关Mock（含spec限制、参考test_w8a8.py模式）
- [ ] 5.3 实现AscendQuantConfig与模型加载集成测试（插件内部协作、量化配置、setUp初始化量化器）
- [ ] 5.4 验证AscendQuantizer与vllm.Quantizer接口兼容（接口兼容性测试）
- [ ] 5.5 实现W8A8量化集成测试（参数化量化类型、异常验证NotImplementedError）
- [ ] 5.6 实现动态量化集成测试（参数化dtype、Mock验证量化调用次数）
- [ ] 5.7 实现Quantization与分布式集成测试（Tensor Parallel、STRunner管理分布式资源）
- [ ] 5.8 参数化双模式测试（exec_mode=["cpu_mock", "npu_real"]）
- [ ] 5.9 实现NPU真实模式精度测试（对比量化前后数值精度、验证量化误差）
- [ ] 5.10 吸纳tests/ut/quantization/test_w8a8.py优秀模式（spec限制Mock、详细异常验证）
- [ ] 5.11 吸纳tests/ut/quantization/test_quantizer.py优秀模式（setUp/tearDown管理全局常量）
- [ ] 5.12 标记双模式测试用例（@pytest.mark.cpu_mock、@pytest.mark.npu_real）
- [ ] 5.13 标记精度测试用例（@pytest.mark.npu_precision）

## 6. Distributed模块集成测试（插件内部协作 + 双模式 + NPU真实测试）

- [ ] 6.1 创建tests/st/distributed/目录和conftest.py（模块级fixtures、execution_environment fixture）
- [ ] 6.2 使用Mock工厂函数创建分布式环境Mock（create_mock_distribution_env参考test_fused_ops.py模式）
- [ ] 6.3 实现NPUCommunicator通信集成测试（插件内部协作、参数化world_size、Mock验证all_reduce调用）
- [ ] 6.4 验证NPUCommunicator与vllm分布式接口兼容（接口兼容性测试）
- [ ] 6.5 实现Tensor Parallel集成测试（权重分片、STRunner管理分布式资源、异常验证）
- [ ] 6.6 实现Expert Parallel集成测试（参数化num_experts、Mock工厂函数复用）
- [ ] 6.7 实现Distributed与KV Connector集成测试（多层Mock协作、Docstring强制）
- [ ] 6.8 参数化双模式测试（exec_mode=["cpu_mock", "npu_real"]）
- [ ] 6.9 实现NPU真实模式HCCL分布式通信测试（验证真实all_reduce、all_gather）
- [ ] 6.10 吸纳tests/ut/ops/test_fused_ops.py优秀模式（Mock工厂函数、多层@patch装饰器）
- [ ] 6.11 标记双模式测试用例（@pytest.mark.cpu_mock、@pytest.mark.npu_real）

## 7. Sample模块集成测试（插件内部协作 + 双模式）

- [ ] 7.1 创建tests/st/sample/目录和conftest.py（模块级fixtures、execution_environment fixture）
- [ ] 7.2 定义全局常量（参考test_rejection_sampler.py的PLACEHOLDER_TOKEN_ID模式）
- [ ] 7.3 实现AscendSampler与AscendRejectionSampler集成测试（插件内部协作、参数化验证场景）
- [ ] 7.4 验证AscendSampler与vllm.Sampler接口兼容（接口兼容性测试）
- [ ] 7.5 实现AscendRejectionSampler不同验证场景集成测试（全接受、全拒绝、部分接受、详细Docstring）
- [ ] 7.6 实现AscendSampler与Temperature集成测试（参数化temperature值、Mock验证采样调用）
- [ ] 7.7 实现AscendSampler与Batch集成测试（参数化batch_size、setUp/tearDown管理采样数据）
- [ ] 7.8 实现Sample模块与Worker集成测试（插件内部协作、异常场景验证、错误传递）
- [ ] 7.9 参数化双模式测试（exec_mode=["cpu_mock", "npu_real"]）
- [ ] 7.10 吸纳tests/ut/sample/test_rejection_sampler.py优秀模式（全局常量、详细Docstring、异常验证）
- [ ] 7.11 标记双模式测试用例（@pytest.mark.cpu_mock、@pytest.mark.npu_real）

## 8. CI集成与测试优化（双模式CI配置）

- [ ] 8.1 编写ST测试运行脚本（run_st_tests.sh：支持单模块和全量执行、双模式执行）
- [ ] 8.2 配置pytest-xdist并行执行（多进程并行，缩短执行时间≤5分钟）
- [ ] 8.3 配置覆盖率报告生成（pytest-cov，生成XML和HTML报告）
- [ ] 8.4 集成ST测试到CI workflow（GitHub Actions：添加ST测试job）
- [ ] 8.5 配置双模式CI job（Job1 CPU Mock快速门禁、Job2 NPU真实深度测试可选）
- [ ] 8.6 添加插件接口兼容性验证到CI（vllm版本升级时运行兼容性测试）
- [ ] 8.7 添加环境检测机制到CI（检测vllm可用性、torch_npu可用性、灵活切换配置策略）
- [ ] 8.8 优化ST测试执行时间（选择性Mock、预生成数据缓存、session scope fixture共享）
- [ ] 8.9 配置CI门禁阈值（执行时间≤5分钟、覆盖率目标设置、NPU测试可选）
- [ ] 8.10 编写ST测试框架文档（README.md：框架说明、STRunner使用指南、插件测试职责说明、双模式使用指南）
- [ ] 8.11 编写ST测试编写规范文档（Docstring模板、参数化指南、异常测试模式、双模式测试指南）
- [ ] 8.12 编写Mock工厂函数使用指南（create_mock_*命名规范、spec限制说明、CPUMockEnvironment使用）
- [ ] 8.13 编写插件架构测试策略文档（继承、替换、注入的测试方法）
- [ ] 8.14 编写pytest命令行参数使用文档（--exec-mode、--enable-mock、--performance-test等参数详细说明）

## 9. 测试用例质量保障

- [ ] 9.1 代码评审ST测试代码（遵循FIRST原则、Right-BICEP原则、强制Docstring）
- [ ] 9.2 执行变异测试验证ST测试有效性（使用mutmut或similar工具）
- [ ] 9.3 从现有UT测试中提取优秀用例模式（参考tests/ut/优秀范例、吸纳双基类、setUp/tearDown）
- [ ] 9.4 从E2E测试中提取优秀模式（参考tests/e2e/的VllmRunner、session scope fixture、插件集成测试、pytest_addoption）
- [ ] 9.5 补充边界场景测试用例（基于CORRECT原则：Conformance、Ordering、Range等）
- [ ] 9.6 补充异常场景测试用例（基于Error Handling：错误传播、资源清理、assertRaises）
- [ ] 9.7 验证Mock对象spec定义完整性（定期检查Mock与真实对象接口一致性）
- [ ] 9.8 验证STRunner资源清理正确性（防止资源泄漏）
- [ ] 9.9 验证配置工厂Fallback配置类完整性（确保Fallback包含ST测试必需字段）
- [ ] 9.10 验证插件接口spec与vllm版本同步（定期检查vllm接口变更，更新spec定义）
- [ ] 9.11 验证插件测试职责边界清晰（ST测试聚焦插件内部协作，不测试vllm核心逻辑）
- [ ] 9.12 验证环境检测机制正确性（HAS_VLLM检测、HAS_TORCH_NPU检测、Fallback切换）
- [ ] 9.13 验证双模式参数化测试正确性（CPU Mock和NPU真实模式结果一致性）
- [ ] 9.14 验证命令行参数默认值合理性（--exec-mode=auto自动检测，适配不同环境）

## 10. 性能测试集成（NPU真实模式）

- [ ] 10.1 创建tests/st/performance/目录和conftest.py（性能测试fixtures、performance_baseline fixture）
- [ ] 10.2 实现性能基准数据管理（performance_baseline.json格式定义、基准加载和保存）
- [ ] 10.3 实现Worker性能测试（执行时间测量、内存占用测量、吞吐量测量）
- [ ] 10.4 实现Attention性能测试（不同序列长度性能对比、不同batch_size性能对比）
- [ ] 10.5 实现Scheduler性能测试（多批次调度性能、优先级调度性能）
- [ ] 10.6 实现Quantization性能测试（量化前后性能对比、不同量化类型性能对比）
- [ ] 10.7 实现Distributed性能测试（分布式通信性能、Tensor Parallel性能）
- [ ] 10.8 实现性能数据收集和分析（JSON格式存储、性能趋势分析）
- [ ] 10.9 实现性能报告自动生成（可视化性能数据、性能对比报告）
- [ ] 10.10 标记性能测试用例（@pytest.mark.npu_performance）
- [ ] 10.11 配置性能基准更新机制（模型升级、硬件升级时更新基准）

## 11. 精度测试集成（NPU真实模式）

- [ ] 11.1 创建tests/st/precision/目录和conftest.py（精度测试fixtures）
- [ ] 11.2 实现精度误差阈值定义（不同算子类型定义不同rtol、atol）
- [ ] 11.3 实现Attention精度测试（CPU vs NPU结果对比、数值精度验证）
- [ ] 11.4 实现Quantization精度测试（量化前后精度对比、量化误差验证）
- [ ] 11.5 实现Sample精度测试（采样精度验证、不同采样策略精度对比）
- [ ] 11.6 实现精度数据收集和分析（精度误差数据存储、精度趋势分析）
- [ ] 11.7 实现精度报告自动生成（可视化精度误差、精度对比报告）
- [ ] 11.8 标记精度测试用例（@pytest.mark.npu_precision）
- [ ] 11.9 配置精度阈值告警机制（精度超出阈值时告警）

## 12. 周边组件集成测试（NPU真实模式）

- [ ] 12.1 创建tests/st/integration/目录和conftest.py（周边组件集成测试fixtures）
- [ ] 12.2 实现CANN版本兼容测试（验证与不同CANN版本兼容）
- [ ] 12.3 实现ACL Graph编译测试（验证ACL Graph编译正确性）
- [ ] 12.4 实现ACL Graph执行测试（验证ACL Graph执行正确性、验证性能）
- [ ] 12.5 实现HCCL分布式通信测试（验证all_reduce、all_gather、分布式推理）
- [ ] 12.6 实现torchair图模式测试（验证torchair图模式编译和执行）
- [ ] 12.7 实现周边组件版本矩阵测试（定义版本矩阵、CI执行多版本兼容测试）
- [ ] 12.8 标记周边组件测试用例（@pytest.mark.npu_real）
- [ ] 12.9 配置周边组件版本兼容告警（版本不兼容时告警）

## 13. Environment-Model-TestCase三层解耦基础设施

- [ ] 13.1 设计三层解耦架构（Environment Layer、Model Layer、Test Case Layer）
- [ ] 13.2 创建tests/st/utils/model_pool.py（ModelPool模型缓存池类）
- [ ] 13.3 实现ModelPool.get_or_create方法（获取或创建模型实例，缓存策略）
- [ ] 13.4 实现ModelPool._create_cache_key方法（考虑模型名、量化、分布式配置）
- [ ] 13.5 实现ModelPool._load_model方法（CPU Mock创建Mock模型、NPU真实加载真实模型）
- [ ] 13.6 实现ModelPool.release方法（手动释放指定模型）
- [ ] 13.7 实现ModelPool.clear方法（清空所有模型，模块结束时清理）
- [ ] 13.8 实现ModelPool.get_stats方法（统计模型数量、缓存命中率）
- [ ] 13.9 创建tests/st/utils/config_loader.py（ConfigLoader配置管理器类）
- [ ] 13.10 实现ConfigLoader.load方法（加载YAML配置文件）
- [ ] 13.11 实现ConfigLoader.load_model_config方法（加载特定模型配置）
- [ ] 13.12 实现ConfigLoader.load_test_config方法（加载特定测试配置）
- [ ] 13.13 实现配置缓存机制（避免重复读取配置文件）
- [ ] 13.14 创建tests/st/configs/目录（配置文件目录）
- [ ] 13.15 创建tests/st/configs/default.yaml（默认配置文件）
- [ ] 13.16 定义default配置节（exec_mode、timeout）
- [ ] 13.17 定义models配置节（default_model、Qwen-0.5B、Qwen-7B等模型配置）
- [ ] 13.18 定义tests配置节（default_test、worker_test、attention_test等测试配置）
- [ ] 13.19 创建tests/st/configs/ci_fast.yaml（CI快速测试配置）
- [ ] 13.20 创建tests/st/configs/npu_deep.yaml（NPU深度测试配置）
- [ ] 13.21 实现pytest_addoption命令行参数（--config-file、--model、--test-config）
- [ ] 13.22 实现config_loader fixture（scope="session"，配置管理器）
- [ ] 13.23 实现model_config fixture（scope="module"，从配置加载模型配置）
- [ ] 13.24 实现test_config fixture（scope="function"，从配置加载测试参数）
- [ ] 13.25 实现st_environment fixture（scope="session"，Layer 1环境层）
- [ ] 13.26 实现model_pool fixture（scope="module"，Layer 2模型层）
- [ ] 13.27 实现cached_model fixture（scope="module"，从ModelPool获取模型）
- [ ] 13.28 实现test_runner fixture（scope="function"，Layer 3测试用例层）
- [ ] 13.29 重构现有Worker测试使用ModelPool（避免重复模型加载）
- [ ] 13.30 重构现有Attention测试使用ModelPool（避免重复模型加载）
- [ ] 13.31 重构现有Quantization测试使用ModelPool（避免重复模型加载）
- [ ] 13.32 验证三层解耦架构正确性（模型不重复加载、测试独立性）
- [ ] 13.33 验证ModelPool缓存命中率（统计缓存效果）
- [ ] 13.34 编写三层解耦架构使用文档（说明如何使用ModelPool和ConfigLoader）
- [ ] 13.35 编写配置文件编写指南（YAML配置文件格式说明）

## 14. Mock模型创建策略（配合ModelPool）

- [ ] 14.1 实现create_mock_model函数（CPU Mock模式创建Mock模型）
- [ ] 14.2 实现Mock模型结构（模拟真实模型的forward、generate等方法）
- [ ] 14.3 实现Mock模型spec限制（确保Mock模型接口与真实模型一致）
- [ ] 14.4 实现Mock模型参数化（支持不同batch_size、seq_len等）
- [ ] 14.5 实现Mock模型缓存策略（同一配置共享Mock模型实例）
- [ ] 14.6 验证Mock模型与真实模型接口兼容性（接口一致性测试）

## 15. 配置驱动的测试用例改造

- [ ] 15.1 改造Worker测试使用配置驱动（从配置加载batch_size、scenario等）
- [ ] 15.2 改造Scheduler测试使用配置驱动（从配置加载调度参数）
- [ ] 15.3 改造Attention测试使用配置驱动（从配置加载seq_len、attention_state）
- [ ] 15.4 改造Quantization测试使用配置驱动（从配置加载量化参数）
- [ ] 15.5 改造Distributed测试使用配置驱动（从配置加载world_size、tp_size）
- [ ] 15.6 改造Sample测试使用配置驱动（从配置加载采样参数）
- [ ] 15.7 验证配置驱动测试正确性（配置参数正确传递到测试用例）
- [ ] 15.8 验证配置文件变更生效（修改配置文件，测试参数变更）

## 16. 精度/性能看护能力（长期优化，5周）

- [ ] 16.1 实现精度看护框架（CPU Mock vs NPU真实对比）
- [ ] 16.2 实现精度误差阈值定义（不同算子类型定义不同rtol、atol）
- [ ] 16.3 实现精度误差数据收集（JSON格式存储精度误差数据）
- [ ] 16.4 实现精度误差告警机制（精度超出阈值时告警）
- [ ] 16.5 实现性能看护框架（执行时间、内存、吞吐量测量）
- [ ] 16.6 实现性能基准数据管理（performance_baseline.json格式定义）
- [ ] 16.7 实现性能数据收集和分析（JSON格式存储性能数据）
- [ ] 16.8 实现性能报告自动生成（可视化性能数据、性能对比报告）
- [ ] 16.9 实现性能基准更新机制（模型升级、硬件升级时更新基准）
- [ ] 16.10 标记精度测试用例（@pytest.mark.npu_precision）
- [ ] 16.11 标记性能测试用例（@pytest.mark.npu_performance）
- [ ] 16.12 配置精度阈值告警机制（精度超出阈值时告警）
- [ ] 16.13 配置性能退化告警机制（性能超出基准时告警）

## 17. 覆盖率数据看护与可视化展示能力（短期优化，5天）

- [ ] 17.1 创建.coveragerc配置文件（定义source、omit、exclude_lines等）
- [ ] 17.2 配置覆盖率排除规则（排除tests/*、__pycache__/*等）
- [ ] 17.3 配置覆盖率报告输出路径（htmlcov、coverage.xml、coverage.json）
- [ ] 17.4 配置覆盖率门禁阈值（fail_under=80）
- [ ] 17.5 配置覆盖率报告显示选项（show_missing=True、skip_covered=True）
- [ ] 17.6 验证pytest-cov命令行参数（--cov、--cov-report、--cov-fail-under）
- [ ] 17.7 验证覆盖率数据收集正确性（pytest tests/st/ --cov=vllm_ascend）
- [ ] 17.8 验证覆盖率报告生成正确性（HTML、XML、JSON报告生成）
- [ ] 17.9 验证覆盖率门禁生效（--cov-fail-under=80门禁控制）
- [ ] 17.10 验证分支覆盖率收集（pytest --cov-branch）
- [ ] 17.11 验证覆盖率数据追加（pytest --cov-append合并UT/ST/E2E覆盖率）
- [ ] 17.12 实现覆盖率门禁分级策略（Level 1: 80%, Level 2: 90%）
- [ ] 17.13 实现模块覆盖率门禁（Worker≥85%, Attention≥85%, Scheduler≥80%）
- [ ] 17.14 实现覆盖率HTML报告可视化（coverage html生成htmlcov/index.html）
- [ ] 17.15 实现覆盖率趋势追踪（coverage_history/*.json存储历史数据）
- [ ] 17.16 实现覆盖率趋势脚本（coverage_trend.py生成趋势图）
- [ ] 17.17 实现覆盖率仪表板（COVERAGE_REPORT.md展示覆盖率统计）
- [ ] 17.18 实现覆盖率ASCII图表（显示覆盖率百分比条形图）
- [ ] 17.19 实现覆盖率JSON报告解析（提取覆盖率数据用于趋势分析）
- [ ] 17.20 实现覆盖率XML报告解析（用于CI集成Codecov）
- [ ] 17.21 配置GitHub Actions覆盖率集成（st_coverage.yml）
- [ ] 17.22 配置覆盖率上传到Codecov（上传coverage.xml）
- [ ] 17.23 配置PR覆盖率评论（PR评论显示覆盖率变化）
- [ ] 17.24 编写覆盖率使用文档（pytest-cov命令行参数使用说明）
- [ ] 17.25 编写覆盖率门禁配置文档（fail_under分级策略说明）
- [ ] 17.26 编写覆盖率报告解读文档（HTML/XML/JSON报告解读）
- [ ] 17.27 编写覆盖率趋势追踪文档（coverage_history使用说明）
- [ ] 17.28 验证覆盖率数据看护完整性（覆盖率收集→报告→门禁→可视化全流程）

## 18. CI/CD集成设计（复用现有workflow架构）

- [ ] 18.1 修正设计文档环境依赖假设（ST测试必须依赖CANN container，源代码强依赖torch_npu）
- [ ] 18.2 验证环境依赖假设修正正确性（platform.py、worker_v1.py、utils.py依赖分析）
- [ ] 18.3 添加st_tracker到changes filter（tests/st/**触发ST测试）
- [ ] 18.4 验证st_tracker触发机制正确性（tests/st/变更触发ST job）
- [ ] 18.5 添加st-cpu-mock job到vllm_ascend_test.yaml（扩展workflow而非独立workflow）
- [ ] 18.6 验证st-cpu-mock job执行正确性（CANN container、pytest执行、覆盖率收集）
- [ ] 18.7 配置st-cpu-mock job环境变量（VLLM_LOGGING_LEVEL、VLLM_WORKER_MULTIPROC_METHOD）
- [ ] 18.8 配置st-cpu-mock pytest参数（--exec-mode=cpu_mock、--cov、--cov-fail-under=80）
- [ ] 18.9 配置st-cpu-mock vllm版本矩阵（vllm_version: [main, v0.10.0]）
- [ ] 18.10 验证双版本ST测试正确性（main和v0.10.0版本ST测试通过）
- [ ] 18.11 配置Codecov flags: st_tests（区分UT和ST覆盖率）
- [ ] 18.12 更新codecov.yml添加st_tests项目状态（target: 85%）
- [ ] 18.13 配置覆盖率上传到Codecov（flags: st_tests、name: vllm-ascend-st）
- [ ] 18.14 验证Codecov flags正确性（UT和ST覆盖率独立显示）
- [ ] 18.15 配置执行顺序（lint → ut → st-cpu-mock → e2e）
- [ ] 18.16 验证执行顺序正确性（ST测试先于E2E、快速门禁）
- [ ] 18.17 配置ST测试执行时间门禁（≤5分钟）
- [ ] 18.18 验证ST测试执行时间（CI执行时间符合门禁要求）
- [ ] 18.19 配置workflow concurrency控制（避免ST测试重复执行）
- [ ] 18.20 验证workflow concurrency正确性（同一ref只执行一次ST测试）
- [ ] 18.21 编写CI/CD集成文档（workflow结构、changes filter、job配置说明）
- [ ] 18.22 编写环境依赖说明文档（CANN container依赖、torch_npu依赖）
- [ ] 18.23 验证CI/CD集成完整性（workflow→changes filter→job→coverage→codecov全流程）
- [ ] 18.24 验证ST测试与UT/E2E协调正确性（并行执行、覆盖率合并）