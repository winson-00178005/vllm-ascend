## 背景

vllm-ascend是vLLM的Ascend NPU硬件插件，当前测试分层包括：
- **UT（tests/ut/）**：使用unittest+pytest框架，验证单个函数/类级别的功能，已有完善的单元测试基础设施（TestBase基类、conftest.py fixtures）
- **E2E（tests/e2e/）**：端到端测试，验证完整推理流程，需要真实NPU环境和模型加载，执行时间较长（分钟级）

**缺失中间层**：缺少系统集成测试（ST）来验证模块间协作关系，例如：
- Worker与ModelRunner的交互流程
- Scheduler与Worker的任务调度协作
- Attention组件间的数据流传递
- 量化模块与模型加载的集成

**约束条件**：
- ST测试必须在CANN环境下执行（源代码强依赖torch_npu，即使CPU Mock模式也需要torch_npu安装）
- CPU Mock模式不依赖真实NPU硬件（使用Mock替代硬件调用）
- 单套测试执行时间≤5分钟（符合CI门禁要求）
- 测试用例需遵循FIRST原则（Fast, Independent, Repeatable, Self-Validating, Timely）
- 不能侵入业务代码，测试代码与产品代码隔离

## 目标与非目标

**Goals：**
- 建立ST测试框架基础设施，包括基类、fixtures、配置管理、Mock工具
- 为核心模块生成首批ST测试用例，覆盖模块间协作场景（首批定义：Worker、Scheduler、Attention模块，共15个核心协作场景）
- ST测试可在CPU Mock模式下执行，无需真实NPU硬件，但需要CANN container环境（torch_npu是必需依赖）
- CPU Mock模式整体执行时间控制在5分钟内（单模块≤1分钟，全量≤5分钟），适合CI门禁
- 测试结果可追溯（pytest日志+覆盖率报告），失败可定位到具体模块交互点（命名规范+Docstring+Mock验证）

**Non-Goals：**
- 不验证完整推理流程（这是E2E测试职责）
- 不测试单个函数内部逻辑（这是UT测试职责）
- 不替代现有UT/E2E测试，而是补充中间测试层
- 不修改业务代码以适配测试需求
- 不追求100%覆盖率，而是聚焦关键模块协作路径

## 设计决策

### 1. 测试框架选择：pytest + unittest混合策略（双基类设计）

**决策**：ST测试框架使用pytest为主，兼容unittest.TestCase风格，采用双基类设计。

**双基类设计**（吸纳现有UT框架模式）：
- **TestSTBase(unittest.TestCase)**：unittest风格，适合传统测试场景
  - 在`__init__`中自动初始化Mock环境
  - 提供`tearDown`清理Mock状态
- **PytestSTBase**：pytest风格，适合复杂fixture和参数化场景
  - 使用`@pytest.fixture(autouse=True)`自动初始化环境
  - 在fixture中使用`yield`实现setup/teardown分离

**理由**：
- pytest提供更强大的fixtures系统，适合ST测试的复杂依赖管理
- pytest的parametrize功能支持多场景测试
- unittest.TestCase与现有UT框架一致，开发人员熟悉度高
- pytest-mock提供便捷的Mock功能，隔离外部依赖
- **双基类设计**解决unittest和pytest兼容性问题（参考tests/ut/base.py优秀模式）

**替代方案考虑**：
- 纯unittest：fixtures功能较弱，不适合复杂集成测试
- 纯pytest：与现有UT风格不一致，增加学习成本
- **选择混合策略**：兼容性最好，兼顾功能强大和团队熟悉度

### 2. 资源管理策略：setUp/tearDown + STRunner上下文管理器

**决策**：使用setUp/tearDown和STRunner上下文管理器双重策略管理测试资源。

**setUp/tearDown模式**（吸纳UT框架模式）：
- **setUp**：初始化共享资源（Mock对象、配置、测试数据）
- **tearDown**：恢复原始状态、清理Mock、释放资源
- 避免测试间状态污染，保证独立性

**STRunner上下文管理器**（吸纳E2E VllmRunner模式）：
```python
class STRunner:
    def __init__(self, module_name: str, config: Config):
        # 初始化模块和依赖
        self.worker = create_mock_worker(config)
        self.model_runner = create_mock_model_runner(config)
        
    def execute_integration(self, scenario: str):
        # 执行集成测试场景
        ...
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 自动清理模块资源
        cleanup_module_resources()
        # 防止资源泄漏

# 使用方式
with STRunner("worker", config) as runner:
    runner.execute_integration("normal_flow")
```

**理由**：
- setUp/tearDown是unittest标准模式，团队熟悉度高
- STRunner上下文管理器自动管理资源，防止资源泄漏（参考tests/e2e/conftest.py的VllmRunner模式）
- 双重策略适应不同测试场景：简单测试用setUp/tearDown，复杂集成用STRunner
- 符合FIRST原则的Independent和Repeatable要求

### 3. Mock策略：模块级Mock + 数据Mock + Mock工厂函数

**决策**：ST测试使用两层Mock策略。

**决策**：ST测试使用三层Mock策略（模块级Mock + 数据Mock + Mock工厂函数）。

**模块级Mock**：
- Mock外部硬件依赖（torch_npu、CANN）
- Mock耗时操作（模型加载、编译）
- Mock分布式通信（hccl）

**数据Mock**：
- 使用预生成的测试数据（tensor、配置）
- 使用内存数据库替代真实存储
- 使用Fake对象替代复杂组件

**Mock工厂函数**（吸纳test_fused_ops.py优秀模式）：
```python
# mock_utils.py
def create_mock_worker(spec_attrs: List[str] = None):
    """创建Mock Worker对象，支持spec限制"""
    default_spec = ['execute_model', 'input_batch', 'device']
    mock_worker = MagicMock(spec=spec_attrs or default_spec)
    mock_worker.execute_model = MagicMock(return_value=torch.randn(...))
    mock_worker.input_batch = create_mock_input_batch()
    return mock_worker

def create_mock_distribution_env(mocker):
    """创建分布式环境Mock"""
    mock_group = mocker.MagicMock()
    mock_group.rank_in_group = 0
    mock_group.world_size = 4
    mock_group.all_reduce = MagicMock(return_value=torch.randn(...))
    return mock_group

def verify_mock_calls(mock_obj, expected_calls: int, 
                     expected_args: dict = None):
    """验证Mock调用次数和参数"""
    assert mock_obj.call_count == expected_calls
    if expected_args:
        mock_obj.assert_called_with(**expected_args)
```

**Mock验证策略**（吸纳现有测试模式）：
- `mock.assert_called_once()`：验证调用次数
- `mock.assert_called_once_with(args)`：验证调用参数
- `self.assertEqual(mock.call_count, 2)`：验证多次调用
- 验证调用顺序、参数内容、返回值

**理由**：
- 模块级Mock隔离外部依赖，保证测试可重复执行
- 数据Mock提升测试执行速度（避免真实计算）
- **Mock工厂函数**集中管理复杂Mock对象，可复用（参考tests/ut/ops/test_fused_ops.py的mock_ep_and_mc2_group模式）
- **spec限制Mock对象**避免过度Mock，提升真实性（参考tests/ut/quantization/test_w8a8.py的MagicMock(spec=[...])模式）
- **Mock验证策略**确保Mock按预期调用，避免测试无效（参考现有测试的assert_called_once模式）
- 符合"易维护"原则，Mock对象可复用

### 4. 测试目录结构：按模块组织 + 共享基础设施 + 层级化fixtures

**决策**：ST测试目录结构为：
```
tests/st/
├── base.py                 # ST测试基类（TestSTBase + PytestSTBase）
├── conftest.py              # pytest全局fixtures（session scope）
├── utils/                   # 测试工具类
│   ├── mock_utils.py       # Mock辅助函数 + Mock工厂
│   ├── data_generator.py   # 测试数据生成器
│   ├── config_factory.py   # 配置对象工厂
│   └── runner_factory.py   # STRunner工厂类
├── fixtures/                # 测试fixtures
│   ├── worker_fixtures.py  # Worker相关fixtures
│   ├── attention_fixtures.py
│   └── ...
└── <module>/                # 各模块ST测试
    ├── conftest.py          # 模块级fixtures（参数化）
    ├── test_<module_integration>.py
```

**层级化fixtures组织**（吸纳E2E conftest.py模式）：
```python
# tests/st/conftest.py（全局级）
@pytest.fixture(scope="session")
def mock_npu_env():
    """会话级：Mock NPU硬件环境，所有测试共享"""
    with patch('torch_npu', ...):
        yield

@pytest.fixture(scope="session")
def st_runner_factory():
    """会话级：返回STRunner类，延迟实例化"""
    return STRunner

# tests/st/worker/conftest.py（模块级）
@pytest.fixture
def worker_runner(st_runner_factory, worker_config):
    """模块级：创建Worker STRunner实例"""
    return st_runner_factory("worker", worker_config)

@pytest.fixture(params=[DEFAULT_CONFIG, LARGE_BATCH_CONFIG])
def worker_config(request):
    """参数化：不同Worker配置"""
    return request.param
```

**理由**：
- 模块组织与源码结构对应，易于定位测试
- 共享基础设施避免重复代码
- **层级化fixtures**（全局 + 模块级）分离关注点（参考tests/e2e/conftest.py的session scope模式）
- **fixture返回类本身**延迟实例化，提升灵活性（参考tests/e2e/conftest.py的vllm_runner fixture）
- **参数化fixture**支持多场景测试（参考tests/e2e/conftest.py的prompt_template fixture）
- conftest.py层级化管理fixtures，避免全局fixtures过多

### 5. 测试用例命名：Test_<功能特性>_<场景描述>

**决策**：ST测试命名遵循`Test_<模块集成特性>_<正常/异常/边界场景>`。

**示例**：
- `Test_WorkerModelRunner_Integration`：Worker与ModelRunner集成测试类
- `test_normal_execution_flow`：正常执行流程场景
- `test_error_handling_when_model_not_loaded`：模型未加载异常场景

**理由**：
- 测试意图明确，易于理解
- 符合开发者测试规范的命名建议
- 测试失败时，名称直接提示问题模块

### 6. 测试数据管理：预生成 + 工厂模式 + 全局常量

**决策**：使用工厂模式生成测试数据，预生成常用数据集，定义全局常量。

**实现方式**：
- `data_generator.py`提供数据工厂类
- 预生成常用tensor形状（例如：[1, 16], [32, 64], [1, 1024, 4096]）
- 预生成常用配置模板（例如：default_config, large_batch_config）
- 支持随机生成可配置范围的数据
- **定义全局常量**：避免数据硬编码（参考tests/ut/sample/test_rejection_sampler.py的PLACEHOLDER_TOKEN_ID模式）

**理由**：
- 工厂模式避免数据硬编码，提升可维护性
- 预生成数据提升执行速度（避免每次测试时生成）
- 随机生成支持探索性测试（配合固定种子保证可重复）
- **全局常量**统一管理测试数据，避免分散定义（参考现有测试的优秀模式）

### 7. 测试用例编写规范：强制Docstring + 参数化 + 异常测试

**决策**：强制要求ST测试遵循编写规范，确保测试质量。

**强制Docstring**（吸纳现有测试优秀模式）：
```python
def test_worker_schedule_to_runner(self, batch_size, scenario):
    """Test Worker调度请求到ModelRunner执行
    
    验证：
    - Worker正确传递请求参数到ModelRunner
    - ModelRunner执行推理并返回输出tensor
    - 异常情况下Worker正确处理错误
    """
    ...
```

**参数化多维度测试**（吸纳E2E测试模式）：
```python
@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("scenario", ["normal", "error"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_worker_model_runner_integration(self, batch_size, scenario, dtype):
    ...
```

**异常场景测试模式**（吸纳UT测试模式）：
```python
def test_worker_handle_runner_error(self):
    """Test Worker处理ModelRunner执行失败"""
    with self.assertRaises(RuntimeError) as cm:
        worker.execute_model(error_input)
    assert "memory allocation failed" in str(cm.exception)
    # 验证Worker正确清理资源
    assert worker.resources_released
```

**理由**：
- **Docstring**清晰说明测试意图和验证点（参考tests/ut/quantization/test_w8a8.py的优秀注释模式）
- **参数化测试**自动生成多组测试场景，提升覆盖率（参考tests/e2e/singlecard/test_offline_inference.py的多维度参数化模式）
- **异常测试**验证错误处理逻辑，确保系统稳定性（参考tests/ut/quantization/test_w8a8.py的assertRaises模式）
- 符合开发者测试规范的"Right-BICEP原则"（验证异常场景）
- 提升测试可读性和可维护性

### 8. 配置管理策略：明确依赖环境（插件架构适配）

**决策**：ST测试框架明确依赖vllm和torch_npu环境，无法在无vllm环境下执行。

**背景**：vllm-ascend是vllm的硬件插件，采用插件架构：
- **继承**：NPUPlatform extends Platform，实现抽象方法
- **替换**：vllm.Worker → NPUWorker，vllm.Attention → AscendAttention
- **注入**：register()注册插件，check_and_update_config()注入配置

**依赖现状分析**（修正之前假设）：
- **源代码依赖**：
  - platform.py: from vllm.platforms import Platform
  - worker_v1.py: from vllm.config import VllmConfig
  - utils.py: import torch_npu（第29行）
  
- **UT测试依赖**：
  - test_ascend_config.py: from vllm.config import ModelConfig, VllmConfig
  - test_platform.py: from vllm.config import CompilationLevel
  - test_fused_ops.py: import torch_npu
  
- **结论**：插件无法独立于宿主vllm执行，torch_npu是必需依赖

**vllm依赖现状**：
- E2E测试：强依赖vllm（import vllm, LLM, SamplingParams），需要真实环境
- UT测试：轻依赖vllm.config（from vllm.config import Config），需要vllm环境
- **ST测试目标**：中等依赖，需要vllm和torch_npu环境执行

**修正后实现方式**：
```python
# config_factory.py - 明确依赖环境
from vllm.config import VllmConfig, ModelConfig, CompilationConfig

# ST测试依赖vllm环境（插件架构要求）
def create_vllm_config(**kwargs):
    """创建VllmConfig，需要vllm环境"""
    return VllmConfig(**kwargs)
        return create_mock_vllm_config(**kwargs)

def create_ascend_config(**kwargs):
    """创建AscendConfig（插件自有配置，无需vllm依赖）"""
    from vllm_ascend.ascend_config import AscendConfig
    return AscendConfig(**kwargs)
```

**理由**：
- 开发环境可能没有安装vllm包，ST测试需要独立执行
- UT测试依赖vllm.config但在某些环境无法执行（参考tests/ut/test_ascend_config.py依赖vllm.config）
- ST测试不应依赖外部包，确保CI门禁可执行
- 提供Fallback配置类或Mock策略，隔离vllm依赖
- 符合插件架构的独立性要求（插件测试不应依赖宿主环境）

### 9. 插件接口兼容性验证：spec限制Mock验证接口

**决策**：ST测试使用spec限制Mock验证插件组件与vllm接口的兼容性。

**背景**：vllm-ascend插件需要替换vllm核心组件：
```
vllm Core Component        →  vllm-ascend Plugin Component
─────────────────────────────────────────────────────────────
vllm.Worker                →  NPUWorker
vllm.Scheduler             →  AscendScheduler  
vllm.AttentionBackend      →  AscendAttentionBackend
vllm.ModelRunner           →  AscendModelRunner
```

**验证插件接口兼容性**：
```python
# mock_utils.py - 定义vllm接口spec
VLLM_WORKER_SPEC = [
    'execute_model',
    'initialize_model',
    'get_model',
    'profile_run',
    'start_worker',
    'stop_worker',
]

VLLM_ATTENTION_BACKEND_SPEC = [
    'get_name',
    'get_impl_cls',
    'get_metadata_cls',
    'get_state_cls',
    'get_kv_cache_shape',
    'swap_blocks',
    'copy_blocks',
]

def create_mock_vllm_worker(spec_attrs=None):
    """创建Mock vllm.Worker，定义接口spec"""
    spec = spec_attrs or VLLM_WORKER_SPEC
    mock_worker = MagicMock(spec=spec)
    mock_worker.execute_model = MagicMock(return_value=torch.randn(...))
    return mock_worker

def verify_plugin_interface_compatibility(plugin_obj, vllm_interface_spec):
    """验证插件组件实现了vllm接口"""
    for attr in vllm_interface_spec:
        assert hasattr(plugin_obj, attr), \
            f"Plugin object missing vllm interface: {attr}"
        # 验证方法可调用
        method = getattr(plugin_obj, attr)
        assert callable(method) or isinstance(method, property), \
            f"Plugin interface {attr} is not callable"
```

**ST测试用例示例**：
```python
def test_npu_worker_interface_compatibility():
    """验证NPUWorker实现了vllm.Worker的所有接口"""
    npu_worker = create_npu_worker(mock_config)
    
    # 验证接口兼容性
    verify_plugin_interface_compatibility(npu_worker, VLLM_WORKER_SPEC)
    
    # 验证NPUWorker与vllm.Worker协作
    mock_vllm_worker = create_mock_vllm_worker()
    # 测试调度协作...
```

**理由**：
- vllm-ascend是插件，需要替换vllm的核心组件
- ST测试应验证插件组件实现了vllm的接口（确保替换逻辑正确）
- spec定义vllm接口规范，Mock验证插件兼容性
- 确保插件与vllm核心版本兼容（避免接口变更导致插件失效）
- 参考tests/e2e/conftest.py的VllmRunner模式（测试插件集成）

### 10. 测试聚焦策略：插件内部协作而非vllm核心逻辑

**决策**：ST测试用例聚焦插件组件间协作，不测试vllm核心逻辑。

**测试分层职责明确**：
```
测试类型    职责范围                          测试内容
───────────────────────────────────────────────────────────────
E2E测试    验证插件与vllm完整集成            插件注入、完整推理流程
           需要真实vllm环境和NPU硬件         端到端正确性

UT测试     验证插件组件独立功能              插件类、函数、配置
           Mock torch_npu硬件               组件内部逻辑

ST测试     验证插件内部模块间协作            Worker+ModelRunner协作
           Mock vllm Core和torch_npu        Scheduler+Worker协作
           不依赖真实vllm和NPU               Attention+Mask协作
```

**ST测试用例设计原则**：
- ✅ **测试插件内部协作**：NPUWorker与AscendModelRunner协作流程
- ✅ **验证替换逻辑正确**：AscendScheduler替换vllm.Scheduler后的协作
- ✅ **验证接口兼容性**：插件组件实现了vllm接口
- ❌ **不测试vllm核心逻辑**：vllm.Scheduler的调度算法（那是vllm的测试职责）
- ❌ **不测试vllm集成**：插件与vllm的完整集成（那是E2E测试职责）

**Worker集成测试示例**：
```python
class Test_NPUWorker_AscendModelRunner_Integration(TestSTBase):
    """测试插件内部：NPUWorker与AscendModelRunner协作"""
    
    def test_worker_initialize_model_runner(self):
        """验证NPUWorker正确初始化AscendModelRunner"""
        # 使用Mock配置（隔离vllm依赖）
        mock_config = create_ascend_config()
        
        # 创建插件组件（真实插件组件）
        npu_worker = NPUWorker(mock_config)
        
        # 验证Worker初始化ModelRunner
        assert hasattr(npu_worker, 'model_runner')
        assert isinstance(npu_worker.model_runner, AscendModelRunner)
        
        # 验证接口兼容性（插件实现了vllm.Worker接口）
        verify_plugin_interface_compatibility(npu_worker, VLLM_WORKER_SPEC)
    
    def test_worker_schedule_to_model_runner(self):
        """验证NPUWorker调度请求到AscendModelRunner执行"""
        with STRunner("worker", mock_config) as runner:
            # Mock输入数据
            input_batch = create_mock_input_batch()
            
            # 执行插件内部协作流程
            result = runner.worker.execute_model(input_batch)
            
            # 验证ModelRunner被调用
            runner.model_runner.execute.assert_called_once()
            # 验证结果传递回Worker
            assert result is not None
```

**理由**：
- E2E测试验证插件与vllm的完整集成（需要真实环境）
- UT测试验证插件组件的独立功能（组件级别）
- ST测试应验证插件内部模块间协作（填补中间测试层）
- 不应测试vllm核心逻辑（那是vllm上游的测试职责）
- 符合"插件独立性"原则（插件测试不应依赖宿主核心）
- 参考tests/ut/test_utils.py设计（测试插件自身功能，不依赖vllm执行）

### 11. 双模式执行策略：CPU Mock + NPU真实

**决策**：ST测试支持双模式执行，通过命令行参数灵活切换。

**CPU Mock模式**（快速执行）：
- Mock torch_npu.npu_*所有NPU算子操作
- Mock torch.npu.*设备、内存、同步操作
- Mock NPU硬件（内存分配、设备属性）
- **保留插件模块协作**（Worker、Scheduler、Attention真实组件）
- 执行环境：x86/arm服务器、MacBook、无torch_npu环境
- 执行时间：快速（秒级），≤5分钟
- 测试目的：验证模块协作、验证接口兼容、快速反馈、CI门禁

**NPU真实模式**（深度测试）：
- 使用真实torch_npu.npu_*算子
- 使用真实torch.npu.*设备操作
- 使用真实NPU硬件（Ascend NPU、Atlas 800、Atlas A2）
- 测试插件与周边组件集成（CANN、ACL Graph、HCCL、torchair）
- 执行时间：较长（分钟级），可选执行
- 测试目的：验证真实性能、验证真实精度、深度集成测试、性能优化验证

**双模式设计理由**：
- CI环境差异：x86/arm服务器无NPU，需要Mock快速执行；NPU环境需要真实测试
- 测试分层适配：CPU Mock适合快速CI门禁，NPU真实适合深度性能/精度测试
- 周边组件集成：NPU真实模式可测试CANN、ACL Graph、HCCL等周边组件集成
- 参考现有模式：UT全部Mock torch_npu（179次patch），E2E使用真实torch_npu，ST双模式

### 12. pytest命令行控制Mock：pytest_addoption设计

**决策**：使用pytest_addoption添加命令行参数控制Mock行为（参考tests/e2e/models/conftest.py）。

**命令行参数设计**：
```python
# tests/st/conftest.py
def pytest_addoption(parser):
    # 执行模式选择
    parser.addoption(
        "--exec-mode",
        action="store",
        default="auto",
        choices=["auto", "cpu_mock", "npu_real"],
        help="Execution mode: auto (detect), cpu_mock, npu_real"
    )
    
    # Mock开关
    parser.addoption(
        "--enable-mock",
        action="store_true",
        default=False,
        help="Force enable Mock even in NPU environment"
    )
    
    parser.addoption(
        "--disable-mock",
        action="store_true",
        default=False,
        help="Force disable Mock even in CPU environment"
    )
    
    # 性能/精度测试开关
    parser.addoption(
        "--performance-test",
        action="store_true",
        default=False,
        help="Enable performance measurement tests"
    )
    
    parser.addoption(
        "--precision-test",
        action="store_true",
        default=False,
        help="Enable precision verification tests"
    )
    
    # NPU设备选择
    parser.addoption(
        "--npu-device",
        action="store",
        default="0",
        help="NPU device ID to use (default: 0)"
    )
    
    # 性能基准文件
    parser.addoption(
        "--performance-baseline",
        action="store",
        default=None,
        help="Path to performance baseline JSON file"
    )

@pytest.fixture(scope="session")
def exec_mode(pytestconfig):
    return pytestconfig.getoption("--exec-mode")

@pytest.fixture(scope="session")
def enable_mock(pytestconfig):
    return pytestconfig.getoption("--enable-mock")

@pytest.fixture(scope="session")
def performance_test(pytestconfig):
    return pytestconfig.getoption("--performance-test")
```

**命令行执行示例**：
```bash
# CI快速执行（CPU Mock）
pytest tests/st/ --exec-mode=cpu_mock
pytest tests/st/ --exec-mode=auto
pytest tests/st/ --enable-mock

# NPU深度测试
pytest tests/st/ --exec-mode=npu_real
pytest tests/st/ --exec-mode=npu_real --npu-device=1
pytest tests/st/ --exec-mode=npu_real --performance-test
pytest tests/st/ --exec-mode=npu_real --precision-test

# 性能测试对比
pytest tests/st/worker/ --exec-mode=npu_real \
    --performance-test \
    --performance-baseline=./baseline/qwen2.json

# 强制Mock（调试用）
pytest tests/st/ --exec-mode=npu_real --enable-mock
```

**理由**：
- pytest_addoption是pytest官方推荐的自定义命令行参数方式
- 参考tests/e2e/models/conftest.py的优秀模式（--config-list-file、--tp-size等）
- 命令行控制灵活，CI和开发者可根据环境选择执行模式
- 与pytest marker互补（marker标记测试类型，命令行控制执行行为）

### 13. pytest marker标记系统：双模式测试标记

**决策**：使用pytest marker标记测试类型，配合命令行参数执行。

**自定义marker定义**（pytest配置）：
```ini
# pytest.ini或pyproject.toml
[tool:pytest]
markers =
    cpu_mock: CPU Mock execution (fast)
    npu_real: NPU real execution (deep)
    npu_performance: NPU performance tests
    npu_precision: NPU precision tests
```

**测试用例标记示例**：
```python
@pytest.mark.cpu_mock
def test_worker_schedule_cpu():
    """CPU Mock模式测试Worker调度"""
    ...

@pytest.mark.npu_real
def test_worker_schedule_npu():
    """NPU真实模式测试Worker调度"""
    ...

@pytest.mark.npu_performance
def test_worker_performance():
    """NPU性能测试"""
    ...

@pytest.mark.npu_precision
def test_attention_precision():
    """NPU精度测试"""
    ...
```

**marker执行命令**：
```bash
pytest -m cpu_mock tests/st/  # 执行CPU Mock测试
pytest -m npu_real tests/st/  # 执行NPU真实测试
pytest -m "cpu_mock or npu_real"  # 双模式执行
pytest -m npu_performance tests/st/  # 性能测试
pytest -m npu_precision tests/st/  # 精度测试
```

**参数化测试支持双模式**（参考tests/ut/ops/test_activation.py）：
```python
@pytest.mark.parametrize("exec_mode", ["cpu_mock", "npu_real"])
def test_worker_execute_model(exec_mode):
    """参数化双模式测试"""
    if exec_mode == "cpu_mock":
        # Mock torch_npu
        worker = create_mock_worker()
    elif exec_mode == "npu_real":
        # 真实torch_npu
        worker = create_real_worker()
    ...
```

**理由**：
- pytest marker是标准的测试分类方式
- 参数化测试支持双模式（参考test_activation.py的is_310p_return参数化）
- marker和命令行参数互补：marker标记测试类型，命令行控制执行行为

### 14. 环境检测机制：自动选择执行模式

**决策**：实现环境检测机制，自动选择合适的执行模式。

**环境检测层级**（tests/st/utils/env_detector.py）：
```python
def detect_execution_environment():
    """检测环境并返回可用执行模式"""
    # Level 1: torch可用性
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False
        return "skip"  # 无torch，跳过
    
    # Level 2: torch_npu可用性
    try:
        import torch_npu
        HAS_TORCH_NPU = True
    except ImportError:
        HAS_TORCH_NPU = False
    
    # Level 3: NPU硬件可用性
    try:
        NPU_AVAILABLE = torch.npu.is_available()
        NPU_COUNT = torch.npu.device_count()
    except:
        NPU_AVAILABLE = False
        NPU_COUNT = 0
    
    # Level 4: NPU类型检测（参考vllm_ascend/utils.py的is_310p()）
    try:
        from vllm_ascend.utils import is_310p
        IS_310P = is_310p()
    except:
        IS_310P = False
    
    # 确定执行模式
    if NPU_AVAILABLE and NPU_COUNT > 0:
        return "npu_real"
    else:
        return "cpu_mock"

@pytest.fixture(scope="session")
def execution_environment(request):
    """根据命令行和环境检测选择执行模式"""
    exec_mode = request.config.getoption("--exec-mode")
    
    if exec_mode == "auto":
        # 自动检测
        mode = detect_execution_environment()
    elif exec_mode == "cpu_mock":
        mode = "cpu_mock"
    elif exec_mode == "npu_real":
        # 验证NPU可用
        if not detect_execution_environment() == "npu_real":
            pytest.skip("NPU not available, skip npu_real tests")
        mode = "npu_real"
    
    return create_environment(mode)
```

**理由**：
- 参考vllm_ascend/utils.py的is_310p()检测模式（from _build_info获取SOC版本）
- 参考tests/ut/test_platform.py Mock torch.npu.is_available()模式
- 自动检测避免手动配置，适配不同CI环境
- 支持命令行override自动检测结果

### 15. 选择性Mock工厂：CPU Mock vs NPU真实环境

**决策**：实现选择性Mock工厂类，动态创建CPU Mock或NPU真实环境。

**Mock工厂设计**（tests/st/utils/env_factory.py）：
```python
class ExecutionEnvironment:
    """执行环境基类"""
    def __init__(self, mode: str):
        self.mode = mode
        self.mock_patches = []
    
    def enter(self):
        """进入环境（启动Mock或初始化真实环境）"""
        raise NotImplementedError
    
    def exit(self):
        """退出环境（清理Mock或释放真实资源）"""
        raise NotImplementedError

class CPUMockEnvironment(ExecutionEnvironment):
    """CPU Mock环境：Mock torch_npu和torch.npu"""
    def __init__(self):
        super().__init__("cpu_mock")
        # 定义Mock patches（参考tests/ut/quantization/test_w8a8.py）
        self.mock_patches = [
            patch("torch_npu.npu_quantize", side_effect=lambda x: x),
            patch("torch_npu.npu_swiglu", side_effect=lambda x: x),
            patch("torch_npu.npu_flash_attention", ...),
            patch("torch.npu.set_device", ...),
            patch("torch.npu.empty_cache", ...),
            patch("torch.npu.synchronize", ...),
            # Mock is_310p（参考tests/ut/ops/test_activation.py）
            patch("vllm_ascend.utils.is_310p", return_value=False),
        ]
    
    def enter(self):
        for p in self.mock_patches:
            p.start()
    
    def exit(self):
        for p in self.mock_patches:
            p.stop()

class NPURealEnvironment(ExecutionEnvironment):
    """NPU真实环境：使用真实torch_npu"""
    def __init__(self, device_id=0):
        super().__init__("npu_real")
        self.device_id = device_id
        # 验证NPU可用
        assert torch.npu.is_available(), "NPU not available"
        torch.npu.set_device(device_id)
    
    def measure_performance(self, func):
        """性能测量：执行时间、内存、吞吐量"""
        import time
        start_time = time.time()
        start_mem = torch.npu.memory_allocated()
        result = func()
        end_time = time.time()
        end_mem = torch.npu.memory_allocated()
        return {
            "execution_time": end_time - start_time,
            "memory_usage": end_mem - start_mem,
            "result": result
        }
    
    def measure_precision(self, result_cpu, result_npu):
        """精度测量：对比CPU和NPU结果"""
        return torch.allclose(result_cpu, result_npu,
                             rtol=1e-3, atol=1e-5)
    
    def exit(self):
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()

def create_environment(mode: str, **kwargs):
    """根据模式创建执行环境"""
    if mode == "cpu_mock":
        return CPUMockEnvironment()
    elif mode == "npu_real":
        return NPURealEnvironment(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
```

**测试用例使用示例**：
```python
def test_worker_execute_model(execution_environment):
    """测试Worker执行模型"""
    with execution_environment as env:
        if env.mode == "cpu_mock":
            # Mock模式测试
            worker = create_mock_worker()
            result = worker.execute_model(input_batch)
            # 验证协作逻辑
            ...
        elif env.mode == "npu_real":
            # NPU真实模式测试
            worker = create_real_worker()
            result = worker.execute_model(input_batch)
            # 性能测量（如果--performance-test启用）
            if performance_test:
                perf = env.measure_performance(
                    lambda: worker.execute_model(input_batch))
                assert perf["execution_time"] < threshold
            # 精度测量（如果--precision-test启用）
            if precision_test:
                result_cpu = get_cpu_reference_result()
                assert env.measure_precision(result_cpu, result)
```

**理由**：
- 参考tests/ut中179次patch torch_npu的优秀模式
- 参考tests/ut/ops/test_activation.py的patch("vllm_ascend.utils.is_310p")模式
- 工厂模式集中管理Mock，避免每个测试文件重复patch
- 上下文管理器自动清理Mock或NPU资源

### 16. 性能/精度测试集成：周边组件深度测试

**决策**：在NPU真实模式下集成性能/精度测试和周边组件集成测试。

**性能测试集成**：

### 17. Environment-Model-TestCase三层解耦架构

**决策**：ST测试框架采用三层解耦架构，分离环境层、模型层和测试用例层，避免重复模型加载。

**背景**：现有E2E测试存在严重耦合问题：
- VllmRunner在`__init__`中调用`LLM(model)`加载模型
- VllmRunner在`__exit__`中删除模型
- 同一模型被多次加载（例如Qwen-0.5B在test_offline_inference.py和test_aclgraph.py中重复加载）
- **总计43次模型加载/卸载**，浪费大量时间和资源

**三层解耦架构设计**：
```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Environment Layer (Session Scope)              │
│ - NPU环境初始化（一次性）                                 │
│ - torch_npu初始化                                        │
│ - CANN/HCCL环境准备                                      │
│ - pytest fixture scope="session"                         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 2: Model Layer (ModelPool with Cache)             │
│ - ModelPool管理模型实例                                  │
│ - 模型缓存避免重复加载                                   │
│ - 支持多模型并发                                         │
│ - pytest fixture scope="module" or "class"              │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Test Case Layer (Function Scope)               │
│ - 从ModelPool获取模型                                    │
│ - 不直接加载模型                                         │
│ - 测试用例独立                                           │
│ - pytest fixture scope="function"                        │
└─────────────────────────────────────────────────────────┘
```

**实现方式**：
```python
# tests/st/conftest.py

# Layer 1: Environment Layer (session scope)
@pytest.fixture(scope="session")
def st_environment(request):
    """会话级：初始化NPU环境，只执行一次"""
    exec_mode = request.config.getoption("--exec-mode")
    env = create_environment(exec_mode)
    env.enter()
    yield env
    env.exit()

# Layer 2: Model Layer (module scope, with ModelPool)
@pytest.fixture(scope="module")
def model_pool(st_environment):
    """模块级：ModelPool管理模型实例"""
    pool = ModelPool()
    yield pool
    pool.clear()

@pytest.fixture(scope="module")
def cached_model(model_pool, model_name):
    """模块级：从ModelPool获取模型，避免重复加载"""
    return model_pool.get_or_create(model_name, 
                                    config=create_model_config(model_name))

# Layer 3: Test Case Layer (function scope)
@pytest.fixture(scope="function")
def test_runner(cached_model, st_environment):
    """函数级：创建测试Runner，使用缓存模型"""
    return STRunner(cached_model, st_environment)
```

**理由**：
- Layer 1只初始化一次环境，避免重复初始化NPU（节省时间）
- Layer 2使用ModelPool缓存模型，避免重复加载（解决43次加载问题）
- Layer 3测试用例从缓存获取模型，不直接加载（保证独立性）
- 三层解耦保证测试独立性和执行效率
- 参考E2E测试的session scope fixture模式，但引入ModelPool解决模型重复加载

### 18. ModelPool模型缓存池设计

**决策**：实现ModelPool模型缓存池，管理模型实例生命周期，避免重复加载。

**ModelPool设计**：
```python
# tests/st/utils/model_pool.py

class ModelPool:
    """模型缓存池，管理模型实例生命周期"""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._configs: Dict[str, Config] = {}
        
    def get_or_create(self, model_name: str, config: Config) -> Any:
        """获取或创建模型实例（缓存策略）"""
        cache_key = self._create_cache_key(model_name, config)
        
        if cache_key not in self._models:
            # 首次加载模型
            model = self._load_model(model_name, config)
            self._models[cache_key] = model
            self._configs[cache_key] = config
        else:
            # 使用缓存模型
            model = self._models[cache_key]
            
        return model
    
    def _create_cache_key(self, model_name: str, config: Config) -> str:
        """创建缓存key（考虑模型名、量化、分布式配置）"""
        key_parts = [model_name]
        if hasattr(config, 'quantization'):
            key_parts.append(config.quantization)
        if hasattr(config, 'tensor_parallel_size'):
            key_parts.append(f"tp{config.tensor_parallel_size}")
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    def _load_model(self, model_name: str, config: Config) -> Any:
        """加载模型（Mock或真实）"""
        if config.exec_mode == "cpu_mock":
            # CPU Mock模式：创建Mock模型
            return create_mock_model(model_name, config)
        else:
            # NPU真实模式：加载真实模型
            return load_real_model(model_name, config)
    
    def release(self, model_name: str):
        """释放指定模型（允许手动释放）"""
        for key in list(self._models.keys()):
            if key.startswith(model_name):
                del self._models[key]
                del self._configs[key]
    
    def clear(self):
        """清空所有模型（模块结束时清理）"""
        for model in self._models.values():
            if hasattr(model, 'cleanup'):
                model.cleanup()
        self._models.clear()
        self._configs.clear()
    
    def get_stats(self) -> Dict:
        """获取统计信息（模型数量、缓存命中率）"""
        return {
            "total_models": len(self._models),
            "model_names": list(set(k.split("|")[0] for k in self._models.keys())),
        }
```

**ModelPool使用示例**：
```python
# tests/st/worker/conftest.py

@pytest.fixture(scope="module")
def worker_model(model_pool, worker_config):
    """模块级：获取Worker测试模型"""
    return model_pool.get_or_create(
        worker_config.model_name,
        create_worker_config(worker_config)
    )

# tests/st/worker/test_worker_integration.py
class TestWorkerIntegration:
    def test_worker_normal_flow(self, worker_model, test_runner):
        """使用缓存模型，无需重新加载"""
        result = test_runner.execute(worker_model)
        assert result is not None
```

**理由**：
- ModelPool缓存模型实例，避免重复加载（解决43次加载问题）
- 支持多配置缓存（量化、分布式等不同配置）
- 支持手动释放和自动清理（灵活管理）
- 统计信息支持调试和优化（缓存命中率）
- 参考现有VllmRunner模式，但改为缓存池设计

### 19. ConfigLoader配置管理器设计

**决策**：实现ConfigLoader配置管理器，支持独立配置文件，实现配置驱动的测试。

**配置驱动测试背景**：
- 当前测试参数硬编码在代码中（batch_size=[1,16,32]）
- 配置变更需要修改代码，维护成本高
- 不同环境需要不同配置（CI快速配置、深度测试配置）

**ConfigLoader设计**：
```python
# tests/st/utils/config_loader.py

class ConfigLoader:
    """配置管理器，支持独立配置文件"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "tests/st/configs/default.yaml"
        self._config_cache: Dict[str, Any] = {}
        
    def load(self, config_name: str = None) -> Dict:
        """加载配置文件"""
        if config_name and config_name in self._config_cache:
            return self._config_cache[config_name]
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config_name:
            # 加载特定配置
            config = config.get(config_name, config.get('default', {}))
            self._config_cache[config_name] = config
        
        return config
    
    def load_model_config(self, model_name: str) -> ModelConfig:
        """加载模型配置"""
        config = self.load('models')
        model_config = config.get(model_name, config.get('default_model'))
        return create_model_config(model_config)
    
    def load_test_config(self, test_module: str) -> TestConfig:
        """加载测试配置"""
        config = self.load('tests')
        test_config = config.get(test_module, config.get('default_test'))
        return create_test_config(test_config)

# tests/st/configs/default.yaml
default:
  exec_mode: "auto"
  timeout: 300
  
models:
  default_model:
    name: "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: "float16"
    max_model_len: 2048
    
  Qwen-0.5B:
    name: "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: "float16"
    max_model_len: 2048
    
  Qwen-7B:
    name: "Qwen/Qwen2.5-7B-Instruct"
    dtype: "bfloat16"
    max_model_len: 4096
    tensor_parallel_size: 2
    
tests:
  default_test:
    batch_size: [1, 16]
    scenario: ["normal"]
    
  worker_test:
    batch_size: [1, 16, 32]
    scenario: ["normal", "error"]
    dtype: ["float16", "bfloat16"]
    
  attention_test:
    seq_len: [128, 512, 1024]
    attention_state: ["normal", "mla"]
```

**命令行配置选择**：
```python
# tests/st/conftest.py
def pytest_addoption(parser):
    parser.addoption(
        "--config-file",
        action="store",
        default="tests/st/configs/default.yaml",
        help="Path to test configuration file"
    )
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Model name to use (from config file)"
    )
    parser.addoption(
        "--test-config",
        action="store",
        default=None,
        help="Test configuration name (from config file)"
    )

@pytest.fixture(scope="session")
def config_loader(request):
    """会话级：配置管理器"""
    config_file = request.config.getoption("--config-file")
    return ConfigLoader(config_file)

@pytest.fixture(scope="module")
def model_config(config_loader, request):
    """模块级：模型配置"""
    model_name = request.config.getoption("--model")
    if model_name:
        return config_loader.load_model_config(model_name)
    return config_loader.load_model_config("default_model")
```

**命令行执行示例**：
```bash
# 使用默认配置
pytest tests/st/

# 指定配置文件
pytest tests/st/ --config-file=tests/st/configs/ci_fast.yaml

# 指定模型
pytest tests/st/worker/ --model=Qwen-7B

# 指定测试配置
pytest tests/st/worker/ --test-config=worker_test

# 组合使用
pytest tests/st/ --config-file=tests/st/configs/npu_deep.yaml \
    --model=Qwen-7B --exec-mode=npu_real
```

**理由**：
- 配置驱动测试，避免参数硬编码（提升可维护性）
- 支持多环境配置（CI快速配置、深度测试配置）
- 命令行灵活选择配置（适配不同测试场景）
- 参考tests/e2e/models/conftest.py的pytest_addoption模式
- YAML格式配置文件，易于编辑和理解

### 20. 覆盖率数据看护与可视化展示能力

**决策**：ST测试框架集成pytest-cov覆盖率能力，支持命令行参数输出覆盖率报告。

**覆盖率工具现状**：
- pytest-cov已存在于requirements-dev.txt（未实际使用）
- 缺少.coveragerc配置文件
- 缺少覆盖率门禁设置
- 缺少覆盖率报告生成
- CI中缺少覆盖率配置

**pytest-cov命令行参数支持**：

基本覆盖率命令：
```bash
pytest tests/st/ --cov=vllm_ascend                     # 覆盖vllm_ascend模块
pytest tests/st/ --cov=vllm_ascend.worker              # 覆盖特定子模块
pytest tests/st/ --cov-branch                          # 分支覆盖率
```

覆盖率报告格式：
```bash
pytest tests/st/ --cov-report=term                     # 终端文本报告
pytest tests/st/ --cov-report=term-missing             # 显示缺失行号
pytest tests/st/ --cov-report=html                     # HTML报告
pytest tests/st/ --cov-report=html:htmlcov             # HTML报告（指定目录）
pytest tests/st/ --cov-report=xml                      # XML报告（CI集成）
pytest tests/st/ --cov-report=xml:coverage.xml         # XML报告（指定文件）
pytest tests/st/ --cov-report=json                     # JSON报告
pytest tests/st/ --cov-report=json:coverage.json       # JSON报告（指定文件）
pytest tests/st/ --cov-report=lcov                     # LCOV报告
```

覆盖率门禁：
```bash
pytest tests/st/ --cov-fail-under=80                   # 覆盖率≥80%
pytest tests/st/ --cov-fail-under=90                   # 覆盖率≥90%（严格）
pytest tests/st/ --cov-fail-under=100                  # 覆盖率100%（最严格）
```

覆盖率数据追加：
```bash
pytest tests/ut/ --cov=vllm_ascend --cov-append        # 收集UT覆盖率
pytest tests/st/ --cov=vllm_ascend --cov-append        # 收集ST覆盖率
pytest tests/e2e/ --cov=vllm_ascend --cov-append       # 收集E2E覆盖率
# 合并UT、ST、E2E覆盖率数据
```

综合命令示例：
```bash
pytest tests/st/ \
  --cov=vllm_ascend \
  --cov-branch \
  --cov-report=html:htmlcov \
  --cov-report=xml:coverage.xml \
  --cov-report=json:coverage.json \
  --cov-fail-under=80 \
  --cov-config=.coveragerc
```

**.coveragerc配置文件设计**：
```ini
[run]
source = vllm_ascend
omit =
    tests/*
    vllm_ascend/__init__.py
    */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:

fail_under = 80
show_missing = True
skip_covered = True

[html]
directory = htmlcov

[xml]
output = coverage.xml

[json]
output = coverage.json
pretty_print = True
```

**覆盖率门禁分级策略**：

Level 1：CI快速门禁（CPU Mock模式）
- 覆盖率目标：≥80%
- 测试代码排除
- 执行时间：≤5分钟

Level 2：深度测试门禁（NPU真实模式）
- 覆盖率目标：≥90%
- 包含测试代码
- 执行时间：分钟级

Level 3：模块覆盖率门禁
- Worker模块：≥85%
- Attention模块：≥85%
- Scheduler模块：≥80%
- Quantization模块：≥80%

**覆盖率可视化展示方案**：

方案1：HTML报告可视化
- coverage html
- 输出：htmlcov/index.html
- 内容：模块覆盖率列表、文件覆盖率统计、源代码高亮显示、缺失行标记、分支覆盖率显示

方案2：覆盖率趋势追踪
- JSON数据存储：tests/st/coverage_history/coverage_*.json
- 覆盖率趋势脚本：tests/st/scripts/coverage_trend.py
- 生成覆盖率趋势图、输出覆盖率变化报告

方案3：覆盖率仪表板
- Markdown报告：tests/st/COVERAGE_REPORT.md
- ASCII图表：显示覆盖率百分比条形图

方案4：PR覆盖率评论
- GitHub Actions集成：.github/workflows/st_coverage.yml
- PR评论显示覆盖率变化

**理由**：
- pytest-cov是pytest标准覆盖率插件，已在requirements-dev.txt
- 命令行参数灵活控制覆盖率收集和报告生成
- 覆盖率门禁分级策略适配不同测试场景
- HTML报告可视化便于开发者理解覆盖率情况
- 覆盖率趋势追踪帮助发现覆盖率退化
- 参考coverage.py和pytest-cov官方最佳实践

### 21. CI/CD集成设计：复用现有workflow架构

**决策**：ST测试集成到现有vllm_ascend_test.yaml workflow，复用现有CANN container、vllm安装流程、覆盖率上传机制。

**现有CI架构分析**：

```
当前workflow结构：
┌────────────────────────────────────────────────────────────────┐
│  vllm_ascend_test.yaml                                         │
│                                                                │
│  jobs:                                                         │
│    lint ──▶ changes ──▶ ut ──▶ e2e ──▶ e2e-2-cards             │
│              │            │        │          │                │
│           outputs:      container NPU-1卡   NPU-2卡            │
│           e2e_tracker   CPU+CANN 真实硬件   真实硬件           │
│           ut_tracker                                            │
│                                                                │
│  changes filter机制：                                          │
│  • e2e_tracker: vllm_ascend/**, tests/e2e/**, csrc/**         │
│  • ut_tracker: tests/ut/**                                     │
│                                                                │
│  ❌ 缺失：无st_tracker                                          │
│  ❌ 缺失：无st job                                              │
└────────────────────────────────────────────────────────────────┘
```

**核心问题发现**：

1. **环境依赖假设错误**
   - 设计文档声称："CPU Mock可在无vllm环境执行"
   - 实际情况：源代码强依赖vllm和torch_npu
   - platform.py: from vllm.platforms import Platform
   - worker_v1.py: from vllm.config import VllmConfig
   - utils.py: import torch_npu（第29行）
   - 结论：ST测试必须在CANN container执行

2. **CI workflow未定义**
   - 无st_coverage.yml
   - 无st_tracker
   - 无双模式CI job设计

3. **覆盖率集成缺失**
   - codecov.yml只有patch/project
   - 无unittests/st_tests flags定义
   - 无分级门禁定义

**解决方案：扩展vllm_ascend_test.yaml**

```
新增workflow结构：
┌────────────────────────────────────────────────────────────────┐
│  vllm_ascend_test.yaml                                         │
│                                                                │
│  jobs:                                                         │
│    lint ──▶ changes ──▶ ut ──▶ st-cpu-mock ──▶ e2e ──▶ e2e-2   │
│              │            │        │            │        │      │
│           outputs:      container container    NPU-1卡   NPU-2  │
│           e2e_tracker  CPU+CANN CPU+CANN       真实硬件  真实   │
│           ut_tracker                           硬件             │
│           st_tracker                                            │
│                                                                │
│  新增st_tracker：                                              │
│  • tests/st/**                                                 │
│  • vllm_ascend/**                                              │
│  • csrc/**                                                     │
│                                                                │
│  新增st-cpu-mock job：                                         │
│  • runs-on: ubuntu-latest                                      │
│  • container: quay.io/ascend/cann:8.2.rc1-910b-...            │
│  • pytest tests/st/ --exec-mode=cpu_mock --cov                │
│  • 上传覆盖率：flags: st_tests                                 │
└────────────────────────────────────────────────────────────────┘
```

**changes filter更新**：

```yaml
outputs:
  e2e_tracker: ${{ steps.filter.outputs.e2e_tracker }}
  ut_tracker: ${{ steps.filter.outputs.ut_tracker }}
  st_tracker: ${{ steps.filter.outputs.st_tracker }}

steps:
  - uses: dorny/paths-filter@v3
    id: filter
    with:
      filters: |
        e2e_tracker:
          - '.github/workflows/vllm_ascend_test.yaml'
          - 'vllm_ascend/**'
          - 'csrc/**'
          - 'tests/e2e/**'
        ut_tracker:
          - 'tests/ut/**'
        st_tracker:
          - '.github/workflows/vllm_ascend_test.yaml'
          - 'vllm_ascend/**'
          - 'tests/st/**'
          - 'csrc/**'
```

**st-cpu-mock job设计**：

```yaml
st:
  needs: [lint, changes]
  name: st test (cpu mock)
  if: ${{ needs.lint.result == 'success' && needs.changes.outputs.st_tracker == 'true' }}
  runs-on: ubuntu-latest
  container:
    image: quay.io/ascend/cann:8.2.rc1-910b-ubuntu22.04-py3.11
    env:
      VLLM_LOGGING_LEVEL: ERROR
      VLLM_USE_MODELSCOPE: True
  strategy:
    matrix:
      vllm_version: [main, v0.10.0]
  steps:
    - name: Install packages
      run: |
        apt-get update -y
        apt-get install -y python3-pip git vim wget net-tools gcc g++ cmake

    - name: Checkout vllm-project/vllm repo
      uses: actions/checkout@v4
      with:
        repository: vllm-project/vllm
        ref: ${{ matrix.vllm_version }}
        path: ./vllm-empty

    - name: Install vllm-project/vllm from source
      working-directory: ./vllm-empty
      run: |
        VLLM_TARGET_DEVICE=empty python3 -m pip install .

    - name: Checkout vllm-project/vllm-ascend repo
      uses: actions/checkout@v4

    - name: Install vllm-project/vllm-ascend
      run: |
        python3 -m pip install -r requirements-dev.txt
        python3 -m pip install -v .

    - name: Run ST test
      env:
        VLLM_WORKER_MULTIPROC_METHOD: spawn
        TORCH_DEVICE_BACKEND_AUTOLOAD: 0
      run: |
        pytest -sv tests/st/ --exec-mode=cpu_mock \
          --cov=vllm_ascend \
          --cov-report=xml:st-coverage.xml \
          --cov-fail-under=80

    - name: Upload coverage to Codecov
      if: ${{ matrix.vllm_version == 'main' }}
      uses: codecov/codecov-action@v5
      env:
        CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
      with:
        flags: st_tests
        name: vllm-ascend-st
        files: st-coverage.xml
        verbose: true
```

**codecov.yml更新**：

```yaml
coverage:
  status:
    patch:
      default:
        target: 80%
    project:
      default:
        informational: true
      st_tests:
        target: 85%
        flags:
          - st_tests

flags:
  unittests:
    paths:
      - "tests/ut/"
  st_tests:
    paths:
      - "tests/st/"
```

**执行顺序设计**：

```
当前：lint → ut → e2e → e2e-2-cards

新增：lint → ut → st-cpu-mock → e2e → e2e-2-cards

  ┌────────┐   ┌────────┐   ┌────────────┐   ┌───────┐
  │  lint  │──▶│   ut   │──▶│st-cpu-mock │──▶│  e2e  │
  │ ubuntu │   │container│   │  container │   │  NPU  │
  │ ~1min  │   │ ~2min   │   │   ~5min    │   │~30min │
  └────────┘   └────────┘   └────────────┘   └───────┘

关键点：
• ST测试独立于E2E（可并行）
• ST测试先于E2E（快速门禁）
• ST测试≤5分钟
```

**理由**：
- 复用优先原则：复用现有CANN container、vllm安装流程、覆盖率上传机制
- 环境依赖现实：ST测试必须依赖CANN container（源代码强依赖torch_npu）
- 轻量执行原则：CPU Mock模式≤5分钟
- 按需触发原则：changes filter + st_tracker
- 覆盖率合并原则：UT+ST覆盖率合并上传Codecov

**替代方案考虑**：
- 独立workflow：创建st_coverage.yml（清晰职责，但增加workflow数量）
- 选择扩展workflow：复用优先，减少维护成本

## 风险与权衡

### Risk 1: Mock过度导致测试不真实

**风险**：过度使用Mock可能测试的是Mock代码而非真实模块协作。

**缓解**：
- 明确Mock边界：只Mock外部依赖（硬件、网络），保留模块间真实调用
- 定期验证ST测试与E2E测试结果一致性
- 代码评审时检查Mock合理性

### Risk 2: 测试数据与真实场景脱节

**风险**：预生成数据可能无法覆盖真实场景的边界情况。

**缓解**：
- 从E2E测试中提取真实数据样本，纳入ST测试数据集
- 支持随机生成数据，定期刷新测试数据种子
- 从缺陷案例中补充测试数据

### Risk 3: 测试维护成本高

**风险**：模块接口变更时，ST测试可能需大量修改。

**缓解**：
- 使用工厂模式封装测试数据生成，集中管理变更点
- 使用fixtures封装模块初始化逻辑，减少重复代码
- 测试代码同步评审，与业务代码变更同步更新
- **使用Mock工厂函数**，Mock对象变更时只需修改工厂函数（吸纳test_fused_ops.py模式）

### Risk 4: 测试用例数量难以把握

**风险**：不清楚应该编写多少ST用例，可能过多或过少。

**缓解**：
- 参考开发者测试规范：聚焦核心协作路径，每个协作路径至少覆盖正常、异常场景
- 从缺陷历史中识别高风险协作点，优先覆盖
- 定期度量ST测试有效性（变异测试）



### Risk 5: NPU真实模式测试不稳定

**风险**：NPU真实模式依赖硬件环境，可能因硬件故障、资源竞争、环境差异导致测试不稳定。

**缓解**：
- **NPU环境健康检查**：测试前验证NPU可用性、内存充足、设备正常
- **资源隔离策略**：NPU测试使用独立设备，避免资源竞争
- **失败重试机制**：NPU测试失败时自动重试（pytest-rerunfailures）
- **跳过机制**：NPU不可用时自动跳过NPU测试，不影响CPU Mock测试
- **硬件故障告警**：NPU测试失败时告警，及时排查硬件问题



### Risk 6: 三层解耦架构fixture依赖链过长

**风险**：三层解耦架构导致fixture依赖链过长（st_environment → model_pool → cached_model → test_runner），调试困难。

**缓解**：
- **fixture依赖可视化**：提供fixture_dependency_graph.py生成依赖图，便于调试
- **fixture日志记录**：每个fixture执行时记录日志，显示依赖链执行顺序
- **fixture失败定位**：fixture失败时显示完整依赖链和失败点
- **简化fixture设计**：对于简单测试用例，允许绕过三层架构直接使用简化fixture
- **文档说明fixture依赖**：在ST测试文档中说明三层fixture依赖链和执行顺序

### Risk 7: Mock模型与真实模型行为不一致

**风险**：create_mock_model创建的Mock模型可能与真实模型行为不一致，导致测试结果不可靠。

**缓解**：
- **spec限制Mock模型**：create_mock_model使用spec定义真实模型接口，确保接口一致
- **Mock模型行为验证**：定期验证Mock模型输出与真实模型输出结构一致（tensor shape、dtype等）
- **Mock模型更新机制**：真实模型接口变更时同步更新Mock模型
- **双模式对比测试**：对于关键测试用例，同时执行CPU Mock和NPU真实模式，对比结果

### Risk 8: 配置文件参数过多导致使用困难

**风险**：配置文件支持大量参数（batch_size、scenario、dtype、seq_len等），开发者可能不知道如何选择合适的参数组合。

**缓解**：
- **配置预设模板**：提供常见测试场景的配置预设模板（快速测试、深度测试、性能测试等）
- **参数建议文档**：配置编写指南文档中提供参数选择建议和示例
- **配置验证工具**：验证配置参数组合合理性，不合理组合给出警告
- **配置继承和覆盖**：允许子配置继承父配置并覆盖部分参数，简化配置定义




### 22. 模块集成问题拦截度量机制

**决策**：建立模块集成问题拦截度量机制，量化"提前拦截"的效果。

**度量维度定义**：

| 维度 | 定义 | 度量方法 |
|------|------|----------|
| 拦截时间 | 从问题引入到被ST测试发现的时间 | CI执行时间（分钟级） |
| 拦截位置 | 问题被发现的测试层级 | ST vs E2E vs 生产环境 |
| 问题类型 | 模块集成问题的分类 | 接口不兼容、数据流错误、状态不一致 |
| 拦截率 | ST测试拦截的集成问题占比 | ST拦截数 / 总集成问题数 |

**问题类型分类**：
```
模块集成问题分类：
┌─────────────────────────────────────────────────────────────────────────────┐
│  类型1: 接口不兼容                                                           │
│  - 插件组件未实现vllm接口                                                    │
│  - 接口参数类型不匹配                                                        │
│  - 接口返回值格式不正确                                                      │
│  - 检测方法: verify_plugin_interface_compatibility                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  类型2: 数据流错误                                                           │
│  - 模块间数据传递格式错误                                                    │
│  - 数据转换丢失精度                                                          │
│  - 数据维度不一致                                                            │
│  - 检测方法: Mock验证策略 + tensor shape检查                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  类型3: 状态不一致                                                           │
│  - 模块间状态传递丢失                                                        │
│  - 资源清理不完整                                                            │
│  - 并发执行状态冲突                                                          │
│  - 检测方法: setUp/tearDown + STRunner资源清理验证                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  类型4: 调度协作错误                                                         │
│  - Worker与Scheduler调度顺序错误                                            │
│  - 任务分发逻辑错误                                                          │
│  - 优先级处理不正确                                                          │
│  - 检测方法: Mock验证调用顺序 + 调度流程测试                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

**拦截效果目标**：
- 拦截时间目标：≤5分钟（CPU Mock模式）
- 拦截率目标：≥70%（ST测试应拦截70%以上的集成问题）
- 问题定位时间：≤10分钟（从测试失败到定位具体交互点）

**理由**：
- 量化"提前拦截"的效果，验证ST测试有效性
- 问题类型分类帮助针对性设计测试用例
- 拦截率度量帮助评估ST测试覆盖程度
- 参考软件测试度量最佳实践（缺陷发现时间、缺陷定位效率）

### 23. 重构时模块交互稳定性保护机制

**决策**：建立重构时模块交互稳定性保护机制，包括CI触发策略、快速回归测试、接口变更检测。

**重构场景定义**：
```
重构场景分类：
┌─────────────────────────────────────────────────────────────────────────────┐
│  场景1: 模块内部重构                                                         │
│  - 重构模块内部实现逻辑                                                      │
│  - 不改变模块对外接口                                                        │
│  - ST测试应全部通过（验证接口兼容性）                                        │
│                                                                             │
│  场景2: 模块接口变更                                                         │
│  - 改变模块对外接口（参数、返回值）                                          │
│  - 可能影响协作模块                                                          │
│  - ST测试可能失败（接口不兼容）                                              │
│  - 需要同步更新协作模块的测试用例                                            │
│                                                                             │
│  场景3: 模块间协作流程变更                                                   │
│  - 改变模块间调用顺序或数据流                                                │
│  - 不改变单个模块接口                                                        │
│  - ST测试可能失败（调用顺序错误）                                            │
│  - 需要同步更新协作流程测试用例                                              │
│                                                                             │
│  场景4: 新增模块或删除模块                                                   │
│  - 新增协作模块或删除现有模块                                                │
│  - 影响协作关系                                                              │
│  - 需要新增或删除对应的ST测试                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**稳定性保护目标**：
- 重构后ST测试通过率：≥95%（核心模块重构）
- 接口变更自动检测覆盖率：≥80%
- 快速回归测试执行时间：≤1分钟

**理由**：
- 重构是常见开发场景，ST测试应提供稳定性保护
- 快速回归测试减少重构等待时间
- 接口变更检测提前预警协作模块影响
- 稳定性度量帮助评估重构风险

### 24. 测试结果可追溯与失败定位机制

**决策**：建立测试结果可追溯与失败定位机制，包括日志记录、覆盖率追踪、失败定位工具。

**可追溯架构**：
```
可追溯架构：
┌─────────────────────────────────────────────────────────────────────────────┐
│  Level 1: 测试执行日志                                                       │
│  - pytest输出日志（--tb=long详细错误）                                       │
│  - 测试执行时间记录                                                          │
│  - Mock调用记录                                                              │
│  - 存储位置: tests/st/logs/<timestamp>/                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Level 2: 覆盖率历史追踪                                                     │
│  - 每次CI执行覆盖率数据                                                      │
│  - 覆盖率趋势变化                                                            │
│  - 覆盖率下降告警                                                            │
│  - 存储位置: tests/st/coverage_history/                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Level 3: 失败定位工具                                                       │
│  - 从测试失败定位到具体模块交互点                                            │
│  - 显示fixture依赖链                                                         │
│  - 显示Mock调用链                                                            │
│  - 输出: 失败分析报告                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**可追溯目标**：
- 日志覆盖率：100%（所有ST测试执行都有日志）
- 失败定位时间：≤10分钟（从失败到定位具体交互点）
- 覆盖率历史保存：≥30天（可追溯最近30天覆盖率变化）

**理由**：
- 可追溯机制帮助事后分析问题原因
- 失败定位工具减少人工排查时间
- 覆盖率历史追踪发现覆盖率退化
- pytest集成实现自动追踪，无需额外配置

### 25. 模块间接口与数据流的边界定义

**决策**：明确定义"模块间接口"和"模块间数据流"的区别，以及对应的测试方法。

**边界定义**：
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         模块间接口 vs 数据流边界                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  模块间接口:                                                                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│  定义: 模块对外暴露的调用入口（函数签名、类方法）                             │
│  关注点: 接口是否兼容、参数是否匹配、返回值是否正确                           │
│  验证方法:                                                                  │
│  ├─ verify_plugin_interface_compatibility (接口兼容性验证)                  │
│  ├─ spec限制Mock (接口规范验证)                                             │
│  ├─ assert_called_with (接口调用参数验证)                                   │
│  └─ 返回值类型检查                                                          │
│                                                                             │
│  模块间数据流:                                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│  定义: 模块间传递的数据内容和格式（tensor、config、state）                    │
│  关注点: 数据格式是否正确、数据是否完整、数据流向是否正确                     │
│  验证方法:                                                                  │
│  ├─ tensor shape检查 (数据格式验证)                                         │
│  ├─ 数据内容检查 (数据完整性验证)                                           │
│  ├─ 数据流追踪 (数据流向验证)                                               │
│  └─ 状态一致性检查                                                          │
│                                                                             │
│  边界区别:                                                                   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  ├─ 接口关注"调用是否正确"，数据流关注"数据是否正确"                         │
│  ├─ 接口是静态规范，数据流是动态内容                                         │
│  ├─ 接口不兼容导致调用失败，数据流错误导致结果错误                            │
│  └─ 接口测试用spec验证，数据流测试用tensor验证                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**理由**：
- 明确边界帮助开发者选择正确的测试方法
- 分类命名规范帮助快速识别测试类型
- 两种验证方法互补，全面覆盖模块协作

### 26. 插件内部协作与vllm集成的边界定义

**决策**：明确定义"插件内部协作"和"vllm集成"的区别，以及ST测试的职责边界。

**边界定义**：
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         插件内部协作 vs vllm集成边界                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  插件内部协作 (ST测试职责):                                                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│  定义: vllm-ascend插件内部模块间的协作关系                                   │
│  范围:                                                                      │
│  ├─ NPUWorker与AscendModelRunner协作                                       │
│  ├─ AscendScheduler与NPUWorker协作                                         │
│  ├─ AscendAttention与AscendAttentionMask协作                               │
│  ├─ AscendQuantizer与模型加载协作                                           │
│  └─ 不包含vllm核心组件                                                      │
│  测试方法:                                                                  │
│  ├─ 使用真实插件组件（NPUWorker、AscendScheduler等）                        │
│  ├─ Mock vllm核心依赖                                                       │
│  ├─ Mock torch_npu硬件依赖                                                  │
│  └─ 验证插件内部数据流和调用链                                               │
│                                                                             │
│  vllm集成 (E2E测试职责):                                                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  定义: vllm-ascend插件与vllm核心的完整集成                                   │
│  范围:                                                                      │
│  ├─ 插件注入vllm（register()、check_and_update_config()）                  │
│  ├─ 插件替换vllm核心组件（Worker、Scheduler、AttentionBackend）              │
│  ├─ 完整推理流程（LLM(model).generate()）                                  │
│  ├─ 端到端正确性验证                                                        │
│  └─ 需要真实vllm环境和模型                                                  │
│                                                                             │
│  接口兼容性验证 (ST测试职责，边界模糊):                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  定义: 验证插件组件实现了vllm接口规范                                         │
│  性质: 验证插件与vllm接口的静态兼容性（不涉及运行时集成）                      │
│  归属理由:                                                                  │
│  ├─ 验证的是"插件是否实现了接口"（插件内部职责）                              │
│  ├─ 不依赖vllm运行时（不需要真实vllm环境）                                   │
│  ├─ 可以用spec定义接口（静态验证）                                          │
│  └─ 这是插件组件自身的质量保障，属于ST测试                                   │
│                                                                             │
│  边界判断规则:                                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│  ├─ 测试是否依赖真实vllm？ → 是=E2E，否=ST                                  │
│  ├─ 测试是否验证完整推理流程？ → 是=E2E，否=ST                              │
│  ├─ 测试是否只涉及插件组件？ → 是=ST                                        │
│  ├─ 测试是否验证静态接口规范？ → 是=ST（接口兼容性）                         │
│  └─ 测试是否验证运行时集成行为？ → 是=E2E                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**职责边界对照表**：
| 测试内容 | ST职责 | E2E职责 |
|---------|--------|---------|
| NPUWorker内部逻辑 | ✅ | ❌ |
| NPUWorker与AscendModelRunner协作 | ✅ | ❌ |
| NPUWorker与vllm.Worker接口兼容 | ✅ | ❌ |
| NPUWorker替换vllm.Worker后的完整推理 | ❌ | ✅ |
| 插件注入vllm流程 | ❌ | ✅ |
| 完整推理流程 | ❌ | ✅ |

**理由**：
- 明确边界避免ST测试越界到E2E职责
- 接口兼容性验证归属ST的理由明确
- 职责边界表帮助开发者判断测试归属

### 27. 首批测试用例定义与优先级排序

**决策**：明确定义"首批ST测试用例"的范围和模块优先级。

**首批定义**：
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         首批ST测试用例定义                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  首批范围:                                                                   │
│  ├─ 核心模块: Worker、Scheduler、Attention                                 │
│  ├─ 测试用例数量: 15个核心协作场景                                          │
│  ├─ 执行时间目标: ≤5分钟（全量）                                            │
│  └─ 完成时间: Phase 2（第3-4周）                                            │
│                                                                             │
│  后续扩展（Phase 3）:                                                        │
│  ├─ 扩展模块: Quantization、Distributed、Sample                            │
│  ├─ 测试用例数量: +10个协作场景                                             │
│  ├─ 执行时间目标: ≤10分钟（全量+扩展）                                      │
│  └─ 完成时间: Phase 3（第5周）                                              │
│                                                                             │
│  首批交付标准:                                                               │
│  ├─ 每个核心模块至少5个测试用例                                             │
│  ├─ 每个协作场景覆盖正常+异常场景                                           │
│  ├─ 所有测试用例有Docstring                                                 │
│  ├─ CPU Mock模式全部通过                                                    │
│  └─ 覆盖率≥80%                                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**模块优先级排序**：

```
优先级矩阵：
┌─────────────────────────────────────────────────────────────────────────────┐
│  P0 (首批核心，必须完成):                                                    │
│  ├─ Worker模块                                                              │
│  │   ├─ 理由: Worker是插件核心，与多个模块协作                              │
│  │   ├─ 协作场景: Worker-ModelRunner、Worker-InputBatch、Worker-Proposer    │
│  │   ├─ 测试用例数: 6个                                                     │
│  │   └─ 预计工作量: 1周                                                     │
│  │                                                                          │
│  ├─ Scheduler模块                                                           │
│  │   ├─ 理由: Scheduler控制推理流程，与Worker协作                           │
│  │   ├─ 协作场景: Scheduler-Worker、Scheduler-Config                       │
│  │   ├─ 测试用例数: 5个                                                     │
│  │   └─ 预计工作量: 1周                                                     │
│  │                                                                          │
│  └─ Attention模块                                                           │
│  │   ├─ 理由: Attention是推理核心算子，多个组件协作                         │
│  │   ├─ 协作场景: Attention-Mask、Attention-MLA、Attention-Torchair        │
│  │   ├─ 测试用例数: 4个                                                     │
│  │   └─ 预计工作量: 1周                                                     │
│                                                                             │
│  P1 (后续扩展，Phase 3):                                                     │
│  ├─ Quantization模块 (3个用例)                                              │
│  ├─ Distributed模块 (3个用例)                                               │
│  └─ Sample模块 (4个用例)                                                    │
│                                                                             │
│  P2 (长期优化，Phase 5):                                                     │
│  ├─ 性能测试 (NPU真实模式)                                                  │
│  ├─ 精度测试 (NPU真实模式)                                                  │
│  └─ 周边组件集成测试                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**首批测试用例清单**：

| 模块 | 协作场景 | 测试用例名 | 优先级 |
|------|---------|-----------|--------|
| Worker | Worker-ModelRunner初始化 | test_worker_init_model_runner | P0 |
| Worker | Worker-ModelRunner调度 | test_worker_schedule_to_runner | P0 |
| Worker | Worker-InputBatch数据流 | test_worker_input_batch_data_flow | P0 |
| Worker | Worker-Proposer协作 | test_worker_mt_proposer_collaboration | P0 |
| Worker | Worker异常处理 | test_worker_error_handling | P0 |
| Worker | Worker接口兼容性 | test_worker_interface_compatibility | P0 |
| Scheduler | Scheduler-Worker协作 | test_scheduler_worker_collaboration | P0 |
| Scheduler | Scheduler-Config集成 | test_scheduler_config_integration | P0 |
| Scheduler | Scheduler多批次调度 | test_scheduler_multi_batch_scheduling | P0 |
| Scheduler | Scheduler解码调度 | test_scheduler_decode_scheduling | P0 |
| Scheduler | Scheduler接口兼容性 | test_scheduler_interface_compatibility | P0 |
| Attention | Attention-Mask协作 | test_attention_mask_integration | P0 |
| Attention | Attention-MLA协作 | test_attention_mla_integration | P0 |
| Attention | Attention不同序列长度 | test_attention_seq_len_variations | P0 |
| Attention | Attention接口兼容性 | test_attention_interface_compatibility | P0 |

**理由**：
- 首批定义明确避免"首批"概念模糊
- 模块优先级基于协作复杂度和重要性排序
- 测试用例清单具体可执行
- 交付标准量化验收





