## ADDED Requirements

### Requirement: ST测试框架提供测试基类

ST测试框架 SHALL 提供统一的测试基类（TestSTBase），支持：
- 自动初始化测试环境（Mock外部依赖）
- 提供测试上下文管理（配置、日志）
- 支持unittest.TestCase和pytest混合风格

#### Scenario: 测试基类自动Mock外部依赖

- **WHEN** ST测试类继承TestSTBase基类
- **THEN** 测试环境自动Mock torch_npu、CANN等硬件依赖

#### Scenario: 测试基类提供配置管理

- **WHEN** 测试用例需要访问测试配置
- **THEN** 测试基类提供get_test_config()方法返回标准配置对象

### Requirement: ST测试框架提供fixtures系统

ST测试框架 SHALL 提供pytest fixtures系统，包括：
- 全局fixtures（conftest.py）：环境初始化、Mock管理
- 模块fixtures（模块级conftest.py）：模块特定测试数据、对象初始化
- fixtures支持自动清理和资源释放

#### Scenario: 全局fixtures自动初始化测试环境

- **WHEN** pytest启动执行ST测试
- **THEN** 全局fixtures自动初始化Mock环境和测试配置

#### Scenario: 模块fixtures提供模块特定测试对象

- **WHEN** 测试用例声明使用worker_fixtures
- **THEN** fixtures自动提供Mock的Worker对象和相关依赖

### Requirement: ST测试框架提供Mock工具类

ST测试框架 SHALL 提供Mock工具类（mock_utils.py），支持：
- 模块级Mock：Mock torch_npu、hccl等外部依赖
- 数据Mock：生成预定义的tensor、配置对象
- Mock对象验证：验证Mock调用次数、参数

#### Scenario: Mock工具生成标准tensor数据

- **WHEN** 测试用例调用generate_mock_tensor(shape=(32, 64))
- **THEN** Mock工具返回预生成的符合形状要求的tensor对象

#### Scenario: Mock工具验证模块调用

- **WHEN** 测试用例调用verify_mock_calls(mock_obj, expected_calls=3)
- **THEN** Mock工具验证mock_obj被调用3次，返回验证结果

### Requirement: ST测试框架提供数据生成器

ST测试框架 SHALL 提供数据生成器（data_generator.py），支持：
- 预生成常用tensor形状（支持多种数据类型）
- 预生成常用配置模板（VllmConfig、AscendConfig等）
- 支持随机生成数据（可配置种子保证可重复性）

#### Scenario: 数据生成器生成预定义tensor

- **WHEN** 测试用例调用DataGenerator.get_standard_tensor("small_batch")
- **THEN** 数据生成器返回预定义的小批次tensor（例如：[1, 16]）

#### Scenario: 数据生成器生成配置对象

- **WHEN** 测试用例调用DataGenerator.get_vllm_config("default")
- **THEN** 数据生成器返回默认配置的VllmConfig对象

### Requirement: ST测试框架提供配置工厂

ST测试框架 SHALL 提供配置工厂（config_factory.py），支持：
- 创建各种配置对象（VllmConfig、ModelConfig、ParallelConfig等）
- 支持配置参数自定义（batch_size、tensor_parallel_size等）
- 支持配置验证和合法性检查

#### Scenario: 配置工厂创建自定义VllmConfig

- **WHEN** 测试用例调用ConfigFactory.create_vllm_config(batch_size=32)
- **THEN** 配置工厂返回batch_size=32的VllmConfig对象

#### Scenario: 配置工厂验证配置合法性

- **WHEN** 测试用例调用ConfigFactory.validate_config(config_obj)
- **THEN** 配置工厂检查配置合法性，返回验证结果

### Requirement: ST测试框架支持pytest执行和覆盖率统计

ST测试框架 SHALL 支持pytest执行，包括：
- 单模块测试执行
- 全量ST测试执行
- 覆盖率统计和报告生成
- CI门禁集成（执行时间≤5分钟）

#### Scenario: pytest执行单模块ST测试

- **WHEN** 执行命令pytest tests/st/worker/ -v
- **THEN** pytest仅执行Worker模块ST测试，输出测试结果和详细日志

#### Scenario: pytest生成覆盖率报告

- **WHEN** 执行命令pytest tests/st/ --cov=vllm_ascend --cov-report=xml
- **THEN** pytest生成覆盖率报告文件coverage.xml

#### Scenario: ST测试执行时间满足CI门禁要求

- **WHEN** 执行全量ST测试（tests/st/）
- **THEN** 测试执行时间不超过5分钟