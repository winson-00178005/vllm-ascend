## ADDED Requirements

### Requirement: Scheduler与ScheduleConfig集成测试验证调度策略

ST测试 SHALL 验证Scheduler与ScheduleConfig的协作，包括：
- ScheduleConfig正确配置调度参数（max_num_seqs、max_num_batched_tokens等）
- Scheduler读取ScheduleConfig参数执行调度
- 配置变更时Scheduler动态调整调度策略

#### Scenario: ScheduleConfig配置调度参数

- **WHEN** ScheduleConfig初始化（max_num_seqs=256, max_num_batched_tokens=2048）
- **THEN** ScheduleConfig返回配置对象，Scheduler读取配置参数

#### Scenario: Scheduler根据配置参数调度请求

- **WHEN** Scheduler调度推理请求，使用ScheduleConfig中的max_num_seqs限制
- **THEN** Scheduler正确限制并发序列数量，不超过max_num_seqs

### Requirement: Scheduler与Worker集成测试验证任务调度

ST测试 SHALL 验证Scheduler与Worker_v1的任务调度协作，包括：
- Scheduler将调度请求发送到Worker
- Worker执行请求并返回状态
- Scheduler处理Worker返回的执行结果

#### Scenario: Scheduler调度请求到Worker执行

- **WHEN** Scheduler调度推理请求，调用Worker.execute_model()
- **THEN** Worker接收请求参数，执行推理，返回输出结果

#### Scenario: Scheduler处理Worker执行失败

- **WHEN** Worker执行推理失败（例如：内存不足）
- **THEN** Scheduler接收失败状态，重新调度请求或返回错误

### Requirement: Scheduler多批次调度集成测试

ST测试 SHALL 验证Scheduler调度多个批次的集成正确性，包括：
- Scheduler优先级调度策略（优先级、arrival time）
- 多批次调度的公平性和效率
- Scheduler处理批次抢占和恢复

#### Scenario: Scheduler按优先级调度批次

- **WHEN** Scheduler接收两个批次请求，优先级分别为high和low
- **THEN** Scheduler优先调度high优先级批次，随后调度low优先级批次

#### Scenario: Scheduler处理批次抢占

- **WHEN** Scheduler正在调度低优先级批次，高优先级批次到达
- **THEN** Scheduler抢占低优先级批次，优先调度高优先级批次

### Requirement: Scheduler连续解码调度集成测试

ST测试 SHALL 验证Scheduler调度连续解码（continuous decoding）的集成正确性，包括：
- Scheduler处理prefill和decode阶段调度
- 连续解码的KV cache管理调度
- Scheduler处理解码阶段的序列终止

#### Scenario: Scheduler调度prefill阶段

- **WHEN** Scheduler调度新请求的prefill阶段
- **THEN** Scheduler分配KV cache，调度到Worker执行prefill推理

#### Scenario: Scheduler调度decode阶段

- **WHEN** Prefill完成，Scheduler调度decode阶段
- **THEN** Scheduler使用已分配的KV cache，调度decode推理，直到序列终止