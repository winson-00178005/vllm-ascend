## ADDED Requirements

### Requirement: Sampler与RejectionSampler集成测试验证推测解码

ST测试 SHALL 验证Sampler与RejectionSampler的推测解码（Speculative Decoding）协作，包括：
- Sampler生成目标模型token概率
- RejectionSampler执行推测token验证
- RejectionSampler返回验证结果和接受token数量

#### Scenario: Sampler生成目标模型token概率

- **WHEN** Sampler接收模型输出logits，执行采样
- **THEN** Sampler返回目标模型token概率分布（probs）

#### Scenario: RejectionSampler验证推测token

- **WHEN** RejectionSampler接收目标probs和推测draft_probs
- **THEN** RejectionSampler执行验证，返回接受token数量和拒绝原因

### Requirement: RejectionSampler不同验证场景集成测试

ST测试 SHALL 验证RejectionSampler处理不同验证场景的集成正确性，包括：
- 全接受场景：所有推测token都接受
- 全拒绝场景：所有推测token都拒绝
- 部分接受场景：部分推测token接受，部分拒绝

#### Scenario: RejectionSampler全接受推测token

- **WHEN** RejectionSampler验证推测token，目标probs与draft_probs完全匹配
- **THEN** RejectionSampler返回接受全部推测token，无需重新采样

#### Scenario: RejectionSampler部分接受推测token

- **WHEN** RejectionSampler验证推测token，前3个匹配，第4个不匹配
- **THEN** RejectionSampler返回接受前3个token，第4个token重新采样

### Requirement: Sampler与Temperature集成测试

ST测试 SHALL 验证Sampler与Temperature参数的协作，包括：
- Temperature参数影响采样概率分布
- 不同Temperature值（temperature=0, 0.5, 1.0）的采样行为
- Temperature与Top-P、Top-K采样的协作

#### Scenario: Temperature=0确定性采样

- **WHEN** Sampler使用temperature=0采样
- **THEN** Sampler返回概率最高的token，确定性采样

#### Scenario: Temperature=1.0随机采样

- **WHEN** Sampler使用temperature=1.0采样
- **THEN** Sampler按原始概率分布随机采样，输出随机token

### Requirement: Sampler与Batch集成测试

ST测试 SHALL 验证Sampler处理多batch采样的集成正确性，包括：
- Sampler并行处理多个batch的采样
- Batch采样结果的独立性和正确性
- Sampler处理不同batch大小的采样

#### Scenario: Sampler并行采样多batch

- **WHEN** Sampler接收2个batch的logits，执行并行采样
- **THEN** Sampler返回2个batch的采样token，batch间结果独立

#### Scenario: Sampler处理不同batch大小

- **WHEN** Sampler处理batch_size=1和batch_size=32的采样
- **THEN** Sampler正确处理不同batch大小，返回对应数量的采样token

### Requirement: Sample模块与Worker集成测试

ST测试 SHALL 验证Sample模块与Worker的协作，包括：
- Worker调用Sampler执行采样
- Worker处理Sampler返回的采样结果
- Worker将采样结果用于下一轮decode

#### Scenario: Worker调用Sampler采样

- **WHEN** Worker执行推理后，调用Sampler采样模型输出
- **THEN** Sampler返回采样token，Worker准备下一轮decode输入

#### Scenario: Worker处理RejectionSampler结果

- **WHEN** Worker使用推测解码，RejectionSampler返回接受token数量
- **THEN** Worker根据接受数量调整下一轮decode的输入token序列