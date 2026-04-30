## ADDED Requirements

### Requirement: Attention与AttentionMask集成测试验证mask生成

ST测试 SHALL 验证Attention_v1与AttentionMask的协作，包括：
- AttentionMask正确生成注意力mask
- Attention将mask应用到注意力计算
- 不同mask类型（causal、sliding_window）的正确性

#### Scenario: AttentionMask生成因果mask

- **WHEN** Attention请求生成causal mask（seq_len=16）
- **THEN** AttentionMask返回因果mask tensor，形状为[16, 16]，下三角为True

#### Scenario: Attention应用mask到注意力计算

- **WHEN** Attention计算注意力分数，使用AttentionMask提供的mask
- **THEN** Attention正确应用mask，mask区域注意力分数为0或-inf

### Requirement: Attention与MLA集成测试验证多头注意力

ST测试 SHALL 验证Attention_v1与MLA_v1（Multi-Latent Attention）的协作，包括：
- MLA正确压缩KV cache
- Attention读取压缩的KV cache执行注意力计算
- MLA解压缩后的注意力正确性验证

#### Scenario: MLA压缩KV cache

- **WHEN** Attention请求MLA压缩KV cache（key_states, value_states）
- **THEN** MLA返回压缩后的latent vector，维度小于原始KV维度

#### Scenario: Attention读取压缩KV cache执行计算

- **WHEN** Attention使用MLA压缩的latent vector计算注意力
- **THEN** Attention正确解压缩latent vector，执行多头注意力计算

### Requirement: Attention不同序列长度集成测试

ST测试 SHALL 验证Attention处理不同序列长度的集成正确性，包括：
- 短序列注意力计算（seq_len < 128）
- 长序列注意力计算（seq_len > 1024）
- 动态序列长度变化时的内存管理

#### Scenario: Attention处理短序列

- **WHEN** Attention接收短序列输入（seq_len=16, num_heads=8）
- **THEN** Attention正确计算注意力，输出形状为[16, 8, 16]

#### Scenario: Attention处理长序列

- **WHEN** Attention接收长序列输入（seq_len=2048, num_heads=32）
- **THEN** Attention正确计算注意力，输出形状为[2048, 32, 2048]

### Requirement: Attention与Torchair集成测试

ST测试 SHALL 验证Attention_v1_torchair与TorchairWorker的协作，包括：
- TorchairAttention正确编译为图模式
- Attention在图模式下的执行正确性
- 图模式与eager模式结果一致性

#### Scenario: TorchirAttention编译为图模式

- **WHEN** TorchairAttention初始化，启用torchair_graph_config
- **THEN** Attention编译为静态图，支持图模式执行

#### Scenario: 图模式与eager模式结果一致性

- **WHEN** 同样的输入分别使用图模式和eager模式执行Attention
- **THEN** 两种模式的输出tensor数值一致（误差<1e-5）