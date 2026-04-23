# ST Framework NPU 验证指南

## 一、环境检查

### 1.1 基础环境

```bash
# 检查 Python 版本
python --version

# 检查必要的包
pip list | grep -E "pytest|pyyaml"

# 检查 NPU 驱动
npu-smi info
```

### 1.2 vllm-ascend 环境

```bash
# 检查 vllm-ascend 安装
pip show vllm-ascend

# 检查 WORKER_MULTIPROC_METHOD 环境变量支持
echo $VLLM_WORKER_MULTIPROC_METHOD
```

---

## 二、框架功能验证（无需模型）

### 2.1 运行框架验证脚本

```bash
cd /path/to/vllm-ascend

python tests/e2e/st/test_framework_validation.py
```

**预期输出：**
```
============================================================
ST Framework 装饰器功能验证
============================================================

[1] 装饰器行为测试
----------------------------------------
  [PASS] require_scene: 场景匹配时正常执行
  [PASS] require_scene: 场景不匹配时跳过
  [PASS] require_model: 模型匹配时正常执行
  [PASS] require_model: 模型不匹配时跳过
  [PASS] require_scene_and_model: 条件匹配时正常执行
  [PASS] require_scene_and_model: 场景不匹配时跳过
  [PASS] require_scene_and_model: 模型不匹配时跳过

[2] 兼容层功能测试
----------------------------------------
  [PASS] get_models_for_current_scene: 正确保留支持的模型
  [PASS] get_models_for_current_scene: 正确过滤不支持的模型
  [PASS] get_models_for_current_scene: 4卡正确过滤Qwen3-8B
  [PASS] get_models_for_current_scene: 4卡正确保留30B和235B
  [PASS] get_models_for_current_scene: 310P场景正确过滤

[3] 场景管理器行为测试
----------------------------------------
  [PASS] MULTICARD_2Cards 环境变量设置正确
  [PASS] MULTICARD_4Cards 环境变量设置正确
  [PASS] 环境变量正确恢复

============================================================
Total: 15 | Passed: 15 | Failed: 0 | Skipped: 0
============================================================

[SUCCESS] 所有验证通过!
```

### 2.2 pytest 测试发现

```bash
# 测试发现（无需运行）
export VLLM_TEST_SCENE="SINGLECARD"
pytest tests/e2e/st/testcases/ --collect-only -q

# 预期：发现 25+ 个测试用例
```

### 2.3 场景切换验证

```bash
# 单卡场景
export VLLM_TEST_SCENE="SINGLECARD"
python -c "from tests.e2e.st.framework.scene_manager import SceneManager; import os; m=SceneManager.get_instance(); print('Scene:', m.get_current_scene()); print('TP_SIZE:', m.get_scene_info().tp_size); print('ASCEND_RT_VISIBLE_DEVICES:', os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))"

# 2卡场景
export VLLM_TEST_SCENE="MULTICARD_2Cards"
python -c "from tests.e2e.st.framework.scene_manager import SceneManager; import os; m=SceneManager.get_instance(); print('Scene:', m.get_current_scene()); print('TP_SIZE:', m.get_scene_info().tp_size); print('ASCEND_RT_VISIBLE_DEVICES:', os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))"

# 4卡场景
export VLLM_TEST_SCENE="MULTICARD_4Cards"
python -c "from tests.e2e.st.framework.scene_manager import SceneManager; import os; m=SceneManager.get_instance(); print('Scene:', m.get_current_scene()); print('TP_SIZE:', m.get_scene_info().tp_size); print('ASCEND_RT_VISIBLE_DEVICES:', os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))"
```

---

## 三、场景过滤验证（无需模型）

### 3.1 单卡场景下测试收集

```bash
export VLLM_TEST_SCENE="SINGLECARD"

# 应该跳过多卡测试
pytest tests/e2e/st/testcases/ -v --tb=no 2>&1 | grep -E "PASSED|SKIPPED" | head -30
```

### 3.2 2卡场景下测试收集

```bash
export VLLM_TEST_SCENE="MULTICARD_2Cards"

# 应该跳过4卡测试
pytest tests/e2e/st/testcases/ -v --tb=no 2>&1 | grep -E "PASSED|SKIPPED" | head -30
```

---

## 四、E2E 推理验证（需要模型）

### 4.1 模型准备

```bash
# 检查模型路径配置
ls -la tests/e2e/models/

# 或使用 HuggingFace 模型（需要网络）
# Qwen/Qwen3-8B
# Qwen/Qwen3-30B-A3B
```

### 4.2 单卡 E2E 测试

```bash
export VLLM_TEST_SCENE="SINGLECARD"
export VLLM_TEST_MODEL="Qwen/Qwen3-8B"

pytest -sv tests/e2e/st/testcases/test_basic.py::TestBasicInference::test_single_token_generation --tb=short
```

**预期：** PASSED

### 4.3 多卡 E2E 测试

```bash
# 2卡测试
export VLLM_TEST_SCENE="MULTICARD_2Cards"
export VLLM_TEST_MODEL="Qwen/Qwen3-30B-A3B"

pytest -sv tests/e2e/st/testcases/test_basic.py::TestBasicInference::test_multicard_2cards_basic --tb=short

# 4卡测试
export VLLM_TEST_SCENE="MULTICARD_4Cards"
export VLLM_TEST_MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"

pytest -sv tests/e2e/st/testcases/test_basic.py::TestBasicInference::test_multicard_4cards_basic --tb=short
```

### 4.4 量化测试

```bash
export VLLM_TEST_SCENE="SINGLECARD"
export VLLM_TEST_MODEL="vllm-ascend/Qwen3-0.6B-W8A8"

pytest -sv tests/e2e/st/testcases/test_basic.py::TestQuantization::test_w8a8_quantization_singlecard --tb=short
```

---

## 五、验证检查清单

```
□ 框架验证脚本 15 PASS
□ pytest 测试发现正常
□ 场景切换环境变量正确
□ 单卡测试 PASSED
□ 2卡测试 PASSED (多卡机器)
□ 4卡测试 PASSED (多卡机器)
□ 310P测试 PASSED (310P机器)
□ 量化测试 PASSED
```

---

## 六、常见问题

### Q1: pytest 找不到模块

```bash
# 设置 PYTHONPATH
export PYTHONPATH=/path/to/vllm-ascend:$PYTHONPATH
```

### Q2: 缺少 huggingface_hub

```bash
pip install huggingface_hub
```

### Q3: 模型加载失败

```bash
# 检查模型路径
export VLLM_MODEL_PATH=/path/to/model
# 或使用 HuggingFace
export HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## 七、验证结果记录

| 验证项 | 状态 | 备注 |
|--------|------|------|
| 框架验证脚本 | □ PASS / □ FAIL | |
| pytest 发现 | □ PASS / □ FAIL | 发现 N 个测试 |
| 场景切换 | □ PASS / □ FAIL | |
| 单卡 E2E | □ PASS / □ FAIL | |
| 2卡 E2E | □ PASS / □ FAIL | |
| 4卡 E2E | □ PASS / □ FAIL | |
| 量化 E2E | □ PASS / □ FAIL | |

**总体结论：** □ 通过 / □ 未通过
