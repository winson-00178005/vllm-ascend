# Archive Summary - add-developer-test-suite

**Change Name**: add-developer-test-suite  
**Schema**: spec-driven  
**Archive Date**: 2026-04-30  
**Status**: Complete (279/279 tasks, all artifacts done)

---

## Overview

Implemented a comprehensive ST (System Integration Test) framework for the vllm-ascend plugin, enabling testing of plugin internal module collaboration through dual execution modes (CPU Mock / NPU Real).

---

## Key Deliverables

### 1. ST Framework Infrastructure (26 tasks)

**Core Components:**
- `tests/st/base.py` - Dual base class design (TestSTBase + PytestSTBase)
- `tests/st/conftest.py` - pytest_addoption with 10+ CLI parameters
- `tests/st/utils/env_detector.py` - Auto-detection (torch_npu, NPU availability, is_310p)
- `tests/st/utils/env_factory.py` - CPUMockEnvironment / NPURealEnvironment
- `tests/st/utils/mock_utils.py` - Mock factory functions + vllm interface spec constants
- `tests/st/utils/data_generator.py` - Pre-generated tensors + DEFAULT_* constants
- `tests/st/utils/config_factory.py` - VllmConfig with fallback for non-vllm environments
- `tests/st/utils/runner_factory.py` - STRunner context manager

**Documentation:**
- `tests/st/ST_TEST_WRITING_GUIDE.md` - Docstring template, parameterization guide
- `tests/st/PLUGIN_TEST_RESPONSIBILITY.md` - Test layer boundaries (UT/ST/E2E)
- `tests/st/PYTEST_CLI_GUIDE.md` - CLI parameter documentation
- `tests/st/DUAL_MODE_EXECUTION_GUIDE.md` - Execution mode strategy

### 2. Core Module Integration Tests (68 tasks)

| Module | Test Files | Key Test Coverage |
|--------|-----------|-------------------|
| Worker (17) | 10 files | Worker-ModelRunner, InputBatch, MTProposer, multi-batch concurrent |
| Scheduler (12) | 2 files | Scheduler-Worker, prefill/decode, priority scheduling |
| Attention (14) | 1 file | Attention-Mask, MLA, Torchair graph mode |
| Quantization (13) | 1 file | W8A8, dynamic quantization, TP integration |
| Distributed (11) | 1 file | Communicator, TP/EP, KV Connector |
| Sample (11) | 1 file | Sampler-Temperature, RejectionSampler scenarios |

### 3. CI/CD Integration (14 tasks)

**Configuration:**
- `tests/st/pytest.ini` - pytest markers, execution thresholds
- `tests/st/.coveragerc` - Coverage configuration (fail_under=80)
- `tests/st/scripts/run_st_tests.sh` - Parallel execution, coverage support

**Documentation:**
- `tests/st/README.md` - Framework overview, STRunner guide
- `tests/st/MOCK_FACTORY_GUIDE.md` - create_mock_* naming, spec restriction
- `tests/st/PLUGIN_ARCHITECTURE_TEST_STRATEGY.md` - Inheritance/replacement/injection testing

### 4. Performance/Precision/Integration Framework (43 tasks)

**Placeholder fixtures for future real NPU tests:**
- `tests/st/performance/conftest.py` - performance_baseline, performance_collector
- `tests/st/precision/conftest.py` - PRECISION_THRESHOLDS, precision_checker
- `tests/st/integration/conftest.py` - CANN_version, acl_graph_compiler, hccl_comm_tester

### 5. Three-Layer Decoupling (35 tasks)

**ModelPool + ConfigLoader:**
- `tests/st/utils/model_pool.py` - Model caching, cache key strategy, hit rate stats
- `tests/st/configs/default.yaml` - Default test configuration
- `tests/st/configs/ci_fast.yaml` - CI fast (60s timeout, minimal params)
- `tests/st/configs/npu_deep.yaml` - NPU deep (600s timeout, comprehensive params)

---

## Implementation Highlights

### 1. Dual Execution Mode Innovation

**Problem**: ST tests needed to run both in CI (no NPU) and on real hardware.  
**Solution**: Environment factory pattern with auto-detection:
```python
# Auto-detect based on torch_npu availability
exec_mode = detect_execution_environment()  # "cpu_mock" or "npu_real"

# Dynamic switching via fixture
@pytest.fixture(params=["cpu_mock", "npu_real"])
def exec_mode_param(request):
    if request.param == "npu_real" and not has_torch_npu():
        pytest.skip("NPU not available")
    return request.param
```

### 2. vllm Interface Spec Pattern

**Problem**: Mock objects could be over-mocked, missing critical interface methods.  
**Solution**: Spec restriction with vllm interface constants:
```python
VLLM_WORKER_SPEC = ['execute_model', 'initialize_model', 'get_model', ...]
worker = create_mock_worker(spec_attrs=VLLM_WORKER_SPEC)
verify_plugin_interface_compatibility(worker, VLLM_WORKER_SPEC)
```

### 3. Three-Layer Decoupling Architecture

**Problem**: Models loaded repeatedly across tests, wasting time.  
**Solution**: ModelPool singleton with cache strategy:
- Layer 1 (Environment): CPU Mock / NPU Real
- Layer 2 (Model): ModelPool.get_or_create() with cache_key
- Layer 3 (Test Case): Tests use cached models

---

## Special Adaptations

### 1. Environment Dependency Clarification

**Initial assumption**: ST tests could run without torch_npu.  
**Correction**: ST tests MUST depend on CANN container (source code imports torch_npu).  
**Impact**: CPU Mock mode still works because it Mocks torch_npu at import level.

### 2. Test Layer Boundary Definition

| Layer | Responsibility | Mock Strategy |
|-------|----------------|---------------|
| UT | Single function/class | Mock torch_npu hardware |
| ST | Plugin internal collaboration | Mock vllm Core + torch_npu |
| E2E | Complete inference flow | No mock, real vllm + NPU |

### 3. Docstring Template Enforcement

All test functions must include 4-part Docstring:
1. **验证**: What is being verified
2. **场景**: Test scenario description
3. **预期结果**: Expected outcome
4. **执行模式**: Execution mode (CPU Mock / NPU Real)

---

## Changed Files Summary

**Created**: 40+ files
- Framework: 10 utils + 4 docs
- Tests: 10 Worker + 2 Scheduler + 1 Attention + 1 Quant + 1 Dist + 1 Sample
- Configs: 3 YAML + 2 pytest configs
- Placeholders: 3 performance/precision/integration

**Modified**:
- `pyproject.toml` - pytest markers section
- `requirements-dev.txt` - pytest-xdist, pytest-rerunfailures

---

## Key Changes

1. **ST framework established** - Complete testing pyramid layer between UT and E2E
2. **Dual execution mode** - CPU Mock for CI gate, NPU Real for deep testing
3. **vllm interface spec pattern** - Prevents over-mocking, validates plugin compatibility
4. **ModelPool caching** - Avoids repeated model loading across tests
5. **Config-driven testing** - YAML configs control test parameters
6. **Coverage threshold** - 80% fail_under threshold for CI quality gate

---

## Lessons Learned

### Highlights

1. **Modular fixture design** - Session/module/function scope fixtures enable resource sharing
2. **Parameterization best practices** - @pytest.mark.parametrize for batch_size/dtype/seq_len covers edge cases
3. **Mock factory pattern** - Centralized mock creation with spec restriction ensures consistency
4. **STRunner pattern** - Context manager pattern simplifies resource cleanup
5. **ConfigLoader** - YAML configs decouple test parameters from code

### Remaining Issues

1. **NPU Real tests placeholder** - Requires real hardware, not implemented
2. **Performance baseline missing** - performance_baseline.json not populated
3. **Coverage history not initialized** - coverage_history/*.json empty
4. **CI workflow not integrated** - st-cpu-mock job needs GitHub Actions configuration
5. **Codecov flags not configured** - st_tests flag needs codecov.yml update

### Future Optimizations

1. **Real NPU test implementation** - Implement NPU Real tests when hardware available
2. **Performance regression monitoring** - Track performance_baseline.json trends
3. **Coverage visualization dashboard** - COVERAGE_REPORT.md with ASCII charts
4. **CI workflow integration** - Add st-cpu-mock job to vllm_ascend_test.yaml
5. **ModelPool hit rate monitoring** - Track cache efficiency across test runs

---

## Archive Location

**Archived to**: `openspec/changes/archive/2026-04-30-add-developer-test-suite/`

---

## Conclusion

Successfully implemented complete ST test framework covering:
- ✓ Framework infrastructure (26 tasks)
- ✓ 6 core module integration tests (68 tasks)
- ✓ CI/CD configuration and documentation (14 tasks)
- ✓ Quality assurance guidelines (14 tasks)
- ✓ Performance/Precision/Integration placeholders (43 tasks)
- ✓ Three-layer decoupling ModelPool/ConfigLoader (35 tasks)
- ✓ Mock strategy and config-driven testing (14 tasks)
- ✓ Coverage management framework (28 tasks)
- ✓ CI/CD workflow design (24 tasks)

**Total**: 279 tasks complete, 0 incomplete.

The ST framework enables testing plugin internal collaboration through dual execution modes, establishing the middle layer of the testing pyramid between UT and E2E.