#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import pytest

from vllm_ascend.utils import adapt_patch, register_ascend_customop

adapt_patch()
adapt_patch(True)

register_ascend_customop()


def pytest_addoption(parser):
    """Add custom command line options for ST tests.

    Reference: tests/e2e/models/conftest.py pattern
    """
    parser.addoption(
        "--exec-mode",
        action="store",
        default="auto",
        choices=["auto", "cpu_mock", "npu_real"],
        help="Execution mode: auto (detect), cpu_mock, npu_real"
    )

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

    parser.addoption(
        "--npu-device",
        action="store",
        default="0",
        help="NPU device ID to use (default: 0)"
    )

    parser.addoption(
        "--performance-baseline",
        action="store",
        default=None,
        help="Path to performance baseline JSON file"
    )

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
def exec_mode(request):
    """Session scope: execution mode from command line."""
    return request.config.getoption("--exec-mode")


@pytest.fixture(scope="session")
def enable_mock(request):
    """Session scope: mock enable flag from command line."""
    return request.config.getoption("--enable-mock")


@pytest.fixture(scope="session")
def disable_mock(request):
    """Session scope: mock disable flag from command line."""
    return request.config.getoption("--disable-mock")


@pytest.fixture(scope="session")
def performance_test(request):
    """Session scope: performance test flag from command line."""
    return request.config.getoption("--performance-test")


@pytest.fixture(scope="session")
def precision_test(request):
    """Session scope: precision test flag from command line."""
    return request.config.getoption("--precision-test")


@pytest.fixture(scope="session")
def npu_device(request):
    """Session scope: NPU device ID from command line."""
    return request.config.getoption("--npu-device")


@pytest.fixture(scope="session")
def performance_baseline(request):
    """Session scope: performance baseline file path."""
    return request.config.getoption("--performance-baseline")


@pytest.fixture(scope="session")
def config_file(request):
    """Session scope: config file path from command line."""
    return request.config.getoption("--config-file")


@pytest.fixture(scope="session")
def model_name(request):
    """Session scope: model name from command line."""
    return request.config.getoption("--model")


@pytest.fixture(scope="session")
def test_config_name(request):
    """Session scope: test config name from command line."""
    return request.config.getoption("--test-config")


@pytest.fixture(scope="session")
def st_environment(request):
    """Session scope: Layer 1 environment initialization.

    Initialize NPU environment once per session.
    """
    from tests.st.utils.env_factory import create_environment
    from tests.st.utils.env_detector import detect_execution_environment

    exec_mode = request.config.getoption("--exec-mode")

    if exec_mode == "auto":
        mode = detect_execution_environment()
    elif exec_mode == "cpu_mock":
        mode = "cpu_mock"
    elif exec_mode == "npu_real":
        mode = detect_execution_environment()
        if mode != "npu_real":
            pytest.skip("NPU not available, skip npu_real tests")
    else:
        mode = "cpu_mock"

    env = create_environment(mode)
    env.enter()
    yield env
    env.exit()


@pytest.fixture(scope="session")
def mock_npu_env():
    """Session scope: Mock NPU hardware environment.

    All tests share this mock environment.
    """
    import torch
    try:
        import torch_npu
        HAS_TORCH_NPU = True
    except ImportError:
        HAS_TORCH_NPU = False

    if not HAS_TORCH_NPU:
        pytest.skip("torch_npu not available")

    yield