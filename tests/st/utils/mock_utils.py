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

"""Mock utilities for ST tests.

Provides mock factory functions and verification helpers.
Reference: tests/ut/ops/test_fused_ops.py pattern
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import torch


def create_mock_worker(spec_attrs: Optional[List[str]] = None) -> MagicMock:
    """Create Mock Worker object with spec restriction.

    Args:
        spec_attrs: List of attributes to include in spec.
                   Default: ['execute_model', 'input_batch', 'device']

    Returns:
        MagicMock with worker interface
    """
    default_spec = ['execute_model', 'input_batch', 'device']
    spec = spec_attrs or default_spec
    mock_worker = MagicMock(spec=spec)
    mock_worker.execute_model = MagicMock(
        return_value=torch.randn(1, 16, 64))
    mock_worker.device = "npu:0"
    return mock_worker


def create_mock_model_runner(spec_attrs: Optional[List[str]] = None) -> MagicMock:
    """Create Mock ModelRunner object with spec restriction.

    Args:
        spec_attrs: List of attributes to include in spec.
                   Default: ['execute', 'load_model', 'device']

    Returns:
        MagicMock with model_runner interface
    """
    default_spec = ['execute', 'load_model', 'device']
    spec = spec_attrs or default_spec
    mock_runner = MagicMock(spec=spec)
    mock_runner.execute = MagicMock(return_value=torch.randn(1, 16, 64))
    mock_runner.device = "npu:0"
    return mock_runner


def create_mock_input_batch(batch_size: int = 1) -> MagicMock:
    """Create Mock InputBatch object.

    Args:
        batch_size: Batch size for the input

    Returns:
        MagicMock with input_batch interface
    """
    mock_batch = MagicMock()
    mock_batch.batch_size = batch_size
    mock_batch.token_ids = torch.randint(0, 1000, (batch_size, 16))
    return mock_batch


def create_mock_distribution_env(mocker) -> MagicMock:
    """Create Mock distributed environment.

    Args:
        mocker: pytest-mock mocker object

    Returns:
        MagicMock with distributed group interface
    """
    mock_group = mocker.MagicMock()
    mock_group.rank_in_group = 0
    mock_group.world_size = 4
    mock_group.all_reduce = MagicMock(
        return_value=torch.randn(4, 16, 64))
    return mock_group


def create_mock_attention(spec_attrs: Optional[List[str]] = None) -> MagicMock:
    """Create Mock Attention object with spec restriction.

    Args:
        spec_attrs: List of attributes to include in spec.

    Returns:
        MagicMock with attention interface
    """
    default_spec = ['forward', 'get_kv_cache_shape', 'device']
    spec = spec_attrs or default_spec
    mock_attention = MagicMock(spec=spec)
    mock_attention.forward = MagicMock(
        return_value=torch.randn(1, 16, 64))
    return mock_attention


def create_mock_attention_mask() -> MagicMock:
    """Create Mock AttentionMask object.

    Returns:
        MagicMock with attention_mask interface
    """
    mock_mask = MagicMock()
    mock_mask.mask = torch.ones(1, 16, 16)
    return mock_mask


def create_mock_scheduler(spec_attrs: Optional[List[str]] = None) -> MagicMock:
    """Create Mock Scheduler object with spec restriction.

    Args:
        spec_attrs: List of attributes to include in spec.

    Returns:
        MagicMock with scheduler interface
    """
    default_spec = ['schedule', 'get_num_unfinished_seqs', 'device']
    spec = spec_attrs or default_spec
    mock_scheduler = MagicMock(spec=spec)
    mock_scheduler.schedule = MagicMock(
        return_value=MagicMock(scheduler_output=MagicMock()))
    return mock_scheduler


def verify_mock_calls(mock_obj: MagicMock,
                      expected_calls: int,
                      expected_args: Optional[Dict[str, Any]] = None) -> bool:
    """Verify Mock call count and arguments.

    Args:
        mock_obj: Mock object to verify
        expected_calls: Expected number of calls
        expected_args: Expected arguments for the call

    Returns:
        bool: True if verification passes

    Raises:
        AssertionError: If verification fails
    """
    assert mock_obj.call_count == expected_calls, \
        f"Expected {expected_calls} calls, got {mock_obj.call_count}"

    if expected_args:
        mock_obj.assert_called_with(**expected_args)

    return True


def create_spec_mock(spec_attrs: List[str],
                     return_value: Any = None) -> MagicMock:
    """Create Mock object with spec restriction to avoid over-mocking.

    Args:
        spec_attrs: List of attributes to include in spec
        return_value: Default return value for mock calls

    Returns:
        MagicMock with spec restriction
    """
    mock_obj = MagicMock(spec=spec_attrs)
    if return_value is not None:
        for attr in spec_attrs:
            if not attr.startswith('_'):
                setattr(mock_obj, attr, MagicMock(return_value=return_value))
    return mock_obj


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

VLLM_SCHEDULER_SPEC = [
    'schedule',
    'get_num_unfinished_seqs',
    'has_unfinished_seqs',
    'add_seq_group',
    'abort_seq_group',
]


def verify_plugin_interface_compatibility(plugin_obj: Any,
                                           vllm_interface_spec: List[str]) -> bool:
    """Verify plugin component implements vllm interface.

    Args:
        plugin_obj: Plugin object to verify
        vllm_interface_spec: List of interface attributes to check

    Returns:
        bool: True if all interfaces are implemented

    Raises:
        AssertionError: If interface is missing
    """
    for attr in vllm_interface_spec:
        assert hasattr(plugin_obj, attr), \
            f"Plugin object missing vllm interface: {attr}"

        method = getattr(plugin_obj, attr)
        assert callable(method) or isinstance(method, property), \
            f"Plugin interface {attr} is not callable"

    return True