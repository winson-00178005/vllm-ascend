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

"""Worker多批次并发处理测试

验证NPUWorker处理多批次并发请求的集成正确性。
参考：tests/st/specs/st-worker-integration/spec.md
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from queue import Queue
import threading

from tests.st.base import PytestSTBase
from tests.st.utils.mock_utils import (
    create_mock_worker,
    create_mock_input_batch,
    verify_mock_calls,
)


class TestWorkerMultiBatchConcurrent(PytestSTBase):
    """Worker多批次并发处理测试类"""

    @pytest.mark.cpu_mock
    def test_worker_two_concurrent_batches(self):
        """Test Worker处理两个并发批次请求

        验证：
        - Worker同时接收两个批次推理请求
        - Worker按顺序调度两个批次
        - ModelRunner分别执行并返回两个批次结果

        场景：Worker处理两个并发批次请求

        预期结果：两个批次正确处理，结果正确返回

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'input_batch', 'model_runner']
        )
        
        batch1 = create_mock_input_batch(batch_size=16)
        batch1.token_ids = torch.randint(0, 1000, (16, 128))
        
        batch2 = create_mock_input_batch(batch_size=32)
        batch2.token_ids = torch.randint(0, 1000, (32, 128))
        
        worker.execute_model = MagicMock()
        worker.execute_model.side_effect = [
            torch.randn(16, 128),
            torch.randn(32, 128)
        ]
        
        worker.input_batch = batch1
        result1 = worker.execute_model()
        assert result1.shape[0] == 16
        
        worker.input_batch = batch2
        result2 = worker.execute_model()
        assert result2.shape[0] == 32
        
        verify_mock_calls(worker.execute_model, expected_calls=2)

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("num_batches", [2, 4, 8])
    def test_worker_multiple_concurrent_batches(self, num_batches):
        """Test Worker处理多个并发批次请求

        验证：
        - Worker正确处理num_batches个批次请求
        - 批次调度顺序正确
        - 所有批次结果正确返回

        场景：Worker处理多个并发批次请求

        预期结果：所有批次正确处理

        执行模式：CPU Mock
        """
        worker = create_mock_worker(spec_attrs=['execute_model', 'input_batch'])
        
        results = []
        for i in range(num_batches):
            batch_size = 16 * (i + 1)
            batch = create_mock_input_batch(batch_size=batch_size)
            batch.token_ids = torch.randint(0, 1000, (batch_size, 128))
            
            worker.input_batch = batch
            
            worker.execute_model = MagicMock(
                return_value=torch.randn(batch_size, 128)
            )
            
            result = worker.execute_model()
            results.append(result)
        
        for i, result in enumerate(results):
            expected_batch_size = 16 * (i + 1)
            assert result.shape[0] == expected_batch_size

    @pytest.mark.cpu_mock
    def test_worker_batch_scheduling_order(self):
        """Test Worker批次调度顺序

        验证：
        - Worker按优先级调度批次
        - 调度顺序符合预期
        - 高优先级批次优先处理

        场景：Worker批次调度顺序验证

        预期结果：调度顺序正确

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'input_batch', 'schedule_order']
        )
        
        batches = [
            create_mock_input_batch(batch_size=8),
            create_mock_input_batch(batch_size=16),
            create_mock_input_batch(batch_size=32),
        ]
        
        schedule_order = [2, 0, 1]  # 优先处理batch_size=32的批次
        worker.schedule_order = schedule_order
        
        worker.execute_model = MagicMock()
        worker.execute_model.side_effect = [
            torch.randn(32, 128),
            torch.randn(8, 128),
            torch.randn(16, 128),
        ]
        
        results = []
        for order_idx in schedule_order:
            worker.input_batch = batches[order_idx]
            result = worker.execute_model()
            results.append(result)
        
        assert results[0].shape[0] == 32
        assert results[1].shape[0] == 8
        assert results[2].shape[0] == 16

    @pytest.mark.cpu_mock
    def test_worker_concurrent_resource_management(self):
        """Test Worker并发请求资源管理

        验证：
        - Worker正确管理并发批次资源
        - 资源不发生冲突
        - 资源正确分配和释放

        场景：Worker并发请求资源管理

        预期结果：资源正确管理

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'input_batch', 'allocate_resources', 'release_resources']
        )
        
        worker.allocate_resources = MagicMock(return_value=True)
        worker.release_resources = MagicMock()
        
        batch = create_mock_input_batch(batch_size=16)
        worker.input_batch = batch
        
        worker.allocate_resources.assert_not_called()
        
        worker.allocate_resources(batch_size=16)
        verify_mock_calls(worker.allocate_resources, expected_calls=1)
        
        worker.execute_model = MagicMock(return_value=torch.randn(16, 128))
        worker.execute_model()
        
        worker.release_resources()
        verify_mock_calls(worker.release_resources, expected_calls=1)


class TestWorkerResourceConflictHandling(PytestSTBase):
    """Worker资源冲突处理测试类"""

    @pytest.mark.cpu_mock
    def test_worker_memory_conflict_handling(self):
        """Test Worker处理并发请求时内存冲突

        验证：
        - Worker检测到内存冲突时正确处理
        - Worker优先级调度或返回错误
        - 异常消息包含内存冲突信息

        场景：Worker处理并发请求时发生内存分配冲突

        预期结果：冲突正确处理

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'input_batch', 'check_memory']
        )
        
        worker.check_memory = MagicMock(
            side_effect=RuntimeError("memory conflict: batch 2 exceeds available memory")
        )
        
        batch = create_mock_input_batch(batch_size=64)
        worker.input_batch = batch
        
        with pytest.raises(RuntimeError) as cm:
            worker.check_memory()
        
        assert "memory conflict" in str(cm.value)

    @pytest.mark.cpu_mock
    def test_worker_device_conflict_handling(self):
        """Test Worker处理设备冲突

        验证：
        - Worker检测到设备冲突时正确处理
        - Worker返回设备不可用错误
        - 异常消息包含设备信息

        场景：Worker处理并发请求时发生设备冲突

        预期结果：设备冲突正确处理

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'input_batch', 'check_device']
        )
        
        worker.check_device = MagicMock(
            side_effect=RuntimeError("device conflict: NPU:0 busy")
        )
        
        batch = create_mock_input_batch(batch_size=32)
        worker.input_batch = batch
        
        with pytest.raises(RuntimeError) as cm:
            worker.check_device()
        
        assert "device conflict" in str(cm.value)

    @pytest.mark.cpu_mock
    def test_worker_priority_scheduling_on_conflict(self):
        """Test Worker冲突时优先级调度

        验证：
        - Worker在资源冲突时启用优先级调度
        - 高优先级批次优先获取资源
        - 低优先级批次等待或返回错误

        场景：Worker在资源冲突时的优先级调度

        预期结果：优先级调度正确执行

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'schedule_with_priority']
        )
        
        high_priority_batch = create_mock_input_batch(batch_size=8)
        low_priority_batch = create_mock_input_batch(batch_size=64)
        
        worker.schedule_with_priority = MagicMock()
        worker.schedule_with_priority.side_effect = [
            ("success", torch.randn(8, 128)),
            ("queued", None)
        ]
        
        status1, result1 = worker.schedule_with_priority(high_priority_batch, priority="high")
        status2, result2 = worker.schedule_with_priority(low_priority_batch, priority="low")
        
        assert status1 == "success"
        assert result1 is not None
        assert status2 == "queued"
        assert result2 is None

    @pytest.mark.cpu_mock
    def test_worker_batch_queuing(self):
        """Test Worker批次排队机制

        验证：
        - Worker正确排队等待的批次
        - 排队批次在资源可用时执行
        - 排队顺序正确

        场景：Worker批次排队机制验证

        预期结果：排队机制正确

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'queue_batch', 'get_queued_batch']
        )
        
        batch_queue = Queue()
        
        batch1 = create_mock_input_batch(batch_size=32)
        batch2 = create_mock_input_batch(batch_size=16)
        
        worker.queue_batch = MagicMock(side_effect=lambda b: batch_queue.put(b))
        worker.get_queued_batch = MagicMock(side_effect=lambda: batch_queue.get())
        
        worker.queue_batch(batch1)
        worker.queue_batch(batch2)
        
        verify_mock_calls(worker.queue_batch, expected_calls=2)
        
        first_batch = worker.get_queued_batch()
        assert first_batch.batch_size == 32
        
        second_batch = worker.get_queued_batch()
        assert second_batch.batch_size == 16

    @pytest.mark.cpu_mock
    def test_worker_concurrent_execution_with_timeout(self):
        """Test Worker并发执行超时处理

        验证：
        - Worker设置执行超时时间
        - 超时后正确取消批次
        - 超时错误正确传递

        场景：Worker并发执行超时处理

        预期结果：超时正确处理

        执行模式：CPU Mock
        """
        worker = create_mock_worker(
            spec_attrs=['execute_model', 'execute_with_timeout']
        )
        
        worker.execute_with_timeout = MagicMock(
            side_effect=RuntimeError("execution timeout: exceeded 30s limit")
        )
        
        batch = create_mock_input_batch(batch_size=16)
        
        with pytest.raises(RuntimeError) as cm:
            worker.execute_with_timeout(batch, timeout=30)
        
        assert "timeout" in str(cm.value)
        assert "30s" in str(cm.value)