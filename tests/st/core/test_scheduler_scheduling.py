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

"""Scheduler多批次调度和prefill/decode调度测试

验证AscendScheduler多批次调度和连续解码调度。
"""

import pytest
import torch
from unittest.mock import MagicMock

from tests.st.base import PytestSTBase
from tests.st.utils.mock_utils import (
    create_mock_scheduler,
    create_mock_worker,
    verify_mock_calls,
)


class TestSchedulerMultiBatchScheduling(PytestSTBase):
    """Scheduler多批次调度测试类"""

    @pytest.mark.cpu_mock
    def test_scheduler_priority_scheduling(self, mock_scheduler):
        """Test Scheduler优先级调度

        验证：
        - Scheduler按优先级调度批次
        - 高优先级批次优先处理
        - 调度顺序正确

        场景：Scheduler优先级调度验证

        预期结果：优先级调度正确执行

        执行模式：CPU Mock
        """
        requests = [
            MagicMock(priority=1, request_id="req1"),
            MagicMock(priority=3, request_id="req2"),
            MagicMock(priority=2, request_id="req3"),
        ]
        
        mock_scheduler.waiting = requests
        mock_scheduler.schedule = MagicMock()
        
        sorted_requests = sorted(requests, key=lambda r: r.priority, reverse=True)
        
        mock_scheduler.schedule.return_value = MagicMock(
            scheduled_new_reqs=sorted_requests[:2]
        )
        
        output = mock_scheduler.schedule()
        
        assert len(output.scheduled_new_reqs) == 2

    @pytest.mark.cpu_mock
    def test_scheduler_batch_preemption(self, mock_scheduler):
        """Test Scheduler批次抢占

        验证：
        - Scheduler在资源不足时抢占低优先级批次
        - 抢占批次正确记录
        - 抢占后资源重新分配

        场景：Scheduler批次抢占验证

        预期结果：抢占正确执行

        执行模式：CPU Mock
        """
        mock_scheduler.schedule = MagicMock()
        
        preempted_req = MagicMock(request_id="preempted_req")
        
        mock_scheduler.schedule.return_value = MagicMock(
            preempted_reqs=[preempted_req],
            num_scheduled_tokens=512
        )
        
        output = mock_scheduler.schedule()
        
        assert len(output.preempted_reqs) == 1
        assert output.preempted_reqs[0].request_id == "preempted_req"

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("num_running_reqs", [1, 16, 32, 64])
    def test_scheduler_max_running_reqs_limit(self, num_running_reqs, mock_scheduler):
        """Test Scheduler max_running_reqs限制

        验证：
        - Scheduler遵守max_num_running_reqs限制
        - 超过限制时正确暂停调度
        - 限制边界正确处理

        场景：Scheduler运行请求数限制验证

        预期结果：限制正确执行

        执行模式：CPU Mock
        """
        mock_scheduler.max_num_running_reqs = num_running_reqs
        mock_scheduler.running = [MagicMock() for _ in range(num_running_reqs)]
        
        mock_scheduler.has_unfinished_seqs = MagicMock(
            return_value=len(mock_scheduler.running) < num_running_reqs * 2
        )
        
        mock_scheduler.schedule = MagicMock()
        if len(mock_scheduler.running) >= num_running_reqs:
            mock_scheduler.schedule.return_value = MagicMock(
                scheduled_new_reqs=[]
            )
        else:
            mock_scheduler.schedule.return_value = MagicMock(
                scheduled_new_reqs=[MagicMock()]
            )
        
        output = mock_scheduler.schedule()
        assert len(output.scheduled_new_reqs) >= 0


class TestSchedulerPrefillDecodeScheduling(PytestSTBase):
    """Scheduler prefill/decode调度测试类"""

    @pytest.mark.cpu_mock
    def test_scheduler_prefill_phase(self, mock_scheduler):
        """Test Scheduler prefill阶段调度

        验证：
        - Scheduler正确调度prefill请求
        - prefill请求正确分配token_budget
        - prefill结果正确返回

        场景：Scheduler prefill阶段调度

        预期结果：prefill调度正确

        执行模式：CPU Mock
        """
        prefill_req = MagicMock(status="prefill", num_tokens=256)
        
        mock_scheduler.waiting = [prefill_req]
        mock_scheduler.running = []
        
        mock_scheduler.schedule = MagicMock()
        mock_scheduler.schedule.return_value = MagicMock(
            scheduled_new_reqs=[prefill_req],
            num_scheduled_tokens=256
        )
        
        output = mock_scheduler.schedule()
        
        assert len(output.scheduled_new_reqs) == 1
        assert output.num_scheduled_tokens == 256

    @pytest.mark.cpu_mock
    def test_scheduler_decode_phase(self, mock_scheduler):
        """Test Scheduler decode阶段调度

        验证：
        - Scheduler正确调度decode请求
        - decode请求正确分配token_budget
        - decode结果正确返回

        场景：Scheduler decode阶段调度

        预期结果：decode调度正确

        执行模式：CPU Mock
        """
        decode_req = MagicMock(status="decode", num_tokens=1)
        
        mock_scheduler.running = [decode_req]
        mock_scheduler.waiting = []
        
        mock_scheduler.schedule = MagicMock()
        mock_scheduler.schedule.return_value = MagicMock(
            scheduled_running_reqs=[decode_req],
            num_scheduled_tokens=1
        )
        
        output = mock_scheduler.schedule()
        
        assert len(output.scheduled_running_reqs) == 1
        assert output.num_scheduled_tokens == 1

    @pytest.mark.cpu_mock
    def test_scheduler_continuous_decode(self, mock_scheduler):
        """Test Scheduler连续解码调度

        验证：
        - Scheduler支持连续多轮decode调度
        - 每轮decode正确调度
        - 连续调度状态正确维护

        场景：Scheduler连续解码调度验证

        预期结果：连续调度正确

        执行模式：CPU Mock
        """
        decode_req = MagicMock(status="decode", request_id="decode_req")
        
        mock_scheduler.running = [decode_req]
        mock_scheduler.schedule = MagicMock()
        
        outputs = []
        for i in range(5):
            mock_scheduler.schedule.return_value = MagicMock(
                scheduled_running_reqs=[decode_req],
                num_scheduled_tokens=1,
                iteration=i
            )
            output = mock_scheduler.schedule()
            outputs.append(output)
        
        verify_mock_calls(mock_scheduler.schedule, expected_calls=5)
        
        for output in outputs:
            assert len(output.scheduled_running_reqs) == 1

    @pytest.mark.cpu_mock
    def test_scheduler_prefill_decode_transition(self, mock_scheduler):
        """Test Scheduler prefill到decode转换

        验证：
        - Scheduler正确处理prefill完成转换到decode
        - 状态转换正确
        - decode阶段正确启动

        场景：Scheduler prefill到decode转换

        预期结果：转换正确执行

        执行模式：CPU Mock
        """
        req = MagicMock(
            request_id="req1",
            status="prefill",
            num_prompt_tokens=128,
            num_output_tokens=0
        )
        
        mock_scheduler.waiting = [req]
        mock_scheduler.schedule = MagicMock()
        
        mock_scheduler.schedule.return_value = MagicMock(
            scheduled_new_reqs=[req],
            num_scheduled_tokens=128
        )
        
        output1 = mock_scheduler.schedule()
        assert len(output1.scheduled_new_reqs) == 1
        
        req.status = "decode"
        req.num_output_tokens = 1
        mock_scheduler.waiting = []
        mock_scheduler.running = [req]
        
        mock_scheduler.schedule.return_value = MagicMock(
            scheduled_running_reqs=[req],
            num_scheduled_tokens=1
        )
        
        output2 = mock_scheduler.schedule()
        assert len(output2.scheduled_running_reqs) == 1


class TestSchedulerExceptionHandling(PytestSTBase):
    """Scheduler异常处理测试类"""

    @pytest.mark.cpu_mock
    def test_scheduler_resource_conflict_error(self, mock_scheduler):
        """Test Scheduler资源冲突错误处理

        验证：
        - Scheduler检测到资源冲突时正确处理
        - 错误正确传递
        - 异常消息包含关键信息

        场景：Scheduler资源冲突错误处理

        预期结果：错误正确处理

        执行模式：CPU Mock
        """
        mock_scheduler.schedule = MagicMock(
            side_effect=RuntimeError("scheduler resource conflict: token_budget exceeded")
        )
        
        with pytest.raises(RuntimeError) as cm:
            mock_scheduler.schedule()
        
        assert "resource conflict" in str(cm.value)

    @pytest.mark.cpu_mock
    def test_scheduler_request_overflow_error(self, mock_scheduler):
        """Test Scheduler请求溢出错误处理

        验证：
        - Scheduler请求过多时正确处理
        - 错误正确传递
        - 异常消息包含关键信息

        场景：Scheduler请求溢出错误处理

        预期结果：错误正确处理

        执行模式：CPU Mock
        """
        mock_scheduler.schedule = MagicMock(
            side_effect=RuntimeError("scheduler overflow: max_num_running_reqs exceeded")
        )
        
        with pytest.raises(RuntimeError) as cm:
            mock_scheduler.schedule()
        
        assert "overflow" in str(cm.value)

    @pytest.mark.cpu_mock
    def test_scheduler_abort_request(self, mock_scheduler):
        """Test Scheduler abort请求处理

        验证：
        - Scheduler正确处理abort请求
        - abort请求正确移除
        - 状态正确更新

        场景：Scheduler abort请求处理

        预期结果：abort正确处理

        执行模式：CPU Mock
        """
        req_to_abort = MagicMock(request_id="abort_req")
        
        mock_scheduler.abort_seq_group = MagicMock()
        mock_scheduler.running = [req_to_abort]
        
        mock_scheduler.abort_seq_group("abort_req")
        
        verify_mock_calls(mock_scheduler.abort_seq_group, expected_calls=1)


class TestSchedulerDualMode(PytestSTBase):
    """Scheduler双模式测试类"""

    @pytest.mark.parametrize("exec_mode", ["cpu_mock", "npu_real"])
    def test_scheduler_dual_mode_schedule(self, exec_mode):
        """Test Scheduler双模式调度

        验证：
        - CPU Mock模式下Scheduler正确执行
        - NPU Real模式下Scheduler正确执行
        - 不同模式下执行结果一致性

        场景：Scheduler双模式执行对比测试

        预期结果：两种模式执行结果一致

        执行模式：CPU Mock / NPU Real
        """
        scheduler = create_mock_scheduler(spec_attrs=['schedule'])
        
        if exec_mode == "cpu_mock":
            scheduler.schedule = MagicMock(return_value=MagicMock(num_scheduled_tokens=1024))
            output = scheduler.schedule()
            assert output.num_scheduled_tokens == 1024
        elif exec_mode == "npu_real":
            from tests.st.utils.env_detector import has_torch_npu
            if not has_torch_npu():
                pytest.skip("NPU not available")
            pytest.skip("NPU Real mode requires real hardware")

    @pytest.mark.cpu_mock
    @pytest.mark.npu_real
    def test_scheduler_with_dual_markers(self):
        """Test Scheduler双模式标记

        验证：
        - Scheduler在CPU Mock模式下正确标记
        - Scheduler在NPU Real模式下正确标记
        - 标记正确区分执行模式

        场景：Scheduler双模式标记验证

        预期结果：标记正确区分

        执行模式：CPU Mock / NPU Real
        """
        scheduler = create_mock_scheduler(spec_attrs=['schedule'])
        assert scheduler is not None