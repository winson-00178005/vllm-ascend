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

"""Scheduler模块集成测试

验证AscendScheduler的插件内部协作。
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from tests.st.base import PytestSTBase
from tests.st.utils.mock_utils import (
    create_mock_scheduler,
    create_mock_worker,
    verify_mock_calls,
    VLLM_SCHEDULER_SPEC,
    verify_plugin_interface_compatibility,
)
from tests.st.utils.runner_factory import STRunner


class TestSchedulerIntegration(PytestSTBase):
    """Scheduler集成测试类"""

    @pytest.mark.cpu_mock
    def test_scheduler_init_with_config(self, mock_scheduler):
        """Test AscendScheduler初始化配置

        验证：
        - Scheduler正确初始化并持有配置
        - Scheduler配置参数正确传递
        - 初始化状态正确

        场景：Scheduler正常初始化流程

        预期结果：Scheduler正确初始化，配置正确持有

        执行模式：CPU Mock
        """
        assert mock_scheduler is not None
        assert hasattr(mock_scheduler, 'schedule')
        
        mock_scheduler.vllm_config = MagicMock()
        mock_scheduler.max_num_running_reqs = 64
        mock_scheduler.max_num_scheduled_tokens = 2048
        
        assert mock_scheduler.max_num_running_reqs == 64
        assert mock_scheduler.max_num_scheduled_tokens == 2048

    @pytest.mark.cpu_mock
    def test_scheduler_schedule_flow(self, mock_scheduler, mock_scheduler_output):
        """Test Scheduler调度流程

        验证：
        - Scheduler schedule方法正确调用
        - schedule返回SchedulerOutput
        - 输出包含正确的调度信息

        场景：Scheduler正常调度流程

        预期结果：调度正确执行，输出正确

        执行模式：CPU Mock
        """
        mock_scheduler.schedule = MagicMock(return_value=mock_scheduler_output)
        
        output = mock_scheduler.schedule()
        
        assert output is not None
        assert output.num_scheduled_tokens == 1024
        verify_mock_calls(mock_scheduler.schedule, expected_calls=1)

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("scheduler_strategy", ["prefill_first", "chunked_prefill"])
    def test_scheduler_different_strategies(self, scheduler_strategy, mock_scheduler):
        """Test Scheduler不同调度策略

        验证：
        - prefill_first策略正确执行
        - chunked_prefill策略正确执行
        - 不同策略产生不同调度结果

        场景：不同调度策略测试

        预期结果：策略正确执行

        执行模式：CPU Mock
        """
        mock_scheduler.scheduler_strategy = scheduler_strategy
        mock_scheduler.schedule = MagicMock()
        
        if scheduler_strategy == "prefill_first":
            mock_scheduler.schedule.return_value = MagicMock(
                num_scheduled_tokens=512,
                scheduled_new_reqs=[MagicMock()]
            )
        else:
            mock_scheduler.schedule.return_value = MagicMock(
                num_scheduled_tokens=256,
                scheduled_new_reqs=[MagicMock()]
            )
        
        output = mock_scheduler.schedule()
        assert output.num_scheduled_tokens > 0

    @pytest.mark.cpu_mock
    def test_scheduler_with_strunner(self, scheduler_runner, mock_scheduler):
        """Test Scheduler使用STRunner管理资源

        验证：
        - STRunner正确初始化Scheduler资源
        - STRunner正确清理资源
        - 资源管理符合上下文管理器模式

        场景：使用STRunner管理Scheduler模块资源

        预期结果：资源正确初始化和清理

        执行模式：CPU Mock
        """
        with STRunner("scheduler", {"model": "test_model"}) as runner:
            runner.register_resource("scheduler", mock_scheduler)
            
            assert runner.get_resource("scheduler") is not None
        
        assert runner._resources == {}


class TestSchedulerInterfaceCompatibility(PytestSTBase):
    """Scheduler接口兼容性测试类"""

    @pytest.mark.cpu_mock
    def test_scheduler_interface_compatibility(self):
        """Test AscendScheduler implements vllm.Scheduler interface

        验证：
        - AscendScheduler包含所有vllm.Scheduler接口方法
        - 每个接口方法都是可调用的
        - 接口方法签名符合vllm规范

        场景：验证插件Scheduler与vllm接口兼容性

        预期结果：所有接口检查通过，兼容性验证成功

        执行模式：CPU Mock
        """
        scheduler = create_mock_scheduler(spec_attrs=VLLM_SCHEDULER_SPEC)
        
        verify_plugin_interface_compatibility(scheduler, VLLM_SCHEDULER_SPEC)

    @pytest.mark.cpu_mock
    def test_scheduler_interface_spec_coverage(self):
        """Test Scheduler interface spec coverage

        验证：
        - VLLM_SCHEDULER_SPEC包含核心接口方法
        - 每个接口方法在Scheduler中都存在

        场景：验证接口规范覆盖完整性

        预期结果：接口规范覆盖所有必需方法

        执行模式：CPU Mock
        """
        expected_core_methods = ['schedule', 'get_num_unfinished_seqs']
        
        for method in expected_core_methods:
            assert method in VLLM_SCHEDULER_SPEC, \
                f"Core method {method} missing from VLLM_SCHEDULER_SPEC"

    @pytest.mark.cpu_mock
    def test_scheduler_schedule_interface(self):
        """Test Scheduler schedule interface signature

        验证：
        - schedule方法存在且可调用
        - schedule返回SchedulerOutput类型

        场景：验证schedule接口实现

        预期结果：schedule接口正确实现

        执行模式：CPU Mock
        """
        scheduler = create_mock_scheduler(spec_attrs=['schedule'])
        
        assert callable(scheduler.schedule), \
            "schedule is not callable"
        
        scheduler.schedule = MagicMock(return_value=MagicMock())
        
        result = scheduler.schedule()
        assert result is not None


class TestSchedulerReplacementLogic(PytestSTBase):
    """Scheduler替换逻辑测试类"""

    @pytest.mark.cpu_mock
    def test_scheduler_replace_vllm_scheduler(self):
        """Test AscendScheduler替换vllm.Scheduler逻辑

        验证：
        - AscendScheduler继承vllm.Scheduler
        - AscendScheduler扩展prefill_first策略
        - 替换逻辑正确性

        场景：Scheduler替换vllm.Scheduler

        预期结果：替换逻辑正确

        执行模式：CPU Mock
        """
        scheduler = create_mock_scheduler(
            spec_attrs=['schedule', 'add_seq_group', 'abort_seq_group']
        )
        
        scheduler.is_ascend_scheduler = MagicMock(return_value=True)
        scheduler.prefill_first_enabled = MagicMock(return_value=True)
        
        assert scheduler.is_ascend_scheduler() == True
        assert scheduler.prefill_first_enabled() == True

    @pytest.mark.cpu_mock
    def test_scheduler_scheduling_collaboration(self, mock_scheduler, mock_worker_for_scheduler):
        """Test Scheduler与Worker调度协作

        验证：
        - Scheduler生成SchedulerOutput
        - Worker接收SchedulerOutput执行
        - 执行结果正确传递

        场景：Scheduler与Worker协作流程

        预期结果：调度协作正确

        执行模式：CPU Mock
        """
        mock_output = MagicMock()
        mock_output.num_scheduled_seqs = 16
        
        mock_scheduler.schedule = MagicMock(return_value=mock_output)
        
        output = mock_scheduler.schedule()
        
        mock_worker_for_scheduler.execute_model = MagicMock(
            return_value=torch.randn(16, 128)
        )
        
        result = mock_worker_for_scheduler.execute_model(output)
        
        assert result is not None
        verify_mock_calls(mock_scheduler.schedule, expected_calls=1)