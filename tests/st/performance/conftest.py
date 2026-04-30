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

"""Performance test fixtures for ST tests."""

import pytest
import json
import time
from pathlib import Path
from unittest.mock import MagicMock

from tests.st.utils.config_factory import create_vllm_config
from tests.st.utils.env_detector import has_torch_npu


@pytest.fixture
def performance_config():
    """Performance测试配置fixture"""
    return create_vllm_config(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=2048,
        dtype="float16",
    )


@pytest.fixture
def performance_baseline():
    """性能基准数据fixture"""
    baseline_path = Path(__file__).parent / "performance_baseline.json"
    if baseline_path.exists():
        with open(baseline_path, "r") as f:
            return json.load(f)
    return {
        "worker_execute_time_ms": 100,
        "attention_forward_time_ms": 50,
        "scheduler_schedule_time_ms": 10,
        "quantization_apply_time_ms": 20,
    }


@pytest.fixture
def performance_collector():
    """性能数据收集器fixture"""
    collector = MagicMock(spec=['collect', 'analyze', 'report'])
    collector.metrics = {}
    
    def collect_metric(name: str, value: float):
        collector.metrics[name] = value
    
    collector.collect = MagicMock(side_effect=collect_metric)
    collector.analyze = MagicMock(return_value=collector.metrics)
    collector.report = MagicMock(return_value=json.dumps(collector.metrics))
    
    return collector


@pytest.fixture
def timer():
    """计时器fixture"""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
            return self.end_time - self.start_time
        
        def elapsed_ms(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time) * 1000
            return None
    
    return Timer()


@pytest.fixture(params=["worker", "attention", "scheduler", "quantization", "distributed"])
def performance_module(request):
    """参数化性能测试模块"""
    return request.param


@pytest.fixture(params=[1, 16, 32])
def perf_batch_size(request):
    """参数化性能测试batch_size"""
    return request.param


@pytest.fixture(params=[128, 512, 1024])
def perf_seq_len(request):
    """参数化性能测试seq_len"""
    return request.param