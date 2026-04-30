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

"""ST Runner factory for managing test resources.

Provides STRunner context manager for integration test execution.
"""

from typing import Any, Dict, Optional

from tests.st.utils.config_factory import create_ascend_config, create_vllm_config
from tests.st.utils.mock_utils import create_mock_worker, create_mock_model_runner


class STRunner:
    """ST Runner context manager for managing module resources.

    Reference: tests/e2e/conftest.py VllmRunner pattern
    """

    def __init__(self, module_name: str, config: Optional[Dict] = None):
        """Initialize STRunner.

        Args:
            module_name: Module name (worker, scheduler, attention, etc.)
            config: Configuration dict
        """
        self.module_name = module_name
        self.config = config or {}
        self.worker = None
        self.model_runner = None
        self.scheduler = None
        self.attention = None

        self._init_module()

    def _init_module(self):
        """Initialize module based on module_name."""
        ascend_config = create_ascend_config(**self.config)

        if self.module_name == "worker":
            self.worker = create_mock_worker()
            self.model_runner = create_mock_model_runner()
        elif self.module_name == "scheduler":
            self.scheduler = create_mock_scheduler()
        elif self.module_name == "attention":
            self.attention = create_mock_attention()

    def execute_integration(self, scenario: str) -> Any:
        """Execute integration test scenario.

        Args:
            scenario: Scenario name (normal_flow, error_handling, etc.)

        Returns:
            Execution result
        """
        if self.module_name == "worker":
            return self.worker.execute_model(scenario)
        elif self.module_name == "scheduler":
            return self.scheduler.schedule(scenario)
        elif self.module_name == "attention":
            return self.attention.forward(scenario)
        return None

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup resources."""
        self.cleanup_module_resources()

    def cleanup_module_resources(self):
        """Cleanup module resources to prevent leakage."""
        self.worker = None
        self.model_runner = None
        self.scheduler = None
        self.attention = None


def create_st_runner(module_name: str,
                      config: Optional[Dict] = None) -> STRunner:
    """Create STRunner instance.

    Args:
        module_name: Module name
        config: Configuration dict

    Returns:
        STRunner instance
    """
    return STRunner(module_name, config)