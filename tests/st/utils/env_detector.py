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

"""Environment detection for ST tests.

Auto-detect execution environment and choose appropriate mode.
"""


def has_torch_npu() -> bool:
    """Check if torch_npu is available.
    
    Returns:
        True if torch_npu is available, False otherwise
    """
    try:
        import torch_npu
        return True
    except ImportError:
        return False


def detect_execution_environment() -> str:
    """Detect environment and return available execution mode.

    Detection levels:
    1. torch availability
    2. torch_npu availability
    3. NPU hardware availability
    4. NPU type detection (is_310p)

    Returns:
        "npu_real" if NPU available, "cpu_mock" otherwise
    """
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False
        return "skip"

    try:
        import torch_npu
        HAS_TORCH_NPU = True
    except ImportError:
        HAS_TORCH_NPU = False

    try:
        NPU_AVAILABLE = torch.npu.is_available()
        NPU_COUNT = torch.npu.device_count()
    except Exception:
        NPU_AVAILABLE = False
        NPU_COUNT = 0

    try:
        from vllm_ascend.utils import is_310p
        IS_310P = is_310p()
    except Exception:
        IS_310P = False

    if NPU_AVAILABLE and NPU_COUNT > 0:
        return "npu_real"
    else:
        return "cpu_mock"


def get_environment_info() -> dict:
    """Get detailed environment information.

    Returns:
        dict with environment status
    """
    info = {
        "has_torch": False,
        "has_torch_npu": False,
        "npu_available": False,
        "npu_count": 0,
        "is_310p": False,
        "execution_mode": "cpu_mock"
    }

    try:
        import torch
        info["has_torch"] = True
    except ImportError:
        return info

    try:
        import torch_npu
        info["has_torch_npu"] = True
    except ImportError:
        return info

    try:
        info["npu_available"] = torch.npu.is_available()
        info["npu_count"] = torch.npu.device_count()
    except Exception:
        pass

    try:
        from vllm_ascend.utils import is_310p
        info["is_310p"] = is_310p()
    except Exception:
        pass

    info["execution_mode"] = detect_execution_environment()
    return info