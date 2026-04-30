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
"""ST (System Integration Test) framework for vllm-ascend plugin.

ST tests verify module integration and collaboration within the plugin,
focusing on:
- Plugin internal module collaboration (e.g., Worker + ModelRunner)
- Plugin interface compatibility with vllm
- Data flow correctness between modules

ST tests complement UT (unit tests) and E2E (end-to-end tests) to form
a complete testing pyramid for the vllm-ascend plugin.
"""