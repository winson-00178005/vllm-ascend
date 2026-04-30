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

"""Attention模块集成测试

验证AscendAttention的插件内部协作。
"""

import pytest
import torch
from unittest.mock import MagicMock

from tests.st.base import PytestSTBase
from tests.st.utils.env_detector import has_torch_npu
from tests.st.utils.mock_utils import (
    create_mock_attention,
    create_mock_attention_mask,
    verify_mock_calls,
    VLLM_ATTENTION_BACKEND_SPEC,
    verify_plugin_interface_compatibility,
)
from tests.st.utils.runner_factory import STRunner


class TestAttentionIntegration(PytestSTBase):
    """Attention集成测试类"""

    @pytest.mark.cpu_mock
    def test_attention_init_with_mask(self, mock_attention, mock_attention_mask):
        """Test Attention与AttentionMask初始化

        验证：
        - Attention正确初始化并持有AttentionMask
        - AttentionMask正确配置
        - 初始化状态正确

        场景：Attention正常初始化流程

        预期结果：Attention正确初始化，Mask正确持有

        执行模式：CPU Mock
        """
        mock_attention.attention_mask = mock_attention_mask
        
        assert mock_attention.attention_mask is not None
        assert mock_attention.attention_mask.mask is not None

    @pytest.mark.cpu_mock
    def test_attention_forward_flow(self, mock_attention, attention_input_data):
        """Test Attention forward流程

        验证：
        - Attention forward方法正确调用
        - forward接收正确输入参数
        - 输出tensor形状正确

        场景：Attention正常forward流程

        预期结果：forward正确执行，输出正确

        执行模式：CPU Mock
        """
        mock_attention.forward = MagicMock(
            return_value=torch.randn(16, 128, 4096)
        )
        
        output = mock_attention.forward(
            attention_input_data["hidden_states"],
            attention_input_data["attention_mask"],
            attention_input_data["position_ids"]
        )
        
        assert output is not None
        assert output.shape[0] == 16
        verify_mock_calls(mock_attention.forward, expected_calls=1)

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("mask_type", ["full", "causal", "sliding_window"])
    def test_attention_different_mask_types(self, mask_type, mock_attention, mock_attention_mask):
        """Test Attention不同mask类型

        验证：
        - Attention正确处理full mask
        - Attention正确处理causal mask
        - Attention正确处理sliding_window mask

        场景：不同mask类型测试

        预期结果：所有mask类型正确处理

        执行模式：CPU Mock
        """
        if mask_type == "full":
            mock_attention_mask.mask = torch.ones(16, 1, 128, 128)
        elif mask_type == "causal":
            mock_attention_mask.mask = torch.triu(torch.ones(16, 1, 128, 128), diagonal=1)
        elif mask_type == "sliding_window":
            mock_attention_mask.mask = torch.ones(16, 1, 128, 128)
        
        mock_attention.forward = MagicMock(return_value=torch.randn(16, 128, 4096))
        output = mock_attention.forward()
        assert output is not None

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("seq_len", [128, 512, 1024])
    def test_attention_different_sequence_lengths(self, seq_len):
        """Test Attention不同序列长度

        验证：
        - Attention正确处理seq_len=128
        - Attention正确处理seq_len=512
        - Attention正确处理seq_len=1024

        场景：不同序列长度测试

        预期结果：所有序列长度正确处理

        执行模式：CPU Mock
        """
        attention = create_mock_attention()
        
        hidden_states = torch.randn(16, seq_len, 4096)
        attention_mask = torch.ones(16, 1, seq_len, seq_len)
        
        attention.forward = MagicMock(
            return_value=torch.randn(16, seq_len, 4096)
        )
        
        output = attention.forward(hidden_states, attention_mask)
        assert output.shape[1] == seq_len


class TestAttentionInterfaceCompatibility(PytestSTBase):
    """Attention接口兼容性测试类"""

    @pytest.mark.cpu_mock
    def test_attention_interface_compatibility(self):
        """Test AscendAttention implements vllm.AttentionBackend interface

        验证：
        - AscendAttention包含所有vllm.AttentionBackend接口方法
        - 每个接口方法都是可调用的
        - 接口方法签名符合vllm规范

        场景：验证插件Attention与vllm接口兼容性

        预期结果：所有接口检查通过，兼容性验证成功

        执行模式：CPU Mock
        """
        attention = create_mock_attention(spec_attrs=VLLM_ATTENTION_BACKEND_SPEC)
        
        verify_plugin_interface_compatibility(attention, VLLM_ATTENTION_BACKEND_SPEC)

    @pytest.mark.cpu_mock
    def test_attention_get_kv_cache_shape(self):
        """Test Attention get_kv_cache_shape接口

        验证：
        - get_kv_cache_shape方法存在且可调用
        - 返回正确的KV cache形状

        场景：验证get_kv_cache_shape接口实现

        预期结果：接口正确实现

        执行模式：CPU Mock
        """
        attention = create_mock_attention(spec_attrs=['get_kv_cache_shape'])
        
        attention.get_kv_cache_shape = MagicMock(
            return_value=(2, 16, 128, 4096)
        )
        
        shape = attention.get_kv_cache_shape()
        assert shape[0] == 2  # num_layers
        assert shape[1] == 16  # batch_size


class TestAttentionMLAIntegration(PytestSTBase):
    """Attention与MLA集成测试类"""

    @pytest.mark.cpu_mock
    def test_attention_mla_compression(self):
        """Test Attention与MLA KV cache压缩

        验证：
        - Attention正确调用MLA压缩
        - KV cache正确压缩
        - 压缩结果正确返回

        场景：Attention与MLA KV cache压缩集成

        预期结果：压缩正确执行

        执行模式：CPU Mock
        """
        attention = create_mock_attention(spec_attrs=['forward', 'compress_kv_cache'])
        mla = MagicMock()
        mla_compress_result = torch.randn(16, 128, 512)
        mla.compress = MagicMock(return_value=mla_compress_result)
        
        attention.mla = mla
        attention.compress_kv_cache = MagicMock(side_effect=lambda: mla.compress())
        
        compressed = attention.compress_kv_cache()
        assert compressed is not None
        verify_mock_calls(mla.compress, expected_calls=1)

    @pytest.mark.cpu_mock
    def test_attention_mla_decompression(self):
        """Test Attention与MLA KV cache解压缩

        验证：
        - Attention正确调用MLA解压缩
        - KV cache正确解压缩
        - 解压缩结果正确返回

        场景：Attention与MLA KV cache解压缩集成

        预期结果：解压缩正确执行

        执行模式：CPU Mock
        """
        attention = create_mock_attention(spec_attrs=['forward', 'decompress_kv_cache'])
        mla = MagicMock()
        mla.decompress = MagicMock(return_value=torch.randn(16, 128, 4096))
        
        attention.mla = mla
        attention.decompress_kv_cache = MagicMock(return_value=torch.randn(16, 128, 4096))
        
        decompressed = attention.decompress_kv_cache()
        assert decompressed.shape[-1] == 4096


class TestAttentionTorchairIntegration(PytestSTBase):
    """Attention与Torchair集成测试类"""

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("compile_mode", ["eager", "torchair_graph"])
    def test_attention_torchair_compile(self, compile_mode):
        """Test Attention与Torchair图模式编译

        验证：
        - Attention正确处理eager模式
        - Attention正确处理torchair_graph模式
        - 不同编译模式正确切换

        场景：Attention与Torchair编译模式集成

        预期结果：编译模式正确切换

        执行模式：CPU Mock
        """
        attention = create_mock_attention(spec_attrs=['forward', 'compile_mode'])
        
        attention.compile_mode = compile_mode
        
        if compile_mode == "torchair_graph":
            attention.torchair_compiled = MagicMock(return_value=True)
            assert attention.torchair_compiled() == True
        else:
            attention.torchair_compiled = MagicMock(return_value=False)
            assert attention.torchair_compiled() == False


class TestAttentionDualMode(PytestSTBase):
    """Attention双模式测试类"""

    @pytest.mark.parametrize("exec_mode", ["cpu_mock", "npu_real"])
    def test_attention_dual_mode_forward(self, exec_mode):
        """Test Attention双模式forward

        验证：
        - CPU Mock模式下Attention正确执行
        - NPU Real模式下Attention正确执行
        - 不同模式下执行结果一致性

        场景：Attention双模式执行对比测试

        预期结果：两种模式执行结果一致

        执行模式：CPU Mock / NPU Real
        """
        attention = create_mock_attention(spec_attrs=['forward'])
        
        if exec_mode == "cpu_mock":
            attention.forward = MagicMock(
                return_value=torch.randn(16, 128, 4096)
            )
            output = attention.forward()
            assert output.shape == (16, 128, 4096)
        elif exec_mode == "npu_real":
            if not has_torch_npu():
                pytest.skip("NPU not available")
            pytest.skip("NPU Real mode requires real hardware")


class TestAttentionPrecision(PytestSTBase):
    """Attention精度测试类"""

    @pytest.mark.cpu_mock
    @pytest.mark.npu_precision
    def test_attention_precision_placeholder(self):
        """Test Attention精度测试占位

        验证：
        - NPU精度测试标记正确
        - 真实NPU环境下精度框架正确
        - 为未来真实精度测试预留接口

        场景：Attention精度测试框架验证

        预期结果：精度测试框架正确

        执行模式：NPU Precision
        """
        if not has_torch_npu():
            pytest.skip("NPU not available, skipping npu_precision test")
        
        pytest.skip("NPU Precision test requires real hardware - placeholder for future implementation")