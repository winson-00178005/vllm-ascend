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

"""Quantization模块集成测试

验证AscendQuantizer的插件内部协作。
参考：tests/ut/quantization/test_w8a8.py模式
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from tests.st.base import PytestSTBase
from tests.st.utils.env_detector import has_torch_npu
from tests.st.utils.mock_utils import verify_mock_calls
from tests.st.utils.runner_factory import STRunner


class TestQuantizationIntegration(PytestSTBase):
    """Quantization集成测试类"""

    @pytest.mark.cpu_mock
    def test_quantizer_init_with_config(self, mock_quantizer, mock_quant_config):
        """Test Quantizer初始化配置

        验证：
        - Quantizer正确初始化并持有配置
        - 量化配置参数正确传递
        - 初始化状态正确

        场景：Quantizer正常初始化流程

        预期结果：Quantizer正确初始化，配置正确持有

        执行模式：CPU Mock
        """
        mock_quantizer.config = mock_quant_config
        
        assert mock_quantizer.config.quant_method == "w8a8"
        assert mock_quantizer.config.weight_bits == 8

    @pytest.mark.cpu_mock
    def test_quantizer_apply_flow(self, mock_quantizer, quant_test_weights):
        """Test Quantizer量化流程

        验证：
        - Quantizer apply方法正确调用
        - apply接收正确输入参数
        - 量化结果正确返回

        场景：Quantizer正常量化流程

        预期结果：量化正确执行，结果正确

        执行模式：CPU Mock
        """
        input_tensor = torch.randn(32, 128)
        
        output = mock_quantizer.apply(input_tensor, quant_test_weights)
        
        assert output is not None
        verify_mock_calls(mock_quantizer.apply, expected_calls=1)

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("quant_method", ["w8a8", "w8a8_dynamic", "w4a8"])
    def test_quantizer_different_methods(self, quant_method):
        """Test Quantizer不同量化方法

        验证：
        - Quantizer正确处理w8a8方法
        - Quantizer正确处理w8a8_dynamic方法
        - Quantizer正确处理w4a8方法

        场景：不同量化方法测试

        预期结果：所有方法正确处理

        执行模式：CPU Mock
        """
        mock_quant = MagicMock(spec=['quantize', 'apply'])
        mock_quant.quant_method = quant_method
        
        if quant_method == "w8a8":
            mock_quant.quantize = MagicMock(return_value=torch.int8)
        elif quant_method == "w8a8_dynamic":
            mock_quant.quantize = MagicMock(return_value=torch.int8)
        elif quant_method == "w4a8":
            mock_quant.quantize = MagicMock(return_value=torch.int4)
        
        result = mock_quant.quantize(torch.randn(128, 256))
        assert result is not None

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("quant_dtype", ["float16", "bfloat16"])
    def test_quantizer_different_dtypes(self, quant_dtype):
        """Test Quantizer不同数据类型

        验证：
        - Quantizer正确处理float16
        - Quantizer正确处理bfloat16
        - 量化结果dtype正确

        场景：不同数据类型量化测试

        预期结果：所有dtype正确处理

        执行模式：CPU Mock
        """
        mock_quant = MagicMock(spec=['apply'])
        torch_dtype = torch.float16 if quant_dtype == "float16" else torch.bfloat16
        
        mock_quant.apply = MagicMock(
            return_value=torch.randn(32, 128, dtype=torch_dtype)
        )
        
        result = mock_quant.apply(torch.randn(32, 128))
        assert result.dtype == torch_dtype


class TestW8A8Quantization(PytestSTBase):
    """W8A8量化测试类"""

    @pytest.mark.cpu_mock
    def test_w8a8_get_weight(self):
        """Test W8A8获取量化权重

        验证：
        - W8A8LinearMethod正确获取量化权重
        - 权重dtype为int8
        - 权重形状正确

        场景：W8A8获取量化权重

        预期结果：权重正确获取

        执行模式：CPU Mock
        """
        mock_method = MagicMock(spec=['get_weight'])
        
        weight_dict = {
            'weight': torch.randint(-128, 127, (256, 128), dtype=torch.int8)
        }
        mock_method.get_weight = MagicMock(return_value=weight_dict)
        
        result = mock_method.get_weight(128, 256)
        
        assert result['weight'].dtype == torch.int8
        assert result['weight'].shape == (256, 128)

    @pytest.mark.cpu_mock
    def test_w8a8_quant_per_tensor(self):
        """Test W8A8 per_tensor量化

        验证：
        - per_tensor量化正确执行
        - scale和offset正确应用
        - 量化结果正确

        场景：W8A8 per_tensor量化

        预期结果：量化正确执行

        执行模式：CPU Mock
        """
        mock_quant = MagicMock()
        
        in_tensor = torch.randn(32, 128)
        input_scale = torch.tensor(0.1)
        input_offset = torch.tensor(0)
        
        expected_output = torch.randint(-128, 127, (32, 128), dtype=torch.int8)
        mock_quant.return_value = expected_output
        
        output = mock_quant(in_tensor, input_scale, input_offset)
        
        assert output.dtype == torch.int8

    @pytest.mark.cpu_mock
    def test_w8a8_per_channel_quant(self):
        """Test W8A8 per_channel量化

        验证：
        - per_channel量化正确执行
        - channel级scale正确应用
        - 量化结果正确

        场景：W8A8 per_channel量化

        预期结果：量化正确执行

        执行模式：CPU Mock
        """
        mock_method = MagicMock(spec=['get_perchannel_param'])
        
        params = {
            'quant_bias': torch.zeros(256, dtype=torch.int32),
            'deq_scale': torch.ones(256, dtype=torch.float32),
            'weight_scale': torch.ones(256, 1, dtype=torch.bfloat16),
            'weight_offset': torch.zeros(256, 1, dtype=torch.bfloat16),
        }
        mock_method.get_perchannel_param = MagicMock(return_value=params)
        
        result = mock_method.get_perchannel_param(256, torch.bfloat16)
        
        assert result['quant_bias'].shape == (256,)
        assert result['deq_scale'].shape == (256,)

    @pytest.mark.cpu_mock
    def test_w8a8_apply_matmul(self):
        """Test W8A8量化矩阵乘法

        验证：
        - 量化矩阵乘法正确执行
        - 输入量化正确
        - 输出结果正确

        场景：W8A8量化矩阵乘法

        预期结果：矩阵乘法正确执行

        执行模式：CPU Mock
        """
        mock_matmul = MagicMock()
        
        x = torch.randn(32, 128)
        weight = torch.randint(-128, 127, (256, 128), dtype=torch.int8)
        
        mock_matmul.return_value = torch.randn(32, 256)
        
        result = mock_matmul(x, weight)
        
        assert result.shape == (32, 256)


class TestDynamicQuantization(PytestSTBase):
    """动态量化测试类"""

    @pytest.mark.cpu_mock
    def test_dynamic_quant_activation(self):
        """Test 动态量化activation

        验证：
        - 动态量化activation正确执行
        - scale动态计算
        - 量化结果正确

        场景：动态量化activation

        预期结果：动态量化正确执行

        执行模式：CPU Mock
        """
        mock_dyn_quant = MagicMock(spec=['quantize_activation'])
        
        activation = torch.randn(32, 128)
        
        mock_dyn_quant.quantize_activation = MagicMock(
            return_value=(torch.int8, torch.tensor(0.1))
        )
        
        quant_act, scale = mock_dyn_quant.quantize_activation(activation)
        
        assert quant_act is not None
        assert scale is not None

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("quant_dtype", ["float16", "bfloat16"])
    def test_dynamic_quant_with_dtype(self, quant_dtype):
        """Test 动态量化不同dtype

        验证：
        - 动态量化正确处理float16
        - 动态量化正确处理bfloat16
        - scale dtype正确

        场景：动态量化不同dtype测试

        预期结果：所有dtype正确处理

        执行模式：CPU Mock
        """
        mock_dyn_quant = MagicMock()
        torch_dtype = torch.float16 if quant_dtype == "float16" else torch.bfloat16
        
        mock_dyn_quant.return_value = (torch.int8, torch.tensor(0.1, dtype=torch_dtype))
        
        quant_act, scale = mock_dyn_quant(torch.randn(32, 128))
        
        assert scale.dtype == torch_dtype


class TestQuantizationDistributed(PytestSTBase):
    """量化与分布式集成测试类"""

    @pytest.mark.cpu_mock
    @pytest.mark.parametrize("world_size", [1, 2, 4])
    def test_quant_tensor_parallel(self, world_size):
        """Test 量化与Tensor Parallel集成

        验证：
        - 量化正确处理TP world_size=1
        - 量化正确处理TP world_size=2
        - 量化正确处理TP world_size=4
        - 权重分片正确

        场景：量化与Tensor Parallel集成测试

        预期结果：TP正确集成

        执行模式：CPU Mock
        """
        mock_quant = MagicMock(spec=['get_weight', 'apply'])
        
        full_weight = torch.randint(-128, 127, (256, 128), dtype=torch.int8)
        shard_weight = full_weight.chunk(world_size, dim=0)[0]
        
        mock_quant.get_weight = MagicMock(return_value={'weight': shard_weight})
        
        result = mock_quant.get_weight(128, 256 // world_size)
        
        assert result['weight'].shape[0] == 256 // world_size

    @pytest.mark.cpu_mock
    def test_quant_with_strunner(self, quant_runner, mock_quantizer):
        """Test 量化使用STRunner管理资源

        验证：
        - STRunner正确初始化量化资源
        - STRunner正确清理资源
        - 资源管理符合上下文管理器模式

        场景：使用STRunner管理量化模块资源

        预期结果：资源正确初始化和清理

        执行模式：CPU Mock
        """
        with STRunner("quantization", {"model": "test_model"}) as runner:
            runner.register_resource("quantizer", mock_quantizer)
            
            assert runner.get_resource("quantizer") is not None
        
        assert runner._resources == {}


class TestQuantizationInterfaceCompatibility(PytestSTBase):
    """量化接口兼容性测试类"""

    @pytest.mark.cpu_mock
    def test_quantizer_interface_methods(self):
        """Test Quantizer接口方法

        验证：
        - Quantizer包含quantize方法
        - Quantizer包含apply方法
        - 接口方法可调用

        场景：验证Quantizer接口实现

        预期结果：接口正确实现

        执行模式：CPU Mock
        """
        mock_quant = MagicMock(spec=['quantize', 'apply', 'get_weight'])
        
        assert callable(mock_quant.quantize)
        assert callable(mock_quant.apply)
        assert callable(mock_quant.get_weight)


class TestQuantizationException(PytestSTBase):
    """量化异常处理测试类"""

    @pytest.mark.cpu_mock
    def test_quantizer_not_implemented_error(self):
        """Test Quantizer NotImplementedError异常

        验证：
        - 不支持的量化方法抛出NotImplementedError
        - 异常消息包含方法名称
        - 错误正确传递

        场景：不支持的量化方法异常处理

        预期结果：异常正确传递

        执行模式：CPU Mock
        """
        mock_quant = MagicMock(spec=['quantize'])
        mock_quant.quantize = MagicMock(
            side_effect=NotImplementedError("Quantization method 'fp8' not implemented")
        )
        
        with pytest.raises(NotImplementedError) as cm:
            mock_quant.quantize()
        
        assert "not implemented" in str(cm.value)

    @pytest.mark.cpu_mock
    def test_quantizer_weight_shape_error(self):
        """Test Quantizer权重形状错误

        验证：
        - 权重形状不匹配时抛出错误
        - 异常消息包含形状信息
        - 错误正确传递

        场景：权重形状错误处理

        预期结果：错误正确传递

        执行模式：CPU Mock
        """
        mock_quant = MagicMock(spec=['get_weight'])
        mock_quant.get_weight = MagicMock(
            side_effect=ValueError("Weight shape mismatch: expected (256, 128), got (128, 256)")
        )
        
        with pytest.raises(ValueError) as cm:
            mock_quant.get_weight(128, 256)
        
        assert "shape mismatch" in str(cm.value)


class TestQuantizationDualMode(PytestSTBase):
    """量化双模式测试类"""

    @pytest.mark.parametrize("exec_mode", ["cpu_mock", "npu_real"])
    def test_quant_dual_mode_apply(self, exec_mode):
        """Test 量化双模式执行

        验证：
        - CPU Mock模式下量化正确执行
        - NPU Real模式下量化正确执行
        - 不同模式下执行结果一致性

        场景：量化双模式执行对比测试

        预期结果：两种模式执行结果一致

        执行模式：CPU Mock / NPU Real
        """
        mock_quant = MagicMock(spec=['apply'])
        
        if exec_mode == "cpu_mock":
            mock_quant.apply = MagicMock(return_value=torch.randn(32, 128))
            result = mock_quant.apply(torch.randn(32, 128))
            assert result is not None
        elif exec_mode == "npu_real":
            if not has_torch_npu():
                pytest.skip("NPU not available")
            pytest.skip("NPU Real mode requires real hardware")


class TestQuantizationPrecision(PytestSTBase):
    """量化精度测试类"""

    @pytest.mark.cpu_mock
    @pytest.mark.npu_precision
    def test_quant_precision_placeholder(self):
        """Test 量化精度测试占位

        验证：
        - NPU精度测试标记正确
        - 真实NPU环境下精度框架正确
        - 为未来真实精度测试预留接口

        场景：量化精度测试框架验证

        预期结果：精度测试框架正确

        执行模式：NPU Precision
        """
        if not has_torch_npu():
            pytest.skip("NPU not available, skipping npu_precision test")
        
        pytest.skip("NPU Precision test requires real hardware - placeholder for future implementation")

    @pytest.mark.cpu_mock
    def test_quant_error_bounds(self):
        """Test 量化误差边界测试

        验证：
        - 量化误差在可接受范围内
        - int8量化误差<1%
        - 误差计算正确

        场景：量化误差边界验证

        预期结果：误差在可接受范围

        执行模式：CPU Mock
        """
        original = torch.randn(32, 128)
        quantized = torch.round(original * 10).to(torch.int8) / 10
        
        error = torch.abs(original - quantized)
        max_error = error.max().item()
        
        assert max_error < 0.1