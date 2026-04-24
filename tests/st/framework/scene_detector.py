"""
场景自动检测器 - 从测试文件路径和代码自动识别场景

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

功能:
- 从测试文件路径检测场景（tests/e2e/310p/singlecard -> SINGLE_CARD_310P）
- 从测试代码检测参数（tensor_parallel_size等）
- 提供综合场景检测
"""
import re
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class SceneDetector:
    """
    场景自动检测器
    
    针对vllm-ascend的tests/e2e/目录结构进行场景检测
    """
    
    @staticmethod
    def detect_from_path(test_file_path: str) -> Tuple[str, Optional['Scene']]:
        """
        从测试文件路径检测场景
        
        Args:
            test_file_path: 测试文件路径
        
        Returns:
            (场景名称, 场景对象)
        
        检测规则:
        - tests/e2e/310p/singlecard/ -> SINGLE_CARD_310P
        - tests/e2e/310p/multicard/ -> MULTI_CARD_310P（根据子目录进一步区分）
        - tests/e2e/singlecard/ -> SINGLE_CARD_310P（默认）
        - tests/e2e/multicard/2-cards/ -> MULTI_CARD_310P_2
        - tests/e2e/multicard/4-cards/ -> MULTI_CARD_310P_4
        - tests/e2e/nightly/single_node/ -> NIGHTLY_SINGLE_NODE
        - tests/e2e/weekly/single_node/ -> WEEKLY_SINGLE_NODE
        """
        path = Path(test_file_path)
        scene_name = None
        
        # 获取路径的各个部分
        parts = path.parts
        
        # 检测310P场景
        if '310p' in parts:
            if 'singlecard' in parts:
                scene_name = 'SINGLE_CARD_310P'
            elif 'multicard' in parts:
                # 根据子目录进一步区分
                if '2-cards' in parts or '2_cards' in parts:
                    scene_name = 'MULTI_CARD_310P_2'
                elif '4-cards' in parts or '4_cards' in parts:
                    scene_name = 'MULTI_CARD_310P_4'
                else:
                    scene_name = 'MULTI_CARD_310P_2'  # 默认2卡
        
        # 检测常规NPU场景（默认310P）
        elif 'singlecard' in parts:
            scene_name = 'SINGLE_CARD_310P'
        elif 'multicard' in parts:
            if '2-cards' in parts or '2_cards' in parts:
                scene_name = 'MULTI_CARD_310P_2'
            elif '4-cards' in parts or '4_cards' in parts:
                scene_name = 'MULTI_CARD_310P_4'
            else:
                scene_name = 'MULTI_CARD_310P_2'  # 默认2卡
        
        # 检测nightly/weekly场景
        if 'nightly' in parts:
            if 'multi_node' in parts:
                scene_name = 'MULTI_NODE_NPU'
            elif 'single_node' in parts or not scene_name:
                scene_name = 'NIGHTLY_SINGLE_NODE'
        elif 'weekly' in parts:
            if 'single_node' in parts or not scene_name:
                scene_name = 'WEEKLY_SINGLE_NODE'
        
        # 获取场景对象
        from .scene_manager import SceneManager
        scene_manager = SceneManager()
        scene = scene_manager.get_scene(scene_name) if scene_name else None
        
        return scene_name, scene
    
    @staticmethod
    def detect_from_code(test_code: str) -> Tuple[str, Optional[Dict]]:
        """
        从测试代码检测场景参数
        
        Args:
            test_code: 测试函数的源代码
        
        Returns:
            (模型名称, 场景参数字典)
        
        检测内容:
        - tensor_parallel_size参数
        - enforce_eager参数
        - 模型名称
        """
        # 检测tensor_parallel_size参数
        tp_match = re.search(r'tensor_parallel_size\s*=\s*(\d+)', test_code)
        tensor_parallel_size = int(tp_match.group(1)) if tp_match else 1
        
        # 检测模型名称
        # 尝试多种模式匹配模型名称
        model_name = None
        
        # 模式1: 直接在VllmRunner第一个参数
        model_match = re.search(r'VllmRunner\(\s*"([^"]+)"', test_code)
        if model_match:
            model_name = model_match.group(1)
        else:
            # 模式2: model=参数
            model_match = re.search(r'model\s*=\s*"([^"]+)"', test_code)
            if model_match:
                model_name = model_match.group(1)
        
        # 检测enforce_eager参数
        eager_match = re.search(r'enforce_eager\s*=\s*(True|False)', test_code)
        enforce_eager = eager_match.group(1) == 'True' if eager_match else False
        
        # 检测quantization参数
        quant_match = re.search(r'quantization\s*=\s*"([^"]+)"', test_code)
        quantization = quant_match.group(1) if quant_match else None
        
        # 检测max_model_len参数
        len_match = re.search(r'max_model_len\s*=\s*(\d+)', test_code)
        max_model_len = int(len_match.group(1)) if len_match else None
        
        params = {
            'tensor_parallel_size': tensor_parallel_size,
            'enforce_eager': enforce_eager,
            'quantization': quantization,
            'max_model_len': max_model_len
        }
        
        return model_name, params
    
    @staticmethod
    def detect_all(test_file_path: str, test_code: str = '') -> Dict:
        """
        综合检测场景信息
        
        Args:
            test_file_path: 测试文件路径
            test_code: 测试函数源代码（可选）
        
        Returns:
            包含场景名称、场景对象、模型名称、参数的字典
        """
        # 从路径检测场景
        scene_name, scene = SceneDetector.detect_from_path(test_file_path)
        
        # 从代码检测参数
        if test_code:
            model_name, params = SceneDetector.detect_from_code(test_code)
        else:
            model_name = None
            params = {}
        
        return {
            'scene_name': scene_name,
            'scene': scene,
            'model_name': model_name,
            'params': params,
            'test_file': test_file_path
        }
    
    @staticmethod
    def infer_scene_from_params(params: Dict) -> Optional[str]:
        """
        从参数推断场景名称
        
        Args:
            params: 参数字典
        
        Returns:
            推断的场景名称
        
        规则:
        - tensor_parallel_size=1 -> SINGLE_CARD_310P
        - tensor_parallel_size=2 -> MULTI_CARD_310P_2
        - tensor_parallel_size=4 -> MULTI_CARD_310P_4
        """
        tensor_parallel_size = params.get('tensor_parallel_size', 1)
        
        if tensor_parallel_size == 1:
            return 'SINGLE_CARD_310P'
        elif tensor_parallel_size == 2:
            return 'MULTI_CARD_310P_2'
        elif tensor_parallel_size == 4:
            return 'MULTI_CARD_310P_4'
        elif tensor_parallel_size >= 8:
            return 'SINGLE_NODE_MULTI_CARD'
        else:
            return None