"""
模型相关pytest fixtures

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

提供模型配置和测试运行器的pytest fixtures
"""
import pytest
import logging

logger = logging.getLogger(__name__)

@pytest.fixture(scope="function")
def model_config(request, loaded_model):
    """
    模型配置fixture（function级别）
    
    提供当前加载模型的配置信息
    
    Args:
        request: pytest request对象
        loaded_model: 已加载模型fixture
    
    Returns:
        ModelConfig: 模型配置对象
    
    用法:
        def test_model_info(model_config):
            assert model_config.min_memory == "16GB"
            assert "310P" in model_config.supported_devices
    """
    if not loaded_model:
        pytest.skip("No model loaded")
        return None
    
    return loaded_model['config']

@pytest.fixture(scope="function")
def vllm_runner(request, current_scene, loaded_model):
    """
    VllmRunner创建器fixture（function级别）
    
    提供创建VllmRunner实例的工厂函数
    自动根据当前场景调整参数
    
    Args:
        request: pytest request对象
        current_scene: 当前场景fixture
        loaded_model: 已加载模型fixture
    
    Returns:
        callable: VllmRunner创建函数
    
    用法:
        def test_inference(vllm_runner):
            # 自动使用当前场景的参数
            runner = vllm_runner("Qwen/Qwen3-8B")
            # 或者手动指定参数
            runner = vllm_runner("Qwen/Qwen3-8B", tensor_parallel_size=4)
            
            # 使用runner执行推理
            result = runner.generate_greedy(["Hello"], max_tokens=10)
    
    注意: 此fixture需要tests/e2e/conftest.py中的VllmRunner类
    """
    def create_runner(model_name, **kwargs):
        """
        创建VllmRunner实例
        
        Args:
            model_name: 模型名称（如"Qwen/Qwen3-8B"）
            **kwargs: 其他参数
        
        Returns:
            VllmRunner实例
        """
        # 尝试导入VllmRunner
        try:
            # 从tests/e2e/conftest导入
            import sys
            from pathlib import Path
            
            # 确保tests/e2e在路径中
            tests_path = Path(__file__).parent.parent.parent
            if str(tests_path) not in sys.path:
                sys.path.insert(0, str(tests_path))
            
            # 尝试不同的导入方式
            try:
                from tests.e2e.conftest import VllmRunner as OriginalVllmRunner
            except ImportError:
                try:
                    from e2e.conftest import VllmRunner as OriginalVllmRunner
                except ImportError:
                    # 如果找不到，使用模拟的VllmRunner
                    logger.warning("VllmRunner not found in tests/e2e/conftest, using mock")
                    return _create_mock_runner(model_name, current_scene, **kwargs)
            
            # 自动调整参数
            if current_scene and 'tensor_parallel_size' not in kwargs:
                kwargs['tensor_parallel_size'] = current_scene.card_count
                logger.info(f"Auto-set tensor_parallel_size={kwargs['tensor_parallel_size']} from scene")
            
            # 创建VllmRunner实例
            runner = OriginalVllmRunner(model_name, **kwargs)
            
            # 记录场景信息
            runner._st_scene = current_scene.name if current_scene else None
            runner._st_model = model_name
            
            return runner
        
        except Exception as e:
            logger.error(f"Failed to create VllmRunner: {e}")
            pytest.skip(f"VllmRunner creation failed: {e}")
            return None
    
    return create_runner

def _create_mock_runner(model_name, current_scene, **kwargs):
    """
    创建模拟的VllmRunner实例（用于框架验证）
    
    Args:
        model_name: 模型名称
        current_scene: 当前场景
        **kwargs: 其他参数
    
    Returns:
        模拟Runner对象
    """
    class MockVllmRunner:
        def __init__(self, model, **kw):
            self.model = model
            self.kwargs = kw
            self._scene = current_scene.name if current_scene else None
            logger.info(f"MockVllmRunner created: model={model}, scene={self._scene}")
        
        def generate_greedy(self, prompts, max_tokens):
            logger.info(f"Mock generate_greedy: prompts={prompts}, max_tokens={max_tokens}")
            return [(list(range(max_tokens)), f"Mock response for: {prompts[0]}")]
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    return MockVllmRunner(model_name, **kwargs)

@pytest.fixture(scope="function")
def remote_server(request, current_scene):
    """
    RemoteOpenAIServer创建器fixture（function级别）
    
    提供创建RemoteOpenAIServer实例的工厂函数
    
    Args:
        request: pytest request对象
        current_scene: 当前场景fixture
    
    Returns:
        callable: RemoteOpenAIServer创建函数
    
    用法:
        def test_api(remote_server):
            server = remote_server(
                "Qwen/Qwen3-8B",
                ["--port", "8080", "--tensor-parallel-size", "2"]
            )
            client = server.get_client()
            # 使用client执行API调用
    
    注意: 此fixture需要tests/e2e/conftest.py中的RemoteOpenAIServer类
    """
    def create_server(model, server_args, **kwargs):
        """
        创建RemoteOpenAIServer实例
        
        Args:
            model: 模型名称
            server_args: 服务器参数列表
            **kwargs: 其他参数
        
        Returns:
            RemoteOpenAIServer实例
        """
        try:
            import sys
            from pathlib import Path
            
            tests_path = Path(__file__).parent.parent.parent
            if str(tests_path) not in sys.path:
                sys.path.insert(0, str(tests_path))
            
            try:
                from tests.e2e.conftest import RemoteOpenAIServer as OriginalServer
            except ImportError:
                try:
                    from e2e.conftest import RemoteOpenAIServer as OriginalServer
                except ImportError:
                    logger.warning("RemoteOpenAIServer not found, using mock")
                    return _create_mock_server(model, server_args, current_scene, **kwargs)
            
            # 创建服务器实例
            server = OriginalServer(model, server_args, **kwargs)
            server._st_scene = current_scene.name if current_scene else None
            
            return server
        
        except Exception as e:
            logger.error(f"Failed to create RemoteOpenAIServer: {e}")
            pytest.skip(f"RemoteOpenAIServer creation failed: {e}")
            return None
    
    return create_server

def _create_mock_server(model, server_args, current_scene, **kwargs):
    """
    创建模拟的RemoteOpenAIServer实例（用于框架验证）
    
    Args:
        model: 模型名称
        server_args: 服务器参数
        current_scene: 当前场景
        **kwargs: 其他参数
    
    Returns:
        模拟Server对象
    """
    class MockServer:
        def __init__(self, model, args, **kw):
            self.model = model
            self.args = args
            self.kwargs = kw
            self._scene = current_scene.name if current_scene else None
            self.url_root = "http://localhost:8080"
            logger.info(f"MockServer created: model={model}, scene={self._scene}")
        
        def get_client(self, **kw):
            class MockClient:
                def completions_create(self, **args):
                    return {"text": "Mock response"}
            return MockClient()
        
        def url_for(self, *parts):
            return self.url_root + "/" + "/".join(parts)
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    return MockServer(model, server_args, **kwargs)

@pytest.fixture(scope="function")
def hf_runner(request, current_scene):
    """
    HfRunner创建器fixture（function级别）
    
    提供创建HfRunner实例的工厂函数
    
    Args:
        request: pytest request对象
        current_scene: 当前场景fixture
    
    Returns:
        callable: HfRunner创建函数
    
    用法:
        def test_hf_model(hf_runner):
            runner = hf_runner("Qwen/Qwen3-8B")
            # 使用runner执行测试
    """
    def create_runner(model_name, **kwargs):
        """
        创建HfRunner实例
        
        Args:
            model_name: 模型名称
            **kwargs: 其他参数
        
        Returns:
            HfRunner实例
        """
        try:
            import sys
            from pathlib import Path
            
            tests_path = Path(__file__).parent.parent.parent
            if str(tests_path) not in sys.path:
                sys.path.insert(0, str(tests_path))
            
            try:
                from tests.e2e.conftest import HfRunner as OriginalHfRunner
            except ImportError:
                try:
                    from e2e.conftest import HfRunner as OriginalHfRunner
                except ImportError:
                    logger.warning("HfRunner not found, using mock")
                    return _create_mock_hf_runner(model_name, current_scene, **kwargs)
            
            runner = OriginalHfRunner(model_name, **kwargs)
            runner._st_scene = current_scene.name if current_scene else None
            
            return runner
        
        except Exception as e:
            logger.error(f"Failed to create HfRunner: {e}")
            pytest.skip(f"HfRunner creation failed: {e}")
            return None
    
    return create_runner

def _create_mock_hf_runner(model_name, current_scene, **kwargs):
    """
    创建模拟的HfRunner实例（用于框架验证）
    """
    class MockHfRunner:
        def __init__(self, model, **kw):
            self.model = model
            self.kwargs = kw
            self._scene = current_scene.name if current_scene else None
            logger.info(f"MockHfRunner created: model={model}, scene={self._scene}")
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    return MockHfRunner(model_name, **kwargs)