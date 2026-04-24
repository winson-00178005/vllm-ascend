"""
Session级别 Fixtures - Environment First + Model Centric + Parallel

关键特性:
- environment: Session级别，环境信息只加载一次
- loaded_models: Session级别，模型实例共享
- model_runner: Function级别，获取当前模型的runner
"""
import pytest
import logging

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def environment(request):
    """
    Session级别 Fixture - 环境信息
    
    只在 session 开始时设置一次环境
    """
    from framework.environment_manager import EnvironmentManager
    
    runner_label = request.config.getoption("--runner", default=None)
    
    if runner_label is None:
        pytest.skip("No --runner specified")
        return None
    
    env_manager = EnvironmentManager()
    env_manager.load_runners()
    
    try:
        env = env_manager.set_environment(runner_label)
        logger.info(f"Environment fixture: {env}")
        return env
    except ValueError as e:
        pytest.fail(f"Invalid runner: {e}")
        return None


@pytest.fixture(scope="session")
def loaded_models(request, environment):
    """
    Session级别 Fixture - 加载的模型
    
    在 session 开始时加载所有模型
    所有测试共享同一实例
    """
    from framework.model_registry import ModelRegistry
    
    if environment is None:
        pytest.skip("No environment available")
        return {}
    
    models = request.config.getoption("--models", default=None)
    
    if models is None:
        models = ["qwen-7b"]  # Default model
    
    ModelRegistry.initialize()
    loaded = ModelRegistry.load_models(models, environment)
    
    logger.info(f"Loaded models fixture: {loaded}")
    
    yield ModelRegistry
    
    # Cleanup at session end
    ModelRegistry.cleanup()
    logger.info("Session cleanup: models cleaned up")


@pytest.fixture(scope="function")
def model_runner(request, loaded_models):
    """
    Function级别 Fixture - 当前模型runner
    
    获取当前测试使用的模型实例
    """
    from framework.model_registry import ModelRegistry
    
    model_name = request.config.getoption("--model", default=None)
    
    if model_name is None:
        pytest.skip("No --model specified")
        return None
    
    runner = ModelRegistry.get_model(model_name)
    
    if runner is None:
        pytest.skip(f"Model {model_name} not loaded or not available")
        return None
    
    return runner


@pytest.fixture(scope="function")
def model_name(request):
    """Function级别 Fixture - 当前模型名称"""
    return request.config.getoption("--model", default=None)


@pytest.fixture(scope="function")
def model_config(model_name, loaded_models):
    """Function级别 Fixture - 当前模型配置"""
    from framework.model_registry import ModelRegistry
    
    if model_name is None:
        pytest.skip("No model specified")
        return None
    
    return ModelRegistry.get_model_config(model_name)


@pytest.fixture(scope="function")
def environment_info(environment):
    """Function级别 Fixture - 环境信息"""
    return environment


__all__ = [
    "environment",
    "loaded_models",
    "model_runner",
    "model_name",
    "model_config",
    "environment_info",
]