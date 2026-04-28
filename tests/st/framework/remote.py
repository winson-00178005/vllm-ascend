#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#

"""
ST Framework 远程推理与进程管理模块

基于上游 vLLM tests/utils.py 移植:
1. RemoteVLLMServer - 远程 vLLM 服务器生命周期管理
2. 子进程测试管理 - fork/spawn 新进程运行测试
3. 多进程并行测试支持
"""

import contextlib
import copy
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional, TypeVar, Any

import pytest

from tests.st.framework.platform import NPUPlatform


_P = TypeVar("_P")


class RemoteVLLMServer:
    """远程 vLLM 服务器管理

    用于启动和管理 vLLM 推理服务器进程，支持:
    - 自动端口分配
    - 健康检查
    - 进程生命周期管理
    - GPU 内存追踪

    用法:
    with RemoteVLLMServer(model="Qwen/Qwen3-8B") as server:
        client = server.get_client()
        result = client.chat.completions.create(...)
    """

    DUMMY_API_KEY = "token-abc123"

    def __init__(
        self,
        model: str,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.85,
        env_dict: Optional[dict[str, str]] = None,
        seed: int = 0,
        max_wait_seconds: float = 480,
        extra_args: Optional[list[str]] = None,
    ):
        self.model = model
        self.host = host
        self.port = port or self._get_free_port()
        self.tensor_parallel_size = tensor_parallel_size
        self.trust_remote_code = trust_remote_code
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.env_dict = env_dict or {}
        self.seed = seed
        self.max_wait_seconds = max_wait_seconds
        self.extra_args = extra_args or []

        self.proc: Optional[subprocess.Popen] = None
        self._pre_server_memory: Optional[float] = None

    @staticmethod
    def _get_free_port() -> int:
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _build_command(self) -> list[str]:
        cmd = [
            "vllm", "serve",
            self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--trust-remote-code",
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--seed", str(self.seed),
            *self.extra_args,
        ]
        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env.update(self.env_dict)
        return env

    def _wait_until_ready(self):
        import requests
        url = f"http://{self.host}:{self.port}/health"
        start = time.time()

        while True:
            try:
                if requests.get(url, timeout=2).status_code == 200:
                    return
            except Exception:
                pass

            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError("Server process died during startup")

            if time.time() - start > self.max_wait_seconds:
                raise RuntimeError(f"Server failed to start within {self.max_wait_seconds}s")

            time.sleep(0.5)

    def __enter__(self):
        memory_info = NPUPlatform.get_memory_info(0)
        if memory_info:
            self._pre_server_memory = memory_info.used_gb
            print(f"[RemoteVLLMServer] GPU memory before server start: {self._pre_server_memory:.2f} GB")

        cmd = self._build_command()
        env = self._build_env()

        print(f"[RemoteVLLMServer] Launching with: {' '.join(cmd)}")

        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        try:
            self._wait_until_ready()
        except Exception:
            self._shutdown()
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._shutdown()

    def _shutdown(self):
        if self.proc is None:
            return

        pid = self.proc.pid

        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, OSError):
            pgid = None

        with contextlib.suppress(ProcessLookupError, OSError):
            self.proc.terminate()
            print(f"[RemoteVLLMServer] Sent SIGTERM to process {pid}")

        try:
            self.proc.wait(timeout=15)
            print(f"[RemoteVLLMServer] Server {pid} terminated gracefully")
        except subprocess.TimeoutExpired:
            print(f"[RemoteVLLMServer] Server {pid} did not respond to SIGTERM, sending SIGKILL")
            if pgid is not None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(pgid, signal.SIGKILL)
            else:
                self.proc.kill()

            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

        self._wait_for_memory_release()
        self.proc = None

    def _wait_for_memory_release(self, timeout: float = 120.0):
        if self._pre_server_memory is None:
            return

        start = time.time()
        while time.time() - start < timeout:
            memory_info = NPUPlatform.get_memory_info(0)
            if memory_info is None:
                return

            current = memory_info.used_gb
            if current <= self._pre_server_memory + 2.0:
                print(f"[RemoteVLLMServer] GPU memory released: {current:.2f} GB")
                return

            time.sleep(1.0)

        print(f"[RemoteVLLMServer] WARNING: GPU memory not released within {timeout}s")

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def get_client(self, **kwargs):
        from openai import OpenAI
        return OpenAI(
            base_url=f"{self.url_root}/v1",
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )

    def get_async_client(self, **kwargs):
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            base_url=f"{self.url_root}/v1",
            api_key=self.DUMMY_API_KEY,
            max_retries=0,
            **kwargs,
        )


class VllmRunnerProcess:
    """vLLM Runner 进程封装

    用于在独立进程中运行 vLLM 推理任务，支持:
    - fork/spawn 模式
    - 异常传播
    - 内存追踪

    用法:
    @VllmRunnerProcess.spawn()
    def test_inference():
        from vllm import LLM, SamplingParams
        llm = LLM(model="Qwen/Qwen3-8B")
        outputs = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    """

    @staticmethod
    def spawn() -> Callable[[Callable[_P, None]], Callable[_P, None]]:
        """装饰器：在 spawn 子进程中运行测试"""
        def decorator(func: Callable[_P, None]) -> Callable[_P, None]:
            @functools.wraps(func)
            def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
                import multiprocessing as mp
                with suppress(RuntimeError):
                    mp.set_start_method("spawn")

                module_name = func.__module__
                import cloudpickle

                with tempfile.TemporaryDirectory() as tempdir:
                    output_file = os.path.join(tempdir, "output.pkl")

                    input_bytes = cloudpickle.dumps((func, args, kwargs, output_file))

                    repo_root = str(Path(__file__).parent.parent.parent.parent)
                    env = dict(os.environ)
                    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
                    env["RUNNING_IN_SUBPROCESS"] = "1"

                    cmd = [sys.executable, "-m", module_name]

                    result = subprocess.run(
                        cmd,
                        input=input_bytes,
                        capture_output=True,
                        env=env,
                    )

                    try:
                        result.check_returncode()
                    except Exception as e:
                        raise RuntimeError(f"Subprocess error:\n{result.stderr.decode()}") from e

            return wrapper
        return decorator


@contextlib.contextmanager
def suppress(*exceptions):
    """上下文管理器：静默忽略指定异常"""
    try:
        yield
    except exceptions:
        pass


class MultiNodeRunner:
    """多节点测试运行器

    用于在多个节点上运行分布式测试

    用法:
    def worker(rank, size):
        init_distributed_environment(rank, size)
        # ... test logic ...

    runner = MultiNodeRunner(num_nodes=2)
    runner.run(worker)
    """

    def __init__(self, num_nodes: int = 1, tp_size: int = 1):
        self.num_nodes = num_nodes
        self.tp_size = tp_size

    def run(self, target: Callable):
        """在多个进程中运行目标函数"""
        import ray

        os.environ["RAY_RUNTIME_ENV_IGNORE_GITIGNORE"] = "1"
        ray.init(runtime_env={"working_dir": str(Path(__file__).parent.parent.parent.parent)})

        refs = []
        for rank in range(self.num_nodes * self.tp_size):
            refs.append(target.remote(rank, self.tp_size))

        ray.get(refs)
        ray.shutdown()


class CompareSettings:
    """多配置对比测试工具

    用于对比不同配置下的推理结果

    用法:
    def test_quantization():
        compare = CompareSettings(model="Qwen/Qwen3-8B")
        compare.add_setting("--quantization", "ascend_w8a8")
        compare.add_setting("--quantization", "none")
        results = compare.run()
        assert results[0] == results[1]
    """

    def __init__(self, model: str):
        self.model = model
        self.settings: list[dict[str, Any]] = []

    def add_setting(self, **kwargs):
        """添加一个配置组合"""
        self.settings.append(kwargs)

    def run(self, prompt: str = "Hello, my name is"):
        """运行所有配置并返回结果"""
        results = []
        for i, setting in enumerate(self.settings):
            args = []
            for k, v in setting.items():
                args.extend([k, str(v)])

            with RemoteVLLMServer(model=self.model, extra_args=args) as server:
                client = server.get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0,
                )
                results.append(response.choices[0].message.content)

        return results


@pytest.fixture
def remote_vllm_server():
    """远程 vLLM 服务器 fixture

    用法:
    def test_server(remote_vllm_server):
        client = remote_vllm_server.get_client()
        # ... use client ...
    """
    class ServerContext:
        def __init__(self):
            self._server = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            if self._server:
                self._server.__exit__(*args)

        def create(self, **kwargs):
            self._server = RemoteVLLMServer(**kwargs)
            self._server.__enter__()
            return self._server

    return ServerContext()
