"""
ST测试执行脚本 - Environment First + Model Centric + Parallel Execution

Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
Licensed under the Apache License, Version 2.0

执行流程:
1. 设置环境（选择 Runner）
2. 加载模型（Session级别，只加载一次）
3. 收集测试（按模型过滤）
4. 并发执行（pytest-xdist）
5. 清理资源

用法:
    python scripts/run_model_tests.py \
        --runner linux-aarch64-310p-2 \
        --models qwen-7b qwen3-8b \
        --workers 4
"""
import argparse
import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent
ST_DIR = SCRIPTS_DIR.parent
TESTS_DIR = ST_DIR.parent
VLLM_ASCEND_ROOT = TESTS_DIR.parent


def main():
    parser = argparse.ArgumentParser(
        description="ST Test Executor - Environment First + Model Centric + Parallel"
    )
    
    parser.add_argument(
        "--runner",
        required=True,
        help="Runner label (e.g., linux-aarch64-310p-2)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Models to load (e.g., qwen-7b qwen3-8b)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers (default: 4)"
    )
    
    parser.add_argument(
        "--test-dir",
        default=str(ST_DIR / "testcases"),
        help="Test directory"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests without running"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("ST Test Executor - Environment First + Model Centric + Parallel")
    print("=" * 70)
    
    # Step 1: Initialize Environment
    print("\n[Step 1] Initializing Environment")
    print("-" * 70)
    
    sys.path.insert(0, str(ST_DIR))
    from framework.environment_manager import EnvironmentManager
    from framework.model_registry import ModelRegistry
    from framework.test_registry import TestRegistry
    
    env_manager = EnvironmentManager()
    env_manager.load_runners()
    
    try:
        env = env_manager.set_environment(args.runner)
        print(f"Environment: {env}")
        print(f"  NPU Type: {env.npu_type.display_name}")
        print(f"  NPU Count: {env.num_npus}")
    except ValueError as e:
        print(f"ERROR: {e}")
        env_manager.list_environments()
        sys.exit(1)
    
    # Step 2: Load Models (Session level)
    print("\n[Step 2] Loading Models")
    print("-" * 70)
    print(f"Models requested: {args.models}")
    
    loaded_models = ModelRegistry.load_models(args.models, env)
    
    if not loaded_models:
        print("ERROR: No models loaded successfully")
        sys.exit(1)
    
    print(f"Models loaded: {loaded_models}")
    
    # Step 3: Collect Tests per Model
    print("\n[Step 3] Collecting Tests")
    print("-" * 70)
    
    TestRegistry.load_config()
    
    test_plan = {}
    for model_name in loaded_models:
        tests = TestRegistry.get_tests_for_model(model_name)
        test_plan[model_name] = tests
        print(f"Model {model_name}: {len(tests)} tests")
    
    # List mode - just show plan
    if args.list:
        print("\n" + "=" * 70)
        print("Test Plan Summary")
        print("=" * 70)
        
        total_tests = sum(len(t) for t in test_plan.values())
        print(f"\nTotal tests to run: {total_tests}")
        
        for model_name, tests in test_plan.items():
            print(f"\n[{model_name}] {len(tests)} tests:")
            for test in tests:
                print(f"  - {test}")
        
        # Cleanup
        ModelRegistry.cleanup()
        sys.exit(0)
    
    # Step 4: Run Tests (Parallel)
    print("\n[Step 4] Running Tests")
    print("-" * 70)
    print(f"Parallel workers: {args.workers}")
    
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for model_name in loaded_models:
        print(f"\nRunning tests for model: {model_name}")
        
        # Build pytest command
        pytest_cmd = [
            "python", "-m", "pytest",
            args.test_dir,
            "-n", str(args.workers),  # Parallel workers
            "--dist", "loadscope",     # Scope-based distribution
            f"--model={model_name}",   # Current model
            f"--runner={args.runner}", # Environment
            "--tb=short",
            "-v",
        ]
        
        print(f"Command: {' '.join(pytest_cmd)}")
        
        # Run pytest
        result = subprocess.run(pytest_cmd, cwd=str(VLLM_ASCEND_ROOT))
        
        if result.returncode == 0:
            print(f"✅ Tests passed for {model_name}")
        else:
            print(f"❌ Some tests failed for {model_name}")
    
    # Step 5: Cleanup
    print("\n[Step 5] Cleanup")
    print("-" * 70)
    
    ModelRegistry.cleanup()
    print("Model registry cleaned up")
    
    # Summary
    print("\n" + "=" * 70)
    print("Execution Complete")
    print("=" * 70)
    print(f"Environment: {env.label}")
    print(f"Models tested: {loaded_models}")
    print("=" * 70)


if __name__ == "__main__":
    main()