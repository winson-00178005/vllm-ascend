"""
inference测试conftest - 配置inference测试的导入路径
"""
import sys
from pathlib import Path

st_path = Path(__file__).parent.parent.parent
if str(st_path) not in sys.path:
    sys.path.insert(0, str(st_path))