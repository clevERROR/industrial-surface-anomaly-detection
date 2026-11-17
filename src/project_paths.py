"""以 Main 为独立工程根目录解析路径，运行产物默认写入工程内 runs 目录。"""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT
RUNS_ROOT = PROJECT_ROOT / 'runs'
DEFAULT_CONFIG = PROJECT_ROOT / 'configs/config.yaml'


def resolve_path(path) -> Path:
    """项目相对路径按项目根目录解析；外部工具的绝对路径仅在运行时使用。"""
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def relative_path(path) -> str:
    """以独立工程根目录为锚点，输入和外部输出均保存为相对路径。"""
    value = resolve_path(path).resolve()
    return Path(os.path.relpath(value, PROJECT_ROOT)).as_posix()
