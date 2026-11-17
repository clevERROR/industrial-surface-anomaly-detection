"""让测试从项目根目录启动时也能导入 Main 下的模块。"""

from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
