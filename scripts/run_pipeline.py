"""第3部分的完整运行入口：正常训练到测试图像的局部定位导出。"""

from pathlib import Path
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_localization import main


if __name__ == '__main__':
    main()
