"""通过文件字节解码图片，兼容 Windows 中文路径。"""

import cv2
import numpy as np

from src.project_paths import resolve_path


def read_image(path, flags=cv2.IMREAD_COLOR):
    """文件系统路径由 Python 处理，避免 OpenCV 原生文件名编码限制。"""
    data = resolve_path(path).read_bytes()
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
