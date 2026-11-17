"""将局部块异常分数还原为图像空间热力图。"""

import cv2
import numpy as np


def aggregate_scores(scores, coordinates, image_shape) -> np.ndarray:
    """重叠区域按覆盖次数平均；分数越高表示越异常。"""
    scores = np.asarray(scores, dtype=np.float64)
    coordinates = np.asarray(coordinates)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("块分数必须是非空、有限的一维数组")
    if coordinates.shape != (len(scores), 4) or not np.issubdtype(coordinates.dtype, np.integer):
        raise ValueError("每个块必须有四个整数坐标")
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("图像尺寸必须为正")
    total = np.zeros((height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.int32)
    for score, (x1, y1, x2, y2) in zip(scores, coordinates):
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("块坐标越界或面积不为正")
        total[y1:y2, x1:x2] += score
        count[y1:y2, x1:x2] += 1
    if np.any(count == 0):
        raise ValueError("图像存在未被窗口覆盖的像素")
    return (total / count).astype(np.float32)


def resize_heatmap(heatmap: np.ndarray, original_shape) -> np.ndarray:
    """使用线性插值将连续分数还原到原始图像尺寸。"""
    height, width = original_shape[:2]
    if height <= 0 or width <= 0 or heatmap.ndim != 2:
        raise ValueError("热力图及原图尺寸无效")
    return cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
