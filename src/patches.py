"""生成保留空间坐标且完整覆盖图像的滑动窗口。"""

from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np


@dataclass(frozen=True)
class Patch:
    """图像块及其半开区间坐标，坐标顺序为左、上、右、下。"""

    image: np.ndarray
    coordinates: tuple[int, int, int, int]


def _starts(length: int, size: int, stride: int) -> list[int]:
    """补齐最后一个窗口，避免不能整除步长时遗漏右边或下边。"""
    if size <= 0 or stride <= 0 or size > length or stride > size:
        raise ValueError("窗口和步长必须为正，窗口不得超过图像，步长不得大于窗口")
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def iter_patches(image: np.ndarray, patch_size: int = 64, stride: int = 32) -> Iterator[Patch]:
    """按从上到下、从左到右的顺序返回图像块，支持灰度和 RGB 图像。"""
    if image.ndim not in (2, 3):
        raise ValueError("输入必须是二维灰度图或三维彩色图")
    if not isinstance(patch_size, int) or not isinstance(stride, int):
        raise ValueError("窗口和步长必须是整数")
    height, width = image.shape[:2]
    for y in _starts(height, patch_size, stride):
        for x in _starts(width, patch_size, stride):
            yield Patch(image[y:y + patch_size, x:x + patch_size],
                        (x, y, x + patch_size, y + patch_size))
