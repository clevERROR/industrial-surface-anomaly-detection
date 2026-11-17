"""仅用正常验证分数校准阈值并生成缺陷掩膜。"""

import numpy as np


def calibrate_threshold(normal_scores, quantile: float = 0.995, *, per_image: bool = False) -> float:
    """仅用正常验证分位校准，可显式按每图分位再跨图校准。"""
    scores = np.asarray(normal_scores, dtype=np.float64)
    if not 0 < quantile < 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("分位数应在0和1之间，正常验证分数必须非空且有限")
    if per_image:
        if scores.ndim != 3:
            raise ValueError('按图像分层校准需要图数、高、宽组成的三维数组')
        per_image_scores = np.quantile(scores.reshape(scores.shape[0], -1), quantile, axis=1)
        return float(np.quantile(per_image_scores, quantile))
    return float(np.quantile(scores, quantile))


def anomaly_mask(heatmap: np.ndarray, threshold: float) -> np.ndarray:
    """使用严格大于阈值的规则，返回只含0和255的灰度掩膜。"""
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all() or not np.isfinite(threshold):
        raise ValueError("热力图必须是二维有限数组，阈值必须有限")
    return ((heatmap > threshold) * 255).astype(np.uint8)


def top_cell_score(scores, top_cells: int = 3) -> float:
    """按不同空间单元取最高均值，不把同一单元重复像素当多个证据。"""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all() or top_cells <= 0:
        raise ValueError('空间单元分数须非空有限，汇总单元数须为正')
    count = min(int(top_cells), len(scores))
    return float(np.partition(scores, len(scores) - count)[-count:].mean())


def image_score(heatmap: np.ndarray, top_fraction: float = 0.01, *, top_pixels: int | None = None) -> float:
    """取最高比例均值，原图细节可显式指定与物理支持对应的像素数。"""
    values = np.asarray(heatmap).ravel()
    if not values.size or not np.isfinite(values).all() or not 0 < top_fraction <= 1:
        raise ValueError("热力图必须非空且有限，聚合比例应在0和1之间")
    count = max(1, int(np.ceil(values.size * top_fraction)))
    if top_pixels is not None:
        if not isinstance(top_pixels, (int, np.integer)) or isinstance(top_pixels, bool) or top_pixels <= 0:
            raise ValueError('汇总像素数必须为正整数')
        count = min(int(top_pixels), values.size)
    return float(np.partition(values, values.size - count)[-count:].mean())
