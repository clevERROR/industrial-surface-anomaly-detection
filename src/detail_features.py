"""在原始RGB像素上提取紧凑的多尺度颜色和纹理统计。"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


GROUP_SIZES = (6, 2, 4, 4, 10)
FEATURE_FLOORS = np.array([2.] * 6 + [1 / 255] * 2 + [1 / 255] * 8 + [.01] * 10, np.float32)


def detail_features(original, scales=(16, 32, 64), cell_size=16):
    """返回每尺度26维特征与原图空间单元；边缘不足整块也保留。"""
    if original.dtype != np.uint8 or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError('细节特征需要uint8原始RGB图像')
    height, width = original.shape[:2]
    if min(height, width) < 2 or cell_size <= 0 or any(size <= 0 for size in scales):
        raise ValueError('原图尺寸、网格及尺度必须有效')
    coordinates = np.array([(x, y, min(x + cell_size, width), min(y + cell_size, height))
                            for y in range(0, height, cell_size)
                            for x in range(0, width, cell_size)], np.int32)
    x = (coordinates[:, 0] + coordinates[:, 2]) // 2
    y = (coordinates[:, 1] + coordinates[:, 3]) // 2
    gray8 = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    gray = gray8.astype(np.float32) / 255
    lab = cv2.cvtColor(original.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    magnitude = np.sqrt(gx * gx + gy * gy)
    residuals = [gray - cv2.GaussianBlur(gray, (0, 0), sigma) for sigma in (1., 3.)]
    lbp = local_binary_pattern(gray8, 8, 1, method='uniform').astype(np.uint8)
    result = []
    for size in scales:
        def average(field):
            """先在原像素上汇总，再采样网格中心，保留缩放前的局部信号。"""
            return cv2.boxFilter(field, cv2.CV_32F, (size, size), normalize=True,
                                 borderType=cv2.BORDER_REFLECT_101)[y, x]

        fields = []
        means, deviations = [], []
        for channel in cv2.split(lab):
            mean = average(channel)
            means.append(mean)
            deviations.append(np.sqrt(np.maximum(average(channel * channel) - mean * mean, 0)))
        fields.extend(means + deviations)
        mean = average(gray)
        fields.extend([mean, np.sqrt(np.maximum(average(gray * gray) - mean * mean, 0))])
        xx, yy, xy = average(gx * gx), average(gy * gy), average(gx * gy)
        discriminant = np.sqrt(np.maximum((xx - yy) ** 2 + 4 * xy ** 2, 0))
        fields.extend([average(magnitude), np.sqrt(np.maximum(xx + yy, 0)),
                       np.sqrt(np.maximum((xx + yy + discriminant) / 2, 0)),
                       np.sqrt(np.maximum((xx + yy - discriminant) / 2, 0))])
        for residual in residuals:
            fields.extend([average(np.abs(residual)), np.sqrt(np.maximum(average(residual ** 2), 0))])
        fields.extend([average((lbp == code).astype(np.float32)) for code in range(10)])
        features = np.stack(fields, axis=1).astype(np.float32)
        if features.shape[1] != 26 or not np.isfinite(features).all():
            raise ValueError('原图细节特征非有限或维数异常')
        result.append(features)
    return result, coordinates
