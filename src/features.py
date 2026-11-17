import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage import img_as_float
import cv2


def to_gray(img: np.ndarray) -> np.ndarray:
    """
    兼容输入：RGB(H,W,3) 或者已经是灰度(H,W)
    返回单通道灰度图 (H,W)
    约定：预处理输出为 uint8，RGB 顺序。
    """
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def extract_lbp(img: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """提取 LBP 特征，支持 RGB 或灰度输入。"""
    gray = to_gray(img)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2),
    )
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def extract_hog(img: np.ndarray) -> np.ndarray:
    """提取 HOG 特征，支持 RGB 或灰度输入。"""
    gray = to_gray(img)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return hog_feat.astype(np.float32)


def extract_gray_stat(img: np.ndarray) -> np.ndarray:
    """提取灰度均值、标准差、最大值和最小值。"""
    gray = to_gray(img)
    return np.array(
        [np.mean(gray), np.std(gray), np.max(gray), np.min(gray)],
        dtype=np.float32,
    )


def extract_gradient_stat(img: np.ndarray) -> np.ndarray:
    """梯度统计特征。"""
    gray = to_gray(img)
    img_float = img_as_float(gray)
    gx, gy = np.gradient(img_float)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.array(
        [np.mean(grad_mag), np.std(grad_mag), np.max(grad_mag)],
        dtype=np.float32,
    )


def extract_spatial_detail(img: np.ndarray) -> np.ndarray:
    """保留逐像素灰度及带符号梯度，避免块内统计池化丢失细节位置。

    输入为至少2×2的uint8灰度图或RGB图，不缩放；返回float32一维向量，
    依次为灰度/255、水平差分、垂直差分，维数为3×高×宽。
    这是独立可选表达，不替换旧组合特征，不能直接传给已拟合的旧维数模型。
    """
    if (not isinstance(img, np.ndarray) or img.dtype != np.uint8
            or img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] != 3)
            or min(img.shape[:2]) < 2):
        raise ValueError("空间细节特征需要至少2×2的uint8灰度或RGB图")
    gray = to_gray(img).astype(np.float32) / 255.0
    gradient_y, gradient_x = np.gradient(gray)
    return np.concatenate([gray.ravel(), gradient_x.ravel(), gradient_y.ravel()])


def combine_features(
    img: np.ndarray,
    use_lbp: bool = True,
    use_hog: bool = True,
    use_gray_stat: bool = True,
    use_grad_stat: bool = True,
) -> np.ndarray:
    """
    拼接多组特征。
    img: RGB(H,W,3) 或灰度(H,W)。
    return: 一维特征向量。
    """
    feat_list = []
    if use_lbp:
        feat_list.append(extract_lbp(img))
    if use_hog:
        feat_list.append(extract_hog(img))
    if use_gray_stat:
        feat_list.append(extract_gray_stat(img))
    if use_grad_stat:
        feat_list.append(extract_gradient_stat(img))

    if not feat_list:
        raise ValueError("至少需要启用一种特征")

    return np.concatenate(feat_list, axis=0)


class FeatureStandardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X_train: np.ndarray):
        if X_train.size == 0:
            raise ValueError("FeatureStandardizer.fit 输入为空")

        # 高精度累加避免宽float32矩阵的常量列因舍入误差产生伪方差。
        self.mean = np.mean(X_train, axis=0, dtype=np.float64)
        std = np.std(X_train, axis=0, dtype=np.float64)
        # 零方差特征直接置 1，避免除零。
        self.std = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("请先调用 fit")
        return (X - self.mean) / self.std

    def fit_transform(self, X_train: np.ndarray):
        self.fit(X_train)
        return self.transform(X_train)
