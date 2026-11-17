"""使用原图多尺度颜色纹理、局部正常参考和真实空间单元汇总。"""

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from src.detail_features import detail_features, FEATURE_FLOORS, GROUP_SIZES
from src.postprocess import calibrate_threshold, top_cell_score


class DetailColorDetector:
    """只拟合正常图的参考库；四类采用同一套显式参数。"""

    method = 'detail_color'

    def __init__(self, config):
        self.settings = dict(config['localization'])
        self.options = dict(config['detail_color'])
        self.scales = tuple(self.options['scales'])
        self.cell_size = self.options['cell_size']
        self.random_seed = config['dataset']['random_seed']
        self.feature_args = dict(native_lab=True, native_texture=True)
        self.feature_dimension = 26
        self.pixel_threshold = self.image_threshold = None

    @staticmethod
    def _images(value):
        """新模式明确要求原图，避免错误地从缩放图推测细节。"""
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError('detail_color需要（缩放RGB，原始RGB）')
        image, original = value
        if (not isinstance(image, np.ndarray) or not isinstance(original, np.ndarray)
                or original.ndim != 3 or image.ndim != 3 or original.dtype != np.uint8
                or original.shape[2] != 3 or image.shape[2] != 3):
            raise ValueError('图像对必须是同源RGB数组')
        return image, original

    def fit(self, normal_images):
        """逐图正常采样；均值、尺度、距离单位均不使用验证或异常。"""
        rng = np.random.default_rng(self.random_seed)
        sampled = [[] for _ in self.scales]
        normal_count = 0
        total_cells = 0
        for value in normal_images:
            _, original = self._images(value)
            features, coordinates = detail_features(original, self.scales, self.cell_size)
            indexes = rng.choice(len(coordinates), min(self.options['samples_per_image'], len(coordinates)), replace=False)
            for rows, matrix in zip(sampled, features):
                rows.append(matrix[indexes])
            total_cells += len(coordinates)
            normal_count += 1
        if not normal_count:
            raise ValueError('至少需要一张正常训练图')
        weights = np.concatenate([np.full(size, 1 / np.sqrt(len(GROUP_SIZES) * size), np.float32)
                                  for size in GROUP_SIZES])
        self.banks = []
        for rows in sampled:
            raw = np.concatenate(rows)
            if len(raw) > self.options['reference_cap']:
                raw = raw[np.sort(rng.choice(len(raw), self.options['reference_cap'], replace=False))]
            if len(raw) < 4:
                raise ValueError('正常参考块不足4个')
            mean = raw.mean(axis=0)
            scale = np.maximum(raw.std(axis=0), FEATURE_FLOORS)
            reference = np.ascontiguousarray((raw - mean) / scale * weights)
            neighbors = NearestNeighbors(algorithm='brute', n_jobs=1).fit(reference)
            with threadpool_limits(limits=4):
                distances = neighbors.kneighbors(reference, n_neighbors=min(9, len(raw)), return_distance=True)[0][:, 1:]
            unit = max(float(np.median(distances.mean(axis=1))), .1)
            self.banks.append(dict(mean=mean, scale=scale, weights=weights, reference=reference,
                                   neighbors=neighbors, distance_unit=unit))
        self.training_patch_count = sum(len(bank['reference']) for bank in self.banks)
        self.training_image_count = normal_count
        self.training_cell_count = total_cells
        self.pixel_threshold = self.image_threshold = None
        return self

    def score(self, value):
        """原图特征直接匹配正常块，图像汇总与显示热力图分开。"""
        image, original = self._images(value)
        features, source_coordinates = detail_features(original, self.scales, self.cell_size)
        branches = []
        for matrix, bank in zip(features, self.banks):
            normalized = np.ascontiguousarray((matrix - bank['mean']) / bank['scale'] * bank['weights'])
            with threadpool_limits(limits=4):
                distance = bank['neighbors'].kneighbors(normalized, n_neighbors=min(3, len(bank['reference'])), return_distance=True)[0]
            branches.append(distance.mean(axis=1) / bank['distance_unit'])
        scores = np.max(branches, axis=0)
        source_height, source_width = original.shape[:2]
        columns = len(range(0, source_width, self.cell_size))
        grid = scores.reshape(-1, columns)
        full_map = np.repeat(np.repeat(grid, self.cell_size, axis=0), self.cell_size, axis=1)[:source_height, :source_width]
        height, width = image.shape[:2]
        heatmap = cv2.resize(full_map, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
        ratio = np.array([width / source_width, height / source_height] * 2)
        coordinates = source_coordinates.astype(float) * ratio
        coordinates[:, :2] = np.floor(coordinates[:, :2])
        coordinates[:, 2:] = np.ceil(coordinates[:, 2:])
        return dict(patch_scores=scores, coordinates=coordinates.astype(np.int32), heatmap=heatmap,
                    image_score=top_cell_score(scores, self.options['top_cells']),
                    source_coordinates=source_coordinates, source_image_shape=np.array(original.shape[:2]),
                    scale_scores=np.stack(branches), feature_scales=np.array(self.scales),
                    score_method='native_multiscale_color_texture_knn', image_pool_cells=self.options['top_cells'])

    def calibrate(self, normal_validation_images):
        """阈值只来自未扰动正常验证，不更新局部参考或特征变换。"""
        results = [self.score(value) for value in normal_validation_images]
        if not results:
            raise ValueError('正常验证图像不能为空')
        heatmaps = np.stack([r['heatmap'] for r in results])
        scores = np.array([r['image_score'] for r in results])
        self.image_threshold = calibrate_threshold(scores, self.settings['image_quantile'])
        self.pixel_threshold = calibrate_threshold(heatmaps, self.settings['pixel_quantile'], per_image=True)
        return dict(heatmaps=heatmaps, image_scores=scores, image_threshold=self.image_threshold,
                    pixel_threshold=self.pixel_threshold)

    def predict(self, value):
        """预测只使用图像对与已冻结阈值。"""
        if self.image_threshold is None:
            raise RuntimeError('请先用正常验证图像校准')
        result = self.score(value)
        result['image_prediction'] = int(result['image_score'] > self.image_threshold)
        return result

    def metadata(self):
        """导出原图尺度、参考来源和汇总支持，避免与旧协方差混淆。"""
        return dict(estimator='sklearn.neighbors.NearestNeighbors', method=self.method,
                    fit_labels_used=False, training_image_count=self.training_image_count,
                    training_cell_count=self.training_cell_count, training_patch_count=self.training_patch_count,
                    feature_dimension=26, feature_scales=list(self.scales), group_dimensions=list(GROUP_SIZES),
                    original_rgb_required=True, cell_size=self.cell_size, top_cells=self.options['top_cells'],
                    reference_counts=[len(bank['reference']) for bank in self.banks],
                    distance_units=[bank['distance_unit'] for bank in self.banks],
                    pixel_calibration='per_image_then_across_images', source='normal_training_only')
