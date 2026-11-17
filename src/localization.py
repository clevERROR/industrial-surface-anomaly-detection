"""接入正常局部评分，保留IF、官方协方差和可选原图细节支路。"""

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from threadpoolctl import threadpool_limits

from src.features import combine_features, extract_spatial_detail, FeatureStandardizer, to_gray
from src.heatmap import aggregate_scores
from src.patches import iter_patches
from src.postprocess import calibrate_threshold, image_score


def create_detector(config):
    """保留旧检测器接口，新原图颜色模式使用独立实现。"""
    if config['localization'].get('method') == 'detail_color':
        from src.detail_color import DetailColorDetector
        return DetailColorDetector(config)
    return LocalAnomalyDetector(config)


class LocalAnomalyDetector:
    """正常训练与校准；原图方法接收（缩放RGB，原图RGB）。"""

    native_size = 16
    native_reference_cap = 5000
    native_top_cells = 3
    native_context_size = 32

    def __init__(self, config):
        self.settings = dict(config['localization'])
        self.random_seed = config['dataset']['random_seed']
        self.patch_size = self.settings['patch_size']
        self.stride = self.settings['stride']
        features = config['features']
        self.feature_args = dict(use_lbp=features['lbp'], use_hog=features['hog'],
                                 use_gray_stat=features['gray_statistics'],
                                 use_grad_stat=features['gradient_statistics'])
        self.use_spatial_detail = bool(features.get('spatial_detail', False))
        self.standardizer = FeatureStandardizer()
        self.pca = PCA(n_components=0.95, svd_solver='full') if features.get('pca', False) else None
        self.method = self.settings.get('method', 'iforest')
        self.reference_matching = self.settings.get('reference_matching', 'full')
        if self.reference_matching not in ('full', 'exclude_gradient_statistics'):
            raise ValueError(f'未知的正常参照匹配方式: {self.reference_matching}')
        if self.reference_matching != 'full':
            if self.method != 'native_reference':
                raise ValueError('排除梯度统计的参照匹配只适用于native_reference')
            if not any(self.feature_args[name] for name in ('use_lbp', 'use_hog', 'use_gray_stat')):
                raise ValueError('正常参照匹配需要保留LBP、HOG或灰度特征')
        self.reference_neighbors = None
        if self.use_spatial_detail and self.method not in ('native_detail', 'native_context', 'native_reference'):
            raise ValueError('spatial_detail需要使用支持原图的native检测器')
        if self.method == 'native_reference' and self.pca is not None:
            raise ValueError('native_reference需要保留中心与邻域特征，请关闭PCA')
        if self.method in ('iforest', 'native_detail', 'native_context', 'native_reference'):
            self.model = IsolationForest(n_estimators=self.settings['n_estimators'],
                                         random_state=config['dataset']['random_seed'], n_jobs=-1)
        elif self.method == 'metric_covariance':
            from src.normal_metric import CovarianceScorer
            self.model = CovarianceScorer()
        else:
            raise ValueError(f'未知的局部评分方法: {self.method}')
        self.native_mode = False
        self.pixel_threshold = None
        self.image_threshold = None
        self.reference_image_thresholds = None
        self.reference_pixel_thresholds = None

    def _features(self, image):
        """同一尺度内所有窗口使用完全相同的特征维度。"""
        patches = list(iter_patches(image, self.patch_size, self.stride))
        features = np.stack([combine_features(patch.image, **self.feature_args) for patch in patches])
        coordinates = np.array([patch.coordinates for patch in patches], dtype=np.int32)
        return features, coordinates

    def fit(self, normal_images):
        """只在正常训练图像上拟合标准化、可选 PCA 和参考检测器。"""
        if self.method in ('native_detail', 'native_context', 'native_reference'):
            return self._fit_native(normal_images)
        features = np.concatenate([self._features(image)[0] for image in normal_images])
        self.training_patch_count, self.feature_dimension = features.shape
        normal = self.standardizer.fit_transform(features)
        if self.pca is not None:
            normal = self.pca.fit_transform(normal)
        self.model.fit(normal)
        self.pixel_threshold = self.image_threshold = None
        return self

    def score(self, image):
        """评分只接收图像；原图支路及既有方法均不拟合参数。"""
        image, original = self._images(image)
        if getattr(self, 'native_mode', False):
            return self._native_score(image, original)
        features, coordinates = self._features(image)
        features = self.standardizer.transform(features)
        if self.pca is not None:
            features = self.pca.transform(features)
        scores = -self.model.decision_function(features)
        heatmap = aggregate_scores(scores, coordinates, image.shape)
        return dict(patch_scores=scores, coordinates=coordinates, heatmap=heatmap,
                    image_score=image_score(heatmap, self.settings['image_top_fraction']))

    def calibrate(self, normal_validation_images):
        """正常验证只校准阈值；原图像素阈值按图像分层校准。"""
        results = [self.score(image) for image in normal_validation_images]
        if not results:
            raise ValueError('至少需要一张正常验证图像')
        heatmaps = np.stack([result['heatmap'] for result in results])
        scores = np.array([result['image_score'] for result in results])
        native = getattr(self, 'native_mode', False)
        if native and self.method == 'native_reference':
            components = np.stack([item['component_heatmaps'] for item in results])
            image_components = np.stack([item['component_image_scores'] for item in results])
            self.reference_image_thresholds = np.quantile(image_components, self.settings['image_quantile'], axis=0)
            self.reference_pixel_thresholds = np.array([
                calibrate_threshold(components[:, branch], self.settings['pixel_quantile'], per_image=True)
                for branch in range(2)])
            if (self.reference_image_thresholds <= 0).any() or (self.reference_pixel_thresholds <= 0).any():
                raise ValueError('正常验证的两个参照分支阈值必须为正')
            heatmaps = np.max(components / self.reference_pixel_thresholds[None, :, None, None], axis=1)
            scores = np.max(image_components / self.reference_image_thresholds, axis=1)
            self.pixel_threshold = self.image_threshold = 1.0
        else:
            self.pixel_threshold = calibrate_threshold(heatmaps, self.settings['pixel_quantile'], per_image=native)
            self.image_threshold = calibrate_threshold(scores, self.settings['image_quantile'])
        result = dict(heatmaps=heatmaps, image_scores=scores,
                      pixel_threshold=self.pixel_threshold, image_threshold=self.image_threshold)
        if native:
            result.update(score_method='native_foreground_local_density', native_patch_size=self.native_size,
                          image_pool_cells=self.native_top_cells, source_image_shape=np.array(self.native_source_shape),
                          reference_indices=self.native_reference_indices.copy(),
                          reference_pool_size=self.native_reference_pool_size,
                          group_weights=self.group_weights.copy(), geometry_profile=self.native_geometry_profile.copy(),
                          pixel_calibration='imagewise_quantile')
            if self.method in ('native_context', 'native_reference'):
                result.update(score_method='native_context_local_density',
                              context_patch_size=self.native_context_size,
                              image_pool_cells=4, image_pool_method='max_adjacent2x2_mean')
            if self.method == 'native_reference':
                result.update(score_method='native_reference_conditional',
                              image_pool_method='max_adjacent2x2_branch_normalized',
                              component_image_scores=image_components,
                              branch_image_thresholds=self.reference_image_thresholds.copy(),
                              branch_pixel_thresholds=self.reference_pixel_thresholds.copy(),
                              branch_names=np.array(['context_density', 'conditional_center_z']),
                              center_dimension=self.native_center_dimension,
                              conditional_variance_floor=0.01)
                if getattr(self, 'reference_neighbors', None) is not None:
                    result.update(reference_matching=self.reference_matching,
                                  reference_matching_dimension=self.reference_neighbors.n_features_in_,
                                  reference_density_dimension=self.model.n_features_in_)
        return result

    def predict(self, image):
        """预测仅依赖输入图像和已冻结阈值，不读测试标签或标注。"""
        if self.image_threshold is None:
            raise RuntimeError("请先使用正常验证数据调用 calibrate")
        result = self.score(image)
        result['image_prediction'] = int(result['image_score'] > self.image_threshold)
        return result

    @staticmethod
    def _images(value):
        """显式拆分图像对，旧数组调用不引入原图依赖。"""
        if isinstance(value, tuple) and len(value) == 2:
            image, original = value
            if not isinstance(image, np.ndarray) or not isinstance(original, np.ndarray):
                raise ValueError('图像对必须包含缩放图和同源原图数组')
            return image, original
        if not isinstance(value, np.ndarray):
            raise ValueError('输入必须是图像数组或（缩放图，原图）')
        return value, None


    @staticmethod
    def _foreground(image):
        """用最大暗连通区域估计前景，不接收人工标注。"""
        gray = to_gray(image)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return np.zeros(gray.shape, dtype=bool)
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == label


    @classmethod
    def _native_active(cls, original):
        """原图规则网格与前景占比，空前景时保留全部块供评分。"""
        mask = cls._foreground(original)
        height, width = mask.shape
        size = cls.native_size
        active = (mask.reshape(height // size, size, width // size, size)
                  .mean(axis=(1, 3)) >= 0.2).ravel()
        valid = bool(active.any())
        if not valid:
            active[:] = True
        return active, valid


    @classmethod
    def _source_scale(cls, image, original):
        """只启用可精确映射至缩放图整数坐标的高分辨原图。"""
        if original is None or original.dtype != np.uint8 or image.dtype != np.uint8:
            return None
        height, width = image.shape[:2]
        source_height, source_width = original.shape[:2]
        if not height or not width or source_height % height or source_width % width:
            return None
        scale = source_height // height
        if (scale <= 1 or source_width // width != scale or cls.native_size % scale
                or source_height % cls.native_size or source_width % cls.native_size):
            return None
        return scale


    def _weights(self, dimension):
        """在原图密度距离中按特征组维数均衡，不改原提取函数。"""
        sizes = [10 if self.feature_args['use_lbp'] else 0, 0,
                 4 if self.feature_args['use_gray_stat'] else 0,
                 3 if self.feature_args['use_grad_stat'] else 0]
        if self.feature_args['use_hog']:
            sizes[1] = dimension - sum(sizes)
        sizes = [size for size in sizes if size > 0]
        if not sizes or sum(sizes) != dimension:
            raise ValueError('特征组维数不一致')
        return np.concatenate([np.full(size, 1 / np.sqrt(len(sizes) * size), np.float32)
                               for size in sizes])


    def _fit_native(self, normal_images):
        """正常训练决定支路并拟合变换，不从验证或测试读取参考块。"""
        self.reference_neighbors = None
        blocks, areas, aspects, good = [], [], [], []
        priorities = np.empty(0)
        reference_ids = np.empty((0, 3), np.int64)
        crops = None
        rng = np.random.default_rng(self.random_seed)
        pool_size = 0
        compatible = True
        signature = None
        context = self.method in ('native_context', 'native_reference')
        for image_index, value in enumerate(normal_images):
            image, original = self._images(value)
            blocks.append(self._features(image)[0])
            scale = self._source_scale(image, original)
            current = (image.shape[:2], original.shape[:2], scale) if scale else None
            if signature is None and current is not None:
                signature = current
            compatible = compatible and current is not None and current == signature
            if scale is None:
                good.append(False)
                continue
            mask = self._foreground(image)
            y, x = np.where(mask)
            area = float(mask.mean())
            aspect = 0.0
            if len(x):
                _, (width, height), _ = cv2.minAreaRect(np.stack([x, y], axis=1).astype(np.float32))
                aspect = max(width, height) / max(min(width, height), 1.0)
            suitable = 0.05 <= area <= 0.5 and aspect >= 3.0
            areas.append(area)
            aspects.append(aspect)
            good.append(suitable)
            if not suitable:
                continue
            active, _ = self._native_active(original)
            positions = np.flatnonzero(active)
            columns = original.shape[1] // self.native_size
            margin = (self.native_context_size - self.native_size) // 2 if context else 0
            source = np.pad(original, ((margin, margin), (margin, margin), (0, 0)), mode='reflect') if context else original
            crop_size = self.native_context_size if context else self.native_size
            new_crops = np.stack([source[(p // columns) * self.native_size:(p // columns) * self.native_size + crop_size,
                                         (p % columns) * self.native_size:(p % columns) * self.native_size + crop_size]
                                   for p in positions])
            ids = np.column_stack([np.arange(pool_size, pool_size + len(positions)),
                                   np.full(len(positions), image_index), positions])
            pool_size += len(positions)
            priorities = np.concatenate([priorities, rng.random(len(positions))])
            reference_ids = np.concatenate([reference_ids, ids])
            crops = new_crops if crops is None else np.concatenate([crops, new_crops])
            if len(priorities) > self.native_reference_cap:
                keep = np.argpartition(priorities, self.native_reference_cap - 1)[:self.native_reference_cap]
                priorities, reference_ids, crops = priorities[keep], reference_ids[keep], crops[keep]
        if not blocks:
            raise ValueError('至少需要一张正常训练图像')
        self.native_mode = bool(compatible and np.mean(good) >= 0.9 and crops is not None)
        self.native_geometry_profile = np.array([np.mean(good), np.median(areas) if areas else 0,
                                                np.median(aspects) if aspects else 0])
        self.parent_training_patch_count = sum(len(block) for block in blocks)
        if self.native_mode:
            # 按原始池顺序恢复参考块，保证流式采样与正常实验一致。
            order = np.argsort(reference_ids[:, 0])
            crops, reference_ids = crops[order], reference_ids[order]
            features = np.stack([self._native_feature(np.rot90(crop, rotation))
                                 for crop in crops for rotation in (0, 3, 2, 1)])
            self.native_reference_indices = reference_ids
            self.native_reference_pool_size = pool_size
            self.native_image_shape, self.native_source_shape, self.native_scale = signature
            if context:
                fine_dimension = self._native_center_feature(
                    np.zeros((self.native_size, self.native_size, 3), np.uint8)).size
                self.group_weights = np.concatenate([self._native_center_weights(fine_dimension),
                    self._weights(features.shape[1] - fine_dimension)]) / np.sqrt(2)
            else:
                self.group_weights = self._native_center_weights(features.shape[1])
        else:
            features = np.concatenate(blocks)
        self.training_patch_count, self.feature_dimension = features.shape
        normal = self.standardizer.fit_transform(features)
        if self.native_mode:
            normal = normal * self.group_weights
        if self.pca is not None:
            normal = self.pca.fit_transform(normal)
        if self.native_mode:
            if self.method == 'native_reference':
                self.native_center_dimension = fine_dimension
                self.native_center_reference = np.ascontiguousarray(
                    normal[:, :fine_dimension] / self.group_weights[:fine_dimension])
                normal = normal[:, fine_dimension:]
            self.model = LocalOutlierFactor(n_neighbors=min(20, len(normal) - 1), novelty=True,
                                            algorithm='brute', n_jobs=1)
            with threadpool_limits(limits=1):
                self.model.fit(np.ascontiguousarray(normal))
                if (self.method == 'native_reference'
                        and getattr(self, 'reference_matching', 'full') == 'exclude_gradient_statistics'):
                    # 亮度及结构仍参与条件匹配，梯度统计仅保留在完整密度分支。
                    dimension = normal.shape[1] - (3 if self.feature_args['use_grad_stat'] else 0)
                    self.reference_neighbors = NearestNeighbors(n_neighbors=self.model.n_neighbors_,
                                                               algorithm='brute', n_jobs=1)
                    self.reference_neighbors.fit(np.ascontiguousarray(normal[:, :dimension]))
        else:
            self.model = IsolationForest(n_estimators=self.settings['n_estimators'],
                                         random_state=self.random_seed, n_jobs=-1).fit(normal)
        self.pixel_threshold = self.image_threshold = None
        self.reference_image_thresholds = self.reference_pixel_thresholds = None
        return self


    def _native_score(self, image, original):
        """评分先读取原图细块，再映射坐标；不从256图重造细节。"""
        if original is None:
            raise ValueError('此模型使用原图细节，请传入（缩放图，原图）')
        if (image.shape[:2] != self.native_image_shape
                or original.shape[:2] != self.native_source_shape):
            raise ValueError('原图细节预测必须与正常训练的图像尺寸一致')
        active, valid = self._native_active(original)
        size = self.native_size
        height, width = original.shape[:2]
        source_coordinates = np.array([(x, y, x + size, y + size)
                                       for y in range(0, height, size)
                                       for x in range(0, width, size)], np.int32)
        context = self.method in ('native_context', 'native_reference')
        margin = (self.native_context_size - size) // 2 if context else 0
        source = np.pad(original, ((margin, margin), (margin, margin), (0, 0)), mode='reflect') if context else original
        crop_size = self.native_context_size if context else size
        features = np.stack([self._native_feature(source[y1:y1+crop_size, x1:x1+crop_size])
                             for x1, y1, x2, y2 in source_coordinates[active]])
        features = self.standardizer.transform(features) * self.group_weights
        if self.pca is not None:
            features = self.pca.transform(features)
        if self.method == 'native_reference':
            return self._native_reference_result(features, active, source_coordinates, image, original, valid)
        scores = np.zeros(len(source_coordinates))
        with threadpool_limits(limits=1):
            scores[active] = -self.model.score_samples(np.ascontiguousarray(features))
        coordinates = source_coordinates // self.native_scale
        heatmap = aggregate_scores(scores, coordinates, image.shape)
        top_pixels = self.native_top_cells * (size // self.native_scale) ** 2
        parents = np.array([patch.coordinates for patch in iter_patches(image, self.patch_size, self.stride)],
                           np.int32)
        result = dict(patch_scores=scores, coordinates=coordinates, heatmap=heatmap,
                    image_score=image_score(heatmap, self.settings['image_top_fraction'], top_pixels=top_pixels),
                    score_method='native_foreground_local_density', source_coordinates=source_coordinates,
                    source_image_shape=np.array(original.shape[:2]), window_coordinates=parents,
                    native_patch_size=size, image_pool_cells=self.native_top_cells, foreground_valid=valid)
        if context:
            # 要求相邻块共同响应，避免互不相邻的正常极值直接凑成图像异常。
            grid = scores.astype(np.float32).reshape(height // size, width // size)
            adjacent = np.stack([grid[:-1, :-1], grid[1:, :-1], grid[:-1, 1:], grid[1:, 1:]]).mean(axis=0)
            result.update(image_score=float(adjacent.max()), score_method='native_context_local_density',
                          context_patch_size=self.native_context_size,
                          image_pool_cells=4, image_pool_method='max_adjacent2x2_mean')
        return result

    def _native_feature(self, patch):
        """只替换显式启用的中心特征，上下文提取及其他方法保持原样。"""
        if self.method in ('native_context', 'native_reference'):
            margin = (self.native_context_size - self.native_size) // 2
            center = patch[margin:margin+self.native_size, margin:margin+self.native_size]
            return np.concatenate([self._native_center_feature(center),
                                   combine_features(patch, **self.feature_args)])
        return self._native_center_feature(patch)

    def _native_center_feature(self, patch):
        """旧缓存没有新开关时继续使用原组合描述符。"""
        if getattr(self, 'use_spatial_detail', False):
            return extract_spatial_detail(patch)
        return combine_features(patch, **self.feature_args)

    def _native_center_weights(self, dimension):
        """按相同组维数原则均衡三个等长空间通道，不搜索权重。"""
        if getattr(self, 'use_spatial_detail', False):
            return np.full(dimension, 1 / np.sqrt(dimension), np.float32)
        return self._weights(dimension)

    def _reference_values(self, features):
        """用相似正常邻域预测中心均值及方差；不从查询图更新参考。"""
        dimension = self.native_center_dimension
        values = np.zeros((len(features), 2), np.float32)
        for start in range(0, len(features), 512):
            query = features[start:start + 512]
            context = np.ascontiguousarray(query[:, dimension:])
            with threadpool_limits(limits=4):
                matching = getattr(self, 'reference_neighbors', None)
                if matching is None:
                    _, neighbors = self.model.kneighbors(context, n_neighbors=self.model.n_neighbors_)
                else:
                    neighbors = matching.kneighbors(np.ascontiguousarray(context[:, :matching.n_features_in_]),
                                                    return_distance=False)
                density = -self.model.score_samples(context)
            matched = self.native_center_reference[neighbors]
            center = query[:, :dimension] / self.group_weights[:dimension]
            difference = center - matched.mean(axis=1)
            variance = matched.var(axis=1, ddof=1)
            error = np.sqrt(np.sum(difference ** 2 / (variance + .01)
                                   * self.group_weights[:dimension] ** 2, axis=1))
            values[start:start + len(query)] = np.column_stack([density, error])
        return values

    def _native_reference_result(self, features, active, source_coordinates, image, original, valid):
        """保留两条原始证据，校准后按各自正常阈值归一再合并。"""
        component_scores = np.zeros((2, len(source_coordinates)), np.float32)
        component_scores[:, active] = self._reference_values(features).T
        coordinates = source_coordinates // self.native_scale
        maps = np.stack([aggregate_scores(scores, coordinates, image.shape) for scores in component_scores])
        grid = component_scores.reshape(2, original.shape[0] // self.native_size,
                                         original.shape[1] // self.native_size)
        adjacent = np.stack([grid[:, :-1, :-1], grid[:, 1:, :-1],
                             grid[:, :-1, 1:], grid[:, 1:, 1:]]).mean(axis=0)
        image_components = adjacent.max(axis=(1, 2))
        calibrated = self.reference_image_thresholds is not None
        heatmap = np.max(maps / self.reference_pixel_thresholds[:, None, None], axis=0) if calibrated else maps[0]
        patch_scores = (np.max(component_scores / self.reference_pixel_thresholds[:, None], axis=0)
                        if calibrated else component_scores[0])
        value = float(np.max(image_components / self.reference_image_thresholds)) if calibrated else float(image_components[0])
        parents = np.array([patch.coordinates for patch in iter_patches(image, self.patch_size, self.stride)], np.int32)
        result = dict(patch_scores=patch_scores, coordinates=coordinates, heatmap=heatmap, image_score=value,
                    component_patch_scores=component_scores, component_heatmaps=maps,
                    component_image_scores=image_components, score_calibrated=calibrated,
                    score_method='native_reference_conditional', source_coordinates=source_coordinates,
                    source_image_shape=np.array(original.shape[:2]), window_coordinates=parents,
                    native_patch_size=self.native_size, context_patch_size=self.native_context_size,
                    image_pool_cells=4, image_pool_method='max_adjacent2x2_branch_normalized', foreground_valid=valid)
        if getattr(self, 'reference_neighbors', None) is not None:
            result.update(reference_matching=self.reference_matching,
                          reference_matching_dimension=self.reference_neighbors.n_features_in_,
                          reference_density_dimension=self.model.n_features_in_)
        return result

