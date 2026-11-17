"""使用官方无监督协方差度量计算正常训练中心距离。"""

import metric_learn
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


SOURCE_COMMIT = '8eb775ea82c41829bc583629ca3fab0fd8f204fc'


class CovarianceScorer:
    """数值投影和正常中心只在fit时计算，预测不接收标签。"""

    def fit(self, features):
        """先去除正常训练特征的数值零空间，再调用官方Covariance。"""
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or len(features) < 2 or not np.isfinite(features).all():
            raise ValueError('协方差拟合需要至少两个有限特征向量')
        self.training_count, self.input_dimension = features.shape
        if not np.any(np.var(features, axis=0) > 0):
            raise ValueError('正常训练特征没有可用的变化方向')
        self.projection = PCA(svd_solver='full')
        with threadpool_limits(limits=4):
            self.projection.fit(features)
            eigenvalues = self.projection.explained_variance_
            self.keep = eigenvalues > max(float(eigenvalues[0]) * 1e-8, 0.0)
            self.retained_rank = int(self.keep.sum())
            if self.retained_rank == 0:
                raise ValueError('正常训练特征没有可用的变化方向')
            self.components = self.projection.components_[self.keep].copy()
            projected = self.project(features)
            self.metric = metric_learn.Covariance().fit(projected)
            self.center = self.metric.transform(projected).mean(axis=0)
        if not np.isfinite(self.metric.components_).all():
            raise ValueError('官方协方差度量产生非有限结果')
        return self

    def project(self, features):
        """使用冻结的训练投影；单有效维度补零列以使用矩阵伪逆。"""
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.input_dimension or not np.isfinite(features).all():
            raise ValueError('预测特征维度不匹配或包含非有限值')
        projected = (features - self.projection.mean_) @ self.components.T
        if self.retained_rank == 1:
            projected = np.column_stack([projected, np.zeros(len(projected))])
        return projected

    def decision_function(self, features):
        """返回负的正常中心距离，以兼容现有越小越异常的模型接口。"""
        transformed = self.metric.transform(self.project(features))
        return -np.linalg.norm(transformed - self.center, axis=1)

    def metadata(self):
        """保存可复核的官方来源、数值处理与拟合样本数。"""
        return dict(estimator='metric_learn.Covariance', version=metric_learn.__version__,
                    source_commit=SOURCE_COMMIT, input_dimension=self.input_dimension,
                    retained_rank=self.retained_rank, numerical_rank_relative_cutoff=1e-8,
                    training_patch_count=self.training_count, fit_labels_used=False,
                    score='distance_to_normal_training_center')
