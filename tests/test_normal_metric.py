"""验证官方度量数值、低秩处理、冻结边界及评价分母。"""

from copy import deepcopy

import metric_learn
import numpy as np
import pytest

from src.evaluation import classification_rates
from src.localization import LocalAnomalyDetector
from src.normal_metric import CovarianceScorer


def test_score_matches_official_mahalanobis_distance():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(80, 7))
    scorer = CovarianceScorer().fit(train)
    test = rng.normal(size=(6, 7))
    projected = scorer.project(test)
    delta = projected - scorer.project(train).mean(axis=0)
    matrix = scorer.metric.get_mahalanobis_matrix()
    expected = np.sqrt(np.einsum('ij,jk,ik->i', delta, matrix, delta))
    np.testing.assert_allclose(-scorer.decision_function(test), expected, atol=1e-10)
    assert isinstance(scorer.metric, metric_learn.Covariance)
    assert scorer.metadata()['fit_labels_used'] is False


def test_rank_one_and_redundant_features_are_finite():
    x = np.linspace(0, 1, 20)
    train = np.column_stack([x, x * 2, np.ones_like(x), np.zeros_like(x)])
    scorer = CovarianceScorer().fit(train)
    assert scorer.retained_rank == 1
    assert scorer.project(train).shape == (20, 2)
    assert np.isfinite(scorer.decision_function(train)).all()
    assert -scorer.decision_function(train[[0]])[0] > -scorer.decision_function(train[[10]])[0]


def test_zero_rank_is_rejected():
    with pytest.raises(ValueError, match='没有可用'):
        CovarianceScorer().fit(np.zeros((5, 3)))


def test_validation_and_prediction_do_not_fit_metric_again(monkeypatch):
    config = dict(dataset=dict(random_seed=42),
                  features=dict(lbp=False, hog=False, gray_statistics=True, gradient_statistics=True, pca=False),
                  localization=dict(method='metric_covariance', patch_size=16, stride=8,
                                    pixel_quantile=0.995, image_quantile=0.99, image_top_fraction=0.01))
    train = [np.full((32, 32, 3), value, np.uint8) for value in [10, 20, 30]]
    detector = LocalAnomalyDetector(config).fit(iter(train))
    assert detector.training_patch_count == 27
    original_mean = detector.standardizer.mean.copy()
    original_components = detector.model.metric.components_.copy()

    def forbidden_fit(*args, **kwargs):
        """验证或预测如果试图拟合，立即使测试失败。"""
        raise AssertionError('禁止再次拟合')

    monkeypatch.setattr(detector.model.metric, 'fit', forbidden_fit)
    monkeypatch.setattr(detector.model.projection, 'fit', forbidden_fit)
    detector.calibrate([np.full((32, 32, 3), value, np.uint8) for value in [25, 35]])
    thresholds = (detector.image_threshold, detector.pixel_threshold)
    result = detector.predict(np.full((32, 32, 3), 250, np.uint8))
    assert result['image_prediction'] == 1
    assert result['coordinates'].shape == (9, 4)
    assert result['heatmap'].shape == (32, 32)
    assert (detector.image_threshold, detector.pixel_threshold) == thresholds
    np.testing.assert_array_equal(detector.standardizer.mean, original_mean)
    np.testing.assert_array_equal(detector.model.metric.components_, original_components)
    config = deepcopy(config)
    config['localization']['method'] = 'unknown'
    with pytest.raises(ValueError, match='未知'):
        LocalAnomalyDetector(config)


def test_rates_use_class_specific_denominators():
    rates = classification_rates(dict(true_positive=2, false_negative=8, true_negative=15, false_positive=5))
    assert rates['accuracy'] == 17 / 30
    assert rates['miss_rate'] == 8 / 10
    assert rates['false_positive_rate'] == 5 / 20
    empty = classification_rates(dict(true_positive=0, false_negative=0, true_negative=0, false_positive=0))
    assert empty['miss_rate'] is None and empty['false_positive_rate'] is None
