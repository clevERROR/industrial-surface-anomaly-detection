"""验证坐标覆盖、分数还原及正常验证阈值的数值约定。"""

import numpy as np
import pytest

from src.heatmap import aggregate_scores, resize_heatmap
from src.patches import iter_patches
from src.postprocess import anomaly_mask, calibrate_threshold, image_score


def test_default_windows_cover_every_pixel():
    patches = list(iter_patches(np.zeros((256, 256, 3), dtype=np.uint8)))
    assert len(patches) == 49
    assert patches[0].coordinates == (0, 0, 64, 64)
    assert patches[-1].coordinates == (192, 192, 256, 256)
    actual = aggregate_scores(np.ones(49), [p.coordinates for p in patches], (256, 256))
    np.testing.assert_array_equal(actual, np.ones((256, 256)))


def test_non_divisible_edges_are_included():
    image = np.arange(11 * 13).reshape(11, 13)
    patches = list(iter_patches(image, 6, 4))
    assert patches[-1].coordinates == (7, 5, 13, 11)
    np.testing.assert_array_equal(patches[-1].image, image[5:11, 7:13])
    aggregate_scores(np.ones(len(patches)), [p.coordinates for p in patches], image.shape)


@pytest.mark.parametrize('size,stride', [(0, 1), (5, 1), (2, 3), (2, 0)])
def test_invalid_windows_are_rejected(size, stride):
    with pytest.raises(ValueError):
        list(iter_patches(np.zeros((4, 4)), size, stride))


def test_overlap_is_averaged_without_swapping_axes():
    actual = aggregate_scores([2, 6], [(0, 0, 3, 2), (1, 0, 4, 2)], (2, 4))
    np.testing.assert_array_equal(actual, [[2, 4, 4, 6], [2, 4, 4, 6]])


@pytest.mark.parametrize('coords', [[(0, 0, 1, 2)], [(0, 0, 4, 2)]])
def test_gaps_and_out_of_bounds_coordinates_are_rejected(coords):
    with pytest.raises(ValueError):
        aggregate_scores([1], coords, (2, 3))


def test_calibration_and_mask_keep_ties_normal():
    threshold = calibrate_threshold([0, 1, 2, 3], 0.5)
    assert threshold == 1.5
    np.testing.assert_array_equal(anomaly_mask(np.array([[1, 1.5, 2]]), threshold), [[0, 0, 255]])


@pytest.mark.parametrize('values,quantile', [([], 0.5), ([np.nan], 0.5), ([1], 1)])
def test_invalid_calibration_is_rejected(values, quantile):
    with pytest.raises(ValueError):
        calibrate_threshold(values, quantile)


def test_top_fraction_and_original_size():
    heatmap = np.array([[1, 2], [3, 10]], dtype=np.float32)
    assert image_score(heatmap, 0.5) == 6.5
    restored = resize_heatmap(heatmap, (9, 13))
    assert restored.shape == (9, 13)
    assert anomaly_mask(restored, 3).dtype == np.uint8


def test_prediction_keeps_normal_only_calibration_frozen():
    from src.localization import LocalAnomalyDetector

    config = dict(dataset=dict(random_seed=42),
                  features=dict(lbp=False, hog=False, gray_statistics=True, gradient_statistics=True, pca=False),
                  localization=dict(patch_size=16, stride=8, n_estimators=5,
                                    pixel_quantile=0.9, image_quantile=0.9, image_top_fraction=0.1))
    train = [np.full((32, 32), value, np.uint8) for value in [10, 20, 30]]
    validation = [np.full((32, 32), value, np.uint8) for value in [80, 100]]
    detector = LocalAnomalyDetector(config).fit(train)
    training_mean = detector.standardizer.mean.copy()
    assert training_mean[0] == 20
    with pytest.raises(RuntimeError):
        detector.predict(train[0])
    calibration = detector.calibrate(validation)
    thresholds = (detector.pixel_threshold, detector.image_threshold)
    result = detector.predict(np.full((32, 32), 255, np.uint8))
    assert result['heatmap'].shape == (32, 32)
    assert result['coordinates'].shape == (9, 4)
    assert np.isfinite(result['image_score'])
    assert (detector.pixel_threshold, detector.image_threshold) == thresholds
    np.testing.assert_array_equal(detector.standardizer.mean, training_mean)
    assert calibration['heatmaps'].shape == (2, 32, 32)
