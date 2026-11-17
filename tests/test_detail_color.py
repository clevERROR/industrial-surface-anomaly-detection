"""验证原图颜色细节、真实单元汇总、坐标与正常拟合冻结边界。"""

from copy import deepcopy

import cv2
import numpy as np
import pytest

from src.detail_features import detail_features
from src.features import combine_features
from src.localization import create_detector
from src.postprocess import top_cell_score
from src.preprocessing import preprocess_sample


def config():
    """小参考库只用于功能验证，不代替真实实验配置。"""
    return dict(dataset=dict(random_seed=42), features=dict(lbp=True, hog=True, gray_statistics=True,
                gradient_statistics=True, pca=False), localization=dict(method='detail_color',
                patch_size=64, stride=32, n_estimators=5, pixel_quantile=.995, image_quantile=.99,
                image_top_fraction=.01), detail_color=dict(scales=[16, 32], cell_size=16,
                samples_per_image=32, reference_cap=64, top_cells=3))


def test_color_distinguishes_equal_gray_inputs():
    red = np.zeros((64, 64, 3), np.uint8)
    green = np.zeros_like(red)
    red[:, :, 0] = 255
    green[:, :, 1] = 130
    np.testing.assert_array_equal(combine_features(red), combine_features(green))
    first, _ = detail_features(red, [16])
    second, _ = detail_features(green, [16])
    assert first[0].shape == (16, 26)
    assert np.linalg.norm(first[0][:, :3] - second[0][:, :3]) > 100


def test_color_statistics_match_direct_native_crop():
    image = np.random.default_rng(4).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    features, coordinates = detail_features(image, [16])
    index = np.flatnonzero((coordinates[:, 0] == 16) & (coordinates[:, 1] == 16))[0]
    lab = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[16:32, 16:32]
    np.testing.assert_allclose(features[0][index, :3], lab.mean(axis=(0, 1)), atol=1e-4)
    np.testing.assert_allclose(features[0][index, 3:6], lab.std(axis=(0, 1)), atol=1e-3)


def test_original_pixels_survive_preprocessing(tmp_path):
    original = np.full((1024, 1024, 3), 100, np.uint8)
    original[512, 496:528] = 164
    path = tmp_path / '原图细线.png'
    path.write_bytes(cv2.imencode('.png', original)[1].tobytes())
    sample = dict(image_path=str(path), mask_path=None)
    baseline = preprocess_sample(sample)
    result = preprocess_sample(sample, include_original=True)
    np.testing.assert_array_equal(result['original_image'], original)
    np.testing.assert_array_equal(result['image'], baseline['image'])
    assert result['original_image'].max() == 164 and result['image'].max() < 164


def test_top_cells_are_distinct_units():
    assert top_cell_score([1, 2, 3, 4], 3) == 3
    assert top_cell_score([1, 9], 3) == 5
    with pytest.raises(ValueError):
        top_cell_score([np.nan])


def test_non_integer_scale_covers_700_pixels_and_predict_keeps_reference():
    rng = np.random.default_rng(12)
    train = [rng.integers(70, 150, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
    detector = create_detector(config()).fit([(image, image) for image in train])
    reference = [bank['reference'].copy() for bank in detector.banks]
    detector.calibrate([(image, image) for image in train[:2]])
    thresholds = (detector.image_threshold, detector.pixel_threshold)
    raw = rng.integers(70, 150, (700, 700, 3), dtype=np.uint8)
    image = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_AREA)
    result = detector.predict((image, raw))
    assert result['heatmap'].shape == (256, 256)
    assert result['source_coordinates'].shape == (44 * 44, 4)
    coverage = np.zeros((700, 700), np.uint8)
    for x1, y1, x2, y2 in result['source_coordinates']:
        coverage[y1:y2, x1:x2] += 1
    assert np.all(coverage == 1)
    assert result['coordinates'].min() >= 0 and result['coordinates'].max() <= 256
    assert (detector.image_threshold, detector.pixel_threshold) == thresholds
    for bank, frozen in zip(detector.banks, reference):
        np.testing.assert_array_equal(bank['reference'], frozen)
    assert detector.metadata()['fit_labels_used'] is False
    with pytest.raises(ValueError, match='原始RGB'):
        detector.score(image)


def test_default_detector_still_selects_iforest():
    value = deepcopy(config())
    value['localization'].pop('method')
    assert create_detector(value).method == 'iforest'
