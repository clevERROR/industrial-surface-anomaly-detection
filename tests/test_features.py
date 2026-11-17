"""覆盖用户新版特征接口及存储路径的关键兼容行为。"""

import numpy as np
import pytest

from src.features import combine_features, FeatureStandardizer
from src.project_paths import PROJECT_ROOT, DEFAULT_CONFIG, relative_path, resolve_path


def test_zero_variance_does_not_amplify_unseen_values():
    standardizer = FeatureStandardizer().fit(np.array([[3., 1.], [3., 3.]]))
    np.testing.assert_allclose(standardizer.transform(np.array([[4., 2.]])), [[1., 0.]])


def test_unfitted_and_empty_standardizer_are_rejected():
    with pytest.raises(RuntimeError):
        FeatureStandardizer().transform(np.ones((1, 2)))
    with pytest.raises(ValueError):
        FeatureStandardizer().fit(np.empty((0, 2)))


def test_no_feature_selection_is_rejected():
    with pytest.raises(ValueError):
        combine_features(np.zeros((64, 64), np.uint8), False, False, False, False)


def test_feature_dimensions_remain_compatible():
    feature = combine_features(np.zeros((64, 64, 3), np.uint8))
    assert feature.shape == (1781,)
    assert feature.dtype == np.float32 and np.isfinite(feature).all()


def test_project_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = 'data/example.png'
    assert resolve_path(path) == PROJECT_ROOT / path
    assert relative_path(resolve_path(path)) == path


def test_default_config_is_inside_portable_project():
    assert DEFAULT_CONFIG == PROJECT_ROOT / 'configs/config.yaml'
    assert DEFAULT_CONFIG.is_file()


def test_external_output_paths_are_relative_to_project():
    target = PROJECT_ROOT.parent / 'external_outputs/results/example.npz'
    stored = relative_path(target)
    assert stored == '../external_outputs/results/example.npz'
    assert resolve_path(stored).resolve() == target.resolve()


def test_preprocessing_reads_unicode_absolute_paths(tmp_path):
    import cv2
    from src.preprocessing import preprocess_sample

    image = np.full((16, 24, 3), 80, np.uint8)
    mask = np.zeros((16, 24), np.uint8)
    mask[4:8, 5:9] = 255
    image_path, mask_path = tmp_path / '正常与缺陷.png', tmp_path / '缺陷标注.png'
    image_path.write_bytes(cv2.imencode('.png', image)[1].tobytes())
    mask_path.write_bytes(cv2.imencode('.png', mask)[1].tobytes())
    result = preprocess_sample(dict(image_path=str(image_path), mask_path=str(mask_path)), (12, 8))
    assert result['image'].shape == (8, 12, 3)
    assert result['mask'].shape == (8, 12)
    assert set(np.unique(result['mask'])) == {0, 255}
