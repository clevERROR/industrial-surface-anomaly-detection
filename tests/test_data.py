from pathlib import Path
import pytest
from src.data import load_dataset


def create_fake_category(root, category):
    """构造一个最小化的 MVTec AD 测试目录。"""
    category_dir = root / category

    train_dir = category_dir / "train" / "good"
    test_good_dir = category_dir / "test" / "good"
    test_defect_dir = category_dir / "test" / "defect_a"
    mask_dir = category_dir / "ground_truth" / "defect_a"

    for directory in [
        train_dir,
        test_good_dir,
        test_defect_dir,
        mask_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # 4 张正常训练图片
    for i in range(4):
        (train_dir / f"{i:03d}.png").touch()

    # 1 张正常测试图片
    (test_good_dir / "000.png").touch()

    # 1 张异常测试图片及其 Mask
    (test_defect_dir / "000.png").touch()
    (mask_dir / "000_mask.png").touch()


@pytest.fixture
def dataset_config(tmp_path):
    create_fake_category(tmp_path, "grid")

    return {
        "dataset": {
            "root": str(tmp_path),
            "categories": ["grid"],
            "validation_ratio": 0.25,
            "random_seed": 42,
        }
    }


def test_load_dataset_counts(dataset_config):
    datasets = load_dataset(dataset_config)

    grid = datasets["grid"]

    assert len(grid["train"]) == 3
    assert len(grid["val"]) == 1
    assert len(grid["test"]) == 2


def test_train_and_val_only_contain_normal_samples(dataset_config):
    datasets = load_dataset(dataset_config)

    for split in ["train", "val"]:
        for sample in datasets["grid"][split]:
            assert sample["label"] == 0
            assert sample["mask_path"] is None


def test_test_samples_have_correct_labels_and_masks(dataset_config):
    datasets = load_dataset(dataset_config)

    samples = datasets["grid"]["test"]

    normal_samples = [
        sample for sample in samples
        if sample["label"] == 0
    ]

    defect_samples = [
        sample for sample in samples
        if sample["label"] == 1
    ]

    assert len(normal_samples) == 1
    assert len(defect_samples) == 1

    assert defect_samples[0]["defect_type"] == "defect_a"
    assert Path(defect_samples[0]["mask_path"]).is_file()


def test_split_is_reproducible(dataset_config):
    first = load_dataset(dataset_config)
    second = load_dataset(dataset_config)

    first_paths = [
        sample["image_path"]
        for sample in first["grid"]["val"]
    ]

    second_paths = [
        sample["image_path"]
        for sample in second["grid"]["val"]
    ]

    assert first_paths == second_paths


def test_missing_mask_raises_error(dataset_config):
    mask_path = (
        Path(dataset_config["dataset"]["root"])
        / "grid"
        / "ground_truth"
        / "defect_a"
        / "000_mask.png"
    )

    mask_path.unlink()

    with pytest.raises(FileNotFoundError):
        load_dataset(dataset_config)
