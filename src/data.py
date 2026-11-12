
from pathlib import Path
import random


def _create_sample(image_path, category, label, defect_type, mask_path=None):
    return {
        "image_path": str(image_path),
        "label": label,
        "mask_path": str(mask_path) if mask_path else None,
        "category": category,
        "defect_type": defect_type,
    }


def _load_train_samples(category_dir, category):
    """读取正常训练样本。"""
    train_dir = category_dir / "train" / "good"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"训练目录不存在: {train_dir}")

    image_paths = sorted(train_dir.glob("*.png"))

    if len(image_paths) < 2:
        raise ValueError(f"{category} 正常训练图片不足 2 张")

    return [
        _create_sample(
            image_path=path,
            category=category,
            label=0,
            defect_type="good",
        )
        for path in image_paths
    ]


def _split_train_val(samples, validation_ratio, random_seed):
    """将正常训练样本划分为训练集和验证集。"""
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio 必须在 0 和 1 之间")

    samples = samples.copy()

    rng = random.Random(random_seed)
    rng.shuffle(samples)

    val_count = max(1, round(len(samples) * validation_ratio))
    val_count = min(val_count, len(samples) - 1)

    return samples[val_count:], samples[:val_count]


def _load_test_samples(category_dir, category):
    """读取测试图片及对应的像素级标注。"""
    test_dir = category_dir / "test"
    ground_truth_dir = category_dir / "ground_truth"

    if not test_dir.is_dir():
        raise FileNotFoundError(f"测试目录不存在: {test_dir}")

    samples = []

    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue

        defect_type = defect_dir.name
        image_paths = sorted(defect_dir.glob("*.png"))

        if defect_type == "good":
            samples.extend(
                _create_sample(
                    path, category, 0, "good"
                )
                for path in image_paths
            )
            continue

        mask_dir = ground_truth_dir / defect_type

        if not mask_dir.is_dir():
            raise FileNotFoundError(
                f"缺陷标注目录不存在: {mask_dir}"
            )

        image_stems = {path.stem for path in image_paths}
        mask_paths = list(mask_dir.glob("*_mask.png"))
        mask_stems = {
            path.name.removesuffix("_mask.png")
            for path in mask_paths
        }

        missing_masks = image_stems - mask_stems
        extra_masks = mask_stems - image_stems

        if missing_masks:
            raise FileNotFoundError(
                f"{category}/{defect_type} 缺少 Mask，"
                f"对应图片: {sorted(missing_masks)}"
            )

        if extra_masks:
            raise ValueError(
                f"{category}/{defect_type} 存在多余 Mask，"
                f"对应文件: {sorted(extra_masks)}"
            )

        for image_path in image_paths:
            mask_path = mask_dir / f"{image_path.stem}_mask.png"

            samples.append(
                _create_sample(
                    image_path=image_path,
                    category=category,
                    label=1,
                    defect_type=defect_type,
                    mask_path=mask_path,
                )
            )

    if not samples:
        raise ValueError(f"{category} 测试集为空")

    return samples


def load_dataset(config):
    """
    加载 MVTec AD 数据集。

    返回：
        {
            category: {
                "train": [...],
                "val": [...],
                "test": [...]
            }
        }
    """
    dataset_config = config["dataset"]

    root = Path(dataset_config["root"])
    categories = dataset_config["categories"]
    validation_ratio = dataset_config.get("validation_ratio", 0.2)
    random_seed = dataset_config.get("random_seed", 42)

    if not root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在: {root}")

    datasets = {}

    for category in categories:
        category_dir = root / category

        if not category_dir.is_dir():
            raise FileNotFoundError(f"类别目录不存在: {category_dir}")

        all_train_samples = _load_train_samples(
            category_dir, category
        )

        train_samples, val_samples = _split_train_val(
            all_train_samples,
            validation_ratio,
            random_seed,
        )

        test_samples = _load_test_samples(
            category_dir, category
        )

        datasets[category] = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        }

    return datasets
