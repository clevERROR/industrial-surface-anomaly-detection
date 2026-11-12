
from pathlib import Path

import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt

from src.data import load_dataset
from src.preprocessing import preprocess_sample


def validate_dataset(datasets):
    """检查真实数据的图片、Mask 和数据隔离情况。"""
    errors = []
    warnings = []

    total_images = 0
    total_masks = 0

    for category, splits in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Category: {category}")

        # 1. 检查数据划分之间是否存在路径交叉
        split_paths = {
            split: {
                Path(sample["image_path"]).resolve()
                for sample in samples
            }
            for split, samples in splits.items()
        }

        train_val_overlap = (
            split_paths["train"] & split_paths["val"]
        )
        train_test_overlap = (
            split_paths["train"] & split_paths["test"]
        )
        val_test_overlap = (
            split_paths["val"] & split_paths["test"]
        )

        if train_val_overlap or train_test_overlap or val_test_overlap:
            errors.append(
                f"{category}: 数据集划分存在重复图片"
            )

        # 2. 检查所有图片能否正常读取
        for split_name, samples in splits.items():
            normal_count = 0
            anomaly_count = 0

            for sample in samples:
                image_path = Path(sample["image_path"])

                image = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_UNCHANGED,
                )

                if image is None:
                    errors.append(
                        f"图片无法解码: {image_path}"
                    )
                    continue

                total_images += 1

                if sample["label"] == 0:
                    normal_count += 1
                else:
                    anomaly_count += 1

            print(
                f"{split_name:>5}: "
                f"total={len(samples):4d}, "
                f"normal={normal_count:4d}, "
                f"anomaly={anomaly_count:4d}"
            )

        # 3. 检查 Mask
        test_samples = splits["test"]

        for sample in test_samples:
            if sample["label"] == 0:
                continue

            image_path = Path(sample["image_path"])
            mask_path = Path(sample["mask_path"])

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_UNCHANGED,
            )

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_UNCHANGED,
            )

            if mask is None:
                errors.append(
                    f"Mask 无法解码: {mask_path}"
                )
                continue

            total_masks += 1

            if image is None:
                continue

            # 检查图片与 Mask 的空间尺寸
            if image.shape[:2] != mask.shape[:2]:
                errors.append(
                    f"图片与 Mask 尺寸不一致: "
                    f"{image_path} "
                    f"image={image.shape[:2]}, "
                    f"mask={mask.shape[:2]}"
                )
                continue

            # MVTec AD 的缺陷 Mask 应包含非零缺陷区域
            if not np.any(mask > 0):
                errors.append(
                    f"Mask 不包含缺陷像素: {mask_path}"
                )

            # 检查 Mask 是否为二值标注
            unique_values = np.unique(mask)

            if not np.isin(unique_values, [0, 255]).all():
                warnings.append(
                    f"Mask 存在非标准二值像素值: "
                    f"{mask_path}, "
                    f"values={unique_values.tolist()}"
                )

        print(
            f"Test masks checked: "
            f"{sum(s['label'] == 1 for s in test_samples)}"
        )

    # 4. 汇总
    print(f"\n{'=' * 60}")
    print("Dataset Validation Summary")
    print(f"Total images decoded: {total_images}")
    print(f"Anomaly masks decoded: {total_masks}")
    print(f"Errors:                {len(errors)}")
    print(f"Warnings:              {len(warnings)}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"[ERROR] {error}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"[WARNING] {warning}")

    return len(errors) == 0


def main():
    config_path = Path("configs/config.yaml")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    datasets = load_dataset(config)

    success = validate_dataset(datasets)

    if not success:
        raise SystemExit("数据集完整性检查失败")

    print("\n数据集完整性检查通过。")

    # 5. 选择一张缺陷测试样本
    sample = next(
        item for item in datasets["screw"]["test"]
        if item["label"] == 1
    )

    # 6. 预处理
    result = preprocess_sample(sample)

    # 7. 读取原始图像和 Mask
    original_image = cv2.imread(
        sample["image_path"],
        cv2.IMREAD_COLOR,
    )

    if original_image is None:
        raise ValueError(
            f"原始图像读取失败: {sample['image_path']}"
        )

    original_image = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2RGB,
    )

    original_mask = cv2.imread(
        sample["mask_path"],
        cv2.IMREAD_GRAYSCALE,
    )

    if original_mask is None:
        raise ValueError(
            f"原始 Mask 读取失败: {sample['mask_path']}"
        )

    # 8. 获取原始尺寸和预处理后尺寸
    original_height, original_width = original_image.shape[:2]
    processed_height, processed_width = result["image"].shape[:2]

    original_mask_height, original_mask_width = original_mask.shape[:2]
    processed_mask_height, processed_mask_width = result["mask"].shape[:2]

    # 9. 绘制预处理前后对比图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title(
        f"Original Image\n"
        f"{original_width} x {original_height}",
        fontsize=12,
    )

    axes[0, 1].imshow(result["image"])
    axes[0, 1].set_title(
        f"Preprocessed Image\n"
        f"{processed_width} x {processed_height}",
        fontsize=12,
    )

    axes[1, 0].imshow(
        original_mask,
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[1, 0].set_title(
        f"Original Ground Truth Mask\n"
        f"{original_mask_width} x {original_mask_height}",
        fontsize=12,
    )

    axes[1, 1].imshow(
        result["mask"],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[1, 1].set_title(
        f"Preprocessed Mask\n"
        f"{processed_mask_width} x {processed_mask_height}",
        fontsize=12,
    )

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle(
        f"Preprocessing Visualization - "
        f"{sample['category']} / {sample['defect_type']}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()

    # 10. 输出样本信息
    print("\nPreprocessing Result")
    print(f"Image:      {result['image'].shape}")
    print(f"Mask:       {result['mask'].shape}")
    print(f"Label:      {result['label']}")
    print(f"Category:   {result['category']}")
    print(f"Defect type: {result['defect_type']}")


if __name__ == "__main__":
    main()
