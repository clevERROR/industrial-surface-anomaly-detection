from pathlib import Path

import cv2
from src.project_paths import relative_path, resolve_path
from src.image_io import read_image
def preprocess_sample(
    sample: dict,
    target_size: tuple[int, int] = (256, 256),
    *,
    include_original: bool = False,
) -> dict:
    """
    对单个数据样本进行预处理。

    Args:
        sample: Dataset Loader 返回的样本。
        target_size: 目标尺寸，格式为 (width, height)。
        include_original: 是否另返回同次解码的原始RGB，不改默认处理。

    Returns:
        保留原始元数据，并增加 image 和 mask 字段。
        image 为 RGB 图像，mask 为灰度图或 None。
    """
    if (
        len(target_size) != 2
        or any(not isinstance(size, int) or size <= 0 for size in target_size)
    ):
        raise ValueError("target_size 必须是两个正整数")

    image_path = resolve_path(sample["image_path"])

    if not image_path.is_file():
        raise FileNotFoundError(f"图像文件不存在: {relative_path(image_path)}")

    image = read_image(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(f"图像读取失败: {relative_path(image_path)}")

    # OpenCV 默认读取 BGR，统一转换为 RGB。
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image if include_original else None

    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # 根据缩放方向选择插值方式。
    if (
        target_width < original_width
        or target_height < original_height
    ):
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    image = cv2.resize(
        image,
        target_size,
        interpolation=interpolation,
    )

    # 处理 Ground Truth Mask。
    mask = None
    mask_path = sample.get("mask_path")

    if mask_path is not None:
        mask_path = resolve_path(mask_path)

        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Mask 文件不存在: {relative_path(mask_path)}"
            )

        mask = read_image(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise ValueError(f"Mask 读取失败: {relative_path(mask_path)}")

        if mask.shape != (original_height, original_width):
            raise ValueError(
                f"图像与 Mask 尺寸不一致: "
                f"{relative_path(image_path)}, {relative_path(mask_path)}"
            )

        # Mask 必须使用最近邻插值，避免生成新的像素标签。
        mask = cv2.resize(
            mask,
            target_size,
            interpolation=cv2.INTER_NEAREST,
        )

    result = sample.copy()
    result["image"] = image
    result["mask"] = mask
    if include_original:
        result['original_image'] = original_image

    return result
