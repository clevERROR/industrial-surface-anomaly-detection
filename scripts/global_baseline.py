from pathlib import Path
import csv
import json
import platform
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pyod.models.iforest import IsolationForest
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern, hog

from src.data import load_dataset
from src.preprocessing import preprocess_sample
from src.features import combine_features, FeatureStandardizer
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_image_level_metrics(samples, scores):
    labels = np.array([s["label"] for s in samples], dtype=int)
    if len(np.unique(labels)) < 2:
        return None
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "n": int(len(labels)),
        "n_pos": int(labels.sum()),
    }

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

EXP_ROOT = PROJECT_ROOT / "experiments"
EXP_ROOT.mkdir(parents=True, exist_ok=True)
RES_ROOT = PROJECT_ROOT / "results"
RES_ROOT.mkdir(parents=True, exist_ok=True)

# 从配置读取目标尺寸
img_sz = cfg["preprocess"]["image_size"]
target_size = (img_sz, img_sz)

FIELDNAMES = [
    "sample_id",
    "image_path",
    "label",
    "mask_path",
    "category",
    "defect_type",
    "model",
    "feature_set",
    "score",
]

EXP_SET = {
    "lbp_only": {
        "use_lbp": True,
        "use_hog": False,
        "use_gray_stat": False,
        "use_grad_stat": False,
    },
    "hog_only": {
        "use_lbp": False,
        "use_hog": True,
        "use_gray_stat": False,
        "use_grad_stat": False,
    },
    "stat_only": {
        "use_lbp": False,
        "use_hog": False,
        "use_gray_stat": True,
        "use_grad_stat": True,
    },
    "all": {
        "use_lbp": True,
        "use_hog": True,
        "use_gray_stat": True,
        "use_grad_stat": True,
    },
}


def process_samples(sample_list, split_name="unknown"):
    """对一批样本全部执行 preprocess_sample，返回 (处理完毕样本, 跳过记录)。"""
    out = []
    skipped = []

    for s in sample_list:
        try:
            processed = preprocess_sample(s, target_size=target_size)
            out.append(processed)
        except (ValueError, FileNotFoundError) as e:
            print(f"[跳过] split={split_name} image={s.get('image_path')} | {e}")
            skipped.append(
                {"image_path": s.get("image_path"), "reason": str(e)}
            )

    if skipped:
        print(f"[警告] split={split_name} 共跳过 {len(skipped)} 个样本")

    return out, skipped


def extract_feature_for_dataset(processed_samples, **feat_args):
    """
    返回 (feats, samples, avg_time)。
    samples 与 feats 顺序严格一致，跳过特征失败的样本。
    """
    if not processed_samples:
        return np.empty((0, 0), dtype=np.float32), [], 0.0

    feats = []
    samples = []
    t0 = time.time()

    for sample in processed_samples:
        img = sample["image"]
        try:
            feat = combine_features(img, **feat_args)
        except Exception as e:
            print(
                f"[跳过特征] image={sample.get('image_path')} | {e}"
            )
            continue
        feats.append(feat)
        samples.append(sample)

    t_total = time.time() - t0

    if not feats:
        return np.empty((0, 0), dtype=np.float32), [], t_total

    feats = np.stack(feats, axis=0)
    avg_time = t_total / len(feats)
    return feats, samples, avg_time


def _sample_to_record(sample, score, model_name, feature_set, idx):
    return {
        "sample_id": idx,
        "image_path": sample["image_path"],
        "label": int(sample["label"]),
        "mask_path": sample.get("mask_path") or "",
        "category": sample["category"],
        "defect_type": sample["defect_type"],
        "model": model_name,
        "feature_set": feature_set,
        "score": float(score),
    }


def save_scores_csv(path, samples, scores, model_name, feature_set):
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(samples) != len(scores):
        raise ValueError(
            f"samples 与 scores 长度不一致: {len(samples)} vs {len(scores)}"
        )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for idx, (s, score) in enumerate(zip(samples, scores)):
            writer.writerow(
                _sample_to_record(s, score, model_name, feature_set, idx)
            )


def save_feature_cache(cache_dir, prefix, samples, feats):
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{prefix}_features.npy", feats)

    with open(
        cache_dir / f"{prefix}_samples.jsonl", "w", encoding="utf-8"
    ) as f:
        for s in samples:
            f.write(
                json.dumps(
                    {
                        "image_path": s["image_path"],
                        "label": int(s["label"]),
                        "mask_path": s.get("mask_path"),
                        "category": s["category"],
                        "defect_type": s["defect_type"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_feature_vis(processed_sample, category):
    """保存 LBP、HOG 示意图到 results，按类别命名。"""
    if not processed_sample:
        print(f"[跳过可视化] 类别 {category} 没有可用样本")
        return

    img_rgb = processed_sample["image"]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # LBP 图
    lbp_img = local_binary_pattern(gray, P=8, R=1, method="uniform")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(lbp_img, cmap="gray")
    axes[1].set_title("LBP visualization")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(RES_ROOT / f"lbp_vis_{category}.png", dpi=150)
    plt.close()

    # HOG 可视化
    hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
    )[1]

    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
    ax2[0].imshow(img_rgb)
    ax2[0].set_title("Original Image")
    ax2[0].axis("off")
    ax2[1].imshow(hog_image, cmap="gray")
    ax2[1].set_title("HOG visualization")
    ax2[1].axis("off")
    plt.tight_layout()
    plt.savefig(RES_ROOT / f"hog_vis_{category}.png", dpi=150)
    plt.close()


def get_env_info():
    info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }

    try:
        import sklearn

        info["sklearn"] = sklearn.__version__
    except ImportError:
        pass

    try:
        import pyod

        info["pyod"] = pyod.__version__
    except ImportError:
        pass

    try:
        info["opencv"] = cv2.__version__
    except AttributeError:
        pass

    return info


def run_one_category(category, datasets_raw, run_info):
    print(f"\n########## 类别：{category} ##########")

    raw_train = datasets_raw[category]["train"]
    raw_val = datasets_raw[category]["val"]
    raw_test = datasets_raw[category]["test"]

    print("正在预处理训练、验证、测试图片...")
    train_processed, train_skipped = process_samples(raw_train, "train")
    val_processed, val_skipped = process_samples(raw_val, "val")
    test_processed, test_skipped = process_samples(raw_test, "test")

    run_info["categories"][category] = {
        "n_train_raw": len(raw_train),
        "n_val_raw": len(raw_val),
        "n_test_raw": len(raw_test),
        "n_train_processed": len(train_processed),
        "n_val_processed": len(val_processed),
        "n_test_processed": len(test_processed),
        "skipped": {
            "train": train_skipped,
            "val": val_skipped,
            "test": test_skipped,
        },
    }

    if not train_processed or not test_processed:
        print(f"[警告] 类别 {category} 预处理后训练或测试集为空，跳过")
        return

    save_feature_vis(train_processed[0], category)
    print(f"LBP、HOG 示意图已保存到 results/（{category}）")

    cache_root = EXP_ROOT / "cache" / category
    score_root = EXP_ROOT / "scores" / category
    cache_root.mkdir(parents=True, exist_ok=True)
    score_root.mkdir(parents=True, exist_ok=True)

    for exp_name, args in EXP_SET.items():
        print(f"\n===== [{category}] 实验组：{exp_name} =====")

        X_train, train_samples, t_train_avg = extract_feature_for_dataset(
            train_processed, **args
        )
        X_val, val_samples, t_val_avg = extract_feature_for_dataset(
            val_processed, **args
        )
        X_test, test_samples, t_test_avg = extract_feature_for_dataset(
            test_processed, **args
        )

        if X_train.size == 0 or X_test.size == 0:
            print(f"[警告] [{category}] {exp_name} 特征为空，跳过")
            continue

        print(
            f"特征维度: {X_train.shape[1]}，"
            f"训练集单张平均提取时间: {t_train_avg:.4f} s，"
            f"测试集单张平均提取时间: {t_test_avg:.4f} s"
        )

        std = FeatureStandardizer()
        X_train_std = std.fit_transform(X_train)
        X_val_std = std.transform(X_val) if X_val.size else None
        X_test_std = std.transform(X_test)

        model = IsolationForest(n_estimators=100, random_state=42)
        model.fit(X_train_std)

        train_scores = model.decision_function(X_train_std)
        val_scores = (
            model.decision_function(X_val_std)
            if X_val_std is not None
            else np.array([])
        )
        test_scores = model.decision_function(X_test_std)

        save_feature_cache(cache_root / exp_name, "train", train_samples, X_train)
        save_feature_cache(cache_root / exp_name, "val", val_samples, X_val)
        save_feature_cache(cache_root / exp_name, "test", test_samples, X_test)

        save_scores_csv(
            score_root / f"{exp_name}_train_scores.csv",
            train_samples,
            train_scores,
            "iforest",
            exp_name,
        )
        if len(val_samples) > 0:
            save_scores_csv(
                score_root / f"{exp_name}_val_scores.csv",
                val_samples,
                val_scores,
                "iforest",
                exp_name,
            )
        save_scores_csv(
            score_root / f"{exp_name}_test_scores.csv",
            test_samples,
            test_scores,
            "iforest",
            exp_name,
        )
        np.save(score_root / f"{exp_name}_test_score.npy", test_scores)
        print(f"已保存 {category}/{exp_name} 异常分数（train/val/test CSV + test npy）")

        run_info["categories"][category].setdefault(
            "feature_experiments", {}
        )[exp_name] = {
            "feature_dim": int(X_train.shape[1]),
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "n_test": len(test_samples),
            "train_extract_time_avg_s": t_train_avg,
            "val_extract_time_avg_s": t_val_avg,
            "test_extract_time_avg_s": t_test_avg,
        }

        # PCA 只对 all 组做
        if exp_name == "all":
            pca = PCA(n_components=0.95, random_state=42)
            X_train_pca = pca.fit_transform(X_train_std)

            model_pca = IsolationForest(n_estimators=100, random_state=42)
            model_pca.fit(X_train_pca)

            train_pca_scores = model_pca.decision_function(X_train_pca)
            test_pca_scores = model_pca.decision_function(
                pca.transform(X_test_std)
            )

            save_scores_csv(
                score_root / "all_pca_train_scores.csv",
                train_samples,
                train_pca_scores,
                "iforest_pca",
                "all_pca",
            )
            save_scores_csv(
                score_root / "all_pca_test_scores.csv",
                test_samples,
                test_pca_scores,
                "iforest_pca",
                "all_pca",
            )
            np.save(score_root / "all_pca_test_score.npy", test_pca_scores)

            if X_val_std is not None:
                val_pca_scores = model_pca.decision_function(
                    pca.transform(X_val_std)
                )
                save_scores_csv(
                    score_root / "all_pca_val_scores.csv",
                    val_samples,
                    val_pca_scores,
                    "iforest_pca",
                    "all_pca",
                )

            print(
                f"PCA 降维后维度: {X_train_pca.shape[1]}，"
                f"已保存 all_pca 分数"
            )

            run_info["categories"][category]["all_pca"] = {
                "pca_dim": int(X_train_pca.shape[1]),
                "explained_variance_ratio_sum": float(
                    pca.explained_variance_ratio_.sum()
                ),
            }


if __name__ == "__main__":
    t_start = time.time()

    datasets_raw = load_dataset(cfg)

    run_info = {
        "config_path": str(CONFIG_PATH),
        "image_size": img_sz,
        "validation_ratio": cfg["dataset"].get("validation_ratio", 0.2),
        "random_seed": cfg["dataset"].get("random_seed", 42),
        "categories": {},
        "env": get_env_info(),
    }

    for category in cfg["dataset"]["categories"]:
        run_one_category(category, datasets_raw, run_info)

    run_info["total_time_s"] = time.time() - t_start

    with open(EXP_ROOT / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print(f"\n全部实验完成！运行信息已保存到 {EXP_ROOT / 'run_info.json'}")