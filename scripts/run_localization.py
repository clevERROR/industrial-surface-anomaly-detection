"""运行第3部分并保存四类原图尺寸的热力图、掩膜和成功/失败案例。"""

import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml

from src.data import load_dataset
from src.evaluation import classification_rates
from src.heatmap import resize_heatmap
from src.localization import create_detector
from src.postprocess import anomaly_mask
from src.preprocessing import preprocess_sample
from src.project_paths import PROJECT_ROOT, RUNS_ROOT, DEFAULT_CONFIG, relative_path, resolve_path
from src.image_io import read_image


def write_json(path, value):
    """只保存可移植的实验内容和相对路径。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read_rgb(path):
    """读取原图用于尺寸还原和案例展示。"""
    image = read_image(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'图像读取失败: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_png(path, image):
    """通过编码后写字节支持 Windows 中文目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, data = cv2.imencode('.png', image)
    if not success:
        raise ValueError(f'图像编码失败: {path}')
    path.write_bytes(data.tobytes())


def score_overlay(image, heatmap, normal_min, normal_threshold):
    """颜色刻度由正常验证数据固定，避免逐图归一化夸大正常样本。"""
    span = max(normal_threshold - normal_min, 1e-6)
    intensity = np.clip((heatmap - normal_min) / span, 0, 1)
    colored = cv2.applyColorMap(np.uint8(intensity * 255), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 0.55, colored, 0.45, 0)


def export_prediction(sample, prediction, detector, output, normal_min):
    """先独立导出预测，之后才读取测试标注计算案例指标。"""
    image = read_rgb(sample['image_path'])
    original_map = resize_heatmap(prediction['heatmap'], image.shape)
    mask = anomaly_mask(original_map, detector.pixel_threshold)
    relative_name = Path(sample['defect_type']) / Path(sample['image_path']).stem
    mask_path = output / 'masks' / relative_name.with_suffix('.png')
    heatmap_path = output / 'heatmaps' / relative_name.with_suffix('.png')
    scores_path = output / 'scores' / relative_name.with_suffix('.npz')
    overlay = score_overlay(image, original_map, normal_min, detector.pixel_threshold)
    write_png(mask_path, mask)
    write_png(heatmap_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scores_path, **prediction, pixel_threshold=detector.pixel_threshold,
                        image_threshold=detector.image_threshold, original_shape=np.array(image.shape[:2]))
    # 测试标签和 Mask 直到预测完成后才进入此分支。
    label = sample['label']
    ground_truth = np.zeros(image.shape[:2], dtype=np.uint8)
    if sample.get('mask_path') is not None:
        ground_truth = read_image(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
        if ground_truth is None or ground_truth.shape != mask.shape:
            raise ValueError(f"标注无法读取或尺寸不一致: {sample['mask_path']}")
    intersection = int(np.count_nonzero((mask > 0) & (ground_truth > 0)))
    union = int(np.count_nonzero((mask > 0) | (ground_truth > 0)))
    iou = intersection / union if union else 1.0
    kind = {(1, 1): 'true_positive', (0, 0): 'true_negative',
            (0, 1): 'false_positive', (1, 0): 'false_negative'}[(label, prediction['image_prediction'])]
    record = dict(image_id=Path(sample['image_path']).as_posix(), defect_type=sample['defect_type'],
                  label=label, prediction=prediction['image_prediction'], image_score=prediction['image_score'],
                  case_type=kind, mask_iou=iou, heatmap=relative_path(heatmap_path),
                  mask=relative_path(mask_path), scores=relative_path(scores_path))
    return record


def export_cases(records, samples, output):
    """按图像级预测类型整理案例，保留掩膜 IoU 供判断定位质量。"""
    lookup = {Path(sample['image_path']).as_posix(): sample for sample in samples}
    cases = {}
    for kind in ['true_positive', 'true_negative', 'false_positive', 'false_negative']:
        candidates = [record for record in records if record['case_type'] == kind]
        candidates.sort(key=lambda record: record['mask_iou'], reverse=(kind == 'true_positive'))
        cases[kind] = []
        for index, record in enumerate(candidates[:2]):
            sample = lookup[record['image_id']]
            image = read_rgb(record['image_id'])
            overlay = read_rgb(record['heatmap'])
            mask = read_image(record['mask'], cv2.IMREAD_GRAYSCALE)
            truth = read_image(sample['mask_path'], cv2.IMREAD_GRAYSCALE) if sample['mask_path'] else np.zeros(mask.shape, np.uint8)
            panels = [image, overlay, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), cv2.cvtColor(truth, cv2.COLOR_GRAY2RGB)]
            canvas = np.zeros((292, 1024, 3), dtype=np.uint8)
            for column, (panel, title) in enumerate(zip(panels, ['Original', 'Heatmap overlay', 'Prediction mask', 'Ground truth'])):
                canvas[36:, column * 256:(column + 1) * 256] = cv2.resize(panel, (256, 256), interpolation=cv2.INTER_NEAREST)
                cv2.putText(canvas, title, (column * 256 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            case_path = output / 'cases' / f'{kind}_{index + 1}.png'
            write_png(case_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            cases[kind].append(dict(record, comparison=relative_path(case_path)))
    write_json(output / 'cases.json', cases)


def run_category(category, splits, config, output_root, baseline_root=None):
    """对单个类别执行正常训练、正常校准及全部测试图的局部定位。"""
    output = output_root / category
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'splits.json', splits)
    if any(sample['label'] != 0 for split in ['train', 'val'] for sample in splits[split]):
        raise ValueError('训练和验证必须仅含正常图像')
    size = config['preprocess']['image_size']
    use_original = config['localization'].get('method') in ('detail_color', 'native_detail')

    def model_input(sample):
        """模型只接收像素，预测前不读取测试标注。"""
        processed = preprocess_sample(dict(sample, mask_path=None), (size, size), include_original=use_original)
        return (processed['image'], processed['original_image']) if use_original else processed['image']

    def images(samples):
        """逐张读取，避免把全部原始图像同时放入内存。"""
        for sample in samples:
            yield model_input(sample)

    detector = create_detector(config)
    started = perf_counter()
    print(f'{category}: fitting normal patches...', flush=True)
    detector.fit(images(splits['train']))
    fit_seconds = perf_counter() - started
    calibration = detector.calibrate(images(splits['val']))
    np.savez_compressed(output / 'normal_validation.npz', **calibration)
    normal_min = float(calibration['heatmaps'].min())
    print(f'{category}: train_patches={detector.training_patch_count}, pixel_threshold={detector.pixel_threshold:.6f}', flush=True)
    records = []
    seconds = []
    for index, sample in enumerate(splits['test']):
        started = perf_counter()
        prediction = detector.predict(model_input(sample))
        seconds.append(perf_counter() - started)
        records.append(export_prediction(sample, prediction, detector, output, normal_min))
        if (index + 1) % 50 == 0:
            print(f'{category}: predicted {index + 1}/{len(splits["test"])}', flush=True)
    write_json(output / 'predictions.json', records)
    export_cases(records, splits['test'], output)
    labels = [record['label'] for record in records]
    scores = [record['image_score'] for record in records]
    baseline_path = (baseline_root or RUNS_ROOT / 'results/global_baseline') / category / 'summary.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8')) if baseline_path.exists() else None
    # 基线比较固定为相同特征配置的 all 组，不能按测试效果挑选最优组。
    comparable = detector.method in ('iforest', 'metric_covariance') and baseline and baseline['config']['dataset'] == config['dataset'] and all(detector.feature_args.values()) and not config['features'].get('pca', False)
    global_auroc = baseline['experiments']['all']['image_auroc'] if comparable else None
    summary = dict(category=category, config=deepcopy(config), split_counts={key: len(value) for key, value in splits.items()},
                   score_direction='higher_is_more_anomalous', threshold_source='normal_validation',
                   training_patch_count=detector.training_patch_count, feature_dimension=detector.feature_dimension,
                   pixel_threshold=detector.pixel_threshold, image_threshold=detector.image_threshold,
                   image_auroc=float(roc_auc_score(labels, scores)), image_ap=float(average_precision_score(labels, scores)),
                   global_all_image_auroc=global_auroc, fit_seconds=fit_seconds,
                   inference_seconds_per_image=float(np.mean(seconds)),
                   anomaly_mean_mask_iou=float(np.mean([r['mask_iou'] for r in records if r['label'] == 1])),
                   cases={kind: sum(r['case_type'] == kind for r in records) for kind in ['true_positive', 'true_negative', 'false_positive', 'false_negative']},
                   prediction_index=relative_path(output / 'predictions.json'), case_index=relative_path(output / 'cases.json'))
    summary.update(classification_rates(summary['cases']))
    summary['method'] = detector.method
    summary['model_metadata'] = detector.model.metadata() if detector.method == 'metric_covariance' else dict(estimator='sklearn.ensemble.IsolationForest')
    if hasattr(detector, 'metadata'):
        summary['model_metadata'] = detector.metadata()
    summary['skipped_samples'] = []
    write_json(output / 'summary.json', summary)
    print(f'{category}: local AUROC={summary["image_auroc"]:.4f}, global all={global_auroc}, cases={summary["cases"]}', flush=True)
    return summary


def main(argv=None):
    """默认运行全部配置类别，参数化支持单类和窗口尺度的复核。"""
    parser = argparse.ArgumentParser(description='第3部分：局部异常分数、热力图和掩膜')
    parser.add_argument('--config', default=relative_path(DEFAULT_CONFIG))
    parser.add_argument('--category', action='append')
    parser.add_argument('--output-dir', default=relative_path(RUNS_ROOT), help='运行产物根目录，不需要提前存在')
    parser.add_argument('--patch-size', type=int)
    parser.add_argument('--stride', type=int)
    args = parser.parse_args(argv)
    os.chdir(PROJECT_ROOT)
    config = yaml.safe_load(resolve_path(args.config).read_text(encoding='utf-8'))
    if args.category:
        config['dataset']['categories'] = args.category
    for key in ['patch_size', 'stride']:
        if getattr(args, key) is not None:
            config['localization'][key] = getattr(args, key)
    run_root = resolve_path(args.output_dir)
    output = run_root / 'results/localization' / f'patch_{config["localization"]["patch_size"]}_stride_{config["localization"]["stride"]}'
    output.mkdir(parents=True, exist_ok=True)
    datasets = load_dataset(config)
    summaries = [run_category(category, splits, config, output, run_root / 'results/global_baseline') for category, splits in datasets.items()]
    write_json(output / 'summary.json', summaries)
    fields = ['category', 'global_all_image_auroc', 'image_auroc', 'image_ap', 'anomaly_mean_mask_iou', 'inference_seconds_per_image']
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: summary[key] for key in fields} for summary in summaries)


if __name__ == '__main__':
    main()
