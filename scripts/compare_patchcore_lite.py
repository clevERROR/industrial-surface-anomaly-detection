"""对照实验：冻结ImageNet密集特征 + 全量参考库 + top-k聚合。

全程在 GPU 上完成特征提取、补丁采样与精确近邻；只把最终标量分数取回主机。
只评价 AUROC 与阈值准确率，不修改既有基线、默认配置或历史产物。
"""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
from time import perf_counter

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch

from src.data import _load_train_samples, _split_train_val, _load_test_samples
from src.evaluation import classification_rates
from src.image_io import read_image
from src.patchcore_lite import (PatchEncoder, gpu_sample, build_bank_gpu,
                                greedy_coreset_index, bank_neighborhoods,
                                exact_knn_distances, image_score, max_score,
                                finite_sample_threshold, prepare_backbone)
from src.project_paths import RUNS_ROOT, resolve_path


AGGREGATES = {'max': ('max', 1.0), 'top1': ('top', 0.01),
              'top01': ('top', 0.001), 'top001': ('top', 0.0001)}


def load_finetuned_prepared(prepared, category, backbone, weights_dir):
    """把该类的微调骨干权重装入 prepared 元组，保持归一化等元信息。

    payload 由 finetune_patchcore.py 保存，只含张量与基本类型；
    骨干结构由官方 factory 重建并严格加载，缺失或多余键都会报错，
    避免静默退回 ImageNet 权重。
    """
    path = Path(weights_dir) / f'{category}_finetuned_{backbone}.pth'
    if not path.exists():
        raise FileNotFoundError(f'微调权重不存在: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    factory, _, info = prepared
    config = payload.get('config', {})
    info = dict(info, finetuned=True, finetuned_weights=str(path),
                finetuned_epochs=config.get('epochs'),
                finetuned_train_images=config.get('train_images'))
    return factory, payload['backbone'], info


def build_variants(cells, aggregates):
    """按网格与聚合的笛卡尔积生成配置名，便于一次扫清组合。"""
    items = []
    for cell in cells:
        for name in aggregates:
            if name not in AGGREGATES:
                raise ValueError(f'未知聚合方式: {name}')
            aggregate, fraction = AGGREGATES[name]
            items.append(dict(name=f'cell{cell}_{name}', cell=cell,
                              aggregate=aggregate, top_fraction=fraction))
    return items


def write_json(path, value):
    """保存可移植元数据，拒绝非有限JSON数值。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def read_image_rgb(sample):
    """按原分辨率解码，不做缩放，保留细缺陷的高频信息。"""
    image = read_image(str(resolve_path(sample['image_path'])), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图像读取失败: {sample['image_path']}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def channel_mean(samples):
    """扫描正常训练池统计逐通道均值，作为颜色对齐目标，不读取测试图。"""
    total = np.zeros(3, dtype=np.float64)
    count = 0
    for sample in samples:
        image = read_image_rgb(sample)
        total += image.reshape(-1, 3).sum(axis=0)
        count += image.shape[0] * image.shape[1]
    if not count:
        raise ValueError('正常训练池为空，无法统计颜色均值')
    return total / count


def scores_for(encoder, samples, bank, settings, calibrate=False, rw_index=None):
    """逐图在显存内提取补丁并汇总为图像分数，同时记录峰值位置。"""
    rows = []
    for index, sample in enumerate(samples):
        image = read_image_rgb(sample)
        feature, shape = encoder.encode(image)
        distance = exact_knn_distances(feature, bank, settings['k'],
                                       neighborhood_index=rw_index)
        score = max_score(distance) if settings['aggregate'] == 'max' \
            else image_score(distance, settings['top_fraction'], settings['normalize'])
        height, width = image.shape[:2]
        peak_y, peak_x = divmod(int(np.argmax(distance)), shape[1])
        rows.append(dict(image_path=sample['image_path'], category=sample['category'],
                         label=int(sample['label']), defect_type=sample['defect_type'],
                         score=score, patches=int(len(distance)), grid=f'{shape[0]}x{shape[1]}',
                         peak_x=round(peak_x * width / shape[1], 1),
                         peak_y=round(peak_y * height / shape[0], 1)))
        if (index + 1) % 40 == 0 or index + 1 == len(samples):
            print(json.dumps(dict(stage='calibration' if calibrate else 'test',
                                  done=index + 1, total=len(samples)), ensure_ascii=False),
                  flush=True)
    return rows


def evaluate(rows, threshold):
    """按混淆计数给出图像级指标，AUROC 作为主口径。"""
    labels = [row['label'] for row in rows]
    scores = [row['score'] for row in rows]
    cases = Counter({'true_positive': 0, 'false_negative': 0,
                     'false_positive': 0, 'true_negative': 0})
    for row in rows:
        predicted = row['score'] > threshold
        if row['label'] == 1 and predicted:
            cases['true_positive'] += 1
        elif row['label'] == 1:
            cases['false_negative'] += 1
        elif predicted:
            cases['false_positive'] += 1
        else:
            cases['true_negative'] += 1
    result = classification_rates(dict(cases))
    result.update(
        image_auroc=float(roc_auc_score(labels, scores)),
        average_precision=float(average_precision_score(labels, scores)),
        threshold=float(threshold),
        normal_score_median=float(np.median([s for s, l in zip(scores, labels) if l == 0])),
        anomaly_score_median=float(np.median([s for s, l in zip(scores, labels) if l == 1])))
    return result


def build_reference(encoder, samples, settings, generator):
    """只从给定正常样本构建参考库，并记录每个参考补丁的来源图序号。

    来源序号用于留一法校准：评价某张正常图时排除它自己贡献的补丁。
    """
    features, owners = [], []
    for index, sample in enumerate(samples):
        part = gpu_sample(encoder.encode(read_image_rgb(sample))[0],
                          settings['per_image'], generator)
        features.append(part)
        owners.append(torch.full((len(part),), index, dtype=torch.long, device=part.device))
    merged = torch.cat(features, dim=0)
    owner = torch.cat(owners, dim=0)
    del features, owners
    if settings['coreset']:
        pool_limit = settings['bank_size'] * 10
        if len(merged) > pool_limit:
            chosen = torch.randperm(len(merged), device=merged.device, generator=generator)[:pool_limit]
            chosen, _ = torch.sort(chosen)
            merged, owner = merged[chosen], owner[chosen]
        picked = greedy_coreset_index(merged, settings['bank_size'], generator)
        merged, owner = merged[picked], owner[picked]
    elif settings['bank_size'] and len(merged) > settings['bank_size']:
        chosen = torch.randperm(len(merged), device=merged.device, generator=generator)[:settings['bank_size']]
        chosen, _ = torch.sort(chosen)
        merged, owner = merged[chosen], owner[chosen]
    torch.cuda.empty_cache()
    return merged.contiguous(), owner


def leave_one_out_calibration(encoder, normal_samples, bank, owner, settings, rw_index=None):
    """用同一参考库给全部正常图打分，评价时排除该图自己贡献的补丁。

    相比分折重建，它只建一次库，却能让阈值的分位估计覆盖全部正常图。
    """
    rows = []
    for index, sample in enumerate(normal_samples):
        feature = encoder.encode(read_image_rgb(sample))[0]
        excluded = (owner == index).nonzero().flatten()
        distance = exact_knn_distances(feature, bank, settings['k'],
                                       exclude=excluded if len(excluded) else None,
                                       neighborhood_index=rw_index)
        score = max_score(distance) if settings['aggregate'] == 'max' \
            else image_score(distance, settings['top_fraction'], settings['normalize'])
        rows.append(dict(image_path=sample['image_path'], category=sample['category'],
                         label=int(sample['label']), defect_type=sample['defect_type'],
                         score=score, patches=int(len(distance)), excluded=int(len(excluded))))
        if (index + 1) % 40 == 0 or index + 1 == len(normal_samples):
            print(json.dumps(dict(stage='leave_one_out', done=index + 1,
                                  total=len(normal_samples)), ensure_ascii=False), flush=True)
    return rows


def fold_calibration(encoder, normal_samples, settings, generator, folds, seed):
    """分折给出无泄漏的正常分数：每折用不含自身的那部分建库。

    每张正常图只被评一次，其参考库始终不含它本身，因此阈值的分位估计
    样本量等于全部正常图，而不是单折留出的那一小部分。
    """
    order = list(range(len(normal_samples)))
    random.Random(seed).shuffle(order)
    chunks = [order[index::folds] for index in range(folds)]
    rows = []
    for index, chunk in enumerate(chunks):
        if not chunk:
            raise ValueError('分折后出现空折，正常图不足以支撑该折数')
        reference = [normal_samples[position]
                     for other, part in enumerate(chunks) if other != index
                     for position in part]
        bank, _owner = build_reference(encoder, reference, settings, generator)
        rw_index = bank_neighborhoods(bank, settings['rw_neighbors']) \
            if settings['reweight'] else None
        rows.extend(scores_for(encoder, [normal_samples[position] for position in chunk],
                               bank, settings, calibrate=True, rw_index=rw_index))
        del bank, _owner, rw_index
        torch.cuda.empty_cache()
    return rows


def run_variant(variant, category, train_samples, calibration_samples, test_samples,
                config, output, prepared_map, color_target=None):
    """对单个配置完成参考库、正常校准与一次测试评价。

    允许 folds>1 时改用分折校准：参考库最终仍用全部正常图重建。
    允许多个骨干：各自独立建库并打分，再用正常校准集的中位与稳健尺度做标准化后求和。
    融合参数只来自正常校准池，测试标签仍只用于最终评价。
    """
    settings = dict(variant, k=config['knn_k'], bank_size=config['bank_size'],
                    per_image=config['per_image'], coreset=bool(config['coreset']),
                    normalize=None if config['normalize'] == 'none' else config['normalize'],
                    align_color=bool(config['align_color']),
                    reweight=bool(config['reweight']),
                    rw_neighbors=int(config['rw_neighbors']),
                    finetuned=bool(config.get('finetuned_weights_dir')))
    backbones = config['backbones']
    torch.manual_seed(config['seed'])
    generator = torch.Generator(device='cuda').manual_seed(config['seed'])
    folds = int(config['folds'])
    normal_samples = train_samples + calibration_samples if folds > 1 else None
    finetuned_dir = config.get('finetuned_weights_dir')

    calibration_rows, test_rows = None, None
    calibration_parts, test_parts, details = [], [], {}
    for backbone in backbones:
        prepared = prepared_map[backbone]
        if finetuned_dir:
            prepared = load_finetuned_prepared(prepared, category, backbone, finetuned_dir)
        encoder = PatchEncoder(backbone, output / 'weights', variant['cell'],
                               device='cuda', layers=config['layers'],
                               color_mean=color_target if config['align_color'] else None,
                               prepared=prepared)
        started = perf_counter()
        if config['loo']:
            normal = train_samples + calibration_samples
            bank, owner = build_reference(encoder, normal, settings, generator)
            rw_index = bank_neighborhoods(bank, settings['rw_neighbors']) \
                if settings['reweight'] else None
            calibration = leave_one_out_calibration(encoder, normal, bank, owner, settings,
                                                    rw_index=rw_index)
        elif folds > 1:
            calibration = fold_calibration(encoder, normal_samples, settings, generator,
                                           folds, config['seed'])
            bank, owner = build_reference(encoder, normal_samples, settings, generator)
            rw_index = bank_neighborhoods(bank, settings['rw_neighbors']) \
                if settings['reweight'] else None
        else:
            bank, owner = build_reference(encoder, train_samples, settings, generator)
            rw_index = bank_neighborhoods(bank, settings['rw_neighbors']) \
                if settings['reweight'] else None
            calibration = scores_for(encoder, calibration_samples, bank, settings,
                                     calibrate=True, rw_index=rw_index)
        build_seconds = perf_counter() - started
        del owner

        test = scores_for(encoder, test_samples, bank, settings, rw_index=rw_index)
        del rw_index
        calibration_parts.append([row['score'] for row in calibration])
        test_parts.append([row['score'] for row in test])
        details[backbone] = dict(reference_patches=int(len(bank)),
                                 feature_dim=int(bank.shape[1]),
                                 calibration_images=len(calibration),
                                 build_seconds=round(build_seconds, 2),
                                 finetuned=bool(finetuned_dir))
        if calibration_rows is None:
            calibration_rows, test_rows = calibration, test
        del encoder, bank
        torch.cuda.empty_cache()

    if len(backbones) == 1:
        fused_calibration = np.asarray(calibration_parts[0], dtype=np.float64)
        fused_test = np.asarray(test_parts[0], dtype=np.float64)
    else:
        fused_calibration = np.zeros(len(calibration_parts[0]))
        fused_test = np.zeros(len(test_parts[0]))
        for cal, test in zip(calibration_parts, test_parts):
            values = np.asarray(cal, dtype=np.float64)
            center = float(np.median(values))
            scale = float(np.median(np.abs(values - center))) * 1.4826
            if scale <= 0:
                scale = float(values.std()) or 1.0
            fused_calibration += (values - center) / scale
            fused_test += (np.asarray(test, dtype=np.float64) - center) / scale

    for row, score in zip(calibration_rows, fused_calibration):
        row['score'] = float(score)
    for row, score in zip(test_rows, fused_test):
        row['score'] = float(score)

    threshold = finite_sample_threshold(fused_calibration, config['normal_fpr'])
    result = evaluate(test_rows, threshold)
    result.update(variant=variant['name'], category=category, cell=variant['cell'],
                  train_images=len(train_samples),
                  calibration_images=len(calibration_rows),
                  folds=folds, backbones=list(backbones), details=details,
                  reference_patches=details[backbones[0]]['reference_patches'],
                  feature_dim=details[backbones[0]]['feature_dim'],
                  device='cuda')
    write_json(output / f'{category}_{variant["name"]}.json',
               dict(settings=settings, result=result,
                    calibration=calibration_rows, test=test_rows))
    return result


def main():
    parser = argparse.ArgumentParser(description='PatchCore 精简对照实验')
    parser.add_argument('--categories', default='screw')
    parser.add_argument('--cells', default='16')
    parser.add_argument('--aggregates', default='max,top01,top001')
    parser.add_argument('--knn-k', type=int, default=9)
    parser.add_argument('--bank-size', type=int, default=50000)
    parser.add_argument('--per-image', type=int, default=1024)
    parser.add_argument('--validation-ratio', type=float, default=0.2)
    parser.add_argument('--normal-fpr', type=float, default=0.05)
    parser.add_argument('--normalize', default='none', choices=['none', 'ratio', 'subtract'],
                        help='图内分数归一化方式，默认不做')
    parser.add_argument('--layers', default='layer2,layer3',
                        help='参与拼接的骨干层，逗号分隔')
    parser.add_argument('--align-color', action='store_true',
                        help='按正常训练池通道均值做逐图颜色对齐')
    parser.add_argument('--backbones', default='resnet18',
                        help='逗号分隔的冻结骨干；多于一个时按正常校准集标准化后融合')
    parser.add_argument('--coreset', action='store_true',
                        help='参考库改用贪心k中心采样，而不是随机下采样')
    parser.add_argument('--reweight', action='store_true',
                        help='按论文 Eq.7 做库区邻域密度重加权，抑制稠密边缘正常补丁的误报')
    parser.add_argument('--rw-neighbors', type=int, default=3,
                        help='重加权使用的库内近邻数 b')
    parser.add_argument('--finetuned-weights-dir', default=None,
                        help='逐类微调权重目录；存在 <类别>_finetuned_<骨干>.pth 时'
                             '替换冻结 ImageNet 权重，变体名追加 _ft')
    parser.add_argument('--folds', type=int, default=1,
                        help='正常校准分折数；大于1时每折用不含自身的参考库打分')
    parser.add_argument('--loo', action='store_true',
                        help='留一法校准：参考库含全部正常图，评价时排除该图自身补丁')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('本实验要求GPU，当前PyTorch不可用CUDA')
    torch.backends.cudnn.benchmark = True
    args.layers = tuple(name.strip() for name in args.layers.split(',') if name.strip())
    backbones = tuple(name.strip() for name in args.backbones.split(',') if name.strip())
    if not backbones:
        raise ValueError('至少需要一个骨干')

    output = Path(args.output_dir) if args.output_dir else \
        Path(RUNS_ROOT) / 'experiments' / 'patchcore_lite'
    output.mkdir(parents=True, exist_ok=True)
    prepared_map = {name: prepare_backbone(name, output / 'weights') for name in backbones}
    weight_info = prepared_map[backbones[0]][2]
    args.backbones = backbones
    write_json(output / 'protocol.json', dict(
        weight_info=weight_info, args=vars(args), device='cuda',
        source_sha256={'src/patchcore_lite.py': sha256(
            (ROOT / 'src/patchcore_lite.py').read_bytes()).hexdigest()},
        note='参考库仅来自正常训练池，阈值仅来自正常校准池，测试标签只用于最终评价。'))

    root = resolve_path('data/mvtec_ad')
    cells = [int(value) for value in args.cells.split(',') if value.strip()]
    aggregates = [value.strip() for value in args.aggregates.split(',') if value.strip()]
    variants = build_variants(cells, aggregates)
    if args.normalize != 'none':
        for item in variants:
            item['name'] += f'_norm_{args.normalize}'
    if args.align_color:
        for item in variants:
            item['name'] += '_ca'
    if args.coreset:
        for item in variants:
            item['name'] += '_cs'
    if args.reweight:
        for item in variants:
            item['name'] += '_rw'
    if args.finetuned_weights_dir:
        for item in variants:
            item['name'] += '_ft'
    if args.folds > 1:
        for item in variants:
            item['name'] += f'_f{args.folds}'
    if args.loo:
        for item in variants:
            item['name'] += '_loo'
    if len(backbones) > 1:
        for item in variants:
            item['name'] += '_fuse'
    elif backbones[0] != 'resnet18':
        for item in variants:
            item['name'] += f'_{backbones[0]}'
    results = []
    for category in [name.strip() for name in args.categories.split(',') if name.strip()]:
        samples = _load_train_samples(root / category, category)
        train, calibration = _split_train_val(samples, args.validation_ratio, args.seed)
        test = _load_test_samples(root / category, category)
        print(json.dumps(dict(category=category, train=len(train),
                              calibration=len(calibration), test=len(test),
                              normal=sum(1 for s in test if s['label'] == 0)), ensure_ascii=False),
              flush=True)
        target = channel_mean(train) if args.align_color else None
        for variant in variants:
            result = run_variant(variant, category, train, calibration, test,
                                 vars(args), output, prepared_map, target)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    results.sort(key=lambda row: (row['category'], -row['image_auroc']))
    write_json(output / 'metrics.json', results)
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    lines = ['# PatchCore 精简对照实验', '',
             '冻结 ImageNet ResNet18，密集补丁特征不降维；参考库来自正常训练池，',
             '阈值只来自正常校准池的 5% 有限样本秩；全流程在 GPU 上完成，主口径为图像 AUROC。',
             '跨类合并指标与平凡基线见同目录 summary.md，由 summarize_patchcore_lite.py 生成。', '',
             '| 类别 | 配置 | 网格 | 参考补丁 | 准确率 | 漏检 | 误报 | AUROC | AP |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in results:
        lines.append(f"| {row['category']} | {row['variant']} | {row['cell']}px "
                     f"| {row['reference_patches']} | {row['accuracy']:.2%} "
                     f"| {row['miss_rate']:.2%} | {row['false_positive_rate']:.2%} "
                     f"| {row['image_auroc']:.4f} | {row['average_precision']:.4f} |")
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
