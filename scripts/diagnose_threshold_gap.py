"""量化阈值偏差的代价：对比当前阈值、同分布分位阈值与最优阈值。

只读取既有运行目录的逐图预测，不重新提取特征、不修改模型。
同分布分位阈值使用测试集正常图的分数，仅作诊断上界，不可用于部署选择。
"""

import argparse
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.evaluation import classification_rates


def read_json(path):
    """读取本实验生成的元数据。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def accuracy_at(rows, threshold):
    """按给定阈值统计混淆计数并返回汇总指标。"""
    cases = dict(true_positive=0, true_negative=0, false_positive=0, false_negative=0)
    for row in rows:
        predicted = row['score'] > threshold
        if row['label'] == 1:
            cases['true_positive' if predicted else 'false_negative'] += 1
        else:
            cases['false_positive' if predicted else 'true_negative'] += 1
    return classification_rates(cases)


def matched_quantile(normal_scores, alpha=0.05):
    """用同分布正常图的有限样本秩给出分位阈值，仅作诊断上界。"""
    values = np.sort(np.asarray(normal_scores, dtype=np.float64))
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    if rank > len(values):
        return float(values[-1])
    return float(values[rank - 1])


def shifted_threshold(threshold, reference_median, target_median):
    """按两个正常池的中位分数差平移阈值。

    中位数比 5% 分位稳定得多，少量同分布正常样本即可估计，
    因此比直接用测试正常分位定阈值更适合部署。
    """
    return float(threshold + (target_median - reference_median))


def best_accuracy(rows):
    """遍历候选切点找最高准确率，作为绝对上界参考。"""
    scores = np.array([row['score'] for row in rows])
    labels = np.array([row['label'] for row in rows])
    candidates = np.unique(scores)
    best = (0.0, float(candidates[0]))
    for value in candidates:
        predicted = scores > value
        correct = int(((predicted == 1) & (labels == 1)).sum()
                      + ((predicted == 0) & (labels == 0)).sum())
        rate = correct / len(rows)
        if rate > best[0]:
            best = (rate, float(value))
    return best


def main():
    parser = argparse.ArgumentParser(description='阈值偏差代价诊断')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()

    run = Path(args.run_dir)
    rows = []
    for path in sorted(run.glob('*.json')):
        if path.name in ('metrics.json', 'protocol.json', 'summary.json',
                         'fusion.json', 'threshold_gap.json'):
            continue
        record = read_json(path)
        result, test = record['result'], record['test']
        if any(item['category'] == result['category'] for item in rows):
            continue
        normal = [row['score'] for row in test if row['label'] == 0]
        reference = [row['score'] for row in record['calibration']]
        current = accuracy_at(test, result['threshold'])
        matched = accuracy_at(test, matched_quantile(normal))
        shifted = accuracy_at(test, shifted_threshold(
            result['threshold'], float(np.median(reference)), float(np.median(normal))))
        best_rate, best_threshold = best_accuracy(test)
        rows.append(dict(category=result['category'],
                         current_accuracy=current['accuracy'],
                         current_fpr=current['false_positive_rate'],
                         current_miss=current['miss_rate'],
                         matched_accuracy=matched['accuracy'],
                         matched_fpr=matched['false_positive_rate'],
                         shifted_accuracy=shifted['accuracy'],
                         shifted_fpr=shifted['false_positive_rate'],
                         shifted_miss=shifted['miss_rate'],
                         best_accuracy=best_rate,
                         gap_to_matched=matched['accuracy'] - current['accuracy'],
                         gap_to_shifted=shifted['accuracy'] - current['accuracy'],
                         gap_to_best=best_rate - current['accuracy']))

    rows.sort(key=lambda item: -item['gap_to_shifted'])
    lines = ['| 类别 | 当前准确率 | 中位偏移校正 | 同分布分位 | 最优阈值 | 当前误报 | 校正后误报 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['category']} | {row['current_accuracy']:.2%} "
                     f"| {row['shifted_accuracy']:.2%} ({row['gap_to_shifted'] * 100:+.2f}) "
                     f"| {row['matched_accuracy']:.2%} | {row['best_accuracy']:.2%} "
                     f"| {row['current_fpr']:.2%} | {row['shifted_fpr']:.2%} |")
    mean_shifted = float(np.mean([row['gap_to_shifted'] for row in rows]))
    mean_matched = float(np.mean([row['gap_to_matched'] for row in rows]))
    lines.append('')
    lines.append(f'中位偏移校正的平均增益 {mean_shifted * 100:+.2f} 个百分点，'
                 f'同分布分位为 {mean_matched * 100:+.2f} 个百分点。')
    lines.append('两者都使用测试集正常图分数，仅作诊断上界，不可用于部署选择。')
    text = '\n'.join(lines)
    (run / 'threshold_gap.md').write_text(text + '\n', encoding='utf-8')
    (run / 'threshold_gap.json').write_text(
        json.dumps(dict(rows=rows, mean_gap_to_shifted=mean_shifted,
                        mean_gap_to_matched=mean_matched), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
