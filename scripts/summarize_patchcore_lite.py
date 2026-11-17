"""汇总 PatchCore 精简实验的跨类指标，并给出平凡基线对照。

只读取既有运行目录中的逐图预测，不重新计算特征或修改模型。
"""

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluation import classification_rates


def read_json(path):
    """读取本实验生成的元数据。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def tpr_at_fpr(labels, scores, target=0.05):
    """测试集 ROC 上目标误报率处的检出率，只作评价，不参与阈值选择。

    与固定阈值准确率不同，它不依赖校准池与测试池是否同分布。
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    index = int(np.searchsorted(fpr, target))
    return float(tpr[min(index, len(tpr) - 1)])


def main():
    parser = argparse.ArgumentParser(description='汇总跨类指标')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()

    runs = [Path(item.strip()) for item in args.run_dir.split(',') if item.strip()]
    if not runs:
        raise ValueError('至少需要一个运行目录')
    output = runs[-1]
    candidates = []
    for run in runs:
        candidates.extend(path for path in sorted(run.glob('*.json'))
                          if path.name not in ('metrics.json', 'protocol.json',
                                               'summary.json', 'fusion.json'))
    records = [read_json(path) for path in candidates]
    listed = None
    if len(runs) == 1:
        metrics_path = runs[0] / 'metrics.json'
        if metrics_path.exists():
            listed = [(row['category'], row['variant']) for row in read_json(metrics_path)]
            present = {record['result']['category'] for record in records}
            if len({category for category, _ in listed}) < len(present):
                listed = None
    rows = []
    seen = set()
    for record in records:
        result, test = record['result'], record['test']
        key = (result['category'], result['variant'])
        if listed is not None and key not in listed:
            continue
        if key in seen:
            continue
        seen.add(key)
        cases = dict(true_positive=0, true_negative=0, false_positive=0, false_negative=0)
        for row in test:
            predicted = row['score'] > result['threshold']
            if row['label'] == 1:
                cases['true_positive' if predicted else 'false_negative'] += 1
            else:
                cases['false_positive' if predicted else 'true_negative'] += 1
        merged = classification_rates(cases)
        labels = [row['label'] for row in test]
        scores = [row['score'] for row in test]
        merged.update(category=result['category'], variant=result['variant'],
                      cases=dict(cases),
                      image_auroc=result['image_auroc'],
                      average_precision=result['average_precision'],
                      tpr_at_5fpr=tpr_at_fpr(labels, scores),
                      trivial_accuracy=merged['anomaly_count'] / merged['test_count'])
        rows.append(merged)

    total = dict(true_positive=0, true_negative=0, false_positive=0, false_negative=0)
    for row in rows:
        for key in total:
            total[key] += row['cases'][key]
    merged = classification_rates(total)
    macro_auroc = float(np.mean([row['image_auroc'] for row in rows]))
    macro_ap = float(np.mean([row['average_precision'] for row in rows]))
    macro_tpr = float(np.mean([row['tpr_at_5fpr'] for row in rows]))
    merged.update(category='COMBINED', trivial_accuracy=merged['anomaly_count'] / merged['test_count'],
                  image_auroc=macro_auroc, macro_auroc=macro_auroc,
                  average_precision=macro_ap, macro_average_precision=macro_ap,
                  tpr_at_5fpr=macro_tpr, macro_tpr_at_5fpr=macro_tpr)
    rows.append(merged)

    lines = ['| 类别 | 准确率 | 漏检率 | 误报率 | 图像AUROC | AP | TPR@5%FPR | 全判异常基线 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['category']} | {row['accuracy']:.2%} | {row['miss_rate']:.2%} "
                     f"| {row['false_positive_rate']:.2%} | {row['image_auroc']:.4f} "
                     f"| {row['average_precision']:.4f} | {row['tpr_at_5fpr']:.2%} "
                     f"| {row['trivial_accuracy']:.2%} |")
    lines.append('')
    lines.append(f"合并 AUROC 为逐类宏平均 {merged['macro_auroc']:.4f}，"
                 f"TPR@5%FPR 宏平均 {merged['macro_tpr_at_5fpr']:.2%}；"
                 f"合并准确率 {merged['accuracy']:.2%}，高于全判异常 {merged['trivial_accuracy']:.2%} "
                 f"{(merged['accuracy'] - merged['trivial_accuracy']) * 100:.2f} 个百分点。")
    text = '\n'.join(lines)
    (output / 'summary.md').write_text(text + '\n', encoding='utf-8')
    (output / 'summary.json').write_text(
        json.dumps(dict(per_category=rows[:-1], combined=merged), ensure_ascii=False,
                   indent=2) + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
