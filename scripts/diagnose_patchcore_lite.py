"""诊断既有 PatchCore 精简结果的分数分布与漏检构成。

只读取运行目录中的逐图预测，不重新提取特征、不修改阈值或模型。
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def read_json(path):
    """读取本实验生成的元数据。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def describe(values):
    """给出分布摘要，用于比较校准池与测试正常池是否同分布。"""
    array = np.asarray(values, dtype=np.float64)
    return dict(n=int(array.size),
                min=float(array.min()), p05=float(np.percentile(array, 5)),
                p25=float(np.percentile(array, 25)),
                median=float(np.median(array)),
                p75=float(np.percentile(array, 75)),
                p95=float(np.percentile(array, 95)),
                max=float(array.max()))


def main():
    parser = argparse.ArgumentParser(description='诊断分数分布与漏检构成')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()

    run = Path(args.run_dir)
    report = {}
    for path in sorted(run.glob('*.json')):
        if path.name in ('metrics.json', 'protocol.json', 'summary.json'):
            continue
        record = read_json(path)
        result = record['result']
        category, threshold = result['category'], result['threshold']
        calibration = [row['score'] for row in record['calibration']]
        test = record['test']
        normal = [row['score'] for row in test if row['label'] == 0]
        anomaly = [row['score'] for row in test if row['label'] == 1]

        by_type = defaultdict(list)
        for row in test:
            if row['label'] == 1:
                by_type[row['defect_type']].append(row['score'] > threshold)

        report[category] = dict(
            variant=result['variant'], threshold=threshold,
            calibration=describe(calibration),
            test_normal=describe(normal),
            test_anomaly=describe(anomaly),
            calibration_exceed=float(np.mean(np.asarray(calibration) > threshold)),
            test_normal_exceed=float(np.mean(np.asarray(normal) > threshold)),
            median_shift=float(np.median(normal) - np.median(calibration)),
            recall_by_defect={name: dict(count=len(hits), recall=float(np.mean(hits)))
                              for name, hits in sorted(by_type.items())})

    (run / 'diagnosis.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    lines = ['| 类别 | 阈值 | 校准中位 | 测试正常中位 | 偏移 | 校准超阈率 | 测试正常超阈率 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, row in report.items():
        lines.append(f"| {name} | {row['threshold']:.4f} | {row['calibration']['median']:.4f} "
                     f"| {row['test_normal']['median']:.4f} | {row['median_shift']:+.4f} "
                     f"| {row['calibration_exceed']:.2%} | {row['test_normal_exceed']:.2%} |")
    lines.append('')
    lines.append('| 类别 | 缺陷类型 | 数量 | 召回 |')
    lines.append('| --- | --- | ---: | ---: |')
    for name, row in report.items():
        for defect, value in row['recall_by_defect'].items():
            lines.append(f"| {name} | {defect} | {value['count']} | {value['recall']:.2%} |")
    text = '\n'.join(lines)
    (run / 'diagnosis.md').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
