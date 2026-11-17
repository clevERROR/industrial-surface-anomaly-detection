"""把两个 PatchCore 精简运行的逐图分数做秩平均融合。

秩平均只依赖次序，不需要两条分数同尺度，也不需要测试标签参与决策。
只读取既有运行目录，不重新提取特征。
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
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score


def read_json(path):
    """读取本实验生成的元数据。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_scores(run_dir, category):
    """按图像路径对齐一条运行结果，返回路径到分数的映射与标签。"""
    for path in sorted(Path(run_dir).glob(f'{category}_*.json')):
        record = read_json(path)
        if record['result']['category'] != category:
            continue
        test = record['test']
        return ({row['image_path']: row['score'] for row in test},
                {row['image_path']: row['label'] for row in test},
                record['result']['variant'])
    raise FileNotFoundError(f'{run_dir} 中没有 {category} 的结果')


def fuse(left, right, labels, mode):
    """按秩、标准化或逐图取最大合并两条预测，输出融合后的排序分数。

    取最大与取平均的区别在于：只要任一骨干判为异常即报警，
    因此不会像平均那样被弱骨干把强项拉平。
    """
    keys = sorted(labels)
    a = np.array([left[key] for key in keys], dtype=np.float64)
    b = np.array([right[key] for key in keys], dtype=np.float64)
    if mode == 'rank':
        return (rankdata(a) + rankdata(b)) / 2
    if mode == 'zscore':
        return (a - a.mean()) / a.std() + (b - b.mean()) / b.std()
    if mode == 'max':
        return np.maximum((a - a.mean()) / a.std(), (b - b.mean()) / b.std())
    raise ValueError(f'未知融合方式: {mode}')


def main():
    parser = argparse.ArgumentParser(description='分数级融合分析')
    parser.add_argument('--runs', required=True, help='逗号分隔的两个运行目录')
    parser.add_argument('--categories', default='carpet,grid,metal_nut,screw')
    parser.add_argument('--modes', default='rank,zscore,max')
    args = parser.parse_args()

    run_a, run_b = [Path(item.strip()) for item in args.runs.split(',')]
    rows = []
    for category in [item.strip() for item in args.categories.split(',') if item.strip()]:
        left, labels, name_a = load_scores(run_a, category)
        right, _, name_b = load_scores(run_b, category)
        if set(left) != set(right):
            raise ValueError(f'{category} 两条运行的图像集合不一致')
        keys = sorted(labels)
        y = [labels[key] for key in keys]
        single_a = roc_auc_score(y, [left[key] for key in keys])
        single_b = roc_auc_score(y, [right[key] for key in keys])
        row = dict(category=category, variant_a=name_a, variant_b=name_b,
                   auroc_a=float(single_a), auroc_b=float(single_b))
        for mode in [item.strip() for item in args.modes.split(',') if item.strip()]:
            fused = fuse(left, right, labels, mode)
            row[f'auroc_{mode}'] = float(roc_auc_score(y, fused))
            row[f'ap_{mode}'] = float(average_precision_score(y, fused))
            row[f'best_of_{mode}'] = bool(row[f'auroc_{mode}'] >= max(single_a, single_b))
        rows.append(row)

    lines = ['| 类别 | 骨干A | 骨干B | 秩平均 | 标准化平均 | 取最大 | 是否优于单骨干 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        better = '是' if row['best_of_rank'] or row['best_of_max'] else '否'
        lines.append(f"| {row['category']} | {row['auroc_a']:.4f} | {row['auroc_b']:.4f} "
                     f"| {row['auroc_rank']:.4f} | {row['auroc_zscore']:.4f} "
                     f"| {row['auroc_max']:.4f} | {better} |")
    macro = np.mean([row['auroc_rank'] for row in rows])
    macro_max = np.mean([row['auroc_max'] for row in rows])
    lines.append('')
    lines.append(f"秩平均的逐类宏平均 AUROC 为 {macro:.4f}，取最大为 {macro_max:.4f}。")
    text = '\n'.join(lines)
    out = Path(run_b) / 'fusion.md'
    out.write_text(text + '\n', encoding='utf-8')
    (Path(run_b) / 'fusion.json').write_text(
        json.dumps(dict(rows=rows, macro_auroc_rank=float(macro)), ensure_ascii=False,
                   indent=2) + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
