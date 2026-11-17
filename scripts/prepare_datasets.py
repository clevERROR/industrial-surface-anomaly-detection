"""下载、整理并核对完整 MVTec AD 与官方单类划分 VisA。"""

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_setup import download_datasets
from src.project_paths import RUNS_ROOT, relative_path, resolve_path


def main(argv=None):
    """默认使用同级辅助目录缓存归档，训练输入最终整理到 Main/data。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=('mvtec_ad', 'visa'), default=['mvtec_ad', 'visa'])
    parser.add_argument('--cache-dir', default=relative_path(RUNS_ROOT / 'downloads'))
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    if args.download_only and args.verify_only:
        parser.error('--download-only和--verify-only不能同时使用')
    if args.verify_only:
        from src.dataset_setup import audit_dataset, write_json
        for name in args.datasets:
            report = audit_dataset(name)
            write_json(resolve_path(f'data/{name}/_meta/inventory.json'), report)
            print(name, report['totals'], flush=True)
        return
    cache = resolve_path(args.cache_dir)
    result = download_datasets(args.datasets, cache)
    if not args.download_only:
        from src.dataset_setup import prepare_dataset
        for name in args.datasets:
            prepare_dataset(name, cache, result[name])


if __name__ == '__main__':
    main()
