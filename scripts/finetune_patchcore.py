"""在正常训练图上做 SimCLR 风格对比微调，产出逐类骨干权重。

只接触 train split 的正常图（与下游评价同一划分、同一种子），
校准池与测试图不参与训练。权重写入实验目录的 weights/，文件名
`<类别>_finetuned_<骨干>.pth`，由 compare_patchcore_lite.py 的
--finetuned-weights-dir 加载做同协议下游对照。
"""

import argparse
import json
from hashlib import sha256
from pathlib import Path
import sys
from time import perf_counter

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import _load_train_samples, _split_train_val
from src.image_io import read_image
from src.patchcore_finetune import (ContrastiveFineTuner, SimCLRAugment,
                                    save_finetuned)
from src.patchcore_lite import prepare_backbone
from src.project_paths import RUNS_ROOT, resolve_path


class PairViewDataset(Dataset):
    """每张正常图独立增广两次，返回一对视图（float32 CHW，0—1）。"""

    def __init__(self, samples, augment):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = read_image(str(resolve_path(sample['image_path'])), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"图像读取失败: {sample['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return self.augment(tensor), self.augment(tensor)


def main():
    parser = argparse.ArgumentParser(description='PatchCore 骨干自监督微调（仅正常数据）')
    parser.add_argument('--categories', default='screw')
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pairs', type=int, default=32,
                        help='每步的图像对数；实际前向批为 2×pairs 个视图')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--crop-scale', type=float, default=0.5)
    parser.add_argument('--validation-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('本实验要求GPU，当前PyTorch不可用CUDA')
    free_gb = (torch.cuda.mem_get_info()[0]) / 1024 ** 3
    if free_gb < 4:
        raise RuntimeError(f'空闲显存不足 4GB（当前 {free_gb:.1f}GB），'
                           '等待在跑实验结束后再启动，避免互相污染')

    output = Path(args.output_dir) if args.output_dir else \
        Path(RUNS_ROOT) / 'experiments' / 'patchcore_finetune'
    output.mkdir(parents=True, exist_ok=True)
    root = resolve_path('data/mvtec_ad')
    prepared = prepare_backbone(args.backbone, output / 'weights')
    augment = SimCLRAugment(image_size=args.image_size, crop_scale=args.crop_scale)
    protocol = dict(args=vars(args), device='cuda', free_gb_before=round(free_gb, 2),
                    per_category={},
                    source_sha256={'src/patchcore_finetune.py': sha256(
                        (ROOT / 'src/patchcore_finetune.py').read_bytes()).hexdigest()},
                    note='微调只接触 train split 正常图；校准池与测试图不参与训练。')
    protocol_path = output / 'finetune_protocol.json'

    for category in [name.strip() for name in args.categories.split(',') if name.strip()]:
        samples = _load_train_samples(root / category, category)
        train, calibration = _split_train_val(samples, args.validation_ratio, args.seed)
        dataset = PairViewDataset(train, augment)
        loader = DataLoader(dataset, batch_size=args.pairs, shuffle=True,
                            num_workers=args.num_workers, drop_last=True,
                            generator=torch.Generator().manual_seed(args.seed))
        if len(loader) < 1:
            raise RuntimeError(f'{category} 正常训练图不足以组成一个完整批')
        torch.manual_seed(args.seed)
        tuner = ContrastiveFineTuner(prepared, device='cuda', lr=args.lr,
                                     weight_decay=args.weight_decay,
                                     temperature=args.temperature)
        started = perf_counter()
        losses = []
        for epoch in range(args.epochs):
            loss = tuner.train_epoch(loader)
            losses.append(round(loss, 4))
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == args.epochs:
                print(json.dumps(dict(category=category, epoch=epoch + 1,
                                      loss=round(loss, 4),
                                      seconds=round(perf_counter() - started, 1)),
                                  ensure_ascii=False), flush=True)
        weight_path = save_finetuned(
            output / 'weights' / f'{category}_finetuned_{args.backbone}.pth',
            tuner, config=dict(epochs=args.epochs, pairs=args.pairs, lr=args.lr,
                               weight_decay=args.weight_decay,
                               temperature=args.temperature,
                               image_size=args.image_size, crop_scale=args.crop_scale,
                               train_images=len(train), category=category,
                               backbone=args.backbone, seed=args.seed))
        protocol['per_category'][category] = dict(
            train_images=len(train), calibration_untouched=len(calibration),
            epochs=args.epochs, steps_per_epoch=len(loader),
            first_loss=losses[0], last_loss=losses[-1],
            seconds=round(perf_counter() - started, 1),
            weights=str(weight_path.relative_to(output)))
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n',
                                 encoding='utf-8')
        print(json.dumps(dict(category=category, done=True,
                              weights=str(weight_path)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
