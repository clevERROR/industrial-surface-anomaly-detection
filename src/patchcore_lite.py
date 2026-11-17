"""补丁级近邻异常评分：全量正常参考库、密集特征、top-k 聚合。

与既有 KNN 基线的差别只在这三处，其余尽量复用：
1. 特征来自冻结的 ImageNet ResNet18，不做 PCA 压缩；
2. 参考库来自全部正常训练图，不按角色切分；
3. 图像分数取距离分布的高分位均值，而不是单个窗口的极值。
"""

import math
from pathlib import Path

import numpy as np


BACKBONES = ('resnet18', 'resnet50', 'wide_resnet50_2')


def prepare_backbone(name, folder):
    """下载并校验官方 ImageNet 权重，返回状态字典与来源元信息。

    权重直接存实验目录，避免依赖本机全局缓存；文件名指纹由官方校验。
    """
    from torchvision.models import (ResNet18_Weights, ResNet50_Weights,
                                    Wide_ResNet50_2_Weights, resnet18, resnet50,
                                    wide_resnet50_2)

    presets = dict(resnet18=(resnet18, ResNet18_Weights.IMAGENET1K_V1),
                   resnet50=(resnet50, ResNet50_Weights.IMAGENET1K_V1),
                   wide_resnet50_2=(wide_resnet50_2,
                                    Wide_ResNet50_2_Weights.IMAGENET1K_V2))
    if name not in presets:
        raise ValueError(f'不支持的骨干: {name}')
    factory, weights = presets[name]
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    filename = weights.url.rsplit('/', 1)[-1]
    target = folder / filename
    if not target.exists():
        # 离线环境下列表目录会逐个新建，复用同一实验根下已下载的同名权重
        import shutil

        for candidate in sorted(folder.parent.parent.glob(f'*/weights/{filename}')):
            if candidate.resolve().parent != folder.resolve():
                shutil.copy2(candidate, target)
                break
    state = weights.get_state_dict(model_dir=str(folder), map_location='cpu',
                                   progress=True, check_hash=True, weights_only=True)
    transform = weights.transforms()
    return factory, state, dict(
        backbone=name, weights=str(weights),
        source_url=weights.url, pretrained_weights_loaded=True,
        backbone_frozen=True, cnn_training_epochs=0, local_finetuning=False,
        normalization_mean=list(transform.mean), normalization_std=list(transform.std),
        resize_or_center_crop=False)


def select_device(name='auto'):
    """自动优先使用GPU，显式请求不可用GPU时及时失败。"""
    import torch

    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('配置请求CUDA，但当前PyTorch不能使用GPU')
    return torch.device(name)


class PatchEncoder:
    """在 GPU 上完成解码图到补丁特征的整条链路，避免主机与显存往返。

    旧接口会把 16384×768 的 float64 结果拷回主机，单图约 100 MB，
    是整条流程的主要瓶颈；这里直接返回驻留显存的张量。
    """

    def __init__(self, backbone, weight_dir, cell_size, device='auto', neighbor=3,
                 layers=('layer2', 'layer3'), color_mean=None, prepared=None):
        import torch

        if cell_size < 1:
            raise ValueError('网格尺寸必须为正')
        if neighbor < 1 or neighbor % 2 == 0:
            raise ValueError('邻域聚合核必须为正奇数')
        if not layers or any(name not in ('layer1', 'layer2', 'layer3') for name in layers):
            raise ValueError('特征层必须取自 layer1—layer3')
        self.device = select_device(device)
        self.cell = cell_size
        self.neighbor = neighbor
        self.layers = tuple(layers)
        self.color_mean = None if color_mean is None else np.asarray(color_mean, dtype=np.float64)
        if self.color_mean is not None and (self.color_mean.shape != (3,)
                                            or not np.all(self.color_mean > 0)):
            raise ValueError('颜色目标均值必须是三个正数')
        factory, state, self.info = prepared if prepared is not None \
            else prepare_backbone(backbone, weight_dir)
        net = factory(weights=None)
        net.load_state_dict(state)
        self.net = net.requires_grad_(False).eval().to(self.device)
        self.mean = torch.tensor(self.info['normalization_mean'],
                                 device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.info['normalization_std'],
                                device=self.device).view(1, 3, 1, 1)

    def align_color(self, image):
        """把逐图通道均值按比例对齐到正常训练池均值，抵消采集批次偏色。

        只做全局线性增益，不改变局部对比度，因此不直接抹掉局部颜色缺陷。
        """
        if self.color_mean is None:
            return image
        current = image.reshape(-1, 3).mean(axis=0)
        if np.any(current <= 1e-6):
            return image
        gain = self.color_mean / current
        aligned = image.astype(np.float64) * gain
        return np.clip(aligned, 0, 255).astype(np.uint8)

    def encode(self, image):
        """返回 (补丁数, 拼接维度) 的 GPU 张量，行序与网格先行后列一致。

        拼接前对每个尺度的特征图做邻域平均，等价于扩大补丁感受野，
        可抑制单点高频噪声，同时保留缺陷区域的整体偏移。
        """
        import torch
        from torch.nn import functional as F

        height, width = image.shape[:2]
        image = self.align_color(image)
        shape = (math.ceil(height / self.cell), math.ceil(width / self.cell))
        with torch.inference_mode():
            value = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
            value = value.to(self.device, non_blocking=True).float().div_(255)
            value = ((value.unsqueeze(0) - self.mean) / self.std)
            net = self.net
            value = net.maxpool(net.relu(net.bn1(net.conv1(value))))
            parts = []
            for name in ('layer1', 'layer2', 'layer3'):
                value = getattr(net, name)(value)
                if name in self.layers:
                    pooled = F.adaptive_avg_pool2d(value, shape)
                    pooled = F.avg_pool2d(pooled, self.neighbor, stride=1,
                                          padding=self.neighbor // 2)
                    parts.append(pooled)
            merged = torch.cat(parts, dim=1)
            return (merged[0].permute(1, 2, 0).reshape(-1, merged.shape[1]).contiguous(), shape)


def gpu_sample(feature, count, generator):
    """在显存内对补丁行随机采样，不产生主机副本。"""
    import torch

    if count <= 0:
        raise ValueError('每图采样数必须为正')
    if len(feature) <= count:
        return feature
    index = torch.randperm(len(feature), device=feature.device, generator=generator)[:count]
    index, _ = torch.sort(index)
    return feature[index]



def image_score(distances, top_fraction, normalize=None):
    """取距离分布最高若干比例的均值，抑制个别噪声补丁主导整图。

    比例按补丁总数换算为整数个数，至少保留 1 个。
    normalize 指定如何抵消整图统一抬升，均不引入标签信息：
    为 'ratio' 时除以本图中位距离，为 'subtract' 时减去本图中位距离。
    减法保留绝对差距，适合缺陷面积较大、中位距离同时被抬高的类别。
    """
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError('补丁距离必须为非空有限向量')
    if not 0 < top_fraction <= 1:
        raise ValueError('top_fraction 必须在 (0, 1] 之间')
    count = max(1, int(math.ceil(values.size * top_fraction)))
    top = float(np.sort(values)[-count:].mean())
    if normalize is None:
        return top
    baseline = float(np.median(values))
    if normalize == 'subtract':
        return top - baseline
    if normalize == 'ratio':
        if baseline <= 0:
            raise ValueError('补丁距离中位数为非正值，无法做图内归一化')
        return top / baseline
    raise ValueError(f'未知归一化方式: {normalize}')


def max_score(distances):
    """保留旧的单点最大值聚合，用于同口径消融对照。"""
    return float(np.max(np.asarray(distances)))


def bank_neighborhoods(bank, neighbors, block=512):
    """预计算参考库内部每个补丁的 b 近邻索引，供邻域密度重加权查表。

    库点所在区域的局部密度是库的固有属性，与具体查询无关，因此建库后
    算一次即可；分块计算避免整张库内距离矩阵同时驻留显存。
    返回索引张量 (库规模, b)，不含补丁自身。
    """
    import torch

    if neighbors < 1:
        raise ValueError('库内近邻数必须为正')
    if neighbors >= len(bank):
        raise ValueError('库内近邻数必须小于参考库规模')
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(bank), block):
            part = bank[start:start + block]
            squared = torch.cdist(part, bank, compute_mode='use_mm_for_euclid_dist')
            rows = torch.arange(len(part), device=bank.device)
            squared[rows, rows + start] = float('inf')
            outputs.append(torch.topk(squared, neighbors, dim=1, largest=False).indices)
    return torch.cat(outputs)


def exact_knn_distances(query, bank, k, block=512, exclude=None, neighborhood_index=None):
    """在 GPU 上分块计算精确 k 近邻距离，返回每个查询点的第 k 距离。

    bank 以 float32 驻留显存，查询过程不做近似索引，避免引入额外变量。
    exclude 给出与查询点同源的参考行索引时，这些行被排除后再取近邻，
    用于在参考库包含自身图块的情况下得到无泄漏的正常分数。

    neighborhood_index 给出由 bank_neighborhoods 预计算的库内 b 近邻索引时，
    按论文 Eq.7 做邻域密度重加权：比值 r = exp(d(p,m*)) /
    Σ_{m∈𝒩_b(m*)} exp(d(p,m))，返回距离乘 (1 − r)。m* 在稠密区时
    各邻居距离与 d(p,m*) 接近，r≈1/b，分数压低；稀疏区 r→0，分数保持。
    密度取自第 1 近邻所在区域，分数基数仍是第 k 距离；重加权只调制
    分数，不改变近邻选择。库内邻域始终按完整库预计算，LOO 只屏蔽
    查询列，不随评价图重算密度。
    """
    import torch

    if k < 1:
        raise ValueError('k 必须为正')
    if not isinstance(query, torch.Tensor):
        query = torch.as_tensor(query, dtype=torch.float32, device=bank.device)
    valid = len(bank) if exclude is None else len(bank) - len(exclude)
    if valid < k:
        raise ValueError('排除同源参考后剩余补丁不足以计算 k 近邻')
    keep = None
    if exclude is not None and len(exclude):
        keep = torch.ones(len(bank), dtype=torch.bool, device=bank.device)
        keep[exclude] = False
    keep_global = keep.nonzero().flatten() if keep is not None else None
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(query), block):
            part = query[start:start + block]
            squared = torch.cdist(part, bank, compute_mode='use_mm_for_euclid_dist')
            unmasked = squared
            if keep is not None:
                squared = squared[:, keep]
            smallest = torch.topk(squared, k, dim=1, largest=False).values
            distances = torch.sqrt(smallest[:, -1].clamp_min_(0))
            if neighborhood_index is not None:
                nearest = torch.topk(squared, 1, dim=1, largest=False).indices[:, 0]
                if keep_global is not None:
                    nearest = keep_global[nearest]
                # cdist 输出即真实 L2 距离；管线分数口径是 sqrt(L2)（既有约定），
                # 因此 Eq.7 的 exp 比值用真实距离计算，无量纲因子再乘回
                # sqrt 口径的分数，两种口径互不混淆。
                nearest_distance = smallest[:, 0].clamp_min_(0)
                nbr_distance = torch.gather(unmasked, 1, neighborhood_index[nearest]).clamp_min_(0)
                ratio = torch.exp(nearest_distance) / torch.exp(nbr_distance).sum(dim=1)
                distances = distances * (1.0 - ratio)
            outputs.append(distances)
    return torch.cat(outputs).float().cpu().numpy()


def build_bank_gpu(features, limit, generator):
    """合并显存内的逐图采样结果，并整体下采样到目标规模。

    每张图先保证有代表补丁，再整体截断，避免纹理密集图独占参考库。
    """
    import torch

    merged = torch.cat(features, dim=0)
    if limit and len(merged) > limit:
        index = torch.randperm(len(merged), device=merged.device, generator=generator)[:limit]
        index, _ = torch.sort(index)
        merged = merged[index]
    return merged.contiguous()


def greedy_coreset_index(features, target, generator):
    """贪心 k 中心选点，返回被选中的行索引，便于同步参考行的来源标记。"""
    import torch

    total = len(features)
    if target >= total:
        return torch.arange(total, device=features.device)
    if target < 1:
        raise ValueError('coreset 目标数量必须为正')
    first = int(torch.randint(total, (1,), device=features.device, generator=generator))
    selected = [first]
    distance = torch.linalg.vector_norm(features - features[first], dim=1)
    for _ in range(target - 1):
        current = int(torch.argmax(distance))
        selected.append(current)
        delta = torch.linalg.vector_norm(features - features[current], dim=1)
        distance = torch.minimum(distance, delta)
    return torch.tensor(selected, device=features.device, dtype=torch.long)


def finite_sample_threshold(scores, alpha=0.05):
    """用正常校准分数的有限样本秩给出阈值，与既有流程保持一致。"""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or not 0 < alpha < 1:
        raise ValueError('正常分数必须为有限非空向量，alpha 须在 0 与 1 之间')
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    if rank > len(values):
        raise ValueError('正常校准样本不足以给出该误报预算的有限阈值')
    return float(np.sort(values)[rank - 1])
