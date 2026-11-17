"""仅正常数据的自监督对比微调：SimCLR 风格 InfoNCE。

下游 PatchCore 用冻结 ImageNet 特征度量偏离正常分布的距离；本模块在
每类的正常训练图上微调骨干，让特征空间把该类的正常变异收得更紧。
约束：
1. 只接触 train split 的正常图（_split_train_val 的第一份返回值），
   校准池与测试图不参与训练；
2. 增广保持保守（裁剪、翻转、轻度颜色抖动），不引入旋转与强噪声，
   避免抹掉缺陷敏感的线索；颜色抖动同时提供对采集批次偏色的不变性；
3. BN 统计默认冻结（前向保持 eval），只训练仿射参数，防止小数据
   批统计漂移导致表征崩坏；
4. 保存的权重只含骨干，投影头仅在训练期使用，加载路径与
   compare_patchcore_lite.py 的 --finetuned-weights-dir 对接。
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def backbone_features(net, images):
    """过骨干到 avgpool 展平（ResNet18 为 512 维），不经过分类 fc。

    与 PatchEncoder 的手动前向保持同一层序；fc 权重不参与训练与下游，
    保留在状态字典中以保证权重文件可被严格加载。
    """
    value = net.maxpool(net.relu(net.bn1(net.conv1(images))))
    value = net.layer1(value)
    value = net.layer2(value)
    value = net.layer3(value)
    value = net.layer4(value)
    return torch.flatten(net.avgpool(value), 1)


class SimCLRAugment:
    """成对视图增广：同一原始图独立采样两次，得到两个正样本视图。

    输入为 float32 CHW 张量（0—1）；裁剪在原分辨率上进行再缩放到
    目标尺寸，保持缺陷与特征的空间尺度。
    """

    def __init__(self, image_size=512, crop_scale=0.5, jitter=0.2, blur=False):
        from torchvision import transforms

        if not 0 < crop_scale <= 1:
            raise ValueError('裁剪尺度下限必须在 (0, 1]')
        parts = [transforms.RandomResizedCrop(image_size, scale=(crop_scale, 1.0)),
                 transforms.RandomHorizontalFlip()]
        if jitter > 0:
            parts.append(transforms.RandomApply(
                [transforms.ColorJitter(jitter, jitter, jitter, jitter * 0.25)], p=0.8))
        if blur:
            parts.append(transforms.GaussianBlur(5, sigma=(0.1, 1.0)))
        self.transform = transforms.Compose(parts)

    def __call__(self, image):
        return self.transform(image)


class ProjectionHead(nn.Module):
    """SimCLR 投影头：表征先过 MLP 再算对比损失，下游只用骨干输出。"""

    def __init__(self, in_dim=512, hidden=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, out_dim))

    def forward(self, features):
        return self.net(features)


def nt_xent(view_a, view_b, temperature):
    """NT-Xent 对比损失：批内 2N 个视图，只有配对视图互为正样本。

    view_a/view_b 各为 (N, d)，已过投影头；返回标量损失。
    """
    if view_a.shape != view_b.shape or view_a.shape[0] < 2:
        raise ValueError('两个视图批大小必须一致且至少为 2')
    z = F.normalize(torch.cat([view_a, view_b], dim=0), dim=1)
    count = view_a.shape[0]
    similarity = z @ z.t() / temperature
    similarity.fill_diagonal_(float('-inf'))
    targets = torch.cat([torch.arange(count, 2 * count),
                         torch.arange(0, count)]).to(z.device)
    return F.cross_entropy(similarity, targets)


class ContrastiveFineTuner:
    """封装骨干、投影头与优化器的微调器。

    net 从 prepare_backbone 的 (factory, state, info) 构建；训练只更新
    全部参数，但 BN 模块保持 eval 使统计量不漂移（train_bn 可关闭）。
    """

    def __init__(self, prepared, device='cuda', lr=1e-4, weight_decay=1e-4,
                 temperature=0.2, train_bn=False):
        factory, state, _info = prepared
        self.device = device
        self.net = factory(weights=None)
        self.net.load_state_dict(state)
        self.net = self.net.to(device)
        self.head = ProjectionHead().to(device)
        self.temperature = temperature
        self.train_bn = train_bn
        parameters = list(self.net.parameters()) + list(self.head.parameters())
        self.optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    def _set_mode(self):
        self.net.train()
        self.head.train()
        if not self.train_bn:
            for module in self.net.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

    def train_epoch(self, batches):
        """batches 产出 (view_a, view_b) 的 GPU 张量对；返回平均损失。"""
        total, steps = 0.0, 0
        self._set_mode()
        for view_a, view_b in batches:
            view_a = view_a.to(self.device, non_blocking=True)
            view_b = view_b.to(self.device, non_blocking=True)
            features = backbone_features(self.net, view_a)
            features_b = backbone_features(self.net, view_b)
            loss = nt_xent(self.head(features), self.head(features_b), self.temperature)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total += float(loss.detach())
            steps += 1
        if not steps:
            raise ValueError('训练批为空，正常图不足')
        return total / steps

    def backbone_state(self):
        """返回骨干状态字典（投影头不参与下游，不入库）。"""
        return {key: value.detach().cpu() for key, value in self.net.state_dict().items()}


def save_finetuned(path, tuner, config):
    """保存骨干权重与训练配置；payload 仅含张量与基本类型，可 weights_only 加载。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(backbone=tuner.backbone_state(),
                   config=dict(config))
    torch.save(payload, path)
    return path
