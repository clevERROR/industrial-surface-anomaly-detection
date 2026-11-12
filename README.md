# Industrial Surface Anomaly Detection

基于传统特征与单类异常检测的工业表面缺陷检测与定位。

## Project Goal

本项目研究仅使用正常产品图像训练的工业表面异常检测方法。测试阶段同时判断图像是否存在异常，并利用局部异常分数定位异常区域。

核心问题：

1. 传统图像特征能否仅利用正常样本识别工业表面异常？
2. 局部块与多尺度策略是否能够改善小面积缺陷的检测与定位？
3. Isolation Forest、One-Class SVM 和 KNN 在不同工业表面类别上的检测与定位能力有何差异？

## Dataset

使用公开的 MVTec AD 数据集。

初始实验类别：
- carpet
- grid
- metal_nut
- screw

训练阶段仅使用正常图像。测试阶段使用正常与异常图像，异常区域的公开像素级标注仅用于最终评价。

> MVTec AD 原始数据不提交到 Git 仓库。

## Methods

### Feature Engineering
- LBP
- HOG
- grayscale statistics
- gradient statistics
- optional PCA

### Anomaly Detection
- Isolation Forest
- One-Class SVM
- KNN

### Localization
- overlapping sliding-window patches
- patch anomaly scores
- spatial score heatmap
- threshold-based anomaly mask

### Evaluation
- image-level AUROC
- image-level AP
- pixel-level AUROC
- pixel-level AP
- training / inference time

## Project Structure

```text
industrial-surface-anomaly-detection/
├── configs/
│   └── config.yaml
├── data/
├── docs/
│   └── module_interfaces.md
├── evaluation/
├── experiments/
├── results/
├── scripts/
│   └── run_pipeline.py
├── src/
├── tests/
├── .gitignore
└── requirements.txt
```

## Development Principles

1. 训练、验证、测试数据边界必须明确。
2. 测试集异常标注只能用于最终评价，不能参与模型或阈值选择。
3. 实验参数统一通过配置管理，避免散落在代码中。
4. 特征、模型、定位、评价模块保持清晰接口，支持控制变量实验。
5. 所有实验结果应能够追溯到数据划分、特征、patch 配置和检测器配置。

## Development Status

当前阶段：项目初始化。

下一阶段：
1. 固化数据划分与预处理接口。
2. 实现传统特征提取。
3. 实现统一异常检测器接口。
4. 实现 patch → heatmap → mask 定位链路。
5. 建立可复现实验与评价流程。
