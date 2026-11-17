# 工业表面异常检测

MVTec AD / VisA 上的图像级工业异常检测与局部定位。当前主线为 PatchCore 精简实现：冻结 ImageNet ResNet18 提取密集补丁特征，贪心 coreset 参考库只存正常样本，精确 k 近邻 + top 分位聚合给出图像分数，留一法（LOO）校准阈值；另含作业原始交付的第 2 部分（全图基线）与第 3 部分（局部缺陷定位）。

本目录是完整、可单独复制的核心工程：所有代码、配置、依赖清单与数据集都在本目录内，无任何对外部目录的读取依赖；运行产物默认写入工程内 `runs/`。

## 目录结构

```
Main\
├── src\            17 个模块（路径锚点 project_paths、数据、图像 IO、特征、评价、
│                   定位、数据集准备、patchcore_lite、patchcore_finetune 等）
├── scripts\        11 个入口脚本
├── tests\          pytest 单元测试
├── configs\        默认配置 config.yaml 及全量/VisA/协方差变体
├── data\           mvtec_ad（15 类，约 5GB），含 _meta 来源档案
├── requirements*.txt   依赖清单（基础 / ResNet / ResNet-CUDA 三套）
└── pyproject.toml
```

## 环境创建

推荐 Python 3.12（本次验证使用 3.12.14）；虚拟环境位置任意（例如本工程内 `python -m venv .venv`），不要复制原机器的虚拟环境。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
# PatchCore / ResNet 相关入口另需：
.venv\Scripts\python.exe -m pip install -r requirements-resnet-lock.txt
# 有 NVIDIA GPU 时可用 CUDA 版：requirements-resnet-cuda.txt
```

## 数据准备

数据集已随工程附带在 `data\`。换新机器或数据缺失时用下载脚本（断点续传、SHA256 校验）：

```powershell
python -B scripts/prepare_datasets.py --dataset mvtec_ad   # 或 visa（VisA 数据未随工程附带，约 15GB）
```

下载缓存与解压暂存默认写入 `runs/downloads`、`runs/tmp`，可用 `--cache-dir` 指定。

## 运行产物位置

所有脚本的非代码输出默认写入工程内 `runs\`（自动创建，不需要提前存在）：实验结果在 `runs\experiments\<名称>\`，第 2/3 部分结果在 `runs\results\`。可用 `--output-dir` 指定其他根目录；相对路径以本工程根目录解析，不受终端当前目录影响。持久化引用一律记录为相对路径，不写死盘符。

## 第 2 部分：全图基线

传统流程：滑窗补丁 → LBP/HOG/灰度/梯度特征 → iForest/OCSVM/KNN 一类模型 → 正常验证阈值。

```powershell
python -B scripts/inspect_dataset.py                 # 数据检查与预处理预览
python -B scripts/global_baseline.py                 # 全图基线（默认四类）
python -B scripts/global_baseline.py --category grid # 单类
python -B scripts/global_baseline.py --config configs/mvtec_ad_full.yaml   # 15 类全量
# VisA 变体配置保留在 configs/visa.yaml；数据未附带，先执行 prepare_datasets.py --dataset visa
```

## 第 3 部分：局部缺陷定位

```powershell
python -B scripts/run_pipeline.py                    # 热力图/掩码/成败案例，原始分辨率
python -B scripts/run_pipeline.py --config configs/metric_covariance.yaml
python -B scripts/run_pipeline.py --output-dir results_run
```

## PatchCore 主线（当前最优方法）

冻结 ResNet18（layer1—3，448 维）+ cell16 网格 + 3×3 邻域平均 + 颜色对齐 + 贪心 coreset 参考库 + k=3 精确近邻 + top 0.1% 聚合 + 留一法阈值。参考库只来自正常训练池，阈值只来自正常校准分数，测试标签只用于最终评价。

```powershell
# 冻结配置（15 类正式口径，--bank-size 20000 为对照口径）
python -B scripts/compare_patchcore_lite.py --categories carpet,grid,metal_nut,screw ^
    --cells 16 --aggregates top01 --knn-k 3 --per-image 4096 --bank-size 50000 ^
    --coreset --loo --layers layer1,layer2,layer3 --align-color

# 跨类汇总与平凡基线对照（写回运行目录）
python -B scripts/summarize_patchcore_lite.py --run-dir runs/experiments/patchcore_lite

# 逐图分数诊断 / 双运行秩平均融合 / 阈值偏差代价诊断
python -B scripts/diagnose_patchcore_lite.py --run-dir runs/experiments/patchcore_lite
python -B scripts/fuse_patchcore_lite.py --run-dirs <目录A>,<目录B>
python -B scripts/diagnose_threshold_gap.py --run-dir runs/experiments/patchcore_lite
```

参考库 5 万时 15 类结果：准确率 90.26%、漏检 9.46%、误报 10.49%、宏 AUROC 0.9788、TPR@5%FPR 91.99%，高于全判异常基线 17.33 个百分点；screw 单类 AUROC 0.9947 已超过论文 PatchCore-25% 的 0.981。参考库 2 万时准确率 88.93%、宏 AUROC 0.9750，建库耗时约为 5 万的 1/6。准确率在 pill、carpet、cable 上接近或低于平凡基线，主指标应以 AUROC 与 TPR@5%FPR 为准；carpet 与 leather 的误报源于 test/good 与 train/good 的采集批次外观偏移，需部署期同分布样本校准。

辅助开关（默认关闭，均有消融记录）：`--reweight --rw-neighbors 3` 为论文 Eq.7 邻域密度重加权（实测在 top 0.1% 聚合口径下退化，详见任务 022）；`--finetuned-weights-dir` 加载逐类自监督微调权重（实测退化，详见任务 023）。

自监督微调训练入口（仅 train split 正常图）：

```powershell
python -B scripts/finetune_patchcore.py --categories screw,carpet --epochs 100 ^
    --pairs 32 --image-size 512 --output-dir runs/experiments/patchcore_ft_pilot
```

## 指标口径

图像级：正确数/总数（准确率）、漏检数/异常数（漏检率）、误报数/正常数（误报率）、AUROC、AP、TPR@5%FPR；PatchCore 主线以 AUROC 与 TPR@5%FPR 为主口径，准确率仅作参考并同时报告"全判异常"平凡基线。

## 测试

```powershell
python -B -m pytest tests -q
```

缓存已通过 `pyproject.toml` 全局禁用，不会在工程内产生缓存目录。

## 迁移说明

复制整个工程到其他目录或电脑时带上 `configs` 与 `data`（MVTec AD 约 5GB）；在新机器按 `requirements*.txt` 重建环境即可运行，产物默认落在新位置的 `runs\`。
