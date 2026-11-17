"""按混淆计数计算图像级指标，明确每种比率的分母。"""


def classification_rates(cases):
    """合并评价须先相加TP/TN/FP/FN，不能平均类别百分比。"""
    tp = cases['true_positive']
    tn = cases['true_negative']
    fp = cases['false_positive']
    fn = cases['false_negative']
    total = tp + tn + fp + fn
    return dict(accuracy=(tp + tn) / total if total else None,
                miss_rate=fn / (tp + fn) if tp + fn else None,
                false_positive_rate=fp / (fp + tn) if fp + tn else None,
                test_count=total, anomaly_count=tp + fn, normal_count=fp + tn)
