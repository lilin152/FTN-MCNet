"""Four-class and grouped-binary patient-level evaluation metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score


def compute_metrics_4class(truth, preds, probs):
    """
    四分类原生指标。

    Args:
        truth: [N] int, 标签 0/1/2/3
        preds: [N] int, 预测类别
        probs: [N, 4] float, 四类概率矩阵

    Returns:
        dict: {
            'macro_auc': float,
            'acc': float,
            'per_class': [{auc, sen, spe}, ...] (4 个 dict),
            'cm': 4x4 ndarray
        }
    """
    n_classes = 4
    present_classes = sorted(set(truth))

    # ---- 混淆矩阵 ----
    cm = confusion_matrix(truth, preds, labels=list(range(n_classes)))

    # ---- ACC ----
    acc = accuracy_score(truth, preds)

    # ---- 逐类 OvR AUC（缺类守卫）----
    per_class = []
    for c in range(n_classes):
        if c not in present_classes:
            per_class.append({'auc': float('nan'), 'sen': float('nan'), 'spe': float('nan')})
            continue
        # OvR AUC
        y_true_c = (np.array(truth) == c).astype(int)
        y_score_c = probs[:, c]
        if len(np.unique(y_true_c)) < 2:
            per_class.append({'auc': float('nan'), 'sen': float('nan'), 'spe': float('nan')})
            continue
        auc_c = roc_auc_score(y_true_c, y_score_c)
        # SEN / SPE from confusion matrix
        sen, spe = _sen_spe_from_cm(cm, c)
        per_class.append({'auc': auc_c, 'sen': sen, 'spe': spe})

    # ---- macro AUC（仅对出现的类平均）----
    valid_aucs = [p['auc'] for p in per_class if not np.isnan(p['auc'])]
    macro_auc = np.mean(valid_aucs) if valid_aucs else float('nan')

    return {
        'macro_auc': macro_auc,
        'acc': acc,
        'per_class': per_class,
        'cm': cm,
    }


def compute_metrics_binary_folded(truth_4class, probs_4class):
    """Patient-level FTC vs FTA: probability score is sum of FTC subtype
    probabilities; predicted class is the folded FOUR-CLASS argmax. No optimized
    binary threshold is used. The AUC and class metrics use different aspects
    of this same four-class output; these are not the auxiliary-head metrics.
    """
    truth = np.asarray(truth_4class, dtype=int)
    probs = np.asarray(probs_4class, dtype=float)
    true_binary = (truth != 0).astype(int)
    score = probs[:, 1:].sum(axis=1)
    pred_binary = (probs.argmax(axis=1) != 0).astype(int)
    auc_value = float(roc_auc_score(true_binary, score)) if len(np.unique(true_binary)) == 2 else float('nan')
    tn, fp, fn, tp = confusion_matrix(true_binary, pred_binary, labels=[0, 1]).ravel()
    n = tn+fp+fn+tp
    acc = (tn+tp)/n if n else float('nan')
    sen = tp/(tp+fn) if tp+fn else float('nan')
    spe = tn/(tn+fp) if tn+fp else float('nan')
    return {'auc': auc_value, 'acc': acc, 'sen': sen, 'spe': spe}



def _sen_spe_from_cm(cm, c):
    """从混淆矩阵计算类别 c 的 OvR SEN/SPE"""
    tp = cm[c, c]
    fn = cm[c, :].sum() - tp
    fp = cm[:, c].sum() - tp
    tn = cm.sum() - tp - fn - fp
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sen, spe
