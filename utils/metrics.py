"""Patient-level four-class and folded binary diagnostic metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def _safe_divide(numerator, denominator):
    """Return NaN for undefined ratios rather than silently assigning zero."""
    return float(numerator / denominator) if denominator else float('nan')


def _strict_macro(values):
    """Return the arithmetic mean only when every class-specific value is defined."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if len(values) and np.isfinite(values).all() else float('nan')


def compute_metrics_four_class(truth, preds, probs):
    """Calculate overall and one-vs-rest four-class metrics.

    Args:
        truth: ``[N]`` integer network labels in canonical order
            FTA, WI-FTC, EA-FTC, MI-FTC (0, 1, 2, 3).
        preds: ``[N]`` predicted network labels.
        probs: ``[N, 4]`` probabilities in canonical network-label order.

    Returns overall accuracy, macro metrics, per-class AUC/ACC/SEN/SPE/PPV/NPV/F1,
    and the 4x4 confusion matrix. Undefined ratios are NaN. Macro metrics are
    undefined if any of the four class-specific values is undefined.
    """
    truth = np.asarray(truth, dtype=int)
    preds = np.asarray(preds, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if truth.ndim != 1 or preds.ndim != 1 or len(truth) != len(preds):
        raise ValueError('truth and preds must be one-dimensional arrays of equal length')
    if probs.shape != (len(truth), 4):
        raise ValueError(f'probs must have shape ({len(truth)}, 4)')
    if len(truth) == 0:
        raise ValueError('Cannot compute metrics for an empty cohort')
    if not np.isin(truth, range(4)).all() or not np.isin(preds, range(4)).all():
        raise ValueError('truth and preds must contain only labels 0, 1, 2, 3')
    if not np.isfinite(probs).all():
        raise ValueError('probs must contain only finite values')

    cm = confusion_matrix(truth, preds, labels=[0, 1, 2, 3]).astype(np.int64)
    total = int(cm.sum())
    overall_acc = _safe_divide(int(np.trace(cm)), total)
    per_class = []
    for c in range(4):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(total - tp - fn - fp)
        y_true_c = (truth == c).astype(int)
        auc_c = (float(roc_auc_score(y_true_c, probs[:, c]))
                 if len(np.unique(y_true_c)) == 2 else float('nan'))
        per_class.append({
            'auc': auc_c,
            'acc': _safe_divide(tp + tn, total),
            'sen': _safe_divide(tp, tp + fn),
            'spe': _safe_divide(tn, tn + fp),
            'ppv': _safe_divide(tp, tp + fp),
            'npv': _safe_divide(tn, tn + fn),
            'f1': _safe_divide(2 * tp, 2 * tp + fp + fn),
        })

    macro = {key: _strict_macro([row[key] for row in per_class])
             for key in ('auc', 'sen', 'spe', 'ppv', 'npv', 'f1')}
    return {
        'macro_auc': macro['auc'],
        'macro_sen': macro['sen'],
        'macro_spe': macro['spe'],
        'macro_ppv': macro['ppv'],
        'macro_npv': macro['npv'],
        'macro_f1': macro['f1'],
        'acc': overall_acc,
        'per_class': per_class,
        'cm': cm,
    }


def compute_metrics_binary_folded(truth_four_class, probs_four_class):
    """Evaluate FTA vs FTC using the primary four-class branch only.

    Binary AUC uses the sum of the three FTC subtype probabilities. Binary
    ACC/SEN/SPE fold the final four-class argmax prediction. No optimized
    threshold is searched, and these are not auxiliary binary-head metrics.
    """
    truth = np.asarray(truth_four_class, dtype=int)
    probs = np.asarray(probs_four_class, dtype=float)
    if truth.ndim != 1 or probs.shape != (len(truth), 4):
        raise ValueError('truth must be [N] and probs must be [N, 4]')
    if len(truth) == 0:
        raise ValueError('Cannot compute binary metrics for an empty cohort')
    true_binary = (truth != 0).astype(int)
    score = probs[:, 1:].sum(axis=1)
    pred_binary = (probs.argmax(axis=1) != 0).astype(int)
    auc_value = (float(roc_auc_score(true_binary, score))
                 if len(np.unique(true_binary)) == 2 else float('nan'))
    tn, fp, fn, tp = confusion_matrix(true_binary, pred_binary, labels=[0, 1]).ravel()
    n = int(tn + fp + fn + tp)
    return {
        'auc': auc_value,
        'acc': _safe_divide(int(tn + tp), n),
        'sen': _safe_divide(int(tp), int(tp + fn)),
        'spe': _safe_divide(int(tn), int(tn + fp)),
    }
