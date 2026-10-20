"""Full patient-level four-class and folded-binary validation summaries."""
import numpy as np
from utils.metrics import compute_metrics_four_class, compute_metrics_binary_folded
from utils.labels import CLASS_NAMES as FOUR_CLASS_KEYS

CLASS_METRICS = ('ACC', 'SEN', 'SPE', 'AUC', 'PPV', 'NPV', 'F1')
VALIDATION_COLUMNS = (
    ['epoch', 'n_valid', 'ACC_four_class', 'macro_AUC', 'macro_SEN', 'macro_SPE',
     'macro_PPV', 'macro_NPV', 'macro_F1', 'binary_AUC', 'binary_ACC',
     'binary_SEN', 'binary_SPE']
    + [f'{metric}_{name}' for name in FOUR_CLASS_KEYS for metric in CLASS_METRICS]
)


def calculate_validation_row(epoch, truth, probs):
    truth = np.asarray(truth)
    probs = np.asarray(probs)
    four_class = compute_metrics_four_class(truth, probs.argmax(axis=1), probs)
    binary = compute_metrics_binary_folded(truth, probs)
    row = {
        'epoch': int(epoch),
        'n_valid': int(len(truth)),
        'ACC_four_class': float(four_class['acc']),
        'macro_AUC': float(four_class['macro_auc']),
        'macro_SEN': float(four_class['macro_sen']),
        'macro_SPE': float(four_class['macro_spe']),
        'macro_PPV': float(four_class['macro_ppv']),
        'macro_NPV': float(four_class['macro_npv']),
        'macro_F1': float(four_class['macro_f1']),
        'binary_AUC': float(binary['auc']),
        'binary_ACC': float(binary['acc']),
        'binary_SEN': float(binary['sen']),
        'binary_SPE': float(binary['spe']),
    }
    for i, name in enumerate(FOUR_CLASS_KEYS):
        for field in CLASS_METRICS:
            row[f'{field}_{name}'] = float(four_class['per_class'][i][field.lower()])
    return row
