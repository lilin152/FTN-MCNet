"""Print and save consistent patient-level diagnostic metric tables."""
import os
import numpy as np

from utils.labels import CLASS_NAMES as FOUR_CLASS_NAMES, PAPER_CLASS_ORDER

METRIC_FIELDS = (
    ('auc', 'AUC'), ('acc', 'ACC'), ('sen', 'SEN'), ('spe', 'SPE'),
    ('ppv', 'PPV'), ('npv', 'NPV'), ('f1', 'F1'),
)


def _fmt(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 'N/A'
    return f'{value:.4f}' if np.isfinite(value) else 'N/A'


def _four_class_lines(metrics_four_class, class_names=None):
    if class_names is None:
        class_names = FOUR_CLASS_NAMES
    lines = [
        'Four-Class Diagnostic Metrics',
        f"Overall four-class ACC: {_fmt(metrics_four_class.get('acc'))}",
        '',
        f"{'Class':<16}" + ''.join(f'{label:>9}' for _, label in METRIC_FIELDS),
        '-' * (16 + 9 * len(METRIC_FIELDS)),
    ]
    per_class = metrics_four_class.get('per_class', [])
    for c in PAPER_CLASS_ORDER:
        name = class_names[c] if c < len(class_names) else f'Class {c}'
        row = per_class[c] if c < len(per_class) else {}
        lines.append(f'{name:<16}' + ''.join(f'{_fmt(row.get(key)):>9}' for key, _ in METRIC_FIELDS))
    lines.append('-' * (16 + 9 * len(METRIC_FIELDS)))
    macro_values = {
        'auc': metrics_four_class.get('macro_auc'),
        'acc': None,  # Macro-ACC is not a substitute for overall four-class accuracy.
        'sen': metrics_four_class.get('macro_sen'),
        'spe': metrics_four_class.get('macro_spe'),
        'ppv': metrics_four_class.get('macro_ppv'),
        'npv': metrics_four_class.get('macro_npv'),
        'f1': metrics_four_class.get('macro_f1'),
    }
    lines.append(f"{'Macro average':<16}" + ''.join(f'{_fmt(macro_values[key]):>9}' for key, _ in METRIC_FIELDS))
    return lines


def print_test_table(metrics_four_class, metrics_binary, class_names=None):
    print('\n' + '=' * 90)
    print('\n'.join(_four_class_lines(metrics_four_class, class_names)))
    print('\n' + '=' * 70)
    print('Folded Binary Metrics (FTA vs FTC; primary four-class head)')
    print(f"AUC: {_fmt(metrics_binary.get('auc'))} | ACC: {_fmt(metrics_binary.get('acc'))} | "
          f"SEN: {_fmt(metrics_binary.get('sen'))} | SPE: {_fmt(metrics_binary.get('spe'))}")
    print('=' * 90)


def save_test_txt(metrics_four_class, metrics_binary, save_path, class_names=None):
    """Save the same metric definitions displayed in the terminal."""
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    lines = _four_class_lines(metrics_four_class, class_names)
    lines.extend([
        '',
        'Folded Binary Metrics (FTA vs FTC; primary four-class head)',
        f"AUC: {_fmt(metrics_binary.get('auc'))} | ACC: {_fmt(metrics_binary.get('acc'))} | "
        f"SEN: {_fmt(metrics_binary.get('sen'))} | SPE: {_fmt(metrics_binary.get('spe'))}",
    ])
    with open(save_path, 'w', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
