"""Pretty-print and save validation/test metric tables."""
import os
import numpy as np
import pandas as pd

from utils.labels import CLASS_NAMES as CLASS_NAMES_4, PAPER_CLASS_ORDER


def print_test_table(metrics_4class, metrics_binary, class_names=None):
    """终端打印格式化测试报告"""
    if class_names is None:
        class_names = CLASS_NAMES_4

    print("\n" + "=" * 70)
    print("Test Results — 4-Class Metrics")
    print("=" * 70)
    header = f"{'Class':<22} {'AUC':>8} {'ACC':>8} {'SEN':>8} {'SPE':>8}"
    print(header)
    print("-" * 70)

    per_class = metrics_4class.get('per_class', [])
    for c in PAPER_CLASS_ORDER:
        name = class_names[c] if c < len(class_names) else f'Class {c}'
        m = per_class[c] if c < len(per_class) else {}
        auc_s = f"{m.get('auc', 0):.4f}" if not (isinstance(m.get('auc'), float) and np.isnan(m.get('auc'))) else "N/A"
        sen_s = f"{m.get('sen', 0):.4f}"
        spe_s = f"{m.get('spe', 0):.4f}"
        print(f"{name:<22} {auc_s:>8} {metrics_4class['acc']:>8.4f} {sen_s:>8} {spe_s:>8}")

    print("-" * 70)
    auc_s = f"{metrics_4class['macro_auc']:.4f}" if not np.isnan(metrics_4class['macro_auc']) else "N/A"
    print(f"{'Macro Avg':<22} {auc_s:>8} {metrics_4class['acc']:>8.4f}")

    print("\n" + "=" * 70)
    print("Test Results — Folded Binary (FTA vs FTC)")
    print("=" * 70)
    mb = metrics_binary
    print(f"  AUC: {mb['auc']:.4f}  |  ACC: {mb['acc']:.4f}  |  SEN: {mb['sen']:.4f}  |  SPE: {mb['spe']:.4f}")
    print("=" * 70)


def save_test_txt(metrics_4class, metrics_binary, save_path, class_names=None):
    """保存 test_result.txt"""
    import numpy as np
    if class_names is None:
        class_names = CLASS_NAMES_4

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lines = []
    lines.append("Test Results — 4-Class Metrics")
    lines.append("=" * 60)
    header = f"{'Class':<22} {'AUC':>8} {'ACC':>8} {'SEN':>8} {'SPE':>8}"
    lines.append(header)
    lines.append("-" * 60)

    per_class = metrics_4class.get('per_class', [])
    for c in PAPER_CLASS_ORDER:
        name = class_names[c] if c < len(class_names) else f'Class {c}'
        m = per_class[c] if c < len(per_class) else {}
        auc_s = f"{m.get('auc', 0):.4f}" if not (isinstance(m.get('auc'), float) and np.isnan(m.get('auc'))) else "N/A"
        lines.append(f"{name:<22} {auc_s:>8} {metrics_4class['acc']:>8.4f} {m.get('sen', 0):>8.4f} {m.get('spe', 0):>8.4f}")

    lines.append("-" * 60)
    auc_s = f"{metrics_4class['macro_auc']:.4f}" if not np.isnan(metrics_4class['macro_auc']) else "N/A"
    lines.append(f"{'Macro Avg':<22} {auc_s:>8} {metrics_4class['acc']:>8.4f}")

    lines.append("")
    lines.append("Test Results — Folded Binary (FTA vs FTC)")
    lines.append("=" * 60)
    mb = metrics_binary
    lines.append(f"AUC: {mb['auc']:.4f}  |  ACC: {mb['acc']:.4f}  |  SEN: {mb['sen']:.4f}  |  SPE: {mb['spe']:.4f}")

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


