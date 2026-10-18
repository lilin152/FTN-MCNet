"""ROC and confusion-matrix visualizations for model evaluation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


# 四分类类名
from utils.labels import CLASS_NAMES as CLASS_NAMES_4, PAPER_CLASS_ORDER

# 颜色方案（四类 + macro avg）
COLORS_4 = ['#2ca02c', '#d62728', '#ff7f0e', '#9467bd']
MACRO_COLOR = 'black'
MACRO_LINESTYLE = '--'

# 通用绘图参数
FIGSIZE = (8, 6)
DPI = 300


def plot_roc_ovr(truth, probs_4class, class_names=None, save_path='roc_ovr.png'):
    """
    四分类 OvR ROC 曲线，同一张图。
    """
    if class_names is None:
        class_names = CLASS_NAMES_4

    truth = np.asarray(truth)
    probs_4class = np.asarray(probs_4class)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # 逐类 ROC
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for c in PAPER_CLASS_ORDER:
        y_true_c = (truth == c).astype(int)
        y_score_c = probs_4class[:, c]
        if len(np.unique(y_true_c)) < 2:
            auc_dict[c] = float('nan')
            continue
        fpr, tpr, _ = roc_curve(y_true_c, y_score_c)
        auc_val = auc(fpr, tpr)
        fpr_dict[c], tpr_dict[c], auc_dict[c] = fpr, tpr, auc_val
        ax.plot(fpr, tpr, lw=2, color=COLORS_4[c],
                label=f'{class_names[c]} (AUC={auc_val:.3f})')

    # macro avg（插值平均）
    if not fpr_dict:
        ax.text(0.5, 0.5, 'AUC undefined: no valid one-vs-rest comparison', ha='center')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.savefig(save_path); plt.close(fig)
        return
    all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in PAPER_CLASS_ORDER if c in fpr_dict]))
    mean_tpr = np.zeros_like(all_fpr)
    valid_count = 0
    for c in PAPER_CLASS_ORDER:
        if c in tpr_dict:
            mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
            valid_count += 1
    if valid_count > 0:
        mean_tpr /= valid_count
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, lw=2, color=MACRO_COLOR, linestyle=MACRO_LINESTYLE,
                label=f'Macro Avg (AUC={macro_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('1 - Specificity', fontsize=13)
    ax.set_ylabel('Sensitivity', fontsize=13)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_roc_binary(truth, prob, save_path='roc_binary.png'):
    """二分类 ROC 曲线"""
    truth = np.asarray(truth)
    prob = np.asarray(prob)

    if len(np.unique(truth)) < 2:
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        ax.text(0.5, 0.5, 'Only one class present', ha='center', va='center', fontsize=14)
        fig.savefig(save_path)
        plt.close(fig)
        return

    fpr, tpr, _ = roc_curve(truth, prob, pos_label=1)
    auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc_val:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('1 - Specificity', fontsize=13)
    ax.set_ylabel('Sensitivity', fontsize=13)
    ax.legend(fontsize=12, loc='lower right')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names=None, save_path='confusion_matrix.png', normalize=False):
    """4x4 混淆矩阵热力图"""
    if class_names is None:
        class_names = CLASS_NAMES_4

    cm = np.asarray(cm, dtype=float)[np.ix_(PAPER_CLASS_ORDER, PAPER_CLASS_ORDER)]
    class_names = [class_names[i] for i in PAPER_CLASS_ORDER]
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm / row_sums
        fmt = '.2f'
        title = 'Confusion Matrix (Normalized by Row)'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix (Counts)'

    fig, ax = plt.subplots(figsize=(7, 6), dpi=DPI)
    im = ax.imshow(cm_display, cmap='Blues', vmin=0, vmax=1 if normalize else cm_display.max())

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if normalize:
                text = f'{cm_display[i, j]:.2f}'
            else:
                text = f'{int(cm_display[i, j])}'
            ax.text(j, i, text, ha='center', va='center',
                    color='white' if cm_display[i, j] > (0.7 if normalize else cm_display.max() * 0.5) else 'black',
                    fontsize=11)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


