"""Training loss-component and validation metric plots."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history, output_dir):
    """Plot loss terms and patient-level validation AUC from log rows."""
    from pathlib import Path
    output_dir=Path(output_dir)
    if not history: return
    epochs=[int(x['epoch']) for x in history]
    fig, ax=plt.subplots(figsize=(9,6),dpi=180)
    for col in ('loss','binary','four_class','kl'):
        ax.plot(epochs,[float(x[col]) for x in history],label=col)
    ax.set(xlabel='Epoch',ylabel='Loss',title='Training losses')
    ax.legend(); fig.tight_layout(); fig.savefig(output_dir/'loss_curve.png');plt.close(fig)
    valid=[(int(x['epoch']),float(x['val_macro_auc'])) for x in history if x['val_macro_auc']!='']
    if valid:
        fig,ax=plt.subplots(figsize=(9,6),dpi=180)
        ax.plot([p[0] for p in valid],[p[1] for p in valid],marker='o')
        ax.set(xlabel='Epoch',ylabel='Macro AUC',title='Validation macro AUC',ylim=(0,1))
        fig.tight_layout();fig.savefig(output_dir/'validation_auc_curve.png');plt.close(fig)
