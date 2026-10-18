"""Full patient-level four-class and binary validation summaries."""
import numpy as np
from utils.metrics import compute_metrics_4class, compute_metrics_binary_folded

from utils.labels import CLASS_NAMES as CLASS_KEYS
VALIDATION_COLUMNS = (
    ['epoch','n_valid','ACC4','macro_AUC','macro_SEN','macro_SPE',
     'bin_AUC','bin_ACC','bin_SEN','bin_SPE'] +
    [f'{metric}_{name}' for name in CLASS_KEYS for metric in ('ACC','SEN','SPE','AUC')]
)


def calculate_validation_row(epoch, truth, probs):
    truth=np.asarray(truth); probs=np.asarray(probs)
    m=compute_metrics_4class(truth,probs.argmax(axis=1),probs)
    b=compute_metrics_binary_folded(truth,probs)
    scores=m['per_class']
    row={'epoch':int(epoch),'n_valid':int(len(truth)),
         'ACC4':float(m['acc']),'macro_AUC':float(m['macro_auc']),
         'macro_SEN':float(np.nanmean([x['sen'] for x in scores])),
         'macro_SPE':float(np.nanmean([x['spe'] for x in scores])),
         'bin_AUC':float(b['auc']),'bin_ACC':float(b['acc']),
         'bin_SEN':float(b['sen']),'bin_SPE':float(b['spe'])}
    cm=m['cm']; total=cm.sum()
    for i,name in enumerate(CLASS_KEYS):
        tp=cm[i,i]; fn=cm[i,:].sum()-tp; fp=cm[:,i].sum()-tp
        tn=total-tp-fn-fp
        row[f'ACC_{name}']=float((tp+tn)/total) if total else float('nan')
        for field in ('SEN','SPE','AUC'):
            row[f'{field}_{name}']=float(scores[i][field.lower()])
    return row
