"""Evaluate patient-level probabilities and export class metrics and figures."""
import csv
import numpy as np
import torch
from PIL import Image
from datasets.multimodal_dataset import image_paths
from datasets.transforms import square_generated
from utils.metrics import compute_metrics_four_class, compute_metrics_binary_folded
from utils.plots import plot_roc_ovr, plot_roc_binary, plot_confusion_matrix
from utils.report import save_test_txt, print_test_table

from utils.labels import CLASS_NAMES, PAPER_CLASS_ORDER, PROBABILITY_COLUMNS

@torch.no_grad()
def patient_predict(model,patients,text_map,transform,device):
    model.eval()
    truth,probs,ids,image_counts=[],[],[],[]
    for person in patients:
        files=image_paths(person.folder)
        if not files:raise ValueError(f'Patient {person.patient_id} has no images.')
        tensors=[]
        for path in files:
            with Image.open(path) as handle:
                tensors.append(transform(square_generated(handle.convert('L'))))
        images=torch.stack(tensors).to(device)
        tokens=text_map[person.patient_id].unsqueeze(0).repeat(len(tensors),1).to(device)
        p=model.predict_four_class(images,tokens).cpu().numpy()
        truth.append(person.label)
        probs.append(p.mean(axis=0))
        ids.append(person.patient_id)
        image_counts.append(len(files))
    if not probs:raise ValueError('No patients available for evaluation.')
    return np.asarray(truth),np.stack(probs),ids,image_counts


def evaluate_and_save(model,patients,text_map,transform,device,save_dir,split_name):
    from pathlib import Path
    output=Path(save_dir);output.mkdir(parents=True,exist_ok=True)
    truth,probs,ids,counts=patient_predict(model,patients,text_map,transform,device)
    four_class_metrics=compute_metrics_four_class(truth,probs.argmax(axis=1),probs)
    binary_metrics=compute_metrics_binary_folded(truth,probs)
    print_test_table(four_class_metrics,binary_metrics,CLASS_NAMES)
    save_test_txt(four_class_metrics,binary_metrics,str(output/f'{split_name}_metrics.txt'),CLASS_NAMES)
    with (output/f'{split_name}_predictions.csv').open('w',newline='',encoding='utf-8') as fh:
        wr=csv.writer(fh)
        wr.writerow(['patient_id','label','true_class','predicted_class',*PROBABILITY_COLUMNS,'n_images'])
        for pid,label,p,n in zip(ids,truth,probs,counts):
            wr.writerow([pid,int(label),CLASS_NAMES[int(label)],CLASS_NAMES[int(np.argmax(p))],*[float(p[i]) for i in PAPER_CLASS_ORDER],n])
    # A split containing a single class has undefined OvR AUC; metrics still exported.
    if len(set(map(int,truth)))>=2:
        try:plot_roc_ovr(truth,probs,CLASS_NAMES,str(output/f'{split_name}_roc_four_class.png'))
        except ValueError as ex:print('Skipping four-class ROC:',ex)
    try:plot_roc_binary((truth!=0).astype(int),probs[:,1:].sum(axis=1),str(output/f'{split_name}_roc_binary.png'))
    except ValueError as ex:print('Skipping binary ROC:',ex)
    plot_confusion_matrix(four_class_metrics['cm'],CLASS_NAMES,str(output/f'{split_name}_cm.png'))
    plot_confusion_matrix(four_class_metrics['cm'],CLASS_NAMES,str(output/f'{split_name}_cm_normalized.png'),normalize=True)
    return four_class_metrics,binary_metrics
