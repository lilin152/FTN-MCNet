"""Patient-disjoint multimodal training with per-mini-batch Focal weights."""
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.fusion_head import build_model
from datasets.manifest import load_available_splits, check_disjoint_splits
from datasets.reports import load_reports, build_text_map
from datasets.multimodal_dataset import MultimodalDataset
from datasets.transforms import get_transform
from engine.evaluator import patient_predict, evaluate_and_save
from losses.consistency_loss import ConsistencyLoss
from utils.class_weights import compute_training_class_weights
from utils.validation import VALIDATION_COLUMNS,calculate_validation_row
from utils.training_plots import plot_training_curves

TRAIN_COLUMNS=['epoch','loss','root','leaf','kl','val_macro_auc']


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_csv(path, headers, rows):
    with Path(path).open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=headers)
        w.writeheader();w.writerows(rows)


def _persist(output,cfg,weight_info,history,validation_history,best,best_epoch,messages):
    write_csv(output/'train_log.csv',TRAIN_COLUMNS,history)
    write_csv(output/'valid_metrics.csv',VALIDATION_COLUMNS,validation_history)
    log_text='\n'.join(messages)+'\n'
    (output/'train_progress.log').write_text(log_text,encoding='utf-8')
    (output/'train_log.txt').write_text(log_text,encoding='utf-8')
    (output/'run_config.json').write_text(json.dumps({**cfg,'class_weighting': 'batch_inverse_frequency', 'computed_class_weights':weight_info,
             'best_validation_macro_auc':float(best) if np.isfinite(best) else None,
             'best_epoch':best_epoch},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    plot_training_curves(history,output)


def run(cfg, dry_run=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(int(cfg.get('seed',0)))
    splits=load_available_splits(Path(cfg['data_dir']),image_root=cfg.get('image_root'))
    if 'train' not in splits or 'valid' not in splits:
        raise ValueError('Both train.npy and valid.npy are required.')
    check_disjoint_splits(splits)  # Never train with overlapping patient IDs/folders.
    if cfg.get('class_weighting', 'batch_inverse_frequency') != 'batch_inverse_frequency':
        raise ValueError('This release requires batch_inverse_frequency weights.')
    # Distribution summary for audit only; not used as training weights.
    training_counts=compute_training_class_weights(splits['train'])
    weights={k: training_counts[k] for k in ('class_names','patient_counts_4class','image_counts_4class','patient_counts_binary')}
    weights.update(method='batch_inverse_frequency_laplace', source_split='train', laplace_smoothing=float(cfg.get('laplace_smoothing', 1.0)))
    output=Path(cfg['out_dir'])/('dry_run' if dry_run else 'main')
    output.mkdir(parents=True,exist_ok=True)
    messages=[]
    def log(s):
        print(s,flush=True)
        messages.append(s)
    log('Training class weights: '+json.dumps(weights,ensure_ascii=False))
    reports=load_reports(cfg['report_csv'])
    model=build_model(str(device))
    text_map,report_stats=build_text_map(splits,reports,model.backbone)
    log('Report matching: '+json.dumps(report_stats))
    dataset=MultimodalDataset(splits['train'],text_map,get_transform(train=True))
    loader=DataLoader(dataset,batch_size=int(cfg.get('batch_size',8)),shuffle=True,
         num_workers=int(cfg.get('num_workers',0)),pin_memory=(device.type=='cuda'),drop_last=False)
    criterion=ConsistencyLoss(lambda_consistency=cfg.get('lambda_consistency',0.5),
        gamma=cfg.get('focal_gamma',2),smoothing=cfg.get('laplace_smoothing',1.0)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.get('lr',1e-5),weight_decay=cfg.get('weight_decay',0.01))
    epochs=1 if dry_run else int(cfg.get('epochs',100))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    val_every=1 if dry_run else int(cfg.get('val_every',5))
    if val_every<=0 or epochs<=0: raise ValueError('Epochs and val_every must be positive.')
    patience=int(cfg.get('patience',20))
    if patience<=0:raise ValueError('patience must be positive.')
    best,stale,best_epoch=float('-inf'),0,None
    history=[];validation_history=[]
    for epoch in range(1,epochs+1):
        model.train()
        summed=np.zeros(4,dtype=np.float64);n=0
        for images,tokens,target in loader:
            images=images.to(device);target=target.to(device);tokens=tokens.to(device)
            root,leaf=model(images,tokens)
            loss,rl,ll,kl=criterion(root,leaf,target)
            optimizer.zero_grad();loss.backward();optimizer.step()
            summed+=np.asarray([float(loss.detach()),float(rl),float(ll),float(kl)])
            n+=1
        scheduler.step()
        if not n:raise ValueError('Empty training loader.')
        mean=summed/n
        row=dict(zip(TRAIN_COLUMNS[:5],[epoch,*map(float,mean)]))
        row['val_macro_auc']=''
        log(f"Epoch {epoch}/{epochs} | total={row['loss']:.6f} root={row['root']:.6f} leaf={row['leaf']:.6f} kl={row['kl']:.6f}")
        stop=False
        if epoch%val_every==0:
            truth,probs,_,_=patient_predict(model,splits['valid'],text_map,get_transform(False),device)
            valrow=calculate_validation_row(epoch,truth,probs)
            validation_history.append(valrow)
            score=valrow['macro_AUC'];row['val_macro_auc']=score
            log(f'Validation epoch {epoch}: macro AUC={score:.6f}; best={best:.6f}')
            if np.isfinite(score) and score>best:
                best,stale,best_epoch=score,0,epoch
                if not dry_run:torch.save(model.state_dict(),output/'best.pth')
            else:
                stale+=1
                if stale>=max(1,patience//val_every) and not dry_run:
                    log(f'Early stopping at epoch {epoch}')
                    stop=True
        history.append(row)
        # Persist complete history even if training is interrupted after an epoch.
        _persist(output,cfg,weights,history,validation_history,best,best_epoch,messages)
        if stop:break
    if dry_run:
        log('Integration run completed. No trained checkpoint was saved.')
    elif not (output/'best.pth').exists():
        log('No checkpoint saved: validation AUC may be undefined.')
    else:
        model.load_state_dict(torch.load(output/'best.pth',map_location=device,weights_only=True),strict=True)
        evaluate_and_save(model,splits['valid'],text_map,get_transform(False),device,output,'valid')
    _persist(output,cfg,weights,history,validation_history,best,best_epoch,messages)
    return best
