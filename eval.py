"""Evaluate a trained checkpoint on a patient-disjoint split."""
import argparse
from pathlib import Path
from utils.config import load_config

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default='configs/default.json')
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--split',required=True,choices=['valid','test_external1','test_external2'])
    ap.add_argument('--out-dir')
    args=ap.parse_args()
    import torch
    from models.fusion_head import build_model
    from datasets.manifest import load_available_splits, check_disjoint_splits
    from datasets.reports import load_reports, build_text_map
    from datasets.transforms import get_transform
    from engine.evaluator import evaluate_and_save
    cfg=load_config(args.config)
    data_dir=Path(cfg['data_dir'])
    splits=load_available_splits(data_dir,image_root=cfg.get('image_root'))
    check_disjoint_splits(splits)
    if args.split not in splits:
        raise ValueError(f'Missing {args.split}.npy in {data_dir}')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(str(device))
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=True)
    model.load_state_dict(ckpt,strict=True)
    text_map,_=build_text_map({args.split:splits[args.split]},load_reports(cfg['report_csv']),model.backbone)
    evaluate_and_save(model,splits[args.split],text_map,get_transform(False),device,
                      args.out_dir or cfg['out_dir'],args.split)

if __name__=='__main__': main()
