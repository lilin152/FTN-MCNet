"""Train the BiomedCLIP multimodal classifier."""
import argparse
from utils.config import load_config

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default='configs/default.json')
    ap.add_argument('--data-dir')
    ap.add_argument('--report-csv')
    ap.add_argument('--out-dir',help='Override base output directory; training writes to <out-dir>/main')
    ap.add_argument('--epochs',type=int)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    cfg=load_config(args.config)
    for arg,key in [('data_dir','data_dir'),('report_csv','report_csv'),('out_dir','out_dir'),('epochs','epochs')]:
        val=getattr(args,arg)
        if val is not None: cfg[key]=val
    from engine.trainer import run
    run(cfg,dry_run=args.dry_run)

if __name__=='__main__': main()
