"""Prediction without labels using a de-identified patient folder and a report text."""
import argparse
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--patient-folder',required=True)
    ap.add_argument('--patient-id',default='example_patient')
    ap.add_argument('--report-text',required=True,help='De-identified clinical report; avoid shell history for sensitive data')
    args=ap.parse_args()
    import torch
    from models.fusion_head import build_model
    from datasets.manifest import Patient
    from datasets.reports import build_text_map
    from datasets.transforms import get_transform
    from engine.evaluator import patient_predict
    from utils.labels import CLASS_NAMES, PAPER_CLASS_ORDER
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(str(device))
    checkpoint = torch.load(args.checkpoint,map_location=device,weights_only=True)
    model.load_state_dict(checkpoint,strict=True)
    patient=Patient(Path(args.patient_folder),-1,args.patient_id)
    txt,_=build_text_map({'inference':[patient]},{args.patient_id:args.report_text},model.backbone)
    _, probs, _, _=patient_predict(model,[patient],txt,get_transform(False),device)
    for i in PAPER_CLASS_ORDER:
        print(f'{CLASS_NAMES[i]}: {float(probs[0, i]):.6f}')

if __name__=='__main__': main()
