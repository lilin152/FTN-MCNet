"""Map de-identified patient IDs to report-text tokens."""
import csv
from pathlib import Path


def load_reports(csv_path):
    file_path = Path(csv_path)
    if not file_path.exists():
        raise FileNotFoundError(f'Report CSV not found: {file_path}')
    reports = {}
    with file_path.open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        if not {'patient_id', 'report_text'}.issubset(set(reader.fieldnames or [])):
            raise ValueError('Report CSV must have patient_id and report_text columns.')
        for row in reader:
            pid = row['patient_id'].strip()
            if not pid or pid in reports:
                raise ValueError(f'Empty/duplicated patient_id in report CSV: {pid!r}')
            reports[pid] = row['report_text'].strip()
    return reports


def build_text_map(splits, reports, backbone):
    """Use the actual BiomedCLIP tokenizer (online text encoder, LoRA unfrozen)."""
    unique = {p.patient_id for cohort in splits.values() for p in cohort}
    # Reports are required inputs; no synthetic missing-report content is created.
    text_by_patient = {pid: reports[pid] for pid in unique}
    texts = sorted(set(text_by_patient.values()))
    tokenizer = backbone._open_clip.get_tokenizer(f'hf-hub:{backbone.model_id}')
    tokens = {text: tokenizer([text]).squeeze(0).cpu() for text in texts}
    mapped = {pid: tokens[text_by_patient[pid]] for pid in unique}
    stats = {'total': len(unique), 'matched': len(unique)}
    return mapped, stats
