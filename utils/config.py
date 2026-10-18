"""Load JSON configs relative to project root, with CLI overrides."""
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config(path):
    cfg_file = Path(path).expanduser().resolve()
    cfg = json.loads(cfg_file.read_text(encoding='utf8'))
    for key in ('data_dir','report_csv','out_dir'):
        if key in cfg:
            p = Path(cfg[key]).expanduser()
            cfg[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg
