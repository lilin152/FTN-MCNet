"""Patient-level .npy index reader and deterministic split-overlap checks.

Input arrays contain rows [folder, class_label, patient_id] (preferred), or
[folder, class_label] rows (ID inferred from folder name).
``allow_pickle=True`` is used for object npy; load only trusted local files.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class Patient:
    folder: Path
    label: int
    patient_id: str


def load_split(file_path, image_root=None):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing split: {file_path}. Run examples/create_demo_data.py or supply your own data.")
    raw = np.load(file_path, allow_pickle=True)  # trusted local indices only
    if raw.ndim != 2 or raw.shape[1] not in (2, 3):
        raise ValueError(f"Expected [N,2] or [N,3] .npy: {file_path}")
    people = []
    root = Path(image_root) if image_root else file_path.parent
    for row in raw:
        value = Path(str(row[0]))
        folder = value if value.is_absolute() else (root / value)
        folder = folder.resolve()
        label = int(row[1])
        if label not in range(4):
            raise ValueError(f"Label must be 0..3: {label} in {file_path}")
        patient_id = str(row[2]) if len(row) == 3 else folder.name
        if not patient_id.strip():
            raise ValueError('Empty patient_id')
        people.append(Patient(folder, label, patient_id))
    return people


def check_disjoint_splits(splits):
    """Fail on overlap across train/valid/external; also catch contradictory labels."""
    owners = {}
    for split_name, cohort in splits.items():
        for p in cohort:
            for identity in (f'id:{p.patient_id}', f'folder:{p.folder}'):
                prev = owners.get(identity)
                if prev is not None and prev != split_name:
                    raise ValueError(f"Patient overlap between '{prev}' and '{split_name}': {identity}. "
                                     "Do not report this as independent validation.")
                owners[identity] = split_name
        grouped = {}
        for p in cohort:
            if p.patient_id in grouped:
                raise ValueError(f'Duplicate patient_id within {split_name}: {p.patient_id}')
            grouped[p.patient_id] = p.label
    return True


def load_available_splits(data_dir, names=("train", "valid", "test_external1", "test_external2"), image_root=None):
    data_dir = Path(data_dir)
    return {name: load_split(data_dir / f'{name}.npy', image_root)
            for name in names if (data_dir / f'{name}.npy').is_file()}
