"""Image-level training dataset with patient-level report tokens."""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from .transforms import square_generated

EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}

def image_paths(folder):
    p = Path(folder)
    if not p.is_dir():
        raise FileNotFoundError(f'Image directory is missing: {p}')
    return sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in EXTS)


class MultimodalDataset(Dataset):
    def __init__(self, patients, text_map, transform):
        self.samples = []
        self.labels = []
        self.transform = transform
        for patient in patients:
            tok = text_map[patient.patient_id]
            for file in image_paths(patient.folder):
                self.samples.append((file, tok, patient.label))
                self.labels.append(patient.label)
        if not self.samples:
            raise ValueError('No image samples found in the requested training split.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file, token, label = self.samples[index]
        with Image.open(file) as img:
            img = square_generated(img.convert('L'))
            img = self.transform(img)
        return img, token, label
