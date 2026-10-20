"""Compute Focal alpha strictly from the training cohort's four-class labels.

The image-level classifier is optimized on images, whereas binary alpha
uses patient-level labels folded from the four-class patient distribution.
"""
from collections import Counter
from utils.labels import CLASS_NAMES
import numpy as np
from datasets.multimodal_dataset import image_paths


def inverse_frequency(counts):
    """Return alpha[c] = (1/count[c]) / sum_k(1/count[k]); require all classes."""
    array = np.asarray(counts, dtype=np.float64)
    if array.ndim != 1 or len(array) not in (2, 4) or not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError(f'Each class must have a positive finite training count: {array.tolist()}')
    inv = 1. / array
    return (inv / inv.sum()).tolist()


def compute_training_class_weights(patients, image_counts=None):
    """Four-class patient/image counts -> binary/four-class inverse-frequency weights.

    image_counts is optional, for deterministic tests or for a dataset whose
    image files have already been scanned. Default scans training folders.
    """
    patients = list(patients)
    ids = [p.patient_id for p in patients]
    if len(ids) != len(set(ids)):
        raise ValueError('Each training patient_id must appear exactly once.')
    patient_counts = Counter(p.label for p in patients)
    if image_counts is None:
        image_counts = Counter()
        for p in patients:
            n_images = len(image_paths(p.folder))
            if n_images == 0:
                raise ValueError(f'No images for training patient {p.patient_id}')
            image_counts[p.label] += n_images
    else:
        image_counts = Counter(image_counts)
    patient_four_class = [int(patient_counts[c]) for c in range(4)]
    image_four_class = [int(image_counts[c]) for c in range(4)]
    binary_counts = [patient_four_class[0], sum(patient_four_class[1:])]
    return {
        'class_names': list(CLASS_NAMES),
        'patient_counts_four_class': patient_four_class,
        'image_counts_four_class': image_four_class,
        'patient_counts_binary': binary_counts,
        'alpha_binary': inverse_frequency(binary_counts),
        'alpha_four_class': inverse_frequency(image_four_class),
        'method': 'inverse_frequency_normalized',
        'binary_unit': 'patients',
        'four_class_unit': 'images',
        'source_split': 'train',
    }
