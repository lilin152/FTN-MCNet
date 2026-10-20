# FTN-MCNet

FTN-MCNet is a BiomedCLIP-based vision-language model for preoperative classification of follicular thyroid neoplasms into follicular thyroid adenoma (FTA), minimally invasive follicular thyroid carcinoma (MI-FTC), encapsulated angioinvasive FTC (EA-FTC), and widely invasive FTC (WI-FTC).

## File Descriptions

| File | Function |
| --- | --- |
| `train.py` | Model training |
| `eval.py` | Patient-level evaluation |
| `predict.py` | Single-patient inference |
| `models/` | BiomedCLIP encoders, LoRA, and classification heads |
| `losses/` | Focal objectives and KL consistency constraint |
| `datasets/` | Image and report loading |
| `engine/` | Training and evaluation procedures |
| `utils/` | Metrics and supporting utilities |
| `statistical_analysis/bootstrap_predictions.py` | Patient-level Bootstrap analysis |

## How to Start

1. **Environment.** Install a compatible PyTorch/torchvision environment, then run `pip install -r requirements.txt`. BiomedCLIP weights may need to be downloaded or cached locally.
2. **Data preparation.** Prepare patient-level ultrasound images, paired report descriptions, and cohort index files according to the study protocol. Set the paths in `configs/default.json`.
3. **Model training.** Run:

   ```bash
   python train.py --config configs/default.json
   ```

4. **Model evaluation.** Run `eval.py` for the internal validation or external cohorts, for example:

   ```bash
   python eval.py --config configs/default.json --checkpoint results/main/best.pth --split valid
   ```

   Repeat with `--split test_external1` and `--split test_external2` to populate the external-cohort CSVs.

   By default, `eval.py` now writes to `results/main/`, the same run directory used by `train.py`. The prediction CSVs for the run therefore stay together:

   ```text
   results/main/
   ├── best.pth
   ├── valid_predictions.csv
   ├── test_external1_predictions.csv
   └── test_external2_predictions.csv
   ```

   If you override `train.py --out-dir`, pass `--out-dir <base>/main` to `eval.py` so both use the same run directory.

## Method Overview

Image and report text are encoded separately by LoRA-adapted BiomedCLIP encoders. Their normalized representations are concatenated and passed to an auxiliary binary head and a primary four-class head. Training combines binary and four-class Focal losses with a KL probability-consistency constraint. For patients with multiple ultrasound images, four-class probabilities are averaged before assigning the patient-level prediction.
