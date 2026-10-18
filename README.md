# FTN-MCNet

FTN-MCNet is a BiomedCLIP-based vision-language model for preoperative classification of follicular thyroid neoplasms into four pathological categories: follicular thyroid adenoma (FTA), minimally invasive follicular thyroid carcinoma (MI-FTC), encapsulated angioinvasive FTC (EA-FTC), and widely invasive FTC (WI-FTC).

## Method

FTN-MCNet uses a dual encoder initialized from BiomedCLIP. LoRA modules adapt the image and text encoders while the pretrained parameters remain frozen. The image and report representations are concatenated into a joint multimodal representation and passed through fully connected classification layers.

An auxiliary binary branch distinguishes FTA from FTC, while the primary branch predicts the four pathological categories. Training jointly optimizes the binary and four-class focal losses with batch-level class weighting and a fixed-weight KL probability-consistency constraint between the two branches. Final predictions come directly from the four-class branch. For patients with multiple ultrasound images, four-class probabilities are averaged across images before assigning the patient-level class.

## Repository

| Path | Description |
|---|---|
| `train.py` | Model training and validation |
| `eval.py` | Patient-level evaluation |
| `predict.py` | Inference for one patient |
| `models/` | BiomedCLIP encoders, LoRA modules, and classification heads |
| `losses/` | Training losses |
| `datasets/` | Image and report loading |
| `engine/` | Training and evaluation loops |
| `utils/` | Metrics, plots, and configuration helpers |
| `configs/default.json` | Default training settings and data paths |

## Environment

Install a compatible PyTorch and torchvision environment, then install the project dependencies:

```bash
pip install -r requirements.txt
```

The BiomedCLIP checkpoint is loaded through Hugging Face/open-clip when the model is built.

## Data and running

Prepare patient-level ultrasound image folders, paired report descriptions, labels, and the training/validation/external cohort splits described in the manuscript. Set the corresponding paths in `configs/default.json`, then run:

```bash
python train.py --config configs/default.json
python eval.py --config configs/default.json --checkpoint results/main/best.pth --split valid
python predict.py --checkpoint results/main/best.pth --patient-folder /path/to/images --report-text "Ultrasound description"
```

This release provides the core FTN-MCNet method and its patient-level training/evaluation pipeline.
