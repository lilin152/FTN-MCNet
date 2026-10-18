# FTN-MCNet
﻿
This repository provides the core implementation of **FTN-MCNet**, a BiomedCLIP-based vision-language model for preoperative classification of follicular thyroid neoplasms into four pathological categories: follicular thyroid adenoma (FTA), minimally invasive follicular thyroid carcinoma (MI-FTC), encapsulated angioinvasive FTC (EA-FTC), and widely invasive FTC (WI-FTC).
﻿
## File Descriptions
﻿
| **File Name** | **Function Description** |
|---|---|
| `train.py` | Model training and validation |
| `eval.py` | Patient-level model evaluation |
| `predict.py` | Single-patient inference |
| `models/backbone.py` | BiomedCLIP image and text encoders with LoRA |
| `models/fusion_head.py` | Multimodal feature fusion and classification heads |
| `losses/consistency_loss.py` | Focal Loss and KL consistency loss |
| `datasets/` | Patient-level image and report data loading |
| `engine/` | Training and evaluation procedures |
| `utils/` | Evaluation metrics, plots, and helper functions |
| `configs/default.json` | Default training parameters and data paths |
﻿
# How to Start
﻿
The workflow consists of four steps:
﻿
- Step 1: Environment setting
- Step 2: Data preparation
- Step 3: Model training
- Step 4: Model evaluation
﻿
## Step 1: Environment Setting
﻿
Install a compatible PyTorch and torchvision environment for your system, then install the project dependencies:
﻿
```bash
pip install -r requirements.txt
```
﻿
The pretrained BiomedCLIP model is loaded through Hugging Face/open-clip when the model is initialized.
﻿
## Step 2: Data Preparation
﻿
Prepare patient-level ultrasound image folders, corresponding report descriptions, and cohort index files according to the study protocol.
﻿
The expected data structure is:
﻿
```text
data/
├── splits/
│ ├── train.npy
│ ├── valid.npy
│ ├── test_external1.npy
│ └── test_external2.npy
└── reports.csv
```

## Step 3: Model Training
﻿
Set the data paths and training parameters in `configs/default.json`, then run:
﻿
```bash
python train.py --config configs/default.json
```
﻿
The model is trained jointly using binary and four-class Focal Loss with batch-level class weighting and a KL probability-consistency constraint. 
﻿
## Step 4: Model Evaluation
﻿
Evaluate the trained model on the internal validation cohort:
﻿
```bash
python eval.py --config configs/default.json --checkpoint results/main/best.pth --split valid
```
﻿
For external evaluation, use the corresponding split:
﻿
```bash
python eval.py --config configs/default.json --checkpoint results/main/best.pth --split test_external1
python eval.py --config configs/default.json --checkpoint results/main/best.pth --split test_external2
```

﻿
Single-patient inference can be performed using `predict.py` with a trained checkpoint, an image folder, and the corresponding report description.