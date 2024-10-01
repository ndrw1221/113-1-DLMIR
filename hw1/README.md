# DLMIR HW1

## Report
[Link to Report](https://drive.google.com/file/d/1xNHDi0YABiNvcquqlKNnfmRPE7kapkVf/view?usp=sharing)

## Inference Guide

This is the inference guide to perform evaluation on test set using the trained models in hw1.

### 1. Evironment Setup

```bash
conda create -n YOUR_ENV_NAME python=3.9
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```

### 2. Evaluation

```bash
python evaluate.py \
--root_dir "path/to/your/dataset/root/directory"
--model_path "model_classifier-only-finetune_bs32_lr1e-4.pth"
```

Choose the `--model_path` among:

1. `model_classifier-only-finetune_bs32_lr1e-4.pth`
2. `model_full-finetune_bs16_lr1e-4.pth`
3. `model_partial_finetune_bs32_lr1e-4.pth`

#### All options:

```
"--root_dir", type=str, default="dataset"
"--batch_size", type=int, default=4
"--num_workers", type=int, default=0
"--device", type=str, default="cuda"
"--model_path", type=str, default="model.pth"
"--num_classes", type=int, default=9
"--sampling_rate", type=int, default=24000
"--threshold", type=float, default=0.5
```

### 3. Dataset Structure

Your datasets root directory (`root_dir`) should at least contain the **test set audios** and the **test_labels.json**, which will look like:

```
root_dir/
├── test/
│   ├── Track0.npy
│   ├── Track1.npy
│   ├── Track2.npy
│   ├── Track3.npy
│   └── ...
└── test_labels.json
```