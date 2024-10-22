# DLMIR HW2

This code is modified from [sigsep/open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch/tree/master). Please refer to the original code for details and training scripts.

## Report
[Link to Report](https://docs.google.com/presentation/d/1vZwkyOAy6gB2qZyBQJMZfjD9VrDU1PUjimVUWtQaa-k/edit?usp=sharing)

## Inference Guide

### 1. Evironment Setup

```bash
conda env create -f environment.yml -n YOUR_ENV_NAME
conda activate YOUR_ENV_NAME
```

### 2. Evaluation

#### Using Separator

```bash
cd open-unmix-pytorch/openunmix
python eval.py --root ROOT --epoch EPOCH
```
`ROOT` is the path to you local musdb18hq/test split.

Use the `EPOCH` argument to specify which model checkpoint to use, choose among [1, 5, 25, 50, 137, 150].

#### Using Griffin & Lim

```bash
python eval-griffinlim.py --root ROOT
```

`ROOT` is the path to you local musdb18hq/test split.