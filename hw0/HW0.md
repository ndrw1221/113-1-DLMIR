# Report

[Link to report](https://docs.google.com/presentation/d/1PByyBUmborBJNQWZepZ4sHU4tKR5zV2Ln2gSIOJ536Q/edit?usp=sharing)

# Inference Guide

This is the inference guide for **task2** and **task3** of hw0.

## Task 2

### 1. Environment Setup

```bash
cd task2/
conda create --name YOUR_ENV_NAME python=3.10
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```

### 2. Testing Data Preparation

First, correct the path to your local Nsynth dataset in `data_repparation_inference.py` on line 7:

```python
# data_repparation_inference.py

BASE_PATH = Path("path/to/your/nsynth-dataset")  # Correct the path
TEST_PATH = BASE_PATH / "nsynth-test"
FEATURE_VERSION = "v1"
```

Then, run 

```bash
python data_preparation_inference.py
```

You will get the extracted features from the testing data, saved as `.npy` in the `feature/` folder.

### 3. Evalutaion

Make sure your folder sturcture is as follows before running evaluation:

```
task2/
├── features/
│   ├── X_test_v1.npy
│   └── y_test_v1.npy
├── models/
│   └── best_rf_model.joblib
├── data_preparation_inference.py
├── evaluate.py
└── utils.py
```

If correct, run:

```bash
python evaluate.py
``` 

and you should see the result.

## Task3

### 1. Environment Setup

```bash
cd task3/sota-music-tagging-models
conda env create --name YOUR_ENV_NAME  --file environment.yml
conda activate --name YOUR_ENV_NAME
```

### 2. Data Preparation

```bash
cd preprocessing/

python -u nsynth_read.py run YOUR_TEST_DATA_PATH

python -u nsynth_preprocess.py \
--data_path YOUR_TEST_DATA_PATH \
--split test
```

Make sure that YOUR_TEST_DATA_PATH follows the folder structure:

```
YOUR_TEST_DATA_PATH/
├── audio/
│   ├── <audio_0>.wav
│   ├── <audio_1>.wav
│   └── ...
└── examples.json
```


### 3. Evaluate

Make sure your have the necessary files in the correct folder structure before running evaluation:

```
task3/
└── sota-music-tagging-models/
    ├── models/
    │   └── nsynth/
    │       ├── no_to_db/
    │       │   └── best_model.pth
    │       └── to_to/
    │           └── best_model.pth
    ├── split/
    │   └── nsynth/
    │       └── test.npy
    └── training/
        ├── data/
        │   └── nsynth/
        │       └── npy/
        │           ├── <test_audio_0>.npy
        │           ├── <test_audio_1>.npy
        │           └── ...
        ├── attention_modules.py
        ├── eval.py
        ├── model.py
        └── modules.py
```

If correct, run the following and you should see the result:

```bash
cd training/

# With taking the log
python -u eval.py \
--dataset nsynth \
--model_type short \
--model_load_path ./../models/nsynth/to_db/best_model.pth \
--data_path ./data

# Without taking the log
python -u eval.py \
--dataset nsynth \
--model_type short \
--model_load_path ./../models/nsynth/no_to_db/best_model.pth \
--data_path ./data
```

**Note: Comment out line 390 in `model.py` if you want to run the mel-spectrogram without taking the log:**

```python
# model.py

# Spectrogram
x = self.spec(x)
# x = self.to_db(x)  <-- comment out this line
x = x.unsqueeze(1)
x = self.spec_bn(x)
```