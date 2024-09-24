# HW0 Readme

This is the inference guide for task2/ and task3/ of hw0

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
BASE_PATH = Path("path/to/your/nsynth-dataset")
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

### 2. Data Preparation

### 3. Evaluate