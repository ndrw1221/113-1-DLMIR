import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import joblib

# Set random seeds
import random

np.random.seed(42)
random.seed(42)

# Load data
FEATURE_VERSION = "v1"

X_train = np.load(f"features/X_train_{FEATURE_VERSION}.npy")
y_train = np.load(f"features/y_train_{FEATURE_VERSION}.npy")

X_valid = np.load(f"features/X_valid_{FEATURE_VERSION}.npy")
y_valid = np.load(f"features/y_valid_{FEATURE_VERSION}.npy")

# Initialize Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Define parameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": [None, "balanced"],
}

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, scoring="accuracy", cv=cv, n_jobs=-1, verbose=2
)

grid_search.fit(X_train, y_train)

# Get the best model from cross-validation
best_rf = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Evaluate the model on the validation set
valid_accuracy = best_rf.score(X_valid, y_valid)
print(f"Validation accuracy: {valid_accuracy:.4f}")

# Save the best model
joblib.dump(best_rf, "models/best_rf_model.joblib")
