import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    top_k_accuracy_score,
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Load test data
FEATURE_VERSION = "v1"
X_test = np.load(f"features/X_test_{FEATURE_VERSION}.npy")
y_test = np.load(f"features/y_test_{FEATURE_VERSION}.npy")

# Load the best model
best_rf = joblib.load("models/best_rf_model.joblib")

# Evaluate on test set
# Top-1 accuracy
y_test_pred = best_rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Set Top-1 Accuracy: {test_accuracy * 100:.2f}%")

# Top-3 accuracy
y_test_pred_proba = best_rf.predict_proba(X_test)
labels = best_rf.classes_
top3_accuracy = top_k_accuracy_score(y_test, y_test_pred_proba, k=3, labels=labels)
print(f"Test Set Top-3 Accuracy: {top3_accuracy * 100:.2f}%")

# Confusion matrix
cm_test = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix on Test Set:")
print(cm_test)

# Classification report
print("Classification Report on Test Set:")
print(classification_report(y_test, y_test_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.title("Confusion Matrix on Test Set")
plt.savefig("confusion_matrix.png")
