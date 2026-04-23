# ASL Sign Recognition: SVM Implementation Guide

This document outlines the process of training a Support Vector Machine (SVM) model to classify ASL letters (A, B, C, etc.) based on the hand landmarks extracted in Phase 2.

## 1. Prerequisites and Imports

Ensure you have `pandas`, `scikit-learn`, and `joblib` installed in your environment.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
```

## 2. Load and Prepare the ASL Handmarks Data

The dataset `handmarks.csv` contains the normalized features extracted from MediaPipe landmarks.

```python
# Load the dataset
df = pd.read_csv("src/Data/handmarks.csv")

# Separate features (X) and target labels (y)
X = df.drop('label', axis=1)
y = df['label']

# Split into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## 3. Feature Scaling (Critical for SVM)

SVMs are distance-based classifiers. Scaling ensures that features with larger numerical ranges don't dominate the distance calculations.

```python
scaler = StandardScaler()

# Fit on training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Transform validation data using the same scaler
X_val_scaled = scaler.transform(X_val)
```

## 4. Hyperparameter Tuning with Cross-Validation

To ensure the model generalizes well and to find the optimal hyperparameters (`C` and `gamma`), we use `GridSearchCV`. This tool automatically performs **k-fold cross-validation** (default `cv=5`) during the parameter search, reducing the risk of being "lucky" or "unlucky" with a single data split.

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
}

# Initialize GridSearchCV with 5-fold cross-validation and balanced class weights
grid_search = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

# Perform the search (fits on training data using CV)
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_}")
best_model = grid_search.best_estimator_
```

## 5. Evaluate and Save the Best Model

After finding the best estimator via cross-validation, evaluate its final performance on a hold-out test set using metrics that are robust to class imbalance.

```python
from sklearn.metrics import balanced_accuracy_score, f1_score

# Evaluate performance on unseen test data
y_pred = best_model.predict(X_test_scaled)

print(f"Test Accuracy:       {accuracy_score(y_test, y_pred):.4%}")
print(f"Balanced Accuracy:   {balanced_accuracy_score(y_test, y_pred):.4%}")
print(f"Macro F1-Score:      {f1_score(y_test, y_pred, average='macro'):.4f}")

print(classification_report(y_test, y_pred))

# Save for inference
joblib.dump(best_model, 'src/models/asl_svm_model.pkl')
joblib.dump(scaler, 'src/models/asl_scaler.pkl')
```

## Key Considerations for ASL SVM

- **Normalization:** `StandardScaler` is necessary to center the data for the SVM optimizer.
- **Handling Imbalance:** We use `class_weight='balanced'` in the SVC constructor to penalize mistakes on minority classes more heavily.
- **Robust Metrics**: 
    - **Balanced Accuracy**: The average of recall obtained on each class.
    - **Macro F1-Score**: The arithmetic mean of all the per-class F1-scores.
- **Cross-Validation**: `GridSearchCV` provides a more robust estimate of model performance than a single split.
