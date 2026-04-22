import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    balanced_accuracy_score, 
    f1_score, 
    matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "handmarks.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "asl_svm_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "asl_scaler.pkl")

def train_asl_model():
    # 1. Load the data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # 2. Prepare features and target
    X = df.drop('label', axis=1)
    y = df['label']
    classes = sorted(y.unique())

    print(f"Dataset shape: {df.shape}")
    print("\nClass Distribution:")
    dist = y.value_counts(normalize=True) * 100
    for cls, pct in dist.items():
        print(f"  {cls}: {pct:.2f}% ({y.value_counts()[cls]} samples)")

    # 3. Split into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Feature Scaling
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Hyperparameter Tuning with GridSearchCV (incorporates Cross-Validation)
    print("Performing hyperparameter tuning with GridSearchCV (5-fold CV)...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    }

    grid_search = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True),
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation weighted F1-score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_

    # 6. Final Evaluation on Hold-out Test Set
    print("\nEvaluating best model on hold-out test set with robust metrics...")
    y_pred = best_model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    print("\n" + "="*30)
    print(f"Test Accuracy:          {accuracy:.4%}")
    print(f"Balanced Accuracy:      {balanced_acc:.4%}")
    print(f"Macro F1-Score:         {macro_f1:.4f}")
    print(f"Matthews Corr Coeff:    {mcc:.4f}")
    print("="*30)
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred))

    # 7. Visualization: Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('ASL Recognition: Confusion Matrix')
    plt.tight_layout()
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    cm_plot_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plt.savefig(cm_plot_path)
    print(f"Confusion matrix saved to {cm_plot_path}")

    # 8. Save the Model and Scaler
    print(f"\nSaving best model to {MODEL_PATH}...")
    joblib.dump(best_model, MODEL_PATH)
    
    print(f"Saving scaler to {SCALER_PATH}...")
    joblib.dump(scaler, SCALER_PATH)

    print("\nTraining complete!")

if __name__ == "__main__":
    train_asl_model()
