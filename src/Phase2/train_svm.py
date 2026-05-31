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
    f1_score
)
from sklearn.preprocessing import StandardScaler
import joblib
import os
import argparse

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "asl_svm_model_v5.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "asl_scaler_v5.pkl")

def train_asl_model():
    parser = argparse.ArgumentParser(description="Train ASL SVM Model")
    parser.add_argument("--train_data", default=os.path.join(BASE_DIR, "Data", "handmarks.csv"),
                        help="Path to training CSV")
    parser.add_argument("--val_data", default=None,
                        help="Path to optional validation CSV")
    args = parser.parse_args()

    train_path = args.train_data
    val_path = args.val_data

    # 1. Load the data
    if not os.path.exists(train_path):
        print(f"Error: Training dataset not found at {train_path}")
        return

    print(f"Loading training data from {train_path}...")
    df_train = pd.read_csv(train_path)
    X_train_raw = df_train.drop('label', axis=1)
    y_train_raw = df_train['label']

    if val_path and os.path.exists(val_path):
        print(f"Loading validation data from {val_path}...")
        df_val = pd.read_csv(val_path)
        X_val_raw = df_val.drop('label', axis=1)
        y_val_raw = df_val['label']
        
        X_train = X_train_raw
        y_train = y_train_raw
        X_test = X_val_raw
        y_test = y_val_raw
        print(f"Using provided validation set with {len(X_test)} samples.")
    else:
        print("No validation set provided. Splitting training data (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_raw, y_train_raw, test_size=0.2, random_state=42, stratify=y_train_raw
        )

    # 4. Hyperparameter Tuning with GridSearchCV (incorporates Cross-Validation)
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

    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation weighted F1-score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_

    # 6. Final Evaluation on Validation/Test Set
    print("\nEvaluating best model on validation/test set...")
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print("\n" + "="*30)
    print(f"Accuracy:               {accuracy:.4%}")
    print(f"Balanced Accuracy:      {balanced_acc:.4%}")
    print(f"Macro F1-Score:         {macro_f1:.4f}")
    print("="*30)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Visualization: Confusion Matrices (Raw and Percentage)
    print("\nGenerating and saving confusion matrices...")
    
    # Determine labels for the confusion matrix based on what was actually evaluated
    eval_labels = sorted(list(set(y_test.unique()) | set(y_pred)))
    
    # Compute Raw and Percentage Confusion Matrices
    cm = confusion_matrix(y_test, y_pred, labels=eval_labels)
    cm_percentage = confusion_matrix(y_test, y_pred, labels=eval_labels, normalize='true')
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    # Plot and Save Raw Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=eval_labels, 
                yticklabels=eval_labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('ASL Recognition: Raw Confusion Matrix')
    plt.tight_layout()
    cm_plot_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Raw confusion matrix saved to {cm_plot_path}")

    # Plot and Save Percentage Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage * 100, annot=True, fmt='.2f', cmap='Oranges', 
                xticklabels=eval_labels, 
                yticklabels=eval_labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('ASL Recognition: Confusion Matrix (Percentage %)')
    plt.tight_layout()
    cm_pct_plot_path = os.path.join(MODEL_DIR, 'confusion_matrix_percentage.png')
    plt.savefig(cm_pct_plot_path, dpi=300)
    plt.close()
    print(f"Percentage confusion matrix saved to {cm_pct_plot_path}")

    # 8. Save the Model
    print(f"\nSaving best model to {MODEL_PATH}...")
    joblib.dump(best_model, MODEL_PATH)

    print("\nTraining complete!")

if __name__ == "__main__":
    train_asl_model()
