import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
import os
import argparse

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_PATH = os.path.join(MODEL_DIR, "learning_curve_svm.png")

def plot_learning_curve_svm():
    parser = argparse.ArgumentParser(description="Plot Learning Curve for ASL SVM Model")
    parser.add_argument("--train_data", default=os.path.join(BASE_DIR, "Data", "handmarks.csv"),
                        help="Path to training CSV")
    args = parser.parse_args()

    train_path = args.train_data

    # 1. Load the data
    if not os.path.exists(train_path):
        print(f"Error: Training dataset not found at {train_path}")
        return

    print(f"Loading data from {train_path}...")
    df = pd.read_csv(train_path)
    X = df.drop('label', axis=1)
    y = df['label']

    # 2. Setup SVM
    # Using parameters close to what was found in train_svm.py
    svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42)

    # 3. Compute Learning Curve
    print("Computing learning curve (this may take a moment)...")
    train_sizes = np.linspace(0.1, 1.0, 5)
    
    train_sizes, train_scores, test_scores = learning_curve(
        svm, X, y, 
        train_sizes=train_sizes, 
        cv=5, 
        scoring='f1_macro', 
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )

    # 5. Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # 6. Plotting
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    plt.plot(train_sizes, train_mean, 'o-', color="#3498db", label="Training F1-Score", linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="#3498db")
    
    plt.plot(train_sizes, test_mean, 'o-', color="#e74c3c", label="Cross-Validation F1-Score", linewidth=2)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="#e74c3c")

    plt.title("SVM Learning Curve (F1-Score)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Number of Training Samples", fontsize=12, labelpad=10)
    plt.ylabel("Macro F1-Score", fontsize=12, labelpad=10)
    plt.legend(loc="lower right", shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)

    plt.tight_layout()
    os.makedirs(MODEL_DIR, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=300)
    print(f"\nSVM Learning curve saved to {PLOT_PATH}")
    plt.show()

if __name__ == "__main__":
    plot_learning_curve_svm()
