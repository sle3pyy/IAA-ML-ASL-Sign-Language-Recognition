import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "handmarks.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_PATH = os.path.join(MODEL_DIR, "learning_curve.png")

def plot_learning_curve():
    # 1. Load the data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Shuffle the data to ensure all classes are represented even in small subsets
    print("Shuffling data...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 2. Prepare features and target
    X = df.drop('label', axis=1)
    y = df['label']

    # 3. Feature Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Define the model (using best parameters found previously)
    model = SVC(
        kernel='rbf', 
        C=100, 
        gamma=0.001, 
        class_weight='balanced', 
        random_state=42
    )

    # 5. Compute Learning Curve
    print("Computing learning curve (this might take a while)...")
    train_sizes, train_scores, test_scores = learning_curve(
        model, 
        X_scaled, 
        y, 
        cv=5, 
        scoring='f1_weighted', 
        n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        shuffle=True,
        random_state=42,
        verbose=1
    )

    # 6. Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # 7. Plotting
    print("Generating plot...")
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Customize colors and aesthetics
    primary_color = "#3498db" # Blue
    secondary_color = "#e74c3c" # Red
    
    plt.plot(train_sizes, train_mean, 'o-', color=primary_color, label="Training Score", linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color=primary_color)
    
    plt.plot(train_sizes, test_mean, 'o-', color=secondary_color, label="Cross-Validation Score", linewidth=2)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color=secondary_color)

    plt.title("SVM Learning Curve (ASL Recognition)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Number of Training Samples", fontsize=12, labelpad=10)
    plt.ylabel("Weighted F1-Score", fontsize=12, labelpad=10)
    plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Annotate final performance
    plt.annotate(f'Final CV Score: {test_mean[-1]:.4f}', 
                 xy=(train_sizes[-1], test_mean[-1]), 
                 xytext=(train_sizes[-1]*0.7, test_mean[-1]-0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=10, fontweight='bold')

    plt.tight_layout()
    
    # Save the plot
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    plt.savefig(PLOT_PATH, dpi=300)
    print(f"Learning curve saved to {PLOT_PATH}")
    plt.show()

if __name__ == "__main__":
    plot_learning_curve()
