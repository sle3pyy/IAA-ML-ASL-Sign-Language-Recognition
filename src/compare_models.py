import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "asl_svm_model_v4.pkl")
TL_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")

SVM_DATA_PATH = os.path.join(BASE_DIR, "Data", "handmarks.csv")
TL_VAL_DIR = os.path.join(BASE_DIR, "Data", "split", "train") # Using the same dir with validation split

IMG_SIZE = (299, 299)
BATCH_SIZE = 32
SEED = 123
VAL_SPLIT = 0.2

def get_svm_f1_per_class():
    print("Evaluating SVM Model...")
    if not os.path.exists(SVM_MODEL_PATH):
        print("SVM Model not found.")
        return None
    
    df = pd.read_csv(SVM_DATA_PATH)
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = joblib.load(SVM_MODEL_PATH)
    
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_per_class = {k: v['f1-score'] for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']}
    return f1_per_class

def get_tl_f1_per_class():
    print("Evaluating TL Model (InceptionV3)...")
    if not os.path.exists(TL_MODEL_PATH):
        # Try finding any .keras model in MODEL_DIR if bingus_model.keras doesn't exist
        keras_models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
        if keras_models:
            model_path = os.path.join(MODEL_DIR, keras_models[0])
            print(f"Using {model_path} instead.")
        else:
            print("TL Model not found.")
            return None
    else:
        model_path = TL_MODEL_PATH

    model = tf.keras.models.load_model(model_path)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TL_VAL_DIR,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    
    class_names = val_ds.class_names
    y_true = []
    y_pred = []
    
    for x, y in val_ds:
        y_true.extend(y.numpy())
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    f1_per_class = {k: v['f1-score'] for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']}
    return f1_per_class

def plot_comparison():
    svm_f1 = get_svm_f1_per_class()
    tl_f1 = get_tl_f1_per_class()
    
    if svm_f1 is None or tl_f1 is None:
        print("Could not retrieve F1-scores for both models.")
        return

    # Prepare data for plotting
    all_classes = sorted(list(set(svm_f1.keys()) | set(tl_f1.keys())))
    
    data = []
    for cls in all_classes:
        data.append({
            'Class': cls,
            'Model': 'SVM (Handmarks)',
            'F1-Score': svm_f1.get(cls, 0)
        })
        data.append({
            'Class': cls,
            'Model': 'Transfer Learning (InceptionV3)',
            'F1-Score': tl_f1.get(cls, 0)
        })
    
    df_plot = pd.DataFrame(data)
    
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(x='Class', y='F1-Score', hue='Model', data=df_plot, palette='viridis')
    
    plt.title('Comparison of F1-Scores per Class: SVM vs Transfer Learning', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('ASL Class', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(title='Model', loc='upper right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=9, rotation=0)

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "model_comparison_f1.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Comparison plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    plot_comparison()
