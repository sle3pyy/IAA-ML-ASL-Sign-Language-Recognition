import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.augmentation import augment

IMG_SIZE = (299, 299)
BATCH_SIZE = 32 # Increased batch size for faster iterations
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 20
STAGE1_LR = 0.001
STAGE2_LR = 0.0001
FINE_TUNE_LAYERS = 30
SEED = 123

DATA_DIR_TRAIN = os.path.join(BASE_DIR, "Data", "split", "train")
DATA_DIR_VAL = os.path.join(BASE_DIR, "Data", "split", "val")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_PATH = os.path.join(MODEL_DIR, "learning_curve_tl.png")

def build_model(n_classes):
    inception = tf.keras.applications.InceptionV3(
        weights="imagenet",
        input_shape=(299, 299, 3),
        include_top=False,
    )
    inception.trainable = False
    
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1),
        inception,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    return model, inception

def get_learning_curve():
    # 1. Load full datasets
    train_ds_full = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR_TRAIN,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=None, # Load unbatched first for easy splitting
        shuffle=True,
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR_VAL,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    ).prefetch(tf.data.AUTOTUNE)
    
    class_names = train_ds_full.class_names
    n_classes = len(class_names)
    total_samples = tf.data.experimental.cardinality(train_ds_full).numpy()
    
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    results = []

    for frac in fractions:
        num_samples = int(total_samples * frac)
        print(f"\n{'='*60}")
        print(f"Training on {frac*100:.0f}% of data ({num_samples} samples)")
        print(f"{'='*60}")
        
        # Subsample and prepare dataset
        curr_train_ds = train_ds_full.take(num_samples)
        curr_train_ds = curr_train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        curr_train_ds = curr_train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        # Build and train
        model, inception = build_model(n_classes)
        
        # Stage 1
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE1_LR),
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall")
            ]
        )
        
        model.fit(curr_train_ds, epochs=STAGE1_EPOCHS, validation_data=val_ds, 
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
                  verbose=0)
        
        # Stage 2
        for layer in inception.layers[-FINE_TUNE_LAYERS:]:
            layer.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE2_LR),
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall")
            ]
        )
        
        history = model.fit(curr_train_ds, epochs=STAGE2_EPOCHS, validation_data=val_ds,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
                            verbose=0)
        
        # Record best metrics
        best_val_acc = max(history.history['val_accuracy'])
        best_val_prec = history.history['val_precision'][np.argmax(history.history['val_accuracy'])]
        best_val_rec = history.history['val_recall'][np.argmax(history.history['val_accuracy'])]
        
        # Calculate F1-Score for the curve
        if (best_val_prec + best_val_rec) > 0:
            val_f1 = 2 * (best_val_prec * best_val_rec) / (best_val_prec + best_val_rec)
        else:
            val_f1 = 0
            
        train_acc = history.history['accuracy'][-1]
        results.append((num_samples, train_acc, val_f1))
        print(f"Result for {num_samples} samples -> Train Acc: {train_acc:.4f}, Val F1: {val_f1:.4f} (P: {best_val_prec:.4f}, R: {best_val_rec:.4f})")

    # Plotting
    samples, train_scores, val_scores = zip(*results)
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    plt.plot(samples, train_scores, 'o-', label="Training Accuracy", color="#3498db", linewidth=2)
    plt.plot(samples, val_scores, 'o-', label="Validation F1-Score", color="#e74c3c", linewidth=2)
    
    plt.title("Transfer Learning Curve (InceptionV3)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Number of Training Samples", fontsize=12, labelpad=10)
    plt.ylabel("Score", fontsize=12, labelpad=10)
    plt.legend(loc="lower right", shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    os.makedirs(MODEL_DIR, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=300)
    print(f"\nLearning curve saved to {PLOT_PATH}")
    plt.show()

if __name__ == "__main__":
    get_learning_curve()
