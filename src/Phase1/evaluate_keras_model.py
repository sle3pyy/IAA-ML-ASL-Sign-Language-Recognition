import argparse
import os
import sys
from pathlib import Path
import numpy as np

def ensure_cuda_library_path():
    if os.environ.get("TF_CUDA_LIBS_READY") == "1":
        return

    site_packages = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
    )
    lib_dirs = [
        site_packages / "cuda_runtime" / "lib",
        site_packages / "cudnn" / "lib",
        site_packages / "cublas" / "lib",
        site_packages / "cufft" / "lib",
        site_packages / "curand" / "lib",
        site_packages / "cusolver" / "lib",
        site_packages / "cusparse" / "lib",
        site_packages / "nccl" / "lib",
        site_packages / "nvjitlink" / "lib",
    ]
    existing_dirs = [str(path) for path in lib_dirs if path.exists()]

    if not existing_dirs:
        return

    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    current_entries = [entry for entry in current_ld_path.split(":") if entry]
    new_entries = [entry for entry in existing_dirs if entry not in current_entries]
    if not new_entries:
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(new_entries + current_entries)
    os.environ["TF_CUDA_LIBS_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)

ensure_cuda_library_path()

import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
DEFAULT_VAL_DIR = os.path.join(BASE_DIR, "Data", "val")
IMG_SIZE = 299
BATCH_SIZE = 16

def main():
    parser = argparse.ArgumentParser(description="Evaluate Keras model on validation dataset")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved Keras model.",
    )
    parser.add_argument(
        "--val-dir",
        default=DEFAULT_VAL_DIR,
        help="Path to the validation directory organized by class subfolders.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=IMG_SIZE,
        help="Square image dimension to resize images.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for dataset loading.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    if not os.path.exists(args.val_dir):
        print(f"Error: Validation directory not found at {args.val_dir}")
        return

    print(f"Loading Keras model from {args.model}...")
    model = tf.keras.models.load_model(args.model)

    print(f"Loading validation dataset from {args.val_dir}...")
    dataset = tf.keras.utils.image_dataset_from_directory(
        args.val_dir,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
    )
    class_names = dataset.class_names
    print(f"Found classes: {class_names}")

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print("Collecting validation labels...")
    y_true = np.concatenate([labels.numpy() for _, labels in dataset], axis=0)

    print("Predicting with model...")
    y_prob = model.predict(dataset, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    # Only account for specific target classes: A, B, C, L, F, Y
    target_classes = ["A", "B", "C", "L", "F", "Y"]
    target_indices = [i for i, c in enumerate(class_names) if c.upper() in target_classes]
    filtered_class_names = [class_names[i] for i in target_indices]

    # Mask samples belonging to target classes
    mask = np.isin(y_true, target_indices)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    # Calculate metrics
    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    balanced_acc = balanced_accuracy_score(y_true_filtered, y_pred_filtered)
    f1_macro = f1_score(y_true_filtered, y_pred_filtered, average="macro")
    f1_weighted = f1_score(y_true_filtered, y_pred_filtered, average="weighted")

    # Output results
    print("\n" + "="*40)
    print("      Evaluation Results (A, B, C, L, F, Y)")
    print("="*40)
    print(f"Accuracy:          {accuracy:.4%}")
    print(f"Balanced Accuracy: {balanced_acc:.4%}")
    print(f"F1-Score (Macro):  {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print("="*40)

    print("\nDetailed Classification Report:")
    print(classification_report(y_true_filtered, y_pred_filtered, labels=target_indices, target_names=filtered_class_names))

if __name__ == "__main__":
    main()
