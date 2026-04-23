import argparse
import os
import sys
from pathlib import Path


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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "base_datasetABC_full_model.keras")
DEFAULT_VAL_DIR = os.path.join(BASE_DIR, "Data", "val")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "validation_reports")
IMG_SIZE = (299, 299)
BATCH_SIZE = 16


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate a chosen transfer-learning model on a chosen folder."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved Keras model to validate.",
    )
    parser.add_argument(
        "--val-dir",
        default=DEFAULT_VAL_DIR,
        help="Path to the validation folder organized by class subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the confusion matrix and report will be saved.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=299,
        help="Square image size used to load validation images.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size used during validation.",
    )
    return parser


def load_validation_dataset(val_dir, img_size, batch_size):
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )
    class_names = dataset.class_names
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, class_names


def collect_predictions(model, dataset):
    y_true = np.concatenate([labels.numpy() for _, labels in dataset], axis=0)
    y_prob = model.predict(dataset, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred


def save_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_classification_report(y_true, y_pred, class_names, output_path, loss, accuracy):
    report_text = classification_report(y_true, y_pred, target_names=class_names)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"Validation Loss:     {loss:.4f}\n")
        handle.write(f"Validation Accuracy: {accuracy:.4f}\n\n")
        handle.write("Classification Report\n")
        handle.write("=" * 50 + "\n")
        handle.write(report_text)

    return report_text


def main():
    args = build_parser().parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model)

    print(f"Loading validation data from {args.val_dir}...")
    val_ds, class_names = load_validation_dataset(
        args.val_dir, args.img_size, args.batch_size
    )
    print(f"Classes: {class_names}")

    print("\nEvaluating model...")
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=1)
    print(f"Validation Loss:     {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    print("\nCollecting predictions...")
    y_true, y_pred = collect_predictions(model, val_ds)

    report_path = os.path.join(args.output_dir, "classification_report.txt")
    report_text = save_classification_report(
        y_true, y_pred, class_names, report_path, val_loss, val_accuracy
    )
    print("\nClassification Report:")
    print(report_text)
    print(f"Classification report saved to {report_path}")

    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    save_confusion_matrix(y_true, y_pred, class_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
