"""
ASL Sign Language Recognition - Evaluation Script

Loads the best saved model and generates:
  - Confusion matrix (saved as PNG)
  - Classification report (precision, recall, F1 per class)
  - Training history plots (accuracy and loss curves for both stages)
  - Per-class accuracy breakdown

Usage:
  python evaluate.py
  python evaluate.py --model best_model.keras --history training_history.json
"""
import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def load_test_dataset(test_dir, img_size=(299, 299), batch_size=16):
    """Load test dataset from directory."""
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )
    return test_ds


def get_predictions(model, dataset):
    """Get true labels and predicted labels from a dataset."""
    all_labels = []
    all_preds = []

    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        all_labels.extend(labels.numpy())
        all_preds.extend(pred_classes)

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix - ASL Gesture Classification", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_training_history(history, output_path):
    """Plot accuracy and loss curves for both training stages."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["accuracy"]) + 1)
    stage1_end = history.get("stage1_epochs", len(history["accuracy"]))

    # --- Accuracy plot ---
    ax = axes[0]
    ax.plot(epochs, history["accuracy"], label="Train Accuracy", linewidth=2)
    ax.plot(epochs, history["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax.axvline(
        x=stage1_end, color="gray", linestyle="--", alpha=0.7,
        label=f"Fine-tuning start (epoch {stage1_end})"
    )
    ax.set_title("Model Accuracy", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # --- Loss plot ---
    ax = axes[1]
    ax.plot(epochs, history["loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    ax.axvline(
        x=stage1_end, color="gray", linestyle="--", alpha=0.7,
        label=f"Fine-tuning start (epoch {stage1_end})"
    )
    ax.set_title("Model Loss", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training History - Two-Stage Transfer Learning", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved to {output_path}")


def print_per_class_accuracy(y_true, y_pred, class_names):
    """Print accuracy for each class individually."""
    print("\n--- Per-Class Accuracy ---")
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() == 0:
            print(f"  {name}: no samples")
            continue
        correct = (y_pred[mask] == i).sum()
        total = mask.sum()
        acc = correct / total
        print(f"  {name}: {correct}/{total} = {acc:.4f} ({acc * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASL model")
    parser.add_argument(
        "--model", default="best_model.keras",
        help="Path to saved model (default: best_model.keras)"
    )
    parser.add_argument(
        "--history", default="training_history.json",
        help="Path to training history JSON (default: training_history.json)"
    )
    parser.add_argument(
        "--test-dir", default="./Data/split/test",
        help="Path to test data directory"
    )
    parser.add_argument(
        "--output-dir", default="../docs",
        help="Directory to save plots and reports"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model)

    # Load test data
    print(f"Loading test data from {args.test_dir}...")
    test_ds = load_test_dataset(args.test_dir)
    class_names = test_ds.class_names
    print(f"Classes: {class_names}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Get predictions for detailed analysis
    y_true, y_pred = get_predictions(model, test_ds)

    # Classification report
    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # Save report to file
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Loss:     {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Per-class accuracy
    print_per_class_accuracy(y_true, y_pred, class_names)

    # Confusion matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

    # Training history plots
    if os.path.exists(args.history):
        print(f"\nLoading training history from {args.history}...")
        with open(args.history, "r") as f:
            history = json.load(f)
        history_path = os.path.join(args.output_dir, "training_history.png")
        plot_training_history(history, history_path)
    else:
        print(f"\nWarning: Training history file not found at {args.history}")
        print("Skipping training history plots.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    main()
