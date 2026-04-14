"""
Split Data/processed/ into train/val/test directories (70/15/15).
Maintains class folder structure. Uses deterministic seed for reproducibility.
"""
import os
import shutil
import random

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def split_data():
    random.seed(SEED)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    source_dir = os.path.join(src_dir, "Data", "processed")
    output_dirs = {
        "train": os.path.join(src_dir, "Data", "split", "train"),
        "val": os.path.join(src_dir, "Data", "split", "val"),
        "test": os.path.join(src_dir, "Data", "split", "test"),
    }

    # Clean up any previous split
    split_root = os.path.join(src_dir, "Data", "split")
    if os.path.exists(split_root):
        shutil.rmtree(split_root)
        print(f"Removed existing {split_root}/")

    # Get class folders
    class_names = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ])
    print(f"Found classes: {class_names}")

    stats = {"train": {}, "val": {}, "test": {}}

    for class_name in class_names:
        class_path = os.path.join(source_dir, class_name)

        # Get all image files
        images = sorted([
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # Shuffle deterministically
        random.shuffle(images)

        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split_name, split_images in splits.items():
            target_dir = os.path.join(output_dirs[split_name], class_name)
            os.makedirs(target_dir, exist_ok=True)

            for img_name in split_images:
                src = os.path.join(class_path, img_name)
                dst = os.path.join(target_dir, img_name)
                shutil.copy2(src, dst)

            stats[split_name][class_name] = len(split_images)

    # Print summary
    print("\n--- Split Summary ---")
    for split_name, class_counts in stats.items():
        total = sum(class_counts.values())
        detail = ", ".join(f"{c}: {n}" for c, n in class_counts.items())
        print(f"{split_name:>5}: {total:>5} images  ({detail})")

    grand_total = sum(sum(cc.values()) for cc in stats.values())
    print(f"{'total':>5}: {grand_total:>5} images")


if __name__ == "__main__":
    split_data()
