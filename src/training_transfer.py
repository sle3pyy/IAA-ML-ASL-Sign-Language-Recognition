import os
import sys
from pathlib import Path


def ensure_cuda_library_path():
    if os.environ.get("TF_CUDA_LIBS_READY") == "1":
        return

    site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia"
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
from augmentation import augment

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


IMG_SIZE = (299, 299)
BATCH_SIZE = 16
STAGE1_EPOCHS = 30
STAGE2_EPOCHS = 50  
STAGE1_LR = 0.001
STAGE2_LR = 0.0001
FINE_TUNE_LAYERS = 30  
SEED = 123

DATA_DIR_TRAIN = os.path.join(CURRENT_DIR, "Data", "split", "train")
DATA_DIR_VAL = os.path.join(CURRENT_DIR, "Data", "split", "val")
DATA_DIR_TEST = os.path.join(CURRENT_DIR, "Data", "split", "test")

MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "base_datasetABC_model.keras")


def configure_device():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected by TensorFlow. Training will run on CPU.")
        return

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"Using GPU: {[gpu.name for gpu in gpus]}")


configure_device()

print("Loading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_TRAIN,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_VAL,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR_TEST,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
n_classes = len(class_names)
print(f"Classes ({n_classes}): {class_names}")

train_ds = train_ds.unbatch()
train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


print("Building model...")

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

model.summary()


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]



print("\n" + "=" * 60)
print("STAGE 1: Training classifier head (inception frozen)")
print("=" * 60)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE1_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_stage1 = model.fit(
    train_ds,
    epochs=STAGE1_EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
)


print("\n" + "=" * 60)
print(f"STAGE 2: Fine-tuning top {FINE_TUNE_LAYERS} InceptionV3 layers")
print("=" * 60)

for layer in inception.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False
for layer in inception.layers[-FINE_TUNE_LAYERS:]:
    layer.trainable = True

trainable_count = sum(1 for l in inception.layers if l.trainable)
total_count = len(inception.layers)
print(f"Trainable layers in inception: {trainable_count}/{total_count}")

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE2_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

stage1_end_epoch = len(history_stage1.history["loss"])

model.fit(
    train_ds,
    epochs=STAGE2_EPOCHS,
    initial_epoch=stage1_end_epoch,
    validation_data=val_ds,
    callbacks=callbacks,
)



print("\n" + "=" * 60)
print("FINAL EVALUATION ON TEST SET")
print("=" * 60)
best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)
print(f"\nTest Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print(f"Best model saved to {MODEL_SAVE_PATH}")
print("Done.")
