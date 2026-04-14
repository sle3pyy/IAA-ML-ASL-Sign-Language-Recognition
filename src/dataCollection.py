import cv2
import os
from HandTrackingModule import HandDetector
import numpy as np
import time
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--visuals", action="store_true", default=False,
    help="Display hand visuals (landmarks and bounding box)"
)
parser.add_argument(
    "--model", default="bingus_model.keras",
    help="Path to saved model"
)
parser.add_argument(
    "--collect", type=str, default=None,
    help="Class name to collect images for (e.g., A, B, C). "
         "Press SPACE to save an image."
)
args = parser.parse_args()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = args.model
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(CURRENT_DIR, MODEL_PATH)

TRAIN_DIR = os.path.join(CURRENT_DIR, "Data", "split", "train")
COLLECTED_DIR = os.path.join(CURRENT_DIR, "Data", "collected")

cap = cv2.VideoCapture(0)
detector = HandDetector(num_hands=1)
imgSize = 299
show_visuals = args.visuals

folder = COLLECTED_DIR

if MODEL_PATH:
    model = tf.keras.models.load_model(MODEL_PATH)

class_names = sorted([
    entry for entry in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, entry))
])

if model.output_shape[-1] != len(class_names):
    raise ValueError(
        f"Model outputs {model.output_shape[-1]} classes, but {len(class_names)} "
        f"class folders were found in {TRAIN_DIR}: {class_names}"
    )

print(f"Model loaded from {MODEL_PATH}")
print(f"Classes: {class_names}")

last_prediction_time = 0
label = ""
prediction_interval = 1.0  # seconds


while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Detect hands
    hands, img, bbox = detector.findHands(img, draw=show_visuals)

    if bbox:
        x, y, w, h = bbox

        imgCrop = img[y:y + h, x:x + w]
        imgResize = cv2.resize(imgCrop, (imgSize, imgSize))

        # Run prediction if not in collection mode
        if args.collect is None:
            current_time = time.time()
            if current_time - last_prediction_time >= prediction_interval:
                img_input = np.expand_dims(imgResize, axis=0).astype(np.float32)
                predictions = model.predict(img_input, verbose=0)
                pred_class = np.argmax(predictions[0])
                confidence = predictions[0][pred_class]
                label = f"{class_names[pred_class]} ({confidence:.1%})"
                last_prediction_time = current_time

            if label:
                cv2.putText(
                    img, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                )

        cv2.imshow("Hand Crop", imgResize)

    cv2.imshow("Hand Tracking", img)

    key = cv2.waitKey(1) & 0xFF

    # Save image if in collection mode and SPACE is pressed
    if args.collect and key == ord(" "):
        os.makedirs(os.path.join(folder, args.collect), exist_ok=True)
        save_path = f"{folder}/{args.collect}/Image_{time.time()}.jpg"
        cv2.imwrite(save_path, imgResize)
        print(f"Saved: {save_path}")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
