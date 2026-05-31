import cv2
import os
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from utils.HandTrackingModule import HandDetector
from utils.feature_extraction import extract_features_from_landmarks

# Features order (MUST match the order used during training)
FEATURE_FIELDS = [
    'thumb_index_dist', 'index_middle_dist', 'middle_ring_dist', 
    'ring_pinky_dist', 'thumb_middle_dist', 'thumb_ring_dist', 'thumb_pinky_dist', 
    'thumb_curl', 'index_curl', 'middle_curl', 'ring_curl', 'pinky_curl', 
    'thumb_y', 'index_y', 'middle_y', 'ring_y', 'pinky_y', 
    'hand_rotation', 'thumb_angle', 'palm_tilt', 
    'spread_index_ring', 'spread_thumb_pinky', 'y_variance'
]

def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Real-time ASL Classification using SVM")
    parser.add_argument(
        "--model", default="asl_svm_model_v4.pkl",
        help="Name of the saved SVM model in src/models/"
    )
    args = parser.parse_args()

    # Paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(CURRENT_DIR, "models")

    def resolve_model_path(path_arg):
        if os.path.isabs(path_arg):
            return path_arg

        if os.path.exists(path_arg):
            return os.path.abspath(path_arg)

        return os.path.join(MODEL_DIR, path_arg)

    model_path = resolve_model_path(args.model)
    scaler_path = resolve_model_path(args.scaler)

    # Load Model and Scaler
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error: Model or Scaler not found in {MODEL_DIR}")
        print(f"Resolved model path: {model_path}")
        print(f"Resolved scaler path: {scaler_path}")
        return

    print(f"Loading SVM model from {model_path}...")
    model = joblib.load(model_path)

    # Initialize Camera and Detector
    cap = cv2.VideoCapture(0)
    detector = HandDetector(num_hands=1)
    
    label = ""
    last_prediction_time = 0
    prediction_interval = 0.1  # seconds

    print("Starting webcam... Press 'q' to quit.")

    try:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            # Detect hands (draw=True for visualization)
            hands, img, bbox = detector.findHands(img, draw=True)

            if hands:
                landmarks = hands[0]
                
                # Run prediction at intervals
                current_time = time.time()
                if current_time - last_prediction_time >= prediction_interval:
                    # Extract features from landmarks using the shared module
                    features_dict = extract_features_from_landmarks(landmarks, img.shape)
                    
                    if features_dict:
                        # Convert dict to feature vector in the EXACT order the model expects
                        feature_vector = [features_dict[field] for field in FEATURE_FIELDS]
                        
                        # Use feature vector directly (no scaling)
                        feature_df = pd.DataFrame([feature_vector], columns=FEATURE_FIELDS)
                        
                        # Predict class
                        pred_class = model.predict(feature_df)[0]
                        
                        # Handle probabilities (enabled with probability=True during training)
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(feature_df)[0]
                            confidence = np.max(probs)
                            label = f"ASL: {pred_class} ({confidence:.1%})"
                        else:
                            label = f"ASL: {pred_class}"
                            
                        last_prediction_time = current_time

                # Draw the label on the frame
                if label and bbox:
                    x, y, w, h = bbox
                    cv2.putText(
                        img, label, (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
                    )

            # Show the video feed
            cv2.imshow("ASL SVM Real-Time Classification", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

if __name__ == "__main__":
    main()
