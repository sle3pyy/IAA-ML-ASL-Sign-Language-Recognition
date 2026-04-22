import cv2
import os
import time
import argparse
import joblib
import numpy as np
from HandTrackingModule import HandDetector
from feature_extraction import extract_features_from_landmarks

# Features order (MUST match the order used during training in train_svm.py)
FEATURE_FIELDS = [
    'thumb_index_dist', 'index_middle_dist', 'middle_ring_dist', 
    'ring_pinky_dist', 'thumb_ring_dist', 'thumb_pinky_dist', 'thumb_curl', 
    'index_curl', 'middle_curl', 'ring_curl', 'pinky_curl', 'thumb_y', 
    'index_y', 'middle_y', 'ring_y', 'pinky_y', 'hand_rotation', 
    'thumb_angle', 'palm_tilt', 'spread_angle_1', 'spread_angle_2', 'y_variance'
]

def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Real-time ASL Classification using SVM")
    parser.add_argument(
        "--model", default="asl_svm_model.pkl",
        help="Name of the saved SVM model in src/models/"
    )
    parser.add_argument(
        "--scaler", default="asl_scaler.pkl",
        help="Name of the saved Scaler in src/models/"
    )
    args = parser.parse_args()

    # Paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(CURRENT_DIR, "models")
    model_path = os.path.join(MODEL_DIR, args.model)
    scaler_path = os.path.join(MODEL_DIR, args.scaler)

    # Load Model and Scaler
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error: Model or Scaler not found in {MODEL_DIR}")
        print(f"Looked for: {args.model} and {args.scaler}")
        return

    print(f"Loading SVM model from {model_path}...")
    model = joblib.load(model_path)
    print(f"Loading Scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)

    # Initialize Camera and Detector
    cap = cv2.VideoCapture(0)
    detector = HandDetector(num_hands=1)
    
    label = ""
    last_prediction_time = 0
    prediction_interval = 0.1  # seconds

    print("Starting webcam... Press 'q' to quit.")

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
                    
                    # Scale features
                    features_scaled = scaler.transform([feature_vector])
                    
                    # Predict class
                    pred_class = model.predict(features_scaled)[0]
                    
                    # Handle probabilities (enabled with probability=True during training)
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(features_scaled)[0]
                        confidence = np.max(probs)
                        # Only show label if confidence is high enough
                        if confidence > 0.7:
                            label = f"ASL: {pred_class} ({confidence:.1%})"
                        else:
                            label = "Searching..."
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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
