import cv2
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector:
    def __init__(self, model_path=None, num_hands=1, detection_confidence=0.5, tracking_confidence=0.5):
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "hand_landmarker.task")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Define the 21 hand connections manually (standard Mediapipe skeleton)
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (9, 10), (10, 11), (11, 12),         # Middle
            (13, 14), (14, 15), (15, 16),        # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17)            # Palm
        ]

    def findHands(self, img, draw=False, offset=20):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        bbox = None
        
        # Detect landmarks
        detection_result = self.detector.detect(mp_image)
        
        all_hands = []
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                h, w, _ = img.shape
                
                # Convert normalized landmarks to pixel coordinates
                pixel_landmarks = []
                for landmark in hand_landmarks:
                    px_x = int(landmark.x * w)
                    px_y = int(landmark.y * h)
                    pixel_landmarks.append((px_x, px_y))
                
                all_hands.append(pixel_landmarks)

                # 1. Get all x and y coordinates in pixels
                x_coords = [int(lm.x * w) for lm in hand_landmarks]
                y_coords = [int(lm.y * h) for lm in hand_landmarks]

                # 2. Calculate the bounding box with offset and boundary checks
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                x1 = max(0, x_min - offset)
                y1 = max(0, y_min - offset)
                x2 = min(w, x_max + offset)
                y2 = min(h, y_max + offset)
                
                bw, bh = x2 - x1, y2 - y1
                bbox = [x1, y1, bw, bh]

                if draw:
                    # Draw Connections (Lines)
                    for connection in self.HAND_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        cv2.line(img, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (255, 0, 0), 2)

                    # Draw Landmarks (Dots)
                    for point in pixel_landmarks:
                        cv2.circle(img, point, 4, (0, 255, 0), -1)

                    # 3. Draw it to verify
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
        return all_hands, img, bbox

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--visuals', action='store_true', default=False, help='Show visuals')
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while cap.isOpened():
        success, img = cap.read()
        if not success: break
        
        hands, img, bbox = detector.findHands(img, draw=args.visuals)
        cv2.imshow("Hand Tracking Module", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
