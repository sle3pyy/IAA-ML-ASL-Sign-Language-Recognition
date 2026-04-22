import os
import shutil
import cv2
import numpy as np

from .HandTrackingModule import HandDetector


def process_images():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)

    # Configuration
    input_folder = os.path.join(src_dir, 'Data', 'collected')
    output_folder = os.path.join(src_dir, 'Data', 'processed')
    model_path = os.path.join(current_dir, 'hand_landmarker.task')
    img_size = 299
    
    # Initialize the HandDetector
    detector = HandDetector(model_path=model_path, num_hands=1)

    # Remove previous processed dataset so run starts clean.
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print(f"Removed existing output directory: {output_folder}")

    os.makedirs(output_folder)
    print(f"Created output directory: {output_folder}")

    print(f"Starting processing images from {input_folder}...")

    # Statistics
    total_images = 0
    processed_images = 0
    failed_detections = 0

    # Walk through the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_images += 1
                
                # Get paths
                img_path = os.path.join(root, file)
                
                # Determine target directory maintaining structure
                rel_dir = os.path.relpath(root, input_folder)
                target_dir = os.path.join(output_folder, rel_dir)
                
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                
                target_path = os.path.join(target_dir, file)

                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Error: Could not read image {img_path}")
                    continue

                # Use detector to find the hand bounding box (draw=False for no visuals)
                try:
                    hands, _, bbox = detector.findHands(img, draw=False)
                except Exception as e:
                    print(f"Error processing {img_path} with MediaPipe: {e}")
                    continue

                if bbox:
                    x, y, w, h = bbox
                    
                    # Ensure indices are within image bounds (safety)
                    h_img, w_img, _ = img.shape
                    y1, y2 = max(0, y), min(h_img, y + h)
                    x1, x2 = max(0, x), min(w_img, x + w)
                    
                    # Crop the hand region
                    img_crop = img[y1:y2, x1:x2]
                    
                    if img_crop.size == 0:
                        print(f"Warning: Empty crop for {img_path}")
                        failed_detections += 1
                        continue

                    # Resize to target size
                    img_resized = cv2.resize(img_crop, (img_size, img_size))
                    
                    # Save target
                    cv2.imwrite(target_path, img_resized)
                    processed_images += 1
                    print(f"Processed [{processed_images}/{total_images}]: {file}")
                else:
                    failed_detections += 1
                    print(f"No hand detected in: {img_path}")

    print("\nProcessing Complete!")
    print(f"Total images found: {total_images}")
    print(f"Successfully processed: {processed_images}")
    print(f"Hands not detected: {failed_detections}")

if __name__ == "__main__":
    process_images()
