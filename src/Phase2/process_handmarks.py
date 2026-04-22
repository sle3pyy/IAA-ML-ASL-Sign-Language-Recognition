import os
import sys
import cv2
import numpy as np
import csv
import logging

# Add src folder to path to import HandTrackingModule and feature_extraction
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # This is 'src'
sys.path.append(BASE_DIR)
from utils.HandTrackingModule import HandDetector
from utils.feature_extraction import extract_features_from_landmarks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define the paths
DATA_DIR = os.path.join(BASE_DIR, "Data", "collected")
OUTPUT_CSV = os.path.join(BASE_DIR, "Data", "handmarks.csv")

# Initialize HandDetector from HandTrackingModule
detector = HandDetector(num_hands=1)

def process_image(image_path, label):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        hands, img_out, bbox = detector.findHands(image, draw=False)
        
        if not hands:
            return None
            
        # Extract features using shared module
        features = extract_features_from_landmarks(hands[0], image.shape, label)
        
        return features
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None

def main():
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory {DATA_DIR} not found.")
        return
        
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    fieldnames = [
        'label', 'thumb_index_dist', 'index_middle_dist', 'middle_ring_dist', 
        'ring_pinky_dist', 'thumb_ring_dist', 'thumb_pinky_dist', 'thumb_curl', 
        'index_curl', 'middle_curl', 'ring_curl', 'pinky_curl', 'thumb_y', 
        'index_y', 'middle_y', 'ring_y', 'pinky_y', 'hand_rotation', 
        'thumb_angle', 'palm_tilt', 'spread_angle_1', 'spread_angle_2', 'y_variance'
    ]
    
    total_extracted = 0
    class_stats = {}
    
    with open(OUTPUT_CSV, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for class_dir in sorted(os.listdir(DATA_DIR)):
            class_path = os.path.join(DATA_DIR, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            logger.info(f"Processing class: {class_dir}...")
            success_count = 0
            total_count = 0
            
            for img_name in sorted(os.listdir(class_path)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    total_count += 1
                    img_path = os.path.join(class_path, img_name)
                    features = process_image(img_path, class_dir)
                    if features:
                        writer.writerow(features)
                        success_count += 1
                        total_extracted += 1
                        
            class_stats[class_dir] = {"success": success_count, "total": total_count}
            logger.info(f"Class {class_dir} complete: {success_count}/{total_count} extracted.")
            
    if total_extracted > 0:
        logger.info(f"Successfully processed and saved features to {OUTPUT_CSV}")
        logger.info(f"Total rows extracted: {total_extracted}")
        logger.info("Per-class breakdown:")
        for cls, stats in class_stats.items():
            if stats['total'] > 0:
                percentage = (stats['success'] / stats['total']) * 100
            else:
                percentage = 0.0
            logger.info(f"  {cls}: {stats['success']} / {stats['total']} ({percentage:.1f}%)")
    else:
        logger.warning("No features extracted.")

if __name__ == "__main__":
    main()
