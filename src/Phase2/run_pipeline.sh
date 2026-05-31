#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "--------------------------------------------------"
echo "Starting ASL SVM Pipeline"
echo "--------------------------------------------------"

# 1. Process training data
echo "Step 1: Processing training handmarks from collected data..."
python3 process_handmarks.py --input ../Data/collected/ --output ../Data/handmarks.csv

# 2. Process validation data
echo "Step 2: Processing validation handmarks from val data..."
python3 process_handmarks.py --input ../Data/val/ --output ../Data/handmarks_val.csv

# 3. Train the SVM model
echo "Step 3: Training SVM model (v4) using validation data..."
python3 train_svm.py --val_data ../Data/handmarks_val.csv

echo "--------------------------------------------------"
echo "Pipeline completed successfully!"
echo "--------------------------------------------------"
