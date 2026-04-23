Following our transfer learning implementations, we now move into phase 2, focusing on training a linear SVM model with our ASL data as suggested by the professor.

## Data Processing and Model Training Methodology

### 1. Feature Extraction (MediaPipe Hand Landmarks)

Instead of processing images, our SVM model operates on **geometric hand landmarks** extracted via MediaPipe. This significantly reduces the input dimensionality while preserving essential spatial information. The features stored in `handmarks.csv` include:

- **Normalized Distances:** Relative distances between finger tips and key joints.
- **Finger Curls:** Quantified degree of finger flexion.
- **Hand Orientation:** Palm rotation and tilt relative to the wrist.
- **Spatial Variance:** Variability in Y-coordinates to differentiate between open and closed hand states.

### 2. Hand Size Normalization (Geometric Scaling)

To ensure the model is **scale-invariant**, we normalize coordinates during the extraction phase. This allows the system to recognize signs correctly regardless of whether the user is close to or far from the camera.

- **Reference Vector:** We use the distance between **Landmark 1 (Thumb CMC)** and **Landmark 13 (Ring finger MCP)** as the baseline "hand scale." This span is chosen for its relative stability across different hand gestures.
- **Coordinate Translation:** All landmarks are translated relative to the **Wrist (Landmark 0)**, which becomes the origin (0,0).
- **Unit Scaling:** Every raw pixel coordinate is divided by the reference hand scale, converting distances into "hand-relative units."
- **Validation:** Frames where the hand is too small (<5% of image) or too large (>80% of image) are automatically discarded to maintain data integrity.

### 3. Data Preparation and Stratification

To ensure the model learns correctly across all classes:

- **Stratified Split:** Data is divided into Training (80%) and Validation (20%) using stratification to maintain class proportions.
- **Class Balancing:** We use `class_weight='balanced'` in the SVM configuration to handle any class distribution skews.

### 4. Preprocessing: Feature Scaling

Since SVM is a distance-based algorithm, **Feature Scaling** is mandatory. We employ `StandardScaler` to normalize the data (mean=0, variance=1). This prevents features with larger numerical ranges from disproportionately influencing the model's decision boundaries.

### 5. Hyperparameter Optimization (Grid Search & CV)

To identify the optimal configuration for the RBF kernel, we use **GridSearchCV** integrated with **5-fold Cross-Validation**. This rigorous process ensures that our chosen parameters are not just "lucky" for one specific data split.

- **The 5-Fold Process:** The training data is divided into five equal subsets. The model is trained and validated five times; in each iteration, a different fold serves as the validation set while the remaining four are used for training. The final score for a parameter combination is the average of these five runs.
- **Preventing Overfitting:** By validating across the entire training set, we minimize the risk of overfitting to noise in any single subset, leading to a model that generalizes better to new users.
- **Weighted Scoring:** We optimize for `f1_weighted` during the search. This ensures that the grid search prioritizes parameters that achieve high precision and recall across all ASL letters, rather than just maximizing raw accuracy.

The search explores different combinations of:

- **C (Regularization):** Controls the trade-off between decision boundary smoothness and correct classification of training points.
- **Gamma:** Defines how far the influence of a single training example reaches.

### 6. Evaluation with Robust Metrics

Performance is not measured by raw accuracy alone:

- **Balanced Accuracy:** The average recall of all classes, providing a fair assessment even with minor imbalances.
- **Macro F1-Score:** The harmonic mean of precision and recall, averaged across all labels to ensure high performance on every letter.
- **Confusion Matrix Analysis:** Used specifically to identify common misclassifications

Our first attempt at training the model was quite successful, our accuracy
