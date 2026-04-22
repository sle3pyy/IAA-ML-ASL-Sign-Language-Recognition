# ASL Letter Classification: Hand Landmark Processing Guide

## Overview

This document explains how we'll process MediaPipe hand landmarks to extract meaningful features for classifying ASL letters A, B, and C using an SVM classifier.

## What is MediaPipe Hand Detection?

MediaPipe is a framework that detects and tracks hands in images/video. It identifies 21 3D landmarks on each hand (x, y, z coordinates where z is depth/confidence). These landmarks represent:

- 1 wrist point
- 4 finger base points (one for each finger)
- 3 points per finger (base knuckle, middle knuckle, tip)

## The 21 MediaPipe Hand Landmarks

MediaPipe labels landmarks 0-20. Here's the complete map:

```
Landmark Index | Name                      | Location
0              | WRIST                     | Wrist center
1              | THUMB_CMC                 | Thumb base (carpometacarpal joint)
2              | THUMB_MCP                 | Thumb middle (metacarpophalangeal joint)
3              | THUMB_IP                  | Thumb tip (interphalangeal joint)
4              | INDEX_FINGER_MCP          | Index finger base
5              | INDEX_FINGER_PIP          | Index finger middle
6              | INDEX_FINGER_DIP          | Index finger tip
7              | MIDDLE_FINGER_MCP         | Middle finger base
8              | MIDDLE_FINGER_PIP         | Middle finger middle
9              | MIDDLE_FINGER_DIP         | Middle finger tip
10             | RING_FINGER_MCP           | Ring finger base
11             | RING_FINGER_PIP           | Ring finger middle
12             | RING_FINGER_DIP           | Ring finger tip
13             | PINKY_FINGER_MCP          | Pinky finger base
14             | PINKY_FINGER_PIP          | Pinky finger middle
15             | PINKY_FINGER_DIP          | Pinky finger tip
16-20          | THUMB through PINKY tips  | Alias for tips (3, 6, 9, 12, 15)
```

## Which Landmarks Do We Use and Why?

We'll focus on **key landmarks that directly relate to distinguishing A, B, and C** . Not all 21 landmarks are equally important.

### Critical Landmarks: Fingertips (Indices 3, 6, 9, 12, 15)

**Why these matter:**

- ASL letters A, B, and C are primarily distinguished by finger position and extension
- Fingertips show whether fingers are extended (open) or curled (closed)
- The spatial arrangement of fingertips creates the visual signature of each letter

**What each fingertip tells us:**

- **Thumb tip (3)** : Position for A (thumb often forward/sideways) vs B/C (folded)
- **Index tip (6)** : Extended upward in B, curled in A, spread outward in C
- **Middle tip (9)** : Extended upward in B, curled in A, positioned centrally in C
- **Ring tip (12)** : Extended upward in B, curled in A, spread outward in C
- **Pinky tip (15)** : Extended upward in B, curled in A, spread outward in C

### Secondary Landmarks: Finger Bases (Indices 1, 4, 7, 10, 13)

**Why we include these:**

- Establish the "closed" position of curled fingers in letter A
- Help measure finger curl depth (tip to base distance = curl amount)
- Define the palm plane for orientation calculations
- Provide anchor points for normalization

**What each base tells us:**

- **Thumb base (1)** : Palm anchor, defines hand width
- **Index base (4)** : Reference for index finger extension
- **Middle base (7)** : Reference for middle finger extension; hand center reference
- **Ring base (10)** : Reference for ring finger extension
- **Pinky base (13)** : Palm anchor, defines hand width

### Tertiary Landmark: Wrist (Index 0)

**Why we include this:**

- Hand center reference point for all normalization
- Establishes hand orientation and rotation angle
- Helps detect hand roll/tilt relative to camera
- Serves as scale reference (hand size)

## Why We Skip Some Landmarks

We don't use the middle knuckle points (indices 2, 5, 8, 11, 14) because:

- They're redundant — curl can be measured from tip-to-base distance
- They add noise and increase dimensionality without discrimination value
- The two extreme points (base and tip) capture the essential curl information
- Computation is faster with fewer features; SVM benefits from lean feature sets

## The Processing Pipeline

### Step 1: Extract Raw Landmarks from Frame

```python
# Pseudo-code
results = hand_detector.detect(image_frame)
landmarks = results.hand_landmarks[0]  # Get first hand

# Extract only the landmarks we care about
wrist = landmarks[0]  # x, y, z
thumb_tip = landmarks[3]
index_base = landmarks[4]
index_tip = landmarks[6]
middle_base = landmarks[7]
middle_tip = landmarks[9]
ring_base = landmarks[10]
ring_tip = landmarks[12]
pinky_base = landmarks[13]
pinky_tip = landmarks[15]
```

### Step 2: Normalize Coordinates

Raw MediaPipe coordinates are in image pixel space (0-1 range). We normalize them relative to the hand to make features scale and position invariant.

**Normalization approach:**

- Use wrist as origin (0, 0)
- Divide all coordinates by hand scale (palm width)
- This makes features independent of hand size and camera distance

```python
# Hand scale = distance between thumb and pinky bases
hand_scale = distance(landmarks[1], landmarks[13])

# Normalize all points relative to wrist
normalized_points = {}
for landmark_idx in [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15]:
    x = (landmarks[landmark_idx].x - landmarks[0].x) / hand_scale
    y = (landmarks[landmark_idx].y - landmarks[0].y) / hand_scale
    normalized_points[landmark_idx] = (x, y)
```

### Step 3: Calculate Distance-Based Features

From the normalized landmarks, calculate distances between key points.

**Finger-to-finger distances (6 features):**

- Thumb tip to index tip
- Index tip to middle tip
- Middle tip to ring tip
- Ring tip to pinky tip
- Thumb tip to ring tip (diagonal)
- Thumb tip to pinky tip (diagonal)

Why distances: They capture finger spread. A tight fist (A) has very small distances. Open hand (B) has large distances.

```python
thumb_tip = normalized_points[3]
index_tip = normalized_points[6]
middle_tip = normalized_points[9]
ring_tip = normalized_points[12]
pinky_tip = normalized_points[15]

thumb_index_dist = distance(thumb_tip, index_tip)
index_middle_dist = distance(index_tip, middle_tip)
middle_ring_dist = distance(middle_tip, ring_tip)
ring_pinky_dist = distance(ring_tip, pinky_tip)
thumb_ring_dist = distance(thumb_tip, ring_tip)
thumb_pinky_dist = distance(thumb_tip, pinky_tip)
```

### Step 4: Calculate Curl Features

Curl = how much a finger is bent. Measured as distance from tip to base.

**Finger curl (5 features):**

- Thumb curl: distance(thumb_tip, thumb_base)
- Index curl: distance(index_tip, index_base)
- Middle curl: distance(middle_tip, middle_base)
- Ring curl: distance(ring_tip, ring_base)
- Pinky curl: distance(pinky_tip, pinky_base)

Why curl: Letter A has all fingers curled (small curl distances). Letter B has all fingers extended (large curl distances). Letter C has mixed curl.

```python
thumb_curl = distance(normalized_points[3], normalized_points[1])
index_curl = distance(normalized_points[6], normalized_points[4])
middle_curl = distance(normalized_points[9], normalized_points[7])
ring_curl = distance(normalized_points[12], normalized_points[10])
pinky_curl = distance(normalized_points[15], normalized_points[13])
```

### Step 5: Calculate Height Features

Y-coordinate of each fingertip tells us if finger is extended upward or curled down.

**Finger heights (5 features):**

- Thumb y-position (relative to wrist)
- Index y-position
- Middle y-position
- Ring y-position
- Pinky y-position

Why heights: Letter B has all fingers extended upward (high y-values). Letter A has fingers curled (lower y-values). This directly reflects finger extension direction.

```python
thumb_y = normalized_points[3][1]
index_y = normalized_points[6][1]
middle_y = normalized_points[9][1]
ring_y = normalized_points[12][1]
pinky_y = normalized_points[15][1]
```

### Step 6: Calculate Hand Orientation

Hand rotation and thumb position help distinguish letters by hand orientation.

**Orientation features (3 features):**

- Hand rotation angle: angle of palm normal (vector from wrist to middle base)
- Thumb angle: angle of thumb relative to hand center
- Palm tilt: how much hand is tilted left/right

Why orientation: Letters A, B, C can be signed with hand rotated. These features help normalize for rotation while capturing intentional hand tilt.

```python
# Hand rotation = angle from wrist to middle finger base
palm_direction = vector(wrist, middle_base)
hand_rotation = atan2(palm_direction.y, palm_direction.x)

# Thumb angle = angle from palm center to thumb tip
palm_center = mean_position([index_base, middle_base, ring_base])
thumb_vector = vector(palm_center, thumb_tip)
thumb_angle = atan2(thumb_vector.y, thumb_vector.x)

# Palm tilt = angle of thumb base line
palm_tilt = atan2(pinky_base.y - thumb_base.y,
                   pinky_base.x - thumb_base.x)
```

### Step 7: Calculate Spread Features

How spread out are the fingers across the hand?

**Spread features (3 features):**

- Spread angle 1: angle from index to ring (via middle)
- Spread angle 2: angle from thumb to pinky
- Finger variance: how much fingers vary in y-position

Why spread: Letter C has specific finger spread. Letter B has all fingers together. Letter A has fingers curled in.

```python
# Spread angle = cone angle between extended fingers
angle_index_to_ring = angle_between(index_tip, middle_base, ring_tip)
angle_thumb_to_pinky = angle_between(thumb_tip, palm_center, pinky_tip)

# Finger y-variance: do all fingers have same height?
y_positions = [index_y, middle_y, ring_y, pinky_y]
y_variance = variance(y_positions)
```

## Summary of Selected Landmarks

**Used directly:**

- Landmark 0: Wrist (for normalization reference)
- Landmark 1: Thumb base (for curl measurement, hand width)
- Landmark 3: Thumb tip (for distance and position)
- Landmark 4: Index base (for curl measurement)
- Landmark 6: Index tip (for distance and position)
- Landmark 7: Middle base (for curl measurement, hand center)
- Landmark 9: Middle tip (for distance and position)
- Landmark 10: Ring base (for curl measurement)
- Landmark 12: Ring tip (for distance and position)
- Landmark 13: Pinky base (for curl measurement, hand width)
- Landmark 15: Pinky tip (for distance and position)

**Total: 11 out of 21 landmarks used**

## Why This Selection Works for A, B, C

### Letter A Recognition

- Small distances between all fingertips (curled fist)
- Small curl values for all fingers (fingers bent)
- Thumb positioned differently (often forward)
- Low finger spread angles
- Wrist and palm position stable

### Letter B Recognition

- Large curl values for all fingers (extended)
- Large distances between finger tips (spread upward)
- High y-positions for all fingertips (fingers up)
- Minimal spread angles (fingers together vertically)
- Wrist usually rotated palm-forward

### Letter C Recognition

- Moderate curl values (some fingers extended, some bent)
- Index and ring fingers spread wide (C-shape)
- Middle and ring tips positioned to form curve
- Specific angle relationships between fingers
- Thumb positioned outside the curve

## Feature Matrix Output

From one frame, we extract a feature vector of approximately 20-25 values:

```
[thumb_index_dist, index_middle_dist, middle_ring_dist, ring_pinky_dist,
 thumb_ring_dist, thumb_pinky_dist,
 thumb_curl, index_curl, middle_curl, ring_curl, pinky_curl,
 thumb_y, index_y, middle_y, ring_y, pinky_y,
 hand_rotation, thumb_angle, palm_tilt,
 spread_angle_1, spread_angle_2, y_variance]
```

This vector becomes one row in your CSV file, with the label column indicating A, B, or C.

## Processing Multiple Frames

For robustness, capture 5-10 frames per sign instance. This accounts for:

- Natural hand shake/jitter
- Slight variations in how people sign the letter
- Temporal smoothing (average features across frames)

Average the features across frames or create separate CSV rows for each frame with its label. The second approach (one row per frame) gives more training data.

## Confidence Filtering

MediaPipe provides a confidence score (0-1) for each landmark. Filter out frames where any critical landmark has confidence < 0.5:

```python
confidence_threshold = 0.5
for landmark_idx in [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15]:
    if landmarks[landmark_idx].z < confidence_threshold:
        # Skip this frame or request user to reposition hand
        return None
```

## Summary

We use 11 strategic landmarks out of 21 available:

- **Wrist (1)** : Normalization reference
- **Finger bases (4)** : Curl measurement and palm position
- **Finger tips (5)** : Distance and position features
- **Middle base (1)** : Hand center reference

This subset captures the essential geometry that distinguishes A, B, and C while keeping the feature space lean for optimal SVM performance.
