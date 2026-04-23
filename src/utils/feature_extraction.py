import math
import numpy as np

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def vector(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1])

def mean_position(points):
    x = sum([p[0] for p in points]) / len(points)
    y = sum([p[1] for p in points]) / len(points)
    return (x, y)

def angle_between(p1, p2, p3):
    # Angle formed by p1 - p2 - p3
    v1 = vector(p2, p1)
    v2 = vector(p2, p3)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    det = v1[0]*v2[1] - v1[1]*v2[0]
    return math.atan2(det, dot)

def extract_features_from_landmarks(landmarks, image_shape=None, label=None):
    """
    Extracts features from a list of 21 hand landmarks (pixel coordinates).
    landmarks: list of (x, y) tuples
    image_shape: (height, width, channels)
    label: optional class label to include in the dictionary
    """
    wrist_x = landmarks[0][0]
    wrist_y = landmarks[0][1]
    
    raw_1 = landmarks[1]
    raw_13 = landmarks[13]
    
    hand_scale_px = distance(raw_1, raw_13)
    
    # Hand Scale Validation
    if image_shape:
        h, w = image_shape[:2]
        norm_scale = hand_scale_px / min(h, w)
        if norm_scale < 0.05 or norm_scale > 0.8:
            return None
    elif hand_scale_px == 0:
        return None
            
    normalized_points = {}
    for idx in [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15]:
        x = (landmarks[idx][0] - wrist_x) / hand_scale_px
        y = (landmarks[idx][1] - wrist_y) / hand_scale_px
        normalized_points[idx] = (x, y)
        
    # Step 3: Distance-based features
    thumb_tip = normalized_points[3]
    index_tip = normalized_points[6]
    middle_tip = normalized_points[9]
    ring_tip = normalized_points[12]
    pinky_tip = normalized_points[15]
    
    # Landmarks 8 and 5
    index_tip_8 = normalized_points[8]
    index_mcp_5 = normalized_points[5]

    thumb_index_dist = distance(thumb_tip, index_tip)
    thumb_index_8_dist = distance(thumb_tip, index_tip_8)
    index_middle_dist = distance(index_tip, middle_tip)
    middle_ring_dist = distance(middle_tip, ring_tip)
    ring_pinky_dist = distance(ring_tip, pinky_tip)
    thumb_ring_dist = distance(thumb_tip, ring_tip)
    thumb_pinky_dist = distance(thumb_tip, pinky_tip)
    
    # Step 4: Curl features
    thumb_curl = distance(normalized_points[3], normalized_points[1])
    index_curl = distance(normalized_points[6], normalized_points[4])
    index_curl_8 = distance(index_tip_8, index_mcp_5)
    middle_curl = distance(normalized_points[9], normalized_points[7])
    ring_curl = distance(normalized_points[12], normalized_points[10])
    pinky_curl = distance(normalized_points[15], normalized_points[13])
    
    # Step 5: Height features
    thumb_y = normalized_points[3][1]
    index_y = normalized_points[6][1]
    index_8_y = index_tip_8[1]
    index_5_y = index_mcp_5[1]
    middle_y = normalized_points[9][1]
    ring_y = normalized_points[12][1]
    pinky_y = normalized_points[15][1]
    
    # Step 6: Hand Orientation
    wrist = normalized_points[0]
    middle_base = normalized_points[7]
    palm_direction = vector(wrist, middle_base)
    hand_rotation = math.atan2(palm_direction[1], palm_direction[0])
    
    index_base = normalized_points[4]
    ring_base = normalized_points[10]
    palm_center = mean_position([index_base, middle_base, ring_base])
    thumb_vector = vector(palm_center, thumb_tip)
    thumb_angle = math.atan2(thumb_vector[1], thumb_vector[0])
    
    thumb_base = normalized_points[1]
    pinky_base = normalized_points[13]
    palm_tilt = math.atan2(pinky_base[1] - thumb_base[1],
                           pinky_base[0] - thumb_base[0])
                           
    # Step 7: Spread Features
    angle_index_to_ring = angle_between(index_tip, middle_base, ring_tip)
    angle_thumb_to_pinky = angle_between(thumb_tip, palm_center, pinky_tip)
    
    y_positions = [index_y, middle_y, ring_y, pinky_y]
    y_variance = np.var(y_positions)
    
    # Assemble feature vector
    features = {
        'thumb_index_dist': thumb_index_dist,
        'thumb_index_8_dist': thumb_index_8_dist,
        'index_middle_dist': index_middle_dist,
        'middle_ring_dist': middle_ring_dist,
        'ring_pinky_dist': ring_pinky_dist,
        'thumb_ring_dist': thumb_ring_dist,
        'thumb_pinky_dist': thumb_pinky_dist,
        'thumb_curl': thumb_curl,
        'index_curl': index_curl,
        'index_curl_8': index_curl_8,
        'middle_curl': middle_curl,
        'ring_curl': ring_curl,
        'pinky_curl': pinky_curl,
        'thumb_y': thumb_y,
        'index_y': index_y,
        'index_8_y': index_8_y,
        'index_5_y': index_5_y,
        'middle_y': middle_y,
        'ring_y': ring_y,
        'pinky_y': pinky_y,
        'hand_rotation': hand_rotation,
        'thumb_angle': thumb_angle,
        'palm_tilt': palm_tilt,
        'spread_angle_1': angle_index_to_ring,
        'spread_angle_2': angle_thumb_to_pinky,
        'y_variance': y_variance
    }
    
    if label is not None:
        features['label'] = label
        
    return features
