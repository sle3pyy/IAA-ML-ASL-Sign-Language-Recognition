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
    Uses all 21 points for normalization and accurate feature calculation.
    """
    if not landmarks or len(landmarks) < 21:
        return None

    wrist_x = landmarks[0][0]
    wrist_y = landmarks[0][1]
    
    # Hand Scale Reference: Wrist (0) to Middle Finger MCP (9)
    # This is a more stable scale than the previous raw_1 to raw_13
    hand_scale_px = distance(landmarks[0], landmarks[9])
    
    # Hand Scale Validation
    if image_shape:
        h, w = image_shape[:2]
        norm_scale = hand_scale_px / min(h, w)
        if norm_scale < 0.03 or norm_scale > 0.8:
            return None
    elif hand_scale_px == 0:
        return None
            
    # Normalize all 21 points relative to wrist and scale
    normalized_points = {}
    for idx in range(21):
        x = (landmarks[idx][0] - wrist_x) / hand_scale_px
        y = (landmarks[idx][1] - wrist_y) / hand_scale_px
        normalized_points[idx] = (x, y)
        
    # Standard Fingertips (Indices: 4, 8, 12, 16, 20)
    t_tip = normalized_points[4]
    i_tip = normalized_points[8]
    m_tip = normalized_points[12]
    r_tip = normalized_points[16]
    p_tip = normalized_points[20]
    
    # MCP Joints (Base of fingers: 5, 9, 13, 17)
    i_mcp = normalized_points[5]
    m_mcp = normalized_points[9]
    r_mcp = normalized_points[13]
    p_mcp = normalized_points[17]

    # Step 1: Distance-based features (normalized)
    thumb_index_dist = distance(t_tip, i_tip)
    index_middle_dist = distance(i_tip, m_tip)
    middle_ring_dist = distance(m_tip, r_tip)
    ring_pinky_dist = distance(r_tip, p_tip)
    
    thumb_middle_dist = distance(t_tip, m_tip)
    thumb_ring_dist = distance(t_tip, r_tip)
    thumb_pinky_dist = distance(t_tip, p_tip)
    
    # Step 2: Curl features (Distance from tip to MCP)
    # Smaller value = more curled
    thumb_curl = distance(t_tip, normalized_points[2])
    index_curl = distance(i_tip, i_mcp)
    middle_curl = distance(m_tip, m_mcp)
    ring_curl = distance(r_tip, r_mcp)
    pinky_curl = distance(p_tip, p_mcp)
    
    # Step 3: Height features (Relative Y)
    # Negative values are "higher" in screen coordinates
    thumb_y = t_tip[1]
    index_y = i_tip[1]
    middle_y = m_tip[1]
    ring_y = r_tip[1]
    pinky_y = p_tip[1]
    
    # Step 4: Orientation and Spread
    # Hand Rotation: Wrist (0) to Middle MCP (9)
    palm_direction = vector(normalized_points[0], m_mcp)
    hand_rotation = math.atan2(palm_direction[1], palm_direction[0])
    
    # Thumb Angle relative to palm center
    palm_center = mean_position([i_mcp, m_mcp, r_mcp])
    thumb_vector = vector(palm_center, t_tip)
    thumb_angle = math.atan2(thumb_vector[1], thumb_vector[0])
    
    # Palm Tilt: Thumb CMC (1) to Pinky MCP (17)
    palm_tilt = math.atan2(p_mcp[1] - normalized_points[1][1],
                           p_mcp[0] - normalized_points[1][0])
                           
    # Spread Angles
    spread_index_ring = angle_between(i_tip, m_mcp, r_tip)
    spread_thumb_pinky = angle_between(t_tip, palm_center, p_tip)
    
    # Vertical Variance of all fingertips
    y_variance = np.var([thumb_y, index_y, middle_y, ring_y, pinky_y])
    
    # Assemble feature vector
    features = {
        'thumb_index_dist': thumb_index_dist,
        'index_middle_dist': index_middle_dist,
        'middle_ring_dist': middle_ring_dist,
        'ring_pinky_dist': ring_pinky_dist,
        'thumb_middle_dist': thumb_middle_dist,
        'thumb_ring_dist': thumb_ring_dist,
        'thumb_pinky_dist': thumb_pinky_dist,
        'thumb_curl': thumb_curl,
        'index_curl': index_curl,
        'middle_curl': middle_curl,
        'ring_curl': ring_curl,
        'pinky_curl': pinky_curl,
        'thumb_y': thumb_y,
        'index_y': index_y,
        'middle_y': middle_y,
        'ring_y': ring_y,
        'pinky_y': pinky_y,
        'hand_rotation': hand_rotation,
        'thumb_angle': thumb_angle,
        'palm_tilt': palm_tilt,
        'spread_index_ring': spread_index_ring,
        'spread_thumb_pinky': spread_thumb_pinky,
        'y_variance': y_variance
    }
    
    if label is not None:
        features['label'] = label
        
    return features
