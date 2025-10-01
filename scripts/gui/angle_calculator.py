"""
Joint angle calculation utilities
"""
import numpy as np

# COCO17 keypoint indices
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6
LEFT_ELBOW_IDX = 7
RIGHT_ELBOW_IDX = 8
LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12
LEFT_KNEE_IDX = 13
RIGHT_KNEE_IDX = 14
LEFT_ANKLE_IDX = 15
RIGHT_ANKLE_IDX = 16


def calculate_angle(point1, point2, point3):
    """
    Calculate angle at point2 formed by point1-point2-point3.
    
    Args:
        point1: (x, y) coordinates of first point
        point2: (x, y) coordinates of vertex point
        point3: (x, y) coordinates of third point
        
    Returns:
        angle_deg: Angle in degrees (0° = straight, 180° = fully bent)
    """
    # Convert to numpy arrays
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    
    # Calculate vectors from vertex to other points
    vec1 = p1 - p2
    vec2 = p3 - p2
    
    # Calculate angle using dot product
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # Clamp to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Calculate angle in radians then convert to degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    # Return supplementary angle (so 0° = straight, 180° = fully bent)
    return round(180 - angle_deg)


def calculate_hip_angle(keypoints, scores, side='left', confidence_threshold=0.5):
    """
    Calculate hip angle: shoulder - hip - knee
    
    Args:
        keypoints: Array of keypoint coordinates [17, 2]
        scores: Array of keypoint confidence scores [17]
        side: 'left' or 'right'
        confidence_threshold: Minimum confidence for valid calculation
        
    Returns:
        angle: Hip angle in degrees, or None if keypoints not confident
    """
    if side == 'left':
        shoulder_idx = LEFT_SHOULDER_IDX
        hip_idx = LEFT_HIP_IDX
        knee_idx = LEFT_KNEE_IDX
    else:
        shoulder_idx = RIGHT_SHOULDER_IDX
        hip_idx = RIGHT_HIP_IDX
        knee_idx = RIGHT_KNEE_IDX
    
    # Check confidence scores
    if (scores[shoulder_idx] < confidence_threshold or 
        scores[hip_idx] < confidence_threshold or 
        scores[knee_idx] < confidence_threshold):
        return None
    
    # Get keypoint positions
    shoulder = keypoints[shoulder_idx]
    hip = keypoints[hip_idx]
    knee = keypoints[knee_idx]
    
    try:
        angle = calculate_angle(shoulder, hip, knee)
        return angle
    except Exception as e:
        print(f"Error calculating hip angle: {e}")
        return None


def calculate_knee_angle(keypoints, scores, side='left', confidence_threshold=0.5):
    """
    Calculate knee angle: hip - knee - ankle
    
    Args:
        keypoints: Array of keypoint coordinates [17, 2]
        scores: Array of keypoint confidence scores [17]
        side: 'left' or 'right'
        confidence_threshold: Minimum confidence for valid calculation
        
    Returns:
        angle: Knee angle in degrees, or None if keypoints not confident
    """
    if side == 'left':
        hip_idx = LEFT_HIP_IDX
        knee_idx = LEFT_KNEE_IDX
        ankle_idx = LEFT_ANKLE_IDX
    else:
        hip_idx = RIGHT_HIP_IDX
        knee_idx = RIGHT_KNEE_IDX
        ankle_idx = RIGHT_ANKLE_IDX
    
    # Check confidence scores
    if (scores[hip_idx] < confidence_threshold or 
        scores[knee_idx] < confidence_threshold or 
        scores[ankle_idx] < confidence_threshold):
        return None
    
    # Get keypoint positions
    hip = keypoints[hip_idx]
    knee = keypoints[knee_idx]
    ankle = keypoints[ankle_idx]
    
    try:
        angle = calculate_angle(hip, knee, ankle)
        return angle
    except Exception as e:
        print(f"Error calculating knee angle: {e}")
        return None


def calculate_elbow_angle(keypoints, scores, side='left', confidence_threshold=0.5):
    """
    Calculate elbow angle: shoulder - elbow - wrist
    
    Args:
        keypoints: Array of keypoint coordinates [17, 2]
        scores: Array of keypoint confidence scores [17]
        side: 'left' or 'right'
        confidence_threshold: Minimum confidence for valid calculation
        
    Returns:
        angle: Elbow angle in degrees, or None if keypoints not confident
    """
    if side == 'left':
        shoulder_idx = LEFT_SHOULDER_IDX
        elbow_idx = LEFT_ELBOW_IDX
        wrist_idx = LEFT_WRIST_IDX
    else:
        shoulder_idx = RIGHT_SHOULDER_IDX
        elbow_idx = RIGHT_ELBOW_IDX
        wrist_idx = RIGHT_WRIST_IDX
    
    # Check confidence scores
    if (scores[shoulder_idx] < confidence_threshold or 
        scores[elbow_idx] < confidence_threshold or 
        scores[wrist_idx] < confidence_threshold):
        return None
    
    # Get keypoint positions
    shoulder = keypoints[shoulder_idx]
    elbow = keypoints[elbow_idx]
    wrist = keypoints[wrist_idx]
    
    try:
        angle = calculate_angle(shoulder, elbow, wrist)
        return angle
    except Exception as e:
        print(f"Error calculating elbow angle: {e}")
        return None


def calculate_joint_angle(joint_name, keypoints, scores, side='left', confidence_threshold=0.5):
    """
    Calculate angle for specified joint.
    
    Args:
        joint_name: 'hip', 'knee', or 'elbow'
        keypoints: Array of keypoint coordinates [17, 2]
        scores: Array of keypoint confidence scores [17]
        side: 'left' or 'right'
        confidence_threshold: Minimum confidence for valid calculation
        
    Returns:
        angle: Joint angle in degrees, or None if calculation fails
    """
    joint_name = joint_name.lower()
    
    if joint_name == 'hip':
        return calculate_hip_angle(keypoints, scores, side, confidence_threshold)
    elif joint_name == 'knee':
        return calculate_knee_angle(keypoints, scores, side, confidence_threshold)
    elif joint_name == 'elbow':
        return calculate_elbow_angle(keypoints, scores, side, confidence_threshold)
    else:
        raise ValueError(f"Unknown joint name: {joint_name}")
