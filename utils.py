import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from typing import Tuple, List, Dict

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the fingerprint image for better feature extraction.
    
    Args:
        image: Input fingerprint image
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if image is colored
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    image = cv2.equalizeHist(image)
    
    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply adaptive thresholding
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return image

def extract_minutiae(image: np.ndarray) -> List[Dict[str, float]]:
    """
    Extract minutiae points from the fingerprint image.
    
    Args:
        image: Preprocessed fingerprint image
        
    Returns:
        List of dictionaries containing minutiae points and their features
    """
    # Apply local binary pattern for texture analysis
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    
    # Find keypoints using Harris corner detector
    corners = cv2.goodFeaturesToTrack(image, 100, 0.01, 10)
    if corners is None:
        return []
    
    # Convert corners to integer coordinates
    corners = np.int32(corners)
    
    minutiae = []
    for corner in corners:
        x, y = corner.ravel()
        # Extract local features around the corner
        patch = lbp[max(0, y-5):min(lbp.shape[0], y+6),
                   max(0, x-5):min(lbp.shape[1], x+6)]
        if patch.size > 0:
            minutiae.append({
                'x': float(x),
                'y': float(y),
                'orientation': float(np.mean(patch)),
                'quality': float(np.std(patch))
            })
    
    return minutiae

def match_fingerprints(features1: List[Dict[str, float]], 
                      features2: List[Dict[str, float]], 
                      threshold: float = 0.8) -> Tuple[bool, float]:
    """
    Match two sets of fingerprint features.
    
    Args:
        features1: Features from first fingerprint
        features2: Features from second fingerprint
        threshold: Matching threshold
        
    Returns:
        Tuple of (match_result, confidence_score)
    """
    if not features1 or not features2:
        return False, 0.0
    
    # Convert features to numpy arrays for easier computation
    points1 = np.array([[f['x'], f['y'], f['orientation']] for f in features1])
    points2 = np.array([[f['x'], f['y'], f['orientation']] for f in features2])
    
    # Normalize coordinates to [0,1] range for better comparison
    def normalize_points(points):
        if len(points) == 0:
            return points
        mins = np.min(points[:, :2], axis=0)
        maxs = np.max(points[:, :2], axis=0)
        points[:, :2] = (points[:, :2] - mins) / (maxs - mins + 1e-6)
        return points
    
    points1 = normalize_points(points1)
    points2 = normalize_points(points2)
    
    # Calculate pairwise distances (spatial + orientation)
    distances = np.zeros((len(points1), len(points2)))
    for i, p1 in enumerate(points1):
        for j, p2 in enumerate(points2):
            # Spatial distance (weighted more heavily)
            spatial_dist = np.linalg.norm(p1[:2] - p2[:2])
            # Orientation difference (normalized to [0,1])
            orient_diff = abs(p1[2] - p2[2]) / (2 * np.pi)
            # Combined distance (weighted sum)
            distances[i, j] = 0.7 * spatial_dist + 0.3 * orient_diff
    
    # Find matching points using a more lenient threshold
    match_threshold = 0.3  # Adjusted for synthetic fingerprints
    min_distances = np.min(distances, axis=1)
    matches = min_distances < match_threshold
    
    # Calculate confidence score with quality weights
    if len(matches) > 0:
        match_qualities = np.array([f['quality'] for f in features1])[matches]
        base_quality = np.mean([f['quality'] for f in features1])
        if base_quality > 0:
            confidence = np.sum(match_qualities) / (len(features1) * base_quality)
        else:
            confidence = 0.0
    else:
        confidence = 0.0
    
    # Adjust confidence based on number of matches and feature similarity
    match_ratio = np.sum(matches) / max(len(features1), len(features2))
    feature_similarity = 1.0 - np.mean(min_distances[matches]) if len(matches) > 0 else 0.0
    confidence = confidence * match_ratio * feature_similarity
    
    # Ensure confidence is in [0,1] range
    confidence = min(max(confidence, 0.0), 1.0)
    
    return confidence > threshold, confidence

def visualize_features(image: np.ndarray, 
                      minutiae: List[Dict[str, float]]) -> np.ndarray:
    """
    Visualize the extracted features on the fingerprint image.
    
    Args:
        image: Original fingerprint image
        minutiae: List of minutiae points
        
    Returns:
        Image with visualized features
    """
    # Convert to color image if grayscale
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    # Draw minutiae points
    for point in minutiae:
        x, y = int(point['x']), int(point['y'])
        cv2.circle(vis_image, (x, y), 3, (0, 255, 0), -1)
    
    return vis_image 