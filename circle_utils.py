"""
Circle Detection Utilities
Provides robust circle detection and analysis functions
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import pdist, squareform
# Using manual clustering instead of sklearn


class CircleDetector:
    """Robust circle detection with preprocessing and filtering"""
    
    def __init__(self, min_area: int = 50, roundness_threshold: float = 0.1, 
                 ellipse_ratio_threshold: float = 0.9):
        """
        Initialize circle detector with parameters
        
        Args:
            min_area: Minimum contour area to consider
            roundness_threshold: Maximum std/mean ratio for roundness
            ellipse_ratio_threshold: Minimum minor/major axis ratio for circles
        """
        self.min_area = min_area
        self.roundness_threshold = roundness_threshold
        self.ellipse_ratio_threshold = ellipse_ratio_threshold
    
    def preprocess_image(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply various preprocessing techniques to enhance circle detection
        
        Args:
            img: Input grayscale image
            
        Returns:
            Dictionary with preprocessed images
        """
        results = {}
        
        # Gaussian blur to reduce noise
        results['gaussian'] = cv2.GaussianBlur(img, (5, 5), 0)
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(11, 11))
        results['clahe'] = clahe.apply(img)
        
        # Adaptive thresholding - both methods
        results['thresh_gaussian'] = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 63, 1
        )
        results['thresh_mean'] = cv2.adaptiveThreshold(
            results['gaussian'], 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 63, 1
        )
        
        # Inverted image for contour detection
        results['inverted'] = cv2.bitwise_not(results['thresh_mean'])
        
        # Morphological gradient
        kernel = np.ones((3, 3), np.uint8)
        results['morph_gradient'] = cv2.morphologyEx(
            results['thresh_mean'], cv2.MORPH_GRADIENT, kernel
        )
        
        return results
    
    def detect_circles(self, img: np.ndarray) -> List[Dict]:
        """
        Detect circles in the image using contour analysis
        
        Args:
            img: Binary image (preprocessed)
            
        Returns:
            List of detected circles with properties
        """
        circles = []
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            
            # Filter by area
            if area < self.min_area:
                continue
            
            # Check if contour has enough points
            if len(cnt) < 5:
                continue
            
            # Calculate moments and centroid
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
                
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            
            # Calculate roundness using distance variation
            if len(cnt.shape) == 3 and cnt.shape[1] == 1:
                pts = cnt.squeeze()
            else:
                continue
                
            if len(pts.shape) != 2:
                continue
            
            # Calculate distances from centroid
            dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
            mean_r = np.mean(dists)
            std_r = np.std(dists)
            roundness_ratio = std_r / mean_r if mean_r > 0 else float('inf')
            
            # Fit ellipse and check circularity
            try:
                ellipse = cv2.fitEllipse(cnt)
                (center, axes, angle) = ellipse
                major_axis, minor_axis = max(axes), min(axes)
                ellipse_ratio = minor_axis / major_axis if major_axis > 0 else 0
            except:
                continue
            
            # Check if it's a circle
            if roundness_ratio <= self.roundness_threshold and \
               ellipse_ratio >= self.ellipse_ratio_threshold:
                
                # Calculate additional properties
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                circles.append({
                    'centroid': (cx, cy),
                    'radius': mean_r,
                    'area': area,
                    'perimeter': perimeter,
                    'circularity': circularity,
                    'roundness_ratio': roundness_ratio,
                    'ellipse_ratio': ellipse_ratio,
                    'contour': cnt,
                    'hierarchy': hierarchy[0][i] if hierarchy is not None else None
                })
        
        return circles
    
    def find_concentric_pairs(self, circles: List[Dict], 
                            distance_threshold: float = 10.0,
                            radius_ratio_range: Tuple[float, float] = (1.2, 3.0)) -> List[Tuple[Dict, Dict]]:
        """
        Find pairs of concentric circles (inner and outer)
        
        Args:
            circles: List of detected circles
            distance_threshold: Maximum distance between centers to be considered concentric
            radius_ratio_range: Valid range for outer/inner radius ratio
            
        Returns:
            List of concentric circle pairs (inner, outer)
        """
        pairs = []
        used_indices = set()
        
        for i, circle1 in enumerate(circles):
            if i in used_indices:
                continue
                
            for j, circle2 in enumerate(circles):
                if i == j or j in used_indices:
                    continue
                
                # Calculate distance between centers
                dist = np.linalg.norm(
                    np.array(circle1['centroid']) - np.array(circle2['centroid'])
                )
                
                if dist < distance_threshold:
                    # Check radius ratio
                    r1, r2 = circle1['radius'], circle2['radius']
                    if r1 < r2:
                        inner, outer = circle1, circle2
                        ratio = r2 / r1
                    else:
                        inner, outer = circle2, circle1
                        ratio = r1 / r2
                    
                    if radius_ratio_range[0] <= ratio <= radius_ratio_range[1]:
                        pairs.append((inner, outer))
                        used_indices.add(i)
                        used_indices.add(j)
                        break
        
        return pairs
    
    def cluster_circles(self, circles: List[Dict], eps: float = 20.0) -> List[List[Dict]]:
        """
        Cluster circles based on their centroids using simple distance-based clustering
        
        Args:
            circles: List of detected circles
            eps: Maximum distance for clustering
            
        Returns:
            List of circle clusters
        """
        if len(circles) < 2:
            return [circles] if circles else []
        
        clusters = []
        used_indices = set()
        
        for i, circle1 in enumerate(circles):
            if i in used_indices:
                continue
                
            cluster = [circle1]
            used_indices.add(i)
            
            # Find nearby circles
            for j, circle2 in enumerate(circles):
                if j in used_indices:
                    continue
                    
                dist = np.linalg.norm(
                    np.array(circle1['centroid']) - np.array(circle2['centroid'])
                )
                
                if dist <= eps:
                    cluster.append(circle2)
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        return clusters


def calculate_pattern_descriptor(points: np.ndarray) -> Dict:
    """
    Calculate scale and rotation invariant pattern descriptor
    
    Args:
        points: Nx2 array of point coordinates
        
    Returns:
        Dictionary with pattern descriptors
    """
    if len(points) < 2:
        return {}
    
    # Calculate pairwise distances
    distances = squareform(pdist(points))
    
    # Normalize distances by maximum distance (scale invariant)
    max_dist = np.max(distances)
    if max_dist > 0:
        norm_distances = distances / max_dist
    else:
        norm_distances = distances
    
    # Sort distances for each point (rotation invariant)
    sorted_dists = np.sort(norm_distances, axis=1)
    
    # Calculate angles between points
    angles = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[j, 0] - points[i, 0]
            dy = points[j, 1] - points[i, 1]
            angle = np.arctan2(dy, dx)
            angles.append(angle)
    
    # Calculate relative angles (rotation invariant)
    if angles:
        min_angle = min(angles)
        relative_angles = [(a - min_angle) % (2 * np.pi) for a in angles]
    else:
        relative_angles = []
    
    # Calculate area and perimeter
    if len(points) >= 3:
        hull = cv2.convexHull(points.astype(np.float32))
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)
        # Shape factor (scale invariant)
        shape_factor = (perimeter * perimeter) / area if area > 0 else 0
    else:
        area = 0
        perimeter = 0
        shape_factor = 0
    
    return {
        'normalized_distances': norm_distances,
        'sorted_distances': sorted_dists,
        'relative_angles': relative_angles,
        'num_points': len(points),
        'area': area,
        'perimeter': perimeter,
        'shape_factor': shape_factor,
        'distance_histogram': np.histogram(norm_distances.flatten(), bins=10)[0]
    }


def validate_triangular_pattern(points: np.ndarray, tolerance: float = 0.2) -> bool:
    """
    Validate if 4 points form a triangular/lower matrix pattern
    Expected: 3 points forming a right triangle + 1 point forming rectangular pattern
    
    Args:
        points: 4x2 array of point coordinates
        tolerance: Tolerance for pattern validation
        
    Returns:
        True if pattern matches expected triangular/lower matrix arrangement
    """
    if len(points) != 4:
        return False
    
    # Sort points by position to identify triangular structure
    # Sort by y-coordinate first, then x-coordinate
    sorted_points = points[np.lexsort((points[:, 0], points[:, 1]))]
    
    # Check for triangular/lower matrix arrangement
    # Expected pattern: 
    #   o     (top point)
    #  o o    (middle row with 2 points)
    #   o     (bottom point) - forming lower triangular matrix
    
    # Alternative: Check if points form an L-shape or triangular arrangement
    x_coords = sorted_points[:, 0]
    y_coords = sorted_points[:, 1]
    
    # Check if we have a reasonable spread in both dimensions
    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)
    
    if x_range < 10 or y_range < 10:  # Points too close together
        return False
    
    # Check for triangular arrangement by finding if 3 points form a triangle
    # and the 4th point completes a reasonable pattern
    max_area = 0
    best_triangle_idx = None
    
    # Try all combinations of 3 points to find the largest triangle
    from itertools import combinations
    for combo in combinations(range(4), 3):
        triangle_points = sorted_points[list(combo)]
        
        # Calculate triangle area using cross product
        v1 = triangle_points[1] - triangle_points[0]
        v2 = triangle_points[2] - triangle_points[0]
        area = abs(np.cross(v1, v2)) / 2
        
        if area > max_area:
            max_area = area
            best_triangle_idx = combo
    
    # Check if we found a valid triangle (not collinear)
    if max_area < 1000:  # Minimum area threshold
        return False
    
    # Additional check: Look for L-shaped or right-angle patterns
    # Calculate angles between vectors from each point to others
    angle_count = 0
    for i in range(4):
        vectors = []
        for j in range(4):
            if i != j:
                v = sorted_points[j] - sorted_points[i]
                vectors.append(v / np.linalg.norm(v))  # Normalize
        
        # Check for right angles (90 degrees)
        for k in range(len(vectors)):
            for l in range(k + 1, len(vectors)):
                dot_product = np.dot(vectors[k], vectors[l])
                angle = np.arccos(np.clip(dot_product, -1, 1))
                if abs(angle - np.pi/2) < tolerance:  # Near 90 degrees
                    angle_count += 1
    
    # Should have at least one right angle for triangular/rectangular pattern
    return angle_count >= 1


def draw_pattern_visualization(img: np.ndarray, circles: List[Dict], 
                              pairs: List[Tuple[Dict, Dict]], 
                              pattern_points: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Draw visualization of detected circles and pattern
    
    Args:
        img: Original image
        circles: List of detected circles
        pairs: List of concentric circle pairs
        pattern_points: Points forming the detected pattern
        
    Returns:
        Visualization image
    """
    # Create color image for visualization
    if len(img.shape) == 2:
        vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = img.copy()
    
    # Draw all circles in blue
    for circle in circles:
        cv2.circle(vis_img, 
                  (int(circle['centroid'][0]), int(circle['centroid'][1])),
                  int(circle['radius']), 
                  (255, 0, 0), 2)
        # Draw centroid
        cv2.circle(vis_img, 
                  (int(circle['centroid'][0]), int(circle['centroid'][1])),
                  3, (0, 255, 0), -1)
    
    # Draw concentric pairs in red
    for inner, outer in pairs:
        # Draw connection between centers
        cv2.line(vis_img,
                (int(inner['centroid'][0]), int(inner['centroid'][1])),
                (int(outer['centroid'][0]), int(outer['centroid'][1])),
                (0, 0, 255), 2)
        
        # Highlight paired circles
        cv2.circle(vis_img, 
                  (int(inner['centroid'][0]), int(inner['centroid'][1])),
                  int(inner['radius']), 
                  (0, 255, 255), 3)
        cv2.circle(vis_img, 
                  (int(outer['centroid'][0]), int(outer['centroid'][1])),
                  int(outer['radius']), 
                  (0, 255, 255), 3)
    
    # Draw pattern if detected
    if pattern_points is not None and len(pattern_points) == 4:
        # Draw lines connecting pattern points
        for i in range(4):
            for j in range(i + 1, 4):
                cv2.line(vis_img,
                        (int(pattern_points[i, 0]), int(pattern_points[i, 1])),
                        (int(pattern_points[j, 0]), int(pattern_points[j, 1])),
                        (255, 0, 255), 1)
        
        # Mark pattern points
        for pt in pattern_points:
            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 8, (255, 255, 0), -1)
    
    return vis_img
