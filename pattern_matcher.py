"""
Pattern Matching Logic
Identifies a specific 4-point pattern from circle centroids
"""

import cv2
import numpy as np
from itertools import combinations
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import pdist, squareform

from circle_utils import (
    calculate_pattern_descriptor, 
    validate_triangular_pattern
)


class PatternMatcher:
    """Finds a specific geometric pattern from a set of points"""
    
    def __init__(self, num_points: int = 4, 
                 descriptor_tolerance: float = 0.2, 
                 pattern_validation_tolerance: float = 0.2):
        """
        Initialize pattern matcher
        
        Args:
            num_points: Number of points in the target pattern
            descriptor_tolerance: Tolerance for comparing pattern descriptors
            pattern_validation_tolerance: Tolerance for final pattern validation
        """
        self.num_points = num_points
        self.descriptor_tolerance = descriptor_tolerance
        self.pattern_validation_tolerance = pattern_validation_tolerance
        self.reference_descriptor = None
    
    def learn_reference_pattern(self, points: np.ndarray):
        """
        Learn the reference pattern descriptor from a set of points
        
        Args:
            points: Nx2 array of points for the reference pattern
        """
        if len(points) != self.num_points:
            raise ValueError(f"Reference pattern must have {self.num_points} points")
        
        self.reference_descriptor = calculate_pattern_descriptor(points)
        print("Reference pattern descriptor learned:", self.reference_descriptor)
    
    def find_pattern(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the learned pattern in a new set of points
        
        Args:
            points: Mx2 array of points to search for the pattern
            
        Returns:
            4x2 array of points that match the pattern, or None if not found
        """
        if self.reference_descriptor is None:
            print("Warning: Reference pattern not learned. Using default validation.")
        
        # Generate all combinations of points
        if len(points) < self.num_points:
            return None
            
        for combo in combinations(range(len(points)), self.num_points):
            pts_combo = points[list(combo)]
            
            # Compare with reference descriptor if available
            if self.reference_descriptor:
                test_descriptor = calculate_pattern_descriptor(pts_combo)
                if self._compare_descriptors(self.reference_descriptor, test_descriptor):
                    return pts_combo
            
            # If no reference, use geometric validation
            else:
                if validate_triangular_pattern(pts_combo, self.pattern_validation_tolerance):
                    return pts_combo
        
        return None
    
    def _compare_descriptors(self, desc1: Dict, desc2: Dict) -> bool:
        """
        Compare two pattern descriptors
        
        Args:
            desc1: First pattern descriptor
            desc2: Second pattern descriptor
            
        Returns:
            True if descriptors are similar within tolerance
        """
        if desc1['num_points'] != desc2['num_points']:
            return False
        
        # Compare sorted distance matrices
        dist_diff = np.linalg.norm(
            desc1['sorted_distances'] - desc2['sorted_distances']
        )
        if dist_diff > self.descriptor_tolerance:
            return False
        
        # Compare shape factors
        shape_factor_diff = abs(desc1['shape_factor'] - desc2['shape_factor'])
        if shape_factor_diff > self.descriptor_tolerance * 5: # Looser tolerance for shape
            return False
        
        # Compare distance histograms
        hist_diff = np.linalg.norm(
            desc1['distance_histogram'] - desc2['distance_histogram']
        )
        if hist_diff > self.descriptor_tolerance * 10: # Looser tolerance for histogram
            return False
        
        return True


def define_reference_pattern() -> np.ndarray:
    """
    Define the reference pattern for a right-angled/lower matrix arrangement
    
    Returns:
        4x2 numpy array representing the reference pattern
    """
    # Idealized pattern (can be adjusted based on known dimensions)
    # Lower matrix format
    return np.array([
        [0, 0],  # Origin point
        [1, 0],  # Point on x-axis
        [0, 1],  # Point on y-axis
        [1, 1]   # Corner point (completing the square)
    ])


if __name__ == '__main__':
    # Example usage and testing
    
    # 1. Define reference pattern
    reference_pattern = define_reference_pattern()
    print("Reference Pattern:\n", reference_pattern)
    
    # 2. Initialize PatternMatcher and learn reference
    matcher = PatternMatcher()
    matcher.learn_reference_pattern(reference_pattern)
    
    # 3. Create a test case (rotated, scaled, and with noise)
    # Rotation matrix
    angle = np.pi / 4  # 45 degrees
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    
    # Scaling factor
    scale = 50.0
    
    # Apply transformation
    test_pattern = (rotation_matrix @ reference_pattern.T).T * scale
    
    # Add noise points
    noise_points = np.random.rand(5, 2) * 100
    search_points = np.vstack([test_pattern, noise_points])
    np.random.shuffle(search_points) # Shuffle to make it harder
    
    print("\nSearching for pattern in:\n", search_points)
    
    # 4. Find the pattern
    found_pattern = matcher.find_pattern(search_points)
    
    if found_pattern is not None:
        print("\nPattern found!\n", found_pattern)
        
        # Validate that the found points match the test pattern
        # (Note: order might be different)
        
        test_set = {tuple(row) for row in np.round(test_pattern, 5)}
        found_set = {tuple(row) for row in np.round(found_pattern, 5)}
        
        if test_set == found_set:
            print("\nValidation successful: Found pattern matches the test pattern.")
        else:
            print("\nValidation failed: Found pattern does not match the test pattern.")
    else:
        print("\nPattern not found.")

    # Example with geometric validation only
    print("\n-- Testing with geometric validation only --")
    matcher_geom_only = PatternMatcher()
    found_pattern_geom = matcher_geom_only.find_pattern(search_points)
    if found_pattern_geom is not None:
        print("\nPattern found (geom only)!\n", found_pattern_geom)
    else:
        print("\nPattern not found (geom only).")

