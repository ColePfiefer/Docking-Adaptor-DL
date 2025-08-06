"""
Specialized Triangular Pattern Detector
Specifically designed for detecting triangular/lower matrix arrangements
of 4 concentric circle pairs
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from itertools import combinations
from scipy.spatial.distance import pdist, squareform

from pattern_detector import RobustPatternDetector


class TriangularPatternDetector(RobustPatternDetector):
    """
    Specialized detector for triangular/lower matrix patterns
    Inherits from RobustPatternDetector but adds specific triangular pattern logic
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pattern_type = "triangular_lower_matrix"
    
    def _find_triangular_pattern(self, candidates: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the best triangular pattern specifically
        Expected arrangement:
            o       <- top point
           o o      <- middle row  
            o       <- bottom point (forming lower triangular matrix)
        """
        if len(candidates) < 4:
            return None
        
        best_pattern = None
        best_score = -1
        
        for combo in combinations(range(len(candidates)), 4):
            pts = candidates[list(combo)]
            
            # Specific triangular pattern scoring
            score = self._score_triangular_arrangement(pts)
            
            if score > best_score:
                best_score = score
                best_pattern = pts
        
        # More lenient threshold for triangular patterns
        if best_score > 0.3:
            return best_pattern
        else:
            return None
    
    def _score_triangular_arrangement(self, points: np.ndarray) -> float:
        """
        Score how well 4 points form a triangular/lower matrix arrangement
        """
        if len(points) != 4:
            return 0.0
        
        # Sort points by y-coordinate (top to bottom), then x-coordinate
        sorted_points = points[np.lexsort((points[:, 0], points[:, 1]))]
        
        score_components = []
        
        # 1. Check for triangular row structure
        triangular_score = self._check_triangular_rows(sorted_points)
        score_components.append(('triangular', triangular_score, 0.4))
        
        # 2. Check for L-shape or right angles
        l_score = self._evaluate_l_configuration(sorted_points)
        score_components.append(('l_shape', l_score, 0.3))
        
        # 3. Check distance ratios (should have structure)
        distance_score = self._evaluate_distance_structure(sorted_points)
        score_components.append(('distances', distance_score, 0.2))
        
        # 4. Overall compactness
        compactness_score = self._evaluate_compactness(sorted_points)
        score_components.append(('compactness', compactness_score, 0.1))
        
        return sum(score * weight for _, score, weight in score_components)
    
    def _check_triangular_rows(self, points: np.ndarray) -> float:
        """
        Check if points can be arranged in triangular rows
        """
        # Group points by similar y-coordinates (rows)
        y_coords = points[:, 1]
        y_sorted = np.sort(y_coords)
        
        # Try to find row structure
        rows = []
        current_row = [points[0]]
        row_tolerance = 50  # Pixels
        
        for i in range(1, len(points)):
            # Check if this point belongs to current row
            if abs(points[i, 1] - current_row[0][1]) <= row_tolerance:
                current_row.append(points[i])
            else:
                rows.append(current_row)
                current_row = [points[i]]
        rows.append(current_row)
        
        # Score based on row structure
        row_sizes = [len(row) for row in rows]
        row_sizes.sort()
        
        # Perfect triangular: [1, 1, 2] or [1, 3] etc.
        if row_sizes == [1, 1, 2]:
            return 1.0
        elif row_sizes == [1, 3]:
            return 0.9
        elif row_sizes == [2, 2]:
            return 0.7  # Square-ish but acceptable
        elif len(row_sizes) == 4:  # All points in different rows
            return 0.3
        else:
            return 0.1
    
    def _evaluate_l_configuration(self, points: np.ndarray) -> float:
        """
        Check if points form an L-shape (which is common in lower triangular matrices)
        """
        # Find potential corner point (point with most perpendicular connections)
        best_corner_score = 0
        
        for i, center_point in enumerate(points):
            # Get vectors from this point to all others
            vectors = []
            for j, other_point in enumerate(points):
                if i != j:
                    v = other_point - center_point
                    if np.linalg.norm(v) > 0:
                        vectors.append(v / np.linalg.norm(v))
            
            # Count near-perpendicular pairs
            perpendicular_pairs = 0
            for k in range(len(vectors)):
                for l in range(k + 1, len(vectors)):
                    dot_product = np.dot(vectors[k], vectors[l])
                    angle = np.arccos(np.clip(dot_product, -1, 1))
                    if abs(angle - np.pi/2) < np.pi/6:  # Within 30 degrees of 90
                        perpendicular_pairs += 1
            
            corner_score = perpendicular_pairs / 3.0  # Max possible is 3 pairs
            best_corner_score = max(best_corner_score, corner_score)
        
        return best_corner_score
    
    def _evaluate_distance_structure(self, points: np.ndarray) -> float:
        """
        Evaluate if the distance structure makes sense for a triangular pattern
        """
        distances = squareform(pdist(points))
        distance_list = distances[np.triu_indices(4, k=1)]  # Upper triangle only
        
        # Sort distances to analyze structure
        sorted_distances = np.sort(distance_list)
        
        # For triangular patterns, expect some distance grouping
        # Should not be all equal (square) or completely scattered
        std_ratio = np.std(sorted_distances) / np.mean(sorted_distances)
        
        if 0.2 <= std_ratio <= 0.6:  # Good variation for triangular
            return 1.0
        elif std_ratio < 0.2:  # Too uniform
            return 0.4
        else:  # Too scattered
            return max(0, 1 - (std_ratio - 0.6) / 0.4)
    
    def _evaluate_compactness(self, points: np.ndarray) -> float:
        """
        Evaluate if points form a compact triangular arrangement
        """
        # Calculate convex hull and check fill ratio
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            
            # Calculate bounding box
            x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
            y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
            bbox_area = (x_max - x_min) * (y_max - y_min)
            
            if bbox_area > 0:
                fill_ratio = hull_area / bbox_area
                # Triangular patterns should have fill ratio around 0.4-0.7
                if 0.3 <= fill_ratio <= 0.8:
                    return 1.0
                else:
                    return max(0, 1 - abs(fill_ratio - 0.55) / 0.45)
            else:
                return 0.0
        except:
            return 0.5
    
    def detect_triangular_pattern(self, image_path: str, visualize: bool = True) -> Dict:
        """
        Main method optimized for triangular pattern detection
        """
        # Use parent class for basic detection
        result = self.detect_pattern(image_path, visualize=False)
        
        # If pattern found with standard method, validate it's triangular
        if result['pattern_found']:
            triangular_score = self._score_triangular_arrangement(result['pattern_centroids'])
            result['triangular_score'] = triangular_score
            
            # If not triangular enough, try specialized detection
            if triangular_score < 0.5:
                print("Standard pattern not triangular enough, trying specialized detection...")
                specialized_result = self._specialized_triangular_detection(result)
                if specialized_result:
                    result = specialized_result
        
        # Create visualization if requested
        if visualize:
            self._create_triangular_visualization(result)
        
        return result
    
    def _specialized_triangular_detection(self, base_result: Dict) -> Optional[Dict]:
        """
        Specialized detection focusing on triangular arrangements
        """
        circles = base_result['detected_circles']
        if len(circles) < 4:
            return None
        
        # Try different selection strategies
        all_centroids = np.array([c['centroid'] for c in circles])
        
        # Strategy 1: Look specifically for triangular arrangement
        triangular_pattern = self._find_triangular_pattern(all_centroids)
        
        if triangular_pattern is not None:
            # Update result
            base_result['pattern_centroids'] = triangular_pattern
            base_result['pattern_found'] = True
            base_result['triangular_score'] = self._score_triangular_arrangement(triangular_pattern)
            
            # Recalculate confidence with triangular scoring
            self._calculate_triangular_confidence(base_result, triangular_pattern)
            
            return base_result
        
        return None
    
    def _calculate_triangular_confidence(self, result: Dict, pattern_points: np.ndarray):
        """
        Calculate confidence specifically for triangular patterns
        """
        metrics = {}
        
        # Triangular pattern quality
        metrics['triangular_quality'] = self._score_triangular_arrangement(pattern_points)
        
        # Distance structure
        distances = squareform(pdist(pattern_points))
        distance_list = distances[np.triu_indices(4, k=1)]
        metrics['distance_consistency'] = 1.0 / (1.0 + np.std(distance_list) / np.mean(distance_list))
        
        # Overall confidence weighted for triangular patterns
        confidence = (
            metrics['triangular_quality'] * 0.6 +
            metrics['distance_consistency'] * 0.4
        )
        metrics['overall_confidence'] = min(1.0, confidence)
        
        result['confidence_metrics'] = metrics
    
    def _create_triangular_visualization(self, result: Dict):
        """
        Create visualization with triangular pattern emphasis
        """
        if not result['pattern_found']:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image with detected circles
        img = result['original_image']
        circles = result['detected_circles']
        pattern_points = result['pattern_centroids']
        
        # Plot 1: Original with all circles
        axes[0].imshow(img, cmap='gray')
        for circle in circles:
            circle_plot = plt.Circle(circle['centroid'], circle['radius'], 
                                   fill=False, color='blue', linewidth=2)
            axes[0].add_patch(circle_plot)
            axes[0].plot(circle['centroid'][0], circle['centroid'][1], 'go', markersize=4)
        axes[0].set_title('Detected Circles')
        axes[0].axis('off')
        
        # Plot 2: Pattern points only
        axes[1].imshow(img, cmap='gray', alpha=0.7)
        if pattern_points is not None:
            for i, pt in enumerate(pattern_points):
                axes[1].plot(pt[0], pt[1], 'ro', markersize=10, label=f'Point {i+1}')
            
            # Draw triangular connections
            # Connect points to show triangular structure
            for i in range(len(pattern_points)):
                for j in range(i + 1, len(pattern_points)):
                    axes[1].plot([pattern_points[i, 0], pattern_points[j, 0]], 
                               [pattern_points[i, 1], pattern_points[j, 1]], 
                               'm-', alpha=0.6, linewidth=1)
        
        axes[1].set_title('Triangular Pattern')
        axes[1].axis('off')
        
        # Plot 3: Analysis info
        axes[2].axis('off')
        if 'confidence_metrics' in result:
            info_text = "Triangular Pattern Analysis:\\n\\n"
            for key, value in result['confidence_metrics'].items():
                info_text += f"{key}: {value:.3f}\\n"
            
            if 'triangular_score' in result:
                info_text += f"\\nTriangular Score: {result['triangular_score']:.3f}"
        else:
            info_text = "No pattern detected"
        
        axes[2].text(0.1, 0.9, info_text, transform=axes[2].transAxes, 
                    verticalalignment='top', fontsize=12, fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save visualization
        if 'image_path' in result:
            base_name = os.path.splitext(os.path.basename(result['image_path']))[0]
            plt.savefig(f"triangular_pattern_{base_name}.png", dpi=150, bbox_inches='tight')
        
        plt.show()


def main():
    """Test the specialized triangular pattern detector"""
    detector = TriangularPatternDetector()
    
    # Test with sample images
    test_images = [
        "dataset/IMG_10072025161331.png",
        "dataset/IMG_10072025161333.png",
        "dataset/IMG_10072025161339.png"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\\nTesting triangular detection on: {os.path.basename(image_path)}")
            
            result = detector.detect_triangular_pattern(image_path, visualize=True)
            
            if result['pattern_found']:
                print(f"✓ Triangular pattern detected!")
                print(f"✓ Confidence: {result['confidence_metrics']['overall_confidence']:.3f}")
                if 'triangular_score' in result:
                    print(f"✓ Triangular Quality: {result['triangular_score']:.3f}")
            else:
                print("✗ No triangular pattern detected")


if __name__ == "__main__":
    main()
