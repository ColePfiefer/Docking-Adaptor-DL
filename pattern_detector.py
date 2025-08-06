"""
Robust Pattern Detection Algorithm
Detects 4 sets of 2 concentric circles arranged in triangular/lower matrix pattern
Invariant to rotation, scale, and skew transformations
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull

from circle_utils import CircleDetector, calculate_pattern_descriptor, validate_triangular_pattern, draw_pattern_visualization
from pattern_matcher import PatternMatcher


class RobustPatternDetector:
    """
    Main class for robust pattern detection with invariance to transformations
    """
    
    def __init__(self, 
                 min_circle_area: int = 50,
                 roundness_threshold: float = 0.15,
                 ellipse_ratio_threshold: float = 0.85,
                 concentric_distance_threshold: float = 15.0,
                 concentric_radius_ratio_range: Tuple[float, float] = (1.2, 4.0),
                 pattern_tolerance: float = 0.25):
        """
        Initialize the robust pattern detector
        
        Args:
            min_circle_area: Minimum area for circle detection
            roundness_threshold: Maximum std/mean ratio for roundness
            ellipse_ratio_threshold: Minimum minor/major axis ratio
            concentric_distance_threshold: Max distance between concentric centers
            concentric_radius_ratio_range: Valid range for outer/inner radius ratio
            pattern_tolerance: Tolerance for pattern validation
        """
        self.circle_detector = CircleDetector(
            min_area=min_circle_area,
            roundness_threshold=roundness_threshold,
            ellipse_ratio_threshold=ellipse_ratio_threshold
        )
        
        self.pattern_matcher = PatternMatcher(
            num_points=4,
            pattern_validation_tolerance=pattern_tolerance
        )
        
        self.concentric_distance_threshold = concentric_distance_threshold
        self.concentric_radius_ratio_range = concentric_radius_ratio_range
        
        # Storage for analysis results
        self.last_analysis = {}
        
    def detect_pattern(self, image_path: str, visualize: bool = True) -> Dict:
        """
        Main method to detect the pattern in an image
        
        Args:
            image_path: Path to the input image
            visualize: Whether to create visualization plots
            
        Returns:
            Dictionary containing detection results and analysis
        """
        # Load image
        img = cv2.imread(image_path, 0)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Store for analysis
        self.last_analysis = {
            'image_path': image_path,
            'original_image': img,
            'detected_circles': [],
            'concentric_pairs': [],
            'pattern_centroids': None,
            'pattern_found': False,
            'confidence_metrics': {}
        }
        
        # Step 1: Preprocess image
        preprocessed = self.circle_detector.preprocess_image(img)
        
        # Step 2: Detect circles using multiple preprocessing methods
        all_circles = []
        for method_name, processed_img in preprocessed.items():
            if method_name == 'inverted':  # Use inverted image for contour detection
                circles = self.circle_detector.detect_circles(processed_img)
                all_circles.extend(circles)
        
        # Step 3: Remove duplicate circles (merge close centroids)
        unique_circles = self._merge_duplicate_circles(all_circles, distance_threshold=10.0)
        self.last_analysis['detected_circles'] = unique_circles
        
        # Step 4: Find concentric circle pairs
        concentric_pairs = self.circle_detector.find_concentric_pairs(
            unique_circles,
            distance_threshold=self.concentric_distance_threshold,
            radius_ratio_range=self.concentric_radius_ratio_range
        )
        self.last_analysis['concentric_pairs'] = concentric_pairs
        
        # Step 5: Extract centroids from concentric pairs for pattern matching
        if len(concentric_pairs) >= 4:
            # Use the center point between concentric pairs as the pattern point
            pattern_candidates = []
            for inner, outer in concentric_pairs:
                # Calculate weighted centroid (more weight to inner circle)
                weight_inner = 0.7
                weight_outer = 0.3
                cx = (inner['centroid'][0] * weight_inner + outer['centroid'][0] * weight_outer)
                cy = (inner['centroid'][1] * weight_inner + outer['centroid'][1] * weight_outer)
                pattern_candidates.append([cx, cy])
            
            pattern_candidates = np.array(pattern_candidates)
            
            # Step 6: Find the 4-point pattern
            found_pattern = self._find_best_4_point_pattern(pattern_candidates)
            
            if found_pattern is not None:
                self.last_analysis['pattern_centroids'] = found_pattern
                self.last_analysis['pattern_found'] = True
                
                # Step 7: Calculate confidence metrics
                self._calculate_confidence_metrics(found_pattern, concentric_pairs)
        
        # Step 8: Fallback - use all circle centroids if concentric pairs aren't enough
        if not self.last_analysis['pattern_found'] and len(unique_circles) >= 4:
            print("Fallback: Using all circle centroids for pattern detection")
            all_centroids = np.array([c['centroid'] for c in unique_circles])
            found_pattern = self._find_best_4_point_pattern(all_centroids)
            
            if found_pattern is not None:
                self.last_analysis['pattern_centroids'] = found_pattern
                self.last_analysis['pattern_found'] = True
                self._calculate_confidence_metrics(found_pattern, [])
        
        # Step 9: Visualization
        if visualize:
            self._create_visualization_plots(preprocessed)
        
        return self.last_analysis
    
    def _merge_duplicate_circles(self, circles: List[Dict], distance_threshold: float = 10.0) -> List[Dict]:
        """
        Merge circles with centroids that are too close together
        """
        if not circles:
            return []
        
        unique_circles = []
        used_indices = set()
        
        for i, circle1 in enumerate(circles):
            if i in used_indices:
                continue
                
            # Find all circles close to this one
            similar_circles = [circle1]
            used_indices.add(i)
            
            for j, circle2 in enumerate(circles):
                if j in used_indices:
                    continue
                    
                dist = np.linalg.norm(
                    np.array(circle1['centroid']) - np.array(circle2['centroid'])
                )
                
                if dist < distance_threshold:
                    similar_circles.append(circle2)
                    used_indices.add(j)
            
            # Merge similar circles by averaging properties
            if len(similar_circles) == 1:
                unique_circles.append(similar_circles[0])
            else:
                merged_circle = self._merge_circles(similar_circles)
                unique_circles.append(merged_circle)
        
        return unique_circles
    
    def _merge_circles(self, circles: List[Dict]) -> Dict:
        """
        Merge multiple similar circles into one by averaging properties
        """
        n = len(circles)
        
        # Average centroid
        avg_cx = sum(c['centroid'][0] for c in circles) / n
        avg_cy = sum(c['centroid'][1] for c in circles) / n
        
        # Average other properties
        avg_radius = sum(c['radius'] for c in circles) / n
        avg_area = sum(c['area'] for c in circles) / n
        avg_perimeter = sum(c['perimeter'] for c in circles) / n
        avg_circularity = sum(c['circularity'] for c in circles) / n
        avg_roundness = sum(c['roundness_ratio'] for c in circles) / n
        avg_ellipse = sum(c['ellipse_ratio'] for c in circles) / n
        
        return {
            'centroid': (avg_cx, avg_cy),
            'radius': avg_radius,
            'area': avg_area,
            'perimeter': avg_perimeter,
            'circularity': avg_circularity,
            'roundness_ratio': avg_roundness,
            'ellipse_ratio': avg_ellipse,
            'contour': circles[0]['contour'],  # Use first contour
            'hierarchy': circles[0]['hierarchy']
        }
    
    def _find_best_4_point_pattern(self, candidates: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the best 4-point pattern from candidate points
        Uses multiple criteria to rank patterns
        """
        if len(candidates) < 4:
            return None
        
        best_pattern = None
        best_score = -1
        
        # Try all combinations of 4 points
        for combo in combinations(range(len(candidates)), 4):
            pts = candidates[list(combo)]
            
            # Calculate various pattern quality metrics
            score = self._evaluate_pattern_quality(pts)
            
            if score > best_score:
                best_score = score
                best_pattern = pts
        
        # Only return pattern if it meets minimum quality threshold
        if best_score > 0.5:  # Adjustable threshold
            return best_pattern
        else:
            return None
    
    def _evaluate_pattern_quality(self, points: np.ndarray) -> float:
        """
        Evaluate the quality of a 4-point pattern using multiple criteria
        Specifically optimized for triangular/lower matrix patterns
        Returns score between 0-1 (higher is better)
        """
        if len(points) != 4:
            return 0.0
        
        score_components = []
        
        # 1. Triangular arrangement validation (most important)
        geometric_valid = validate_triangular_pattern(points, tolerance=0.3)
        geometric_score = 1.0 if geometric_valid else 0.2  # Give some credit even if not perfect
        score_components.append(('geometric', geometric_score, 0.5))  # Increased weight
        
        # 2. Check for L-shape or right-angle configuration
        l_shape_score = self._evaluate_l_shape(points)
        score_components.append(('l_shape', l_shape_score, 0.2))
        
        # 3. Distance distribution (should have some variation for triangular pattern)
        distances = squareform(pdist(points))
        distance_list = [distances[i, j] for i in range(4) for j in range(i+1, 4)]
        distance_std = np.std(distance_list)
        distance_mean = np.mean(distance_list)
        
        # For triangular patterns, we want some distance variation (not all equal)
        if distance_mean > 0:
            cv = distance_std / distance_mean  # Coefficient of variation
            # Optimal CV for triangular pattern is around 0.2-0.4
            if 0.15 <= cv <= 0.5:
                distance_score = 1.0
            elif cv < 0.15:  # Too uniform (square-like)
                distance_score = 0.3
            else:  # Too scattered
                distance_score = max(0, 1 - (cv - 0.5) / 0.3)
        else:
            distance_score = 0.0
        score_components.append(('distances', distance_score, 0.2))
        
        # 4. Convex hull area (should be reasonable)
        try:
            hull = ConvexHull(points)
            area = hull.volume  # In 2D, volume is area
            # Normalize by bounding box area
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            bbox_area = x_range * y_range
            if bbox_area > 0:
                fill_ratio = area / bbox_area
                # For triangular patterns, fill ratio should be around 0.3-0.7
                if 0.25 <= fill_ratio <= 0.75:
                    area_score = 1.0
                else:
                    area_score = max(0, 1 - abs(fill_ratio - 0.5) / 0.5)
            else:
                area_score = 0.0
        except:
            area_score = 0.0
        score_components.append(('area', area_score, 0.1))
        
        # Calculate weighted final score
        total_score = sum(score * weight for _, score, weight in score_components)
        
        return total_score
    
    def _evaluate_l_shape(self, points: np.ndarray) -> float:
        """
        Evaluate if points form an L-shape or right-angle configuration
        Returns score between 0-1
        """
        if len(points) != 4:
            return 0.0
        
        # Sort points by position
        sorted_points = points[np.lexsort((points[:, 0], points[:, 1]))]
        
        # Check for L-shape by finding corner point
        max_angles = 0
        
        for i in range(4):
            # Count right angles from this point
            vectors_from_i = []
            for j in range(4):
                if i != j:
                    v = sorted_points[j] - sorted_points[i]
                    if np.linalg.norm(v) > 0:
                        vectors_from_i.append(v / np.linalg.norm(v))
            
            # Count near-right angles
            right_angles = 0
            for k in range(len(vectors_from_i)):
                for l in range(k + 1, len(vectors_from_i)):
                    dot_product = np.dot(vectors_from_i[k], vectors_from_i[l])
                    angle = np.arccos(np.clip(dot_product, -1, 1))
                    if abs(angle - np.pi/2) < 0.3:  # Within ~17 degrees of 90
                        right_angles += 1
            
            max_angles = max(max_angles, right_angles)
        
        # Score based on number of right angles found
        return min(1.0, max_angles / 2.0)  # Ideal is 2+ right angles
    
    def _calculate_confidence_metrics(self, pattern_points: np.ndarray, concentric_pairs: List):
        """
        Calculate confidence metrics for the detected pattern
        """
        metrics = {}
        
        # Pattern quality score
        metrics['pattern_quality'] = self._evaluate_pattern_quality(pattern_points)
        
        # Number of concentric pairs found
        metrics['concentric_pairs_count'] = len(concentric_pairs)
        metrics['concentric_pairs_ratio'] = len(concentric_pairs) / 4.0  # Ideal is 4 pairs
        
        # Distance consistency
        distances = squareform(pdist(pattern_points))
        distance_list = [distances[i, j] for i in range(4) for j in range(i+1, 4)]
        metrics['distance_consistency'] = 1.0 / (1.0 + np.std(distance_list) / np.mean(distance_list))
        
        # Overall confidence (weighted combination)
        confidence = (
            metrics['pattern_quality'] * 0.4 +
            metrics['concentric_pairs_ratio'] * 0.3 +
            metrics['distance_consistency'] * 0.3
        )
        metrics['overall_confidence'] = min(1.0, confidence)
        
        self.last_analysis['confidence_metrics'] = metrics
    
    def _create_visualization_plots(self, preprocessed_images: Dict):
        """
        Create comprehensive visualization plots
        """
        fig = plt.figure(figsize=(15, 12))
        
        # Original image
        plt.subplot(4, 4, 1)
        plt.imshow(self.last_analysis['original_image'], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Preprocessed images
        plot_idx = 2
        for name, img in preprocessed_images.items():
            if plot_idx <= 8:
                plt.subplot(4, 4, plot_idx)
                plt.imshow(img, cmap='gray')
                plt.title(name.replace('_', ' ').title())
                plt.axis('off')
                plot_idx += 1
        
        # Detected circles visualization
        plt.subplot(4, 4, 9)
        circles_vis = draw_pattern_visualization(
            self.last_analysis['original_image'],
            self.last_analysis['detected_circles'],
            self.last_analysis['concentric_pairs'],
            self.last_analysis['pattern_centroids']
        )
        plt.imshow(cv2.cvtColor(circles_vis, cv2.COLOR_BGR2RGB))
        plt.title('Pattern Detection Result')
        plt.axis('off')
        
        # Pattern points only
        if self.last_analysis['pattern_found']:
            plt.subplot(4, 4, 10)
            pattern_img = np.zeros_like(self.last_analysis['original_image'])
            for pt in self.last_analysis['pattern_centroids']:
                cv2.circle(pattern_img, (int(pt[0]), int(pt[1])), 8, 255, -1)
            plt.imshow(pattern_img, cmap='gray')
            plt.title('Detected Pattern Points')
            plt.axis('off')
            
            # Confidence metrics text
            plt.subplot(4, 4, 11)
            plt.axis('off')
            metrics_text = "Confidence Metrics:\n\n"
            for key, value in self.last_analysis['confidence_metrics'].items():
                metrics_text += f"{key}: {value:.3f}\n"
            plt.text(0.1, 0.9, metrics_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        plt.tight_layout()
        
        # Extract base filename for saving plot
        plot_base_name = os.path.splitext(os.path.basename(self.last_analysis['image_path']))[0]
        plt.savefig(f"pattern_analysis_{plot_base_name}.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_pattern_results(self, output_dir: str = "results"):
        """
        Save pattern detection results to files
        """
        import os
        import json
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract base filename properly handling both Windows and Unix paths
        base_name = os.path.splitext(os.path.basename(self.last_analysis['image_path']))[0]
        
        # Save pattern points as image
        if self.last_analysis['pattern_found']:
            pattern_img = np.zeros_like(self.last_analysis['original_image'])
            for pt in self.last_analysis['pattern_centroids']:
                cv2.circle(pattern_img, (int(pt[0]), int(pt[1])), 8, 255, -1)
            cv2.imwrite(f"{output_dir}/{base_name}_pattern.png", pattern_img)
            
            # Save pattern points as text
            np.savetxt(f"{output_dir}/{base_name}_pattern_points.txt", 
                      self.last_analysis['pattern_centroids'], fmt='%.2f')
        
        # Save full visualization
        vis_img = draw_pattern_visualization(
            self.last_analysis['original_image'],
            self.last_analysis['detected_circles'],
            self.last_analysis['concentric_pairs'],
            self.last_analysis['pattern_centroids']
        )
        cv2.imwrite(f"{output_dir}/{base_name}_visualization.png", vis_img)
        
        # Save analysis results as JSON
        json_results = {
            'image_path': self.last_analysis['image_path'],
            'pattern_found': self.last_analysis['pattern_found'],
            'num_circles_detected': len(self.last_analysis['detected_circles']),
            'num_concentric_pairs': len(self.last_analysis['concentric_pairs']),
            'confidence_metrics': self.last_analysis['confidence_metrics'],
            'pattern_points': self.last_analysis['pattern_centroids'].tolist() if self.last_analysis['pattern_found'] else None
        }
        
        with open(f"{output_dir}/{base_name}_analysis.json", 'w') as f:
            json.dump(json_results, f, indent=2)


if __name__ == "__main__":
    # Example usage
    detector = RobustPatternDetector()
    
    # Test with the first image
    test_image = "dataset/IMG_10072025161331.png"
    
    try:
        results = detector.detect_pattern(test_image, visualize=True)
        
        print(f"Pattern Detection Results for {test_image}:")
        print(f"Pattern Found: {results['pattern_found']}")
        print(f"Circles Detected: {len(results['detected_circles'])}")
        print(f"Concentric Pairs: {len(results['concentric_pairs'])}")
        
        if results['pattern_found']:
            print(f"Pattern Points:\n{results['pattern_centroids']}")
            print(f"Overall Confidence: {results['confidence_metrics']['overall_confidence']:.3f}")
        
        # Save results
        detector.save_pattern_results()
        
    except Exception as e:
        print(f"Error processing image: {e}")
