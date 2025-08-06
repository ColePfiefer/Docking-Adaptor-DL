"""
Corrected Thermal Image Preprocessing Pipeline for Space Station Docking Pattern Recognition
Author: AI Assistant
Purpose: Detect 4 sets of concentric circles arranged in a diamond/cross pattern
Pattern Description: 4 circular targets, each with 2 concentric circles, arranged in a cross/diamond formation
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import filters, morphology, measure, exposure
from skimage.feature import peak_local_max
import warnings
import sys
import traceback
warnings.filterwarnings('ignore')

class ThermalPatternDetector:
    """
    Specialized detector for 4 sets of concentric circles in thermal images
    """
    
    def __init__(self, output_dir='preprocessed_pattern_detection'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different preprocessing stages
        self.stages = {
            'original': self.output_dir / '01_original',
            'normalized': self.output_dir / '02_normalized',
            'gradient': self.output_dir / '03_gradient',
            'enhanced': self.output_dir / '04_enhanced',
            'binary': self.output_dir / '05_binary',
            'edges': self.output_dir / '06_edges',
            'circles': self.output_dir / '07_circles_detected',
            'pattern': self.output_dir / '08_pattern_analysis'
        }
        
        for stage_dir in self.stages.values():
            stage_dir.mkdir(exist_ok=True)
        
        self.preprocessing_stats = []
        
    def load_thermal_image(self, image_path):
        """
        Load and convert thermal image to appropriate format
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def thermal_normalization(self, image):
        """
        Normalize thermal image using adaptive histogram equalization
        """
        # Convert to 8-bit if necessary
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        normalized = clahe.apply(image)
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(normalized, 9, 75, 75)
        
        return denoised
    
    def compute_gradient_magnitude(self, image):
        """
        Compute gradient magnitude and direction for circle detection
        """
        # Sobel gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Normalize magnitude
        magnitude = ((magnitude - magnitude.min()) / 
                    (magnitude.max() - magnitude.min() + 1e-10) * 255).astype(np.uint8)
        
        return magnitude, direction
    
    def enhance_circular_features(self, image):
        """
        Enhanced circular feature detection specifically for concentric circles
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 1)
        
        # Multi-scale Difference of Gaussians for concentric circle detection
        scales = [(1, 3), (2, 5), (3, 7), (4, 9)]  # Inner and outer sigma pairs
        dog_responses = []
        
        for sigma_inner, sigma_outer in scales:
            gaussian_inner = cv2.GaussianBlur(blurred, (0, 0), sigma_inner)
            gaussian_outer = cv2.GaussianBlur(blurred, (0, 0), sigma_outer)
            dog = np.abs(gaussian_inner.astype(float) - gaussian_outer.astype(float))
            dog_responses.append(dog)
        
        # Combine multi-scale responses
        enhanced = np.mean(dog_responses, axis=0)
        
        # Normalize
        enhanced = ((enhanced - enhanced.min()) / 
                   (enhanced.max() - enhanced.min() + 1e-10) * 255).astype(np.uint8)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # Enhance contrast
        enhanced = cv2.equalizeHist(enhanced)
        
        return enhanced
    
    def adaptive_binary_segmentation(self, image):
        """
        Improved binary segmentation with less noise
        """
        # Apply stronger Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (7, 7), 2)
        
        # Use only Otsu's thresholding for cleaner results
        threshold_value, binary = cv2.threshold(blurred, 0, 255, 
                                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        # Remove small noise
        kernel_small = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Fill small holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # Additional denoising with larger kernel
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_large)
        
        return cleaned
    
    def detect_edges_for_circles(self, image):
        """
        Edge detection optimized for circular patterns
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
        
        # Canny edge detection with carefully tuned thresholds
        edges = cv2.Canny(blurred, 30, 100)
        
        # Dilate edges slightly to connect broken circles
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
    
    def detect_concentric_circle_sets(self, image, enhanced, edges):
        """
        Detect 4 sets of concentric circles in diamond pattern
        """
        # Use enhanced image for better circle detection
        processed = cv2.medianBlur(enhanced, 5)
        
        # Parameters for detecting individual circles
        circles_detected = []
        
        # Try different parameter sets for Hough circles
        param_sets = [
            {'dp': 1, 'minDist': 20, 'param1': 50, 'param2': 25, 
             'minRadius': 8, 'maxRadius': 40},
            {'dp': 1.2, 'minDist': 15, 'param1': 60, 'param2': 20, 
             'minRadius': 10, 'maxRadius': 35},
            {'dp': 1.5, 'minDist': 25, 'param1': 40, 'param2': 30, 
             'minRadius': 12, 'maxRadius': 45},
        ]
        
        for params in param_sets:
            circles = cv2.HoughCircles(
                processed,
                cv2.HOUGH_GRADIENT,
                **params
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    x, y, r = circle
                    circles_detected.append({
                        'center': (int(x), int(y)),
                        'radius': int(r),
                        'area': np.pi * r * r
                    })
        
        # Remove duplicate circles
        unique_circles = self.remove_duplicate_circles(circles_detected)
        
        # Group circles into concentric sets
        circle_sets = self.group_concentric_circles(unique_circles)
        
        # Filter to get the best 4 sets forming a pattern
        pattern_sets = self.find_diamond_pattern(circle_sets)
        
        return pattern_sets, unique_circles
    
    def remove_duplicate_circles(self, circles):
        """
        Remove duplicate circles based on center proximity
        """
        if not circles:
            return []
        
        unique = []
        for circle in circles:
            is_duplicate = False
            for existing in unique:
                dist = np.sqrt((circle['center'][0] - existing['center'][0])**2 + 
                             (circle['center'][1] - existing['center'][1])**2)
                # If centers are very close and radii are similar, it's a duplicate
                if dist < 15 and abs(circle['radius'] - existing['radius']) < 10:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(circle)
        
        return unique
    
    def group_concentric_circles(self, circles):
        """
        Group circles into concentric sets based on center proximity
        """
        if len(circles) < 2:
            return []
        
        # Group circles by center proximity
        circle_groups = []
        used = [False] * len(circles)
        
        for i, circle1 in enumerate(circles):
            if used[i]:
                continue
            
            group = [circle1]
            used[i] = True
            
            for j, circle2 in enumerate(circles):
                if used[j] or i == j:
                    continue
                
                # Check if centers are close (concentric)
                dist = np.sqrt((circle1['center'][0] - circle2['center'][0])**2 + 
                             (circle1['center'][1] - circle2['center'][1])**2)
                
                if dist < 20:  # Centers within 20 pixels
                    group.append(circle2)
                    used[j] = True
            
            # Sort group by radius (inner to outer)
            group.sort(key=lambda x: x['radius'])
            circle_groups.append(group)
        
        # Filter groups to keep only those with 2 circles (concentric pairs)
        concentric_pairs = [g for g in circle_groups if len(g) == 2]
        
        return concentric_pairs
    
    def find_diamond_pattern(self, circle_sets):
        """
        Find 4 circle sets that form a diamond/cross pattern
        """
        if len(circle_sets) < 4:
            return circle_sets  # Return what we have if less than 4
        
        if len(circle_sets) == 4:
            return circle_sets  # Perfect match
        
        # If more than 4, find the best diamond pattern
        # Calculate centroids of each set
        centroids = []
        for circle_set in circle_sets:
            cx = np.mean([c['center'][0] for c in circle_set])
            cy = np.mean([c['center'][1] for c in circle_set])
            centroids.append((cx, cy))
        
        # Find 4 sets that form the most regular pattern
        # This is a simplified approach - you might want to add more sophisticated pattern matching
        best_sets = []
        best_score = float('inf')
        
        # Try all combinations of 4 sets
        from itertools import combinations
        for combo in combinations(range(len(circle_sets)), 4):
            sets = [circle_sets[i] for i in combo]
            cents = [centroids[i] for i in combo]
            
            # Calculate pattern regularity (variance in distances)
            distances = []
            for i in range(4):
                for j in range(i+1, 4):
                    dist = np.sqrt((cents[i][0] - cents[j][0])**2 + 
                                 (cents[i][1] - cents[j][1])**2)
                    distances.append(dist)
            
            # Lower variance means more regular pattern
            if len(distances) > 0:
                score = np.var(distances)
                if score < best_score:
                    best_score = score
                    best_sets = sets
        
        # If no best sets found, return original
        if not best_sets:
            return circle_sets[:4]  # Return first 4
        
        return best_sets
    
    def analyze_pattern(self, circle_sets):
        """
        Analyze the detected pattern of circle sets
        """
        if not circle_sets:
            return None
        
        # Calculate pattern center (centroid of all circle centers)
        all_centers = []
        for circle_set in circle_sets:
            for circle in circle_set:
                all_centers.append(circle['center'])
        
        if not all_centers:
            return None
        
        pattern_center_x = np.mean([c[0] for c in all_centers])
        pattern_center_y = np.mean([c[1] for c in all_centers])
        
        # Calculate set centroids
        set_centroids = []
        for circle_set in circle_sets:
            cx = np.mean([c['center'][0] for c in circle_set])
            cy = np.mean([c['center'][1] for c in circle_set])
            set_centroids.append((cx, cy))
        
        # Calculate pattern metrics
        analysis = {
            'pattern_center': (pattern_center_x, pattern_center_y),
            'num_sets': len(circle_sets),
            'set_centroids': set_centroids,
            'circles_per_set': [len(s) for s in circle_sets],
            'total_circles': sum(len(s) for s in circle_sets)
        }
        
        # Calculate radius ratios for each set
        radius_ratios = []
        for circle_set in circle_sets:
            if len(circle_set) == 2:
                ratio = circle_set[0]['radius'] / circle_set[1]['radius']
                radius_ratios.append(ratio)
        
        if radius_ratios:
            analysis['mean_radius_ratio'] = np.mean(radius_ratios)
            analysis['std_radius_ratio'] = np.std(radius_ratios)
        
        # Calculate pattern symmetry
        if len(set_centroids) == 4:
            # Check for cross/diamond pattern symmetry
            distances_to_center = []
            for centroid in set_centroids:
                dist = np.sqrt((centroid[0] - pattern_center_x)**2 + 
                             (centroid[1] - pattern_center_y)**2)
                distances_to_center.append(dist)
            
            analysis['symmetry_score'] = 1.0 / (1.0 + np.std(distances_to_center))
        
        return analysis
    
    def visualize_preprocessing_stages(self, image_name, stages_data):
        """
        Create visualization of all preprocessing stages
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Pattern Detection Stages: {image_name}', fontsize=16)
        
        stages_to_plot = [
            ('Original', stages_data['original']),
            ('Normalized', stages_data['normalized']),
            ('Gradient Magnitude', stages_data['gradient']),
            ('Enhanced Circular', stages_data['enhanced']),
            ('Binary (Cleaned)', stages_data['binary']),
            ('Edges', stages_data['edges']),
            ('All Circles', stages_data['all_circles_vis']),
            ('Pattern Detected', stages_data['pattern_vis']),
            ('Analysis', stages_data['analysis_vis'])
        ]
        
        for idx, (title, img) in enumerate(stages_to_plot):
            ax = axes[idx // 3, idx % 3]
            if img is not None:
                if len(img.shape) == 3:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_pattern_visualization(self, image, circle_sets, all_circles, analysis):
        """
        Create visualization showing detected pattern
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()
        
        # Colors for different sets
        set_colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        
        # Draw all detected circles faintly
        for circle in all_circles:
            cv2.circle(vis, circle['center'], circle['radius'], (128, 128, 128), 1)
        
        # Draw pattern circles with distinct colors
        for i, circle_set in enumerate(circle_sets):
            color = set_colors[i % len(set_colors)]
            for circle in circle_set:
                cv2.circle(vis, circle['center'], circle['radius'], color, 2)
                cv2.circle(vis, circle['center'], 2, (0, 0, 255), 3)
            
            # Draw set centroid
            if len(circle_set) > 0:
                cx = int(np.mean([c['center'][0] for c in circle_set]))
                cy = int(np.mean([c['center'][1] for c in circle_set]))
                cv2.drawMarker(vis, (cx, cy), color, cv2.MARKER_CROSS, 10, 2)
        
        # Draw pattern center if available
        if analysis and 'pattern_center' in analysis:
            pc = (int(analysis['pattern_center'][0]), int(analysis['pattern_center'][1]))
            cv2.drawMarker(vis, pc, (255, 0, 0), cv2.MARKER_STAR, 15, 2)
            cv2.putText(vis, 'Pattern Center', (pc[0] + 10, pc[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Add text annotations
        text_y = 30
        cv2.putText(vis, f"Pattern Sets: {len(circle_sets)}/4", 
                   (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if analysis:
            text_y += 25
            cv2.putText(vis, f"Total Circles: {analysis.get('total_circles', 0)}", 
                       (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if 'symmetry_score' in analysis:
                text_y += 25
                cv2.putText(vis, f"Symmetry: {analysis['symmetry_score']:.3f}", 
                           (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return vis
    
    def process_single_image(self, image_path):
        """
        Process a single thermal image through all preprocessing stages
        """
        image_name = Path(image_path).stem
        print(f"\nProcessing: {image_name}")
        sys.stdout.flush()  # Force output
        
        try:
            # Load image
            print(f"  Loading image...")
            sys.stdout.flush()
            original = self.load_thermal_image(image_path)
            if original is None:
                print(f"Failed to load {image_path}")
                return None
            print(f"  Image loaded: shape={original.shape}, dtype={original.dtype}")
            sys.stdout.flush()
            
            # Stage 1: Normalization
            print(f"  Stage 1: Normalizing...")
            sys.stdout.flush()
            normalized = self.thermal_normalization(original)
            
            # Stage 2: Gradient computation
            print(f"  Stage 2: Computing gradients...")
            sys.stdout.flush()
            gradient_mag, gradient_dir = self.compute_gradient_magnitude(normalized)
            
            # Stage 3: Enhance circular features
            print(f"  Stage 3: Enhancing circular features...")
            sys.stdout.flush()
            enhanced = self.enhance_circular_features(normalized)
            
            # Stage 4: Binary segmentation (improved)
            print(f"  Stage 4: Binary segmentation...")
            sys.stdout.flush()
            binary = self.adaptive_binary_segmentation(enhanced)
            
            # Stage 5: Edge detection
            print(f"  Stage 5: Detecting edges...")
            sys.stdout.flush()
            edges = self.detect_edges_for_circles(enhanced)
            
            # Stage 6: Detect concentric circle sets
            print(f"  Stage 6: Detecting circles...")
            sys.stdout.flush()
            circle_sets, all_circles = self.detect_concentric_circle_sets(gradient_mag, enhanced, edges)
            
            # Stage 7: Pattern analysis
            print(f"  Stage 7: Analyzing pattern...")
            sys.stdout.flush()
            analysis = self.analyze_pattern(circle_sets)
            
            # Create visualizations
            print(f"  Stage 8: Creating visualizations...")
            sys.stdout.flush()
            
            # All circles visualization
            all_circles_vis = original.copy()
        if len(all_circles_vis.shape) == 2:
            all_circles_vis = cv2.cvtColor(all_circles_vis, cv2.COLOR_GRAY2BGR)
        
        for circle in all_circles:
            cv2.circle(all_circles_vis, circle['center'], circle['radius'], (0, 255, 0), 1)
            cv2.circle(all_circles_vis, circle['center'], 2, (0, 0, 255), 2)
        
        # Pattern visualization
        pattern_vis = self.create_pattern_visualization(original, circle_sets, all_circles, analysis)
        
        # Analysis visualization
        analysis_vis = self.create_pattern_visualization(normalized, circle_sets, all_circles, analysis)
        
        # Save all stages
        cv2.imwrite(str(self.stages['original'] / f"{image_name}.png"), original)
        cv2.imwrite(str(self.stages['normalized'] / f"{image_name}.png"), normalized)
        cv2.imwrite(str(self.stages['gradient'] / f"{image_name}.png"), gradient_mag)
        cv2.imwrite(str(self.stages['enhanced'] / f"{image_name}.png"), enhanced)
        cv2.imwrite(str(self.stages['binary'] / f"{image_name}.png"), binary)
        cv2.imwrite(str(self.stages['edges'] / f"{image_name}.png"), edges)
        cv2.imwrite(str(self.stages['circles'] / f"{image_name}_all.png"), all_circles_vis)
        cv2.imwrite(str(self.stages['pattern'] / f"{image_name}_pattern.png"), pattern_vis)
        cv2.imwrite(str(self.stages['pattern'] / f"{image_name}_analysis.png"), analysis_vis)
        
        # Create comprehensive visualization
        stages_data = {
            'original': original,
            'normalized': normalized,
            'gradient': gradient_mag,
            'enhanced': enhanced,
            'binary': binary,
            'edges': edges,
            'all_circles_vis': all_circles_vis,
            'pattern_vis': pattern_vis,
            'analysis_vis': analysis_vis
        }
        
        # Save comprehensive figure with error handling
        try:
            fig = self.visualize_preprocessing_stages(image_name, stages_data)
            fig.savefig(self.stages['pattern'] / f"{image_name}_all_stages.png", 
                       dpi=100, bbox_inches='tight')
            plt.close(fig)
            plt.clf()
            plt.close('all')
        except Exception as e:
            print(f"Warning: Could not save comprehensive visualization for {image_name}: {e}")
        
        # Store results
        result = {
            'image_name': image_name,
            'image_path': str(image_path),
            'total_circles_detected': len(all_circles),
            'pattern_sets': len(circle_sets),
            'circle_sets': circle_sets,
            'analysis': analysis
        }
        
        self.preprocessing_stats.append(result)
        
        # Print immediate feedback
        print(f"  - Detected {len(all_circles)} total circles")
        print(f"  - Found {len(circle_sets)} concentric sets")
        if analysis:
            print(f"  - Pattern center: ({analysis['pattern_center'][0]:.1f}, {analysis['pattern_center'][1]:.1f})")
            if 'symmetry_score' in analysis:
                print(f"  - Symmetry score: {analysis['symmetry_score']:.3f}")
        
        return result
    
    def process_dataset(self, image_dir):
        """
        Process all images in the dataset
        """
        image_dir = Path(image_dir)
        image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
        
        all_images = []
        for pattern in image_patterns:
            all_images.extend(list(image_dir.rglob(pattern)))
        
        # Exclude already processed images
        all_images = [img for img in all_images if 'preprocessed' not in str(img)]
        
        print(f"Found {len(all_images)} images to process")
        print("Target Pattern: 4 sets of concentric circles in diamond/cross formation")
        print("-" * 60)
        
        results = []
        for i, img_path in enumerate(all_images):
            print(f"\n[{i+1}/{len(all_images)}] ", end="")
            result = self.process_single_image(img_path)
            if result:
                results.append(result)
            
            # Clear matplotlib memory periodically
            if (i + 1) % 10 == 0:
                plt.close('all')
        
        # Save preprocessing statistics
        stats_file = self.output_dir / 'pattern_detection_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results):
        """
        Generate summary report of pattern detection results
        """
        report = []
        report.append("=" * 80)
        report.append("PATTERN DETECTION SUMMARY REPORT")
        report.append("Target: 4 sets of concentric circles in diamond/cross pattern")
        report.append("=" * 80)
        report.append(f"\nTotal images processed: {len(results)}")
        
        # Pattern detection statistics
        pattern_counts = [r['pattern_sets'] for r in results]
        if pattern_counts:
            report.append(f"\nPattern Detection Statistics:")
            report.append(f"  Average sets detected: {np.mean(pattern_counts):.2f}")
            report.append(f"  Images with 4 sets (perfect): {sum(1 for c in pattern_counts if c == 4)}")
            report.append(f"  Images with 3 sets: {sum(1 for c in pattern_counts if c == 3)}")
            report.append(f"  Images with 2 sets: {sum(1 for c in pattern_counts if c == 2)}")
        
        # Circle detection statistics
        circle_counts = [r['total_circles_detected'] for r in results]
        if circle_counts:
            report.append(f"\nCircle Detection Statistics:")
            report.append(f"  Average circles detected: {np.mean(circle_counts):.2f}")
            report.append(f"  Min circles: {np.min(circle_counts)}")
            report.append(f"  Max circles: {np.max(circle_counts)}")
        
        # Symmetry analysis
        symmetry_scores = []
        for r in results:
            if r['analysis'] and 'symmetry_score' in r['analysis']:
                symmetry_scores.append(r['analysis']['symmetry_score'])
        
        if symmetry_scores:
            report.append(f"\nPattern Symmetry Analysis:")
            report.append(f"  Average symmetry score: {np.mean(symmetry_scores):.3f}")
            report.append(f"  Best symmetry: {np.max(symmetry_scores):.3f}")
            report.append(f"  Worst symmetry: {np.min(symmetry_scores):.3f}")
        
        # Best performing images
        perfect_pattern = [r for r in results if r['pattern_sets'] == 4]
        if perfect_pattern:
            report.append(f"\nImages with Perfect Pattern Detection (4 sets):")
            for i, r in enumerate(perfect_pattern[:10]):  # Show top 10
                report.append(f"  - {r['image_name']}")
        
        report.append("\n" + "=" * 80)
        
        # Save report
        report_text = '\n'.join(report)
        report_file = self.output_dir / 'pattern_detection_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        
        return report_text


def main():
    """
    Main execution function
    """
    # Initialize detector
    detector = ThermalPatternDetector(output_dir='preprocessed_pattern_detection')
    
    # Process dataset
    dataset_dir = Path('.')  # Current directory
    results = detector.process_dataset(dataset_dir)
    
    print(f"\nPattern detection complete!")
    print(f"Results saved in: {detector.output_dir}")
    
    return results


if __name__ == "__main__":
    results = main()
