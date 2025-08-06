"""
Batch Testing Script for Pattern Detection
Tests the algorithm on multiple images with various transformations
"""

import os
import glob
import json
import numpy as np
# Using manual analysis instead of pandas
from pattern_detector import RobustPatternDetector
import matplotlib.pyplot as plt


class BatchTester:
    """
    Batch testing class for pattern detection algorithm
    """
    
    def __init__(self, dataset_path: str = "dataset"):
        self.dataset_path = dataset_path
        self.detector = RobustPatternDetector()
        self.results = []
    
    def test_all_images(self, visualize_each: bool = False, save_results: bool = True):
        """
        Test pattern detection on all images in the dataset
        """
        # Find all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            pattern = os.path.join(self.dataset_path, ext)
            image_files.extend(glob.glob(pattern))
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for i, image_path in enumerate(image_files):
            print(f"\nProcessing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            try:
                # Detect pattern
                result = self.detector.detect_pattern(image_path, visualize=visualize_each)
                
                # Store results
                image_result = {
                    'image_name': os.path.basename(image_path),
                    'image_path': image_path,
                    'pattern_found': result['pattern_found'],
                    'num_circles': len(result['detected_circles']),
                    'num_concentric_pairs': len(result['concentric_pairs']),
                    'confidence': result['confidence_metrics'].get('overall_confidence', 0.0),
                    'pattern_quality': result['confidence_metrics'].get('pattern_quality', 0.0),
                    'distance_consistency': result['confidence_metrics'].get('distance_consistency', 0.0)
                }
                
                if result['pattern_found']:
                    # Calculate pattern properties
                    pattern_points = result['pattern_centroids']
                    
                    # Calculate distances and area for scale analysis
                    from scipy.spatial.distance import pdist
                    distances = pdist(pattern_points)
                    image_result.update({
                        'pattern_area': self._calculate_pattern_area(pattern_points),
                        'pattern_mean_distance': np.mean(distances),
                        'pattern_max_distance': np.max(distances),
                        'pattern_min_distance': np.min(distances),
                        'pattern_distance_ratio': np.max(distances) / np.min(distances) if np.min(distances) > 0 else 0
                    })
                else:
                    image_result.update({
                        'pattern_area': 0,
                        'pattern_mean_distance': 0,
                        'pattern_max_distance': 0,
                        'pattern_min_distance': 0,
                        'pattern_distance_ratio': 0
                    })
                
                self.results.append(image_result)
                
                # Save individual results if requested
                if save_results:
                    self.detector.save_pattern_results()
                
                # Progress update
                success_rate = sum(1 for r in self.results if r['pattern_found']) / len(self.results)
                print(f"Current success rate: {success_rate:.1%}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                # Store error result
                error_result = {
                    'image_name': os.path.basename(image_path),
                    'image_path': image_path,
                    'pattern_found': False,
                    'error': str(e),
                    'num_circles': 0,
                    'num_concentric_pairs': 0,
                    'confidence': 0.0,
                    'pattern_quality': 0.0,
                    'distance_consistency': 0.0,
                    'pattern_area': 0,
                    'pattern_mean_distance': 0,
                    'pattern_max_distance': 0,
                    'pattern_min_distance': 0,
                    'pattern_distance_ratio': 0
                }
                self.results.append(error_result)
        
        print(f"\n\nBatch processing complete!")
        print(f"Total images processed: {len(self.results)}")
        print(f"Patterns detected: {sum(1 for r in self.results if r['pattern_found'])}")
        print(f"Success rate: {sum(1 for r in self.results if r['pattern_found']) / len(self.results):.1%}")
        
        return self.results
    
    def _calculate_pattern_area(self, points):
        """Calculate the area of the convex hull of pattern points"""
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            return hull.volume  # In 2D, volume is area
        except:
            return 0
    
    def save_batch_results(self, output_file: str = "batch_results.json"):
        """
        Save batch test results to JSON file
        """
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Batch results saved to {output_file}")
    
    def generate_analysis_report(self):
        """
        Generate comprehensive analysis report
        """
        if not self.results:
            print("No results to analyze")
            return
        
        print("\n" + "="*60)
        print("PATTERN DETECTION ANALYSIS REPORT")
        print("="*60)
        
        total_images = len(self.results)
        successful = [r for r in self.results if r['pattern_found']]
        failed = [r for r in self.results if not r['pattern_found']]
        
        # Basic statistics
        print(f"\nBASIC STATISTICS:")
        print(f"Total images processed: {total_images}")
        print(f"Patterns successfully detected: {len(successful)}")
        print(f"Success rate: {len(successful)/total_images:.1%}")
        
        # Circle detection statistics
        num_circles = [r['num_circles'] for r in self.results]
        num_pairs = [r['num_concentric_pairs'] for r in self.results]
        
        print(f"\nCIRCLE DETECTION STATISTICS:")
        print(f"Average circles per image: {np.mean(num_circles):.1f}")
        print(f"Average concentric pairs: {np.mean(num_pairs):.1f}")
        print(f"Max circles detected: {max(num_circles)}")
        print(f"Min circles detected: {min(num_circles)}")
        
        # Confidence statistics for successful detections
        if successful:
            confidences = [r['confidence'] for r in successful]
            qualities = [r['pattern_quality'] for r in successful]
            consistencies = [r['distance_consistency'] for r in successful]
            
            print(f"\nCONFIDENCE METRICS (successful detections only):")
            print(f"Average overall confidence: {np.mean(confidences):.3f}")
            print(f"Average pattern quality: {np.mean(qualities):.3f}")
            print(f"Average distance consistency: {np.mean(consistencies):.3f}")
            
            # Pattern size analysis
            areas = [r['pattern_area'] for r in successful]
            mean_dists = [r['pattern_mean_distance'] for r in successful]
            dist_ratios = [r['pattern_distance_ratio'] for r in successful]
            
            print(f"\nPATTERN SIZE ANALYSIS:")
            print(f"Average pattern area: {np.mean(areas):.1f}")
            print(f"Average mean distance: {np.mean(mean_dists):.1f}")
            print(f"Average distance ratio: {np.mean(dist_ratios):.2f}")
        
        # Failed detections analysis
        if failed:
            failed_circles = [r['num_circles'] for r in failed]
            failed_pairs = [r['num_concentric_pairs'] for r in failed]
            
            print(f"\nFAILED DETECTIONS ANALYSIS:")
            print(f"Images with 0 circles detected: {sum(1 for c in failed_circles if c == 0)}")
            print(f"Images with >0 but <4 concentric pairs: {sum(1 for p in failed_pairs if 0 < p < 4)}")
            print(f"Average circles in failed detections: {np.mean(failed_circles):.1f}")
        
        # Create visualizations
        self._create_analysis_plots()
    
    def _create_analysis_plots(self):
        """
        Create analysis plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Pattern Detection Analysis', fontsize=16)
        
        successful = [r for r in self.results if r['pattern_found']]
        failed = [r for r in self.results if not r['pattern_found']]
        
        # Success rate pie chart
        success_count = len(successful)
        failed_count = len(failed)
        if success_count + failed_count > 0:
            axes[0, 0].pie([failed_count, success_count], labels=['Failed', 'Success'], 
                          autopct='%1.1f%%', colors=['red', 'green'])
        axes[0, 0].set_title('Detection Success Rate')
        
        # Circle detection histogram
        num_circles = [r['num_circles'] for r in self.results]
        axes[0, 1].hist(num_circles, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribution of Circles Detected')
        axes[0, 1].set_xlabel('Number of Circles')
        axes[0, 1].set_ylabel('Frequency')
        
        # Concentric pairs histogram
        num_pairs = [r['num_concentric_pairs'] for r in self.results]
        axes[0, 2].hist(num_pairs, bins=10, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Distribution of Concentric Pairs')
        axes[0, 2].set_xlabel('Number of Concentric Pairs')
        axes[0, 2].set_ylabel('Frequency')
        
        # Confidence distribution (successful only)
        if successful:
            confidences = [r['confidence'] for r in successful]
            axes[1, 0].hist(confidences, bins=15, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Confidence Distribution\n(Successful Detections)')
            axes[1, 0].set_xlabel('Overall Confidence')
            axes[1, 0].set_ylabel('Frequency')
        
        # Pattern area distribution
        if successful:
            areas = [r['pattern_area'] for r in successful]
            axes[1, 1].hist(areas, bins=15, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Pattern Area Distribution')
            axes[1, 1].set_xlabel('Pattern Area')
            axes[1, 1].set_ylabel('Frequency')
        
        # Circles vs Success scatter
        circles_x = [r['num_circles'] for r in self.results]
        pairs_y = [r['num_concentric_pairs'] for r in self.results]
        success_colors = ['green' if r['pattern_found'] else 'red' for r in self.results]
        axes[1, 2].scatter(circles_x, pairs_y, c=success_colors, alpha=0.6)
        axes[1, 2].set_title('Circles vs Concentric Pairs\n(Green=Success, Red=Failed)')
        axes[1, 2].set_xlabel('Number of Circles')
        axes[1, 2].set_ylabel('Number of Concentric Pairs')
        
        plt.tight_layout()
        plt.savefig('batch_analysis_report.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def find_best_images(self, top_n: int = 5):
        """
        Find the best detected patterns for reference
        """
        successful = [r for r in self.results if r['pattern_found']]
        
        if not successful:
            print("No successful detections found")
            return
        
        # Sort by confidence
        sorted_results = sorted(successful, key=lambda x: x['confidence'], reverse=True)
        
        print(f"\nTOP {top_n} BEST PATTERN DETECTIONS:")
        print("-" * 60)
        
        for i, result in enumerate(sorted_results[:top_n]):
            print(f"{i+1}. {result['image_name']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Pattern Quality: {result['pattern_quality']:.3f}")
            print(f"   Circles: {result['num_circles']}")
            print(f"   Concentric Pairs: {result['num_concentric_pairs']}")
            print()


def main():
    """
    Main function to run batch testing
    """
    # Create batch tester
    tester = BatchTester()
    
    # Test all images (set visualize_each=True to see each image analysis)
    print("Starting batch pattern detection...")
    results = tester.test_all_images(visualize_each=False, save_results=True)
    
    # Save batch results
    tester.save_batch_results("batch_test_results.json")
    
    # Generate analysis report
    tester.generate_analysis_report()
    
    # Find best examples
    tester.find_best_images(top_n=10)
    
    print("\nBatch testing complete!")


if __name__ == "__main__":
    main()
