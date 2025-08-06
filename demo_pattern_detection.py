"""
Demo Script for Pattern Detection
Tests the algorithm on a few sample images to demonstrate functionality
"""

import os
from pattern_detector import RobustPatternDetector


def demo_pattern_detection():
    """
    Run a quick demo of pattern detection on sample images
    """
    print("=== PATTERN DETECTION DEMO ===")
    print("Testing robust circle pattern detection algorithm")
    print("-" * 50)
    
    # Initialize detector
    detector = RobustPatternDetector(
        min_circle_area=50,
        roundness_threshold=0.15,  # Slightly relaxed for more detection
        ellipse_ratio_threshold=0.85,
        concentric_distance_threshold=20.0,
        pattern_tolerance=0.3
    )
    
    # Test on first few images
    test_images = [
        "dataset/IMG_10072025161331.png",
        "dataset/IMG_10072025161333.png", 
        "dataset/IMG_10072025161339.png",
        "dataset/IMG_10072025161342.png",
        "dataset/IMG_10072025161344.png"
    ]
    
    results_summary = []
    
    for i, image_path in enumerate(test_images):
        if not os.path.exists(image_path):
            print(f"Image {i+1}: {image_path} - NOT FOUND")
            continue
            
        print(f"\nProcessing Image {i+1}: {os.path.basename(image_path)}")
        print("-" * 30)
        
        try:
            # Process image
            result = detector.detect_pattern(image_path, visualize=False)
            
            # Print results
            print(f"✓ Pattern Found: {result['pattern_found']}")
            print(f"✓ Circles Detected: {len(result['detected_circles'])}")
            print(f"✓ Concentric Pairs: {len(result['concentric_pairs'])}")
            
            if result['pattern_found']:
                confidence = result['confidence_metrics']['overall_confidence']
                quality = result['confidence_metrics']['pattern_quality']
                print(f"✓ Overall Confidence: {confidence:.3f}")
                print(f"✓ Pattern Quality: {quality:.3f}")
                
                # Show pattern points
                pattern_points = result['pattern_centroids']
                print("✓ Pattern Points:")
                for j, point in enumerate(pattern_points):
                    print(f"   Point {j+1}: ({point[0]:.1f}, {point[1]:.1f})")
            else:
                print("✗ No valid pattern detected")
                
            # Save results
            detector.save_pattern_results("demo_results")
            
            # Store summary
            results_summary.append({
                'image': os.path.basename(image_path),
                'success': result['pattern_found'],
                'circles': len(result['detected_circles']),
                'pairs': len(result['concentric_pairs']),
                'confidence': result['confidence_metrics'].get('overall_confidence', 0.0)
            })
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            results_summary.append({
                'image': os.path.basename(image_path),
                'success': False,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    
    successful = [r for r in results_summary if r.get('success', False)]
    
    print(f"Images Processed: {len(results_summary)}")
    print(f"Successful Detections: {len(successful)}")
    print(f"Success Rate: {len(successful)/len(results_summary)*100:.1f}%")
    
    if successful:
        avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
        print(f"Average Confidence: {avg_confidence:.3f}")
        
        print("\nBest Results:")
        best_results = sorted(successful, key=lambda x: x['confidence'], reverse=True)
        for i, result in enumerate(best_results[:3]):
            print(f"{i+1}. {result['image']} - Confidence: {result['confidence']:.3f}")
    
    print("\nDemo complete! Check 'demo_results' folder for output files.")
    

if __name__ == "__main__":
    demo_pattern_detection()
