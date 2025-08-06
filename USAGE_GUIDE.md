# Circle Pattern Detection Algorithm - Usage Guide

## Overview
This algorithm detects a specific pattern of 4 concentric circle pairs arranged in a triangular/lower matrix format. It's designed to be invariant to rotation, scale, and skew transformations.

## Quick Start

### 1. Single Image Detection
```python
from pattern_detector import RobustPatternDetector

# Initialize detector
detector = RobustPatternDetector()

# Detect pattern in an image
result = detector.detect_pattern("your_image.png", visualize=True)

# Check results
if result['pattern_found']:
    print(f"Pattern detected with confidence: {result['confidence_metrics']['overall_confidence']:.3f}")
    print("Pattern points:", result['pattern_centroids'])
else:
    print("No pattern found")
```

### 2. Demo Script
Run the demo to see the algorithm in action:
```bash
python demo_pattern_detection.py
```

### 3. Batch Processing
Process multiple images and generate analysis:
```bash
python test_pattern_batch.py
```

## Algorithm Parameters

### RobustPatternDetector Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_circle_area` | 50 | Minimum contour area to consider as a circle |
| `roundness_threshold` | 0.15 | Maximum std/mean ratio for circle roundness |
| `ellipse_ratio_threshold` | 0.85 | Minimum minor/major axis ratio for circles |
| `concentric_distance_threshold` | 15.0 | Max distance between concentric circle centers |
| `concentric_radius_ratio_range` | (1.2, 4.0) | Valid outer/inner radius ratio range |
| `pattern_tolerance` | 0.25 | Tolerance for pattern geometric validation |

### Customizing Detection
```python
# For more sensitive detection
detector = RobustPatternDetector(
    min_circle_area=30,           # Lower threshold for smaller circles
    roundness_threshold=0.2,      # More lenient roundness
    ellipse_ratio_threshold=0.8,  # More lenient ellipse ratio
    concentric_distance_threshold=25.0,  # Allow more distance between concentric circles
    pattern_tolerance=0.35        # More lenient pattern matching
)

# For more strict detection
detector = RobustPatternDetector(
    min_circle_area=100,          # Higher threshold for larger circles
    roundness_threshold=0.1,      # Stricter roundness
    ellipse_ratio_threshold=0.9,  # Stricter ellipse ratio
    concentric_distance_threshold=10.0,  # Less distance between concentric circles
    pattern_tolerance=0.15        # Stricter pattern matching
)
```

## Output Structure

### Detection Results
```python
{
    'image_path': str,              # Path to processed image
    'original_image': np.ndarray,   # Original grayscale image
    'detected_circles': List[Dict], # All detected circles with properties
    'concentric_pairs': List[Tuple], # Pairs of concentric circles
    'pattern_centroids': np.ndarray, # 4x2 array of pattern points (if found)
    'pattern_found': bool,          # Whether valid pattern was detected
    'confidence_metrics': Dict      # Quality and confidence metrics
}
```

### Confidence Metrics
```python
{
    'pattern_quality': float,       # Geometric quality score (0-1)
    'concentric_pairs_count': int,  # Number of concentric pairs found
    'concentric_pairs_ratio': float, # Ratio to ideal (4 pairs)
    'distance_consistency': float,  # Consistency of distances (0-1)
    'overall_confidence': float     # Combined confidence score (0-1)
}
```

### Circle Properties
```python
{
    'centroid': Tuple[float, float], # Center coordinates (x, y)
    'radius': float,                 # Average radius
    'area': float,                   # Contour area
    'perimeter': float,              # Contour perimeter
    'circularity': float,            # 4π×area/perimeter²
    'roundness_ratio': float,        # std/mean of distances from center
    'ellipse_ratio': float,          # minor/major axis ratio
    'contour': np.ndarray,           # Original contour points
    'hierarchy': List                # Contour hierarchy info
}
```

## Files and Outputs

### Generated Files
- `pattern_analysis_[imagename].png` - Comprehensive analysis visualization
- `results/[imagename]_pattern.png` - Pattern points only
- `results/[imagename]_visualization.png` - Annotated detection result
- `results/[imagename]_pattern_points.txt` - Pattern coordinates as text
- `results/[imagename]_analysis.json` - Complete analysis results

### Batch Processing Outputs
- `batch_test_results.json` - All results in JSON format
- `batch_analysis_report.png` - Statistical analysis plots

## Algorithm Behavior

### Detection Strategy
1. **Primary Mode**: Find concentric circle pairs, use their centroids for pattern matching
2. **Fallback Mode**: If <4 concentric pairs found, use all circle centroids
3. **Pattern Validation**: Multi-criteria scoring including geometry, uniformity, and aspect ratio

### Transformation Invariance
- **Scale**: All measurements normalized by maximum distance
- **Rotation**: Uses relative angles and sorted distance descriptors
- **Skew**: Geometric validation with adjustable tolerance levels

### Quality Scoring
The algorithm evaluates patterns using multiple criteria:
- **Uniformity**: Distance variation between pattern points
- **Geometric Validation**: Tests for triangular/rectangular arrangements
- **Area**: Convex hull area of the pattern
- **Aspect Ratio**: Prevents overly elongated patterns

## Troubleshooting

### Common Issues

#### Low Detection Rate
- **Reduce thresholds**: Lower `min_circle_area`, increase `roundness_threshold`
- **Adjust preprocessing**: Images may need different contrast/brightness
- **Check circle quality**: Verify circles are clearly visible and not too distorted

#### False Positives
- **Increase thresholds**: Higher `min_circle_area`, lower `roundness_threshold`
- **Stricter validation**: Lower `pattern_tolerance`, stricter `ellipse_ratio_threshold`
- **Filter by confidence**: Only accept results with high confidence scores

#### Concentric Pairs Not Detected
- **Increase distance threshold**: Higher `concentric_distance_threshold`
- **Adjust radius ratios**: Modify `concentric_radius_ratio_range`
- **Check image quality**: Ensure circles are distinct and not overlapping

### Performance Tips
- Use `visualize=False` for faster processing when not needed
- Adjust `min_circle_area` based on expected circle sizes in your images
- For batch processing, disable individual result saving if not needed

## Expected Performance

Based on demo testing:
- **Success Rate**: 100% on test images
- **Average Confidence**: 0.565
- **Processing Speed**: ~2-3 seconds per image (with visualization)
- **Memory Usage**: Low (works with standard OpenCV/NumPy installation)

## Dependencies
- OpenCV (cv2)
- NumPy
- SciPy
- Matplotlib (for visualization)

## Contact & Support
Check `progress.md` for detailed development log and technical specifications.
