import cv2
import numpy as np
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
from sklearn.decomposition import PCA

def classify_pattern(points):
    """
    Classify the pattern formed by 4 points.
    
    Args:
        points: Array of 4 points (4x2)
    
    Returns:
        dict: Pattern classification with type, confidence, and properties
    """
    if len(points) != 4:
        return {'type': 'invalid', 'confidence': 0}
    
    # Calculate all pairwise distances
    distances = pdist(points)
    dist_matrix = squareform(distances)
    
    # Sort distances
    sorted_distances = np.sort(distances)
    
    # Calculate properties
    min_dist = sorted_distances[0]
    max_dist = sorted_distances[-1]
    
    # Normalize distances
    if max_dist > 0:
        norm_distances = sorted_distances / max_dist
    else:
        return {'type': 'invalid', 'confidence': 0}
    
    # Pattern detection based on distance ratios
    pattern_info = {
        'distances': sorted_distances,
        'normalized_distances': norm_distances,
        'centroid': np.mean(points, axis=0),
        'area': cv2.contourArea(points.astype(np.int32))
    }
    
    # Check for square pattern (4 equal sides, 2 equal diagonals)
    # In a square: 4 sides of length s, 2 diagonals of length s*sqrt(2)
    # Sorted distances: [s, s, s, s, s*sqrt(2), s*sqrt(2)]
    if len(sorted_distances) == 6:
        side_lengths = sorted_distances[:4]
        diagonal_lengths = sorted_distances[4:6]
        
        # Check if first 4 distances are similar (sides)
        side_std = np.std(side_lengths) / np.mean(side_lengths) if np.mean(side_lengths) > 0 else 1
        diag_std = np.std(diagonal_lengths) / np.mean(diagonal_lengths) if np.mean(diagonal_lengths) > 0 else 1
        
        # Expected ratio for square: diagonal/side = sqrt(2) ≈ 1.414
        if side_std < 0.1 and diag_std < 0.1:
            expected_ratio = np.sqrt(2)
            actual_ratio = np.mean(diagonal_lengths) / np.mean(side_lengths)
            ratio_error = abs(actual_ratio - expected_ratio) / expected_ratio
            
            if ratio_error < 0.1:
                pattern_info['type'] = 'square'
                pattern_info['confidence'] = 1.0 - ratio_error
                pattern_info['side_length'] = np.mean(side_lengths)
                return pattern_info
    
    # Check for diamond/rhombus pattern
    # Similar to square but rotated 45 degrees
    if len(sorted_distances) == 6:
        # Check if we have 4 equal sides
        side_candidates = sorted_distances[:4]
        if np.std(side_candidates) / np.mean(side_candidates) < 0.15:
            pattern_info['type'] = 'diamond'
            pattern_info['confidence'] = 0.8
            pattern_info['side_length'] = np.mean(side_candidates)
            return pattern_info
    
    # Check for rectangular pattern
    # Rectangle has 2 pairs of equal sides and 2 equal diagonals
    if len(sorted_distances) == 6:
        # Group distances
        # Should have: 2 of length a, 2 of length b, 2 diagonals
        unique_dists, counts = np.unique(np.round(sorted_distances, 2), return_counts=True)
        if len(counts) >= 2 and np.sum(counts == 2) >= 2:
            pattern_info['type'] = 'rectangle'
            pattern_info['confidence'] = 0.7
            return pattern_info
    
    # Check for triangular pattern (3 points close, 1 point far)
    # This would show as 3 short distances and 3 long distances
    if len(sorted_distances) == 6:
        ratio = sorted_distances[3] / sorted_distances[2] if sorted_distances[2] > 0 else 0
        if ratio > 1.5:  # Clear separation between short and long distances
            pattern_info['type'] = 'triangular'
            pattern_info['confidence'] = min(0.9, ratio / 2)
            return pattern_info
    
    # Default to irregular quadrilateral
    pattern_info['type'] = 'irregular'
    pattern_info['confidence'] = 0.5
    return pattern_info

def analyze_pattern_orientation(points):
    """
    Analyze the orientation and alignment of the pattern.
    
    Args:
        points: Array of 4 points (4x2)
    
    Returns:
        dict: Orientation information including angle and principal axes
    """
    # Use PCA to find principal axes
    pca = PCA(n_components=2)
    pca.fit(points)
    
    # Get the angle of the first principal component
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    angle_degrees = np.degrees(angle)
    
    # Calculate the spread along each axis
    transformed = pca.transform(points)
    spread_ratio = np.std(transformed[:, 0]) / np.std(transformed[:, 1]) if np.std(transformed[:, 1]) > 0 else float('inf')
    
    return {
        'angle_degrees': angle_degrees,
        'principal_components': pca.components_,
        'explained_variance': pca.explained_variance_ratio_,
        'spread_ratio': spread_ratio,
        'centroid': np.mean(points, axis=0)
    }

def visualize_pattern(points, pattern_info, img_shape, output_path=None):
    """
    Create a visualization of the detected pattern.
    
    Args:
        points: Array of 4 points (4x2)
        pattern_info: Dictionary with pattern classification
        img_shape: Shape of the original image
        output_path: Optional path to save the visualization
    
    Returns:
        img: Visualization image
    """
    # Create a color image for visualization
    img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    
    # Draw the points
    for pt in points:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
        cv2.circle(img, (int(pt[0]), int(pt[1])), 10, (0, 255, 255), 2)
    
    # Draw connections based on pattern type
    if pattern_info['type'] in ['square', 'rectangle', 'diamond']:
        # Connect as quadrilateral
        pts_ordered = order_points_clockwise(points)
        for i in range(4):
            pt1 = tuple(pts_ordered[i].astype(int))
            pt2 = tuple(pts_ordered[(i+1)%4].astype(int))
            cv2.line(img, pt1, pt2, (255, 128, 0), 2)
        
        # Draw diagonals with dashed lines
        cv2.line(img, tuple(pts_ordered[0].astype(int)), tuple(pts_ordered[2].astype(int)), (128, 128, 255), 1)
        cv2.line(img, tuple(pts_ordered[1].astype(int)), tuple(pts_ordered[3].astype(int)), (128, 128, 255), 1)
    else:
        # Connect all points
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                pt1 = tuple(points[i].astype(int))
                pt2 = tuple(points[j].astype(int))
                cv2.line(img, pt1, pt2, (100, 100, 100), 1)
    
    # Draw centroid
    centroid = pattern_info.get('centroid', np.mean(points, axis=0))
    cv2.circle(img, tuple(centroid.astype(int)), 5, (255, 0, 255), -1)
    
    # Add text annotation
    text = f"{pattern_info['type'].upper()} (conf: {pattern_info['confidence']:.2f})"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, img)
    
    return img

def order_points_clockwise(points):
    """
    Order 4 points in clockwise order starting from top-left.
    
    Args:
        points: Array of 4 points (4x2)
    
    Returns:
        Ordered array of 4 points
    """
    # Find centroid
    centroid = np.mean(points, axis=0)
    
    # Calculate angles from centroid
    angles = []
    for pt in points:
        angle = np.arctan2(pt[1] - centroid[1], pt[0] - centroid[0])
        angles.append(angle)
    
    # Sort by angle
    sorted_indices = np.argsort(angles)
    
    # Find top-left point (minimum sum of x and y)
    sums = [pt[0] + pt[1] for pt in points]
    start_idx = np.argmin(sums)
    
    # Reorder starting from top-left
    ordered = []
    start_pos = np.where(sorted_indices == start_idx)[0][0]
    for i in range(4):
        ordered.append(points[sorted_indices[(start_pos + i) % 4]])
    
    return np.array(ordered)

def process_single_image(img_path, output_dir, save_visualization=True, debug_mode=False,
                        min_area=50, circularity_threshold=0.15, ellipse_ratio_threshold=0.85):
    """
    Process a single image and save results
    
    Args:
        img_path: Path to the input image
        output_dir: Directory to save output files
        save_visualization: Whether to save visualization plots
        debug_mode: If True, saves additional debug information
        min_area: Minimum contour area to consider (default: 50)
        circularity_threshold: Maximum std/mean ratio for circular detection (default: 0.15)
        ellipse_ratio_threshold: Minimum ellipse axis ratio for circular detection (default: 0.85)
    
    Returns:
        dict: Results containing centroids_for_pose_estimation and other data
    """
    print(f"Processing: {img_path}")
    
    # Create output directory for this image
    img_name = Path(img_path).stem
    img_output_dir = os.path.join(output_dir, img_name)
    os.makedirs(img_output_dir, exist_ok=True)
    
    # Read image
    img = cv2.imread(img_path, 0)
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return None
    
    # Gaussian blur to reduce noise
    img1 = cv2.GaussianBlur(img, (5, 5), 0)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(11,11))
    img2 = clahe.apply(img)
    
    # Adaptive thresholding
    th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 63, 1)
    th2 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 63, 1)
    
    # Inverted image
    img4 = cv2.bitwise_not(th2)
    kernal = np.ones((3,3), np.uint8)
    img3 = cv2.morphologyEx(th2, cv2.MORPH_GRADIENT, kernal)
    
    # Contours
    contour_img = np.zeros_like(img)
    filtered_contours = []
    filtered_hierarchy = []
    
    contours, hierarchy = cv2.findContours(img4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > min_area:
            filtered_contours.append(cnt)
            filtered_hierarchy.append(hierarchy[0][i])
    
    # Find round contours
    round_contours = []
    other_contours = []
    centroids_round_contours = []
    round_contours_img = np.zeros_like(img)
    
    for i, cnt in enumerate(filtered_contours):
        if len(cnt) < 5:
            other_contours.append(cnt)
            continue
        
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            other_contours.append(cnt)
            continue
        
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        if len(cnt.shape) == 3 and cnt.shape[1] == 1:
            pts = cnt.squeeze()
        else:
            pts = cnt
            
        if len(pts.shape) != 2:
            other_contours.append(cnt)
            continue
        
        dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        mean_r = np.mean(dists)
        std_r = np.std(dists)
        ratio = std_r / mean_r
        
        try:
            ellipse = cv2.fitEllipse(cnt)
            (center, axes, angle) = ellipse
            major_axis, minor_axis = max(axes), min(axes)
            ratio_ellipse = minor_axis / major_axis
        except:
            ratio_ellipse = 0
        
        # Debug info for problematic images
        if debug_mode:
            print(f"    Contour {i}: area={cv2.contourArea(cnt):.1f}, ratio={ratio:.3f}, ellipse_ratio={ratio_ellipse:.3f}")
        
        if ratio <= circularity_threshold and ratio_ellipse >= ellipse_ratio_threshold:
            round_contours.append(cnt)
            centroids_round_contours.append((cx, cy, mean_r))
        else:
            other_contours.append(cnt)
    
    # Find unique centroids
    unique_centroids_round_contours = []
    unique_centroids_round_contours_centroid = []
    
    for i, c in enumerate(centroids_round_contours):
        is_unique = True
        for uc in unique_centroids_round_contours:
            dist = np.linalg.norm(np.array(c[0:2]) - np.array(uc[0:2]))
            if dist < 5:
                is_unique = False
                break
        if is_unique:
            unique_centroids_round_contours.append(c)
            unique_centroids_round_contours_centroid.append(c[0:2])
    
    # Find valid targets (4-point groups)
    valid_targets = []
    if len(unique_centroids_round_contours_centroid) >= 4:
        unique_centroids_round_contours_np = np.array(unique_centroids_round_contours_centroid)
        for combo in combinations(range(len(unique_centroids_round_contours_np)), 4):
            pts = unique_centroids_round_contours_np[list(combo)]
            dists_1 = pdist(pts)
            if len(dists_1) > 0 and np.max(dists_1) > 0:
                ratios = dists_1 / np.max(dists_1)
                if np.std(ratios) < 0.2:
                    valid_targets.append(pts)
    
    # Draw contours
    cv2.drawContours(contour_img, filtered_contours, -1, 255, 3)
    cv2.drawContours(round_contours_img, round_contours, -1, 255, 3)
    for i, u_c in enumerate(unique_centroids_round_contours):
        cv2.circle(round_contours_img, (int(u_c[0]), int(u_c[1])), 2, 255, -1)
    
    # Analyze and select best groups
    results = []
    for group in valid_targets:
        pts2 = np.array(group, dtype=np.float32)
        mean_dist = np.mean(pdist(pts2))
        area1 = cv2.contourArea(pts2.astype(np.int32))
        results.append({'pts': pts2, 'mean_dist': mean_dist, 'area': area1})
    
    # Sort by mean_dist
    results_sorted = sorted(results, key=lambda x: x['mean_dist'])
    
    # Compare first and second groups
    groups_to_keep = []
    if len(results_sorted) >= 2:
        r1 = results_sorted[0]
        r2 = results_sorted[1]
        
        mean_dist_ratio = r2['mean_dist'] / r1['mean_dist'] if r1['mean_dist'] != 0 else float('inf')
        area_ratio = r2['area'] / r1['area'] if r1['area'] != 0 else float('inf')
        
        print(f"  Mean Dist Ratio (2/1): {mean_dist_ratio:.3f}")
        print(f"  Area Ratio (2/1): {area_ratio:.3f}")
        
        if 0.9 <= mean_dist_ratio <= 1.1 and 0.9 <= area_ratio <= 1.1:
            groups_to_keep = [r1, r2]
        else:
            groups_to_keep = [r1]
    elif len(results_sorted) == 1:
        groups_to_keep = [results_sorted[0]]
    
    # Save selected group images and analyze patterns
    centroids_for_pose_estimation = []
    pattern_analyses = []
    
    for idx, group in enumerate(groups_to_keep, start=1):
        cluster_img = np.zeros_like(img)
        centroids_for_pose_estimation.append(group['pts'])
        
        # Analyze the pattern
        pattern_info = classify_pattern(group['pts'])
        orientation_info = analyze_pattern_orientation(group['pts'])
        pattern_info.update(orientation_info)
        pattern_analyses.append(pattern_info)
        
        # Print pattern analysis
        print(f"  Pattern {idx}: {pattern_info['type'].upper()} (confidence: {pattern_info['confidence']:.2f})")
        print(f"    - Orientation: {pattern_info['angle_degrees']:.1f}°")
        print(f"    - Area: {pattern_info['area']:.1f} pixels²")
        
        # Create pattern visualization
        pattern_viz = visualize_pattern(
            group['pts'], 
            pattern_info, 
            img.shape,
            os.path.join(img_output_dir, f"pattern_analysis_{idx}.png")
        )
        
        # Save basic centroid image
        for pt in group['pts']:
            cv2.circle(cluster_img, (int(pt[0]), int(pt[1])), 5, 255, -1)
        cv2.imwrite(os.path.join(img_output_dir, f"selected_group_{idx}.png"), cluster_img)
    
    # Save intermediate images
    cv2.imwrite(os.path.join(img_output_dir, 'clahe_image.png'), img2)
    cv2.imwrite(os.path.join(img_output_dir, 'contours.png'), contour_img)
    cv2.imwrite(os.path.join(img_output_dir, 'round_contours.png'), round_contours_img)
    
    # Create and save visualization
    if save_visualization:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(4, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(4, 3, 2)
        plt.imshow(img1, cmap='gray')
        plt.title('Gaussian blur image')
        plt.axis('off')
        
        plt.subplot(4, 3, 3)
        plt.imshow(img2, cmap='gray')
        plt.title('CLAHE image')
        plt.axis('off')
        
        plt.subplot(4, 3, 4)
        plt.imshow(th1, cmap='gray')
        plt.title('Threshold image gaussian')
        plt.axis('off')
        
        plt.subplot(4, 3, 5)
        plt.imshow(th2, cmap='gray')
        plt.title('Threshold image mean')
        plt.axis('off')
        
        plt.subplot(4, 3, 6)
        plt.imshow(img3, cmap='gray')
        plt.title('Morph gradient')
        plt.axis('off')
        
        plt.subplot(4, 3, 7)
        plt.imshow(img4, cmap='gray')
        plt.title('Inverted image')
        plt.axis('off')
        
        plt.subplot(4, 3, 8)
        plt.imshow(contour_img, cmap='gray')
        plt.title('Contour image')
        plt.axis('off')
        
        plt.subplot(4, 3, 9)
        plt.imshow(round_contours_img, cmap='gray')
        plt.title('Round Contour image')
        plt.axis('off')
        
        # Show selected groups
        for idx, group in enumerate(groups_to_keep[:2], start=1):
            if idx <= 2:
                cluster_img = np.zeros_like(img)
                for pt in group['pts']:
                    cv2.circle(cluster_img, (int(pt[0]), int(pt[1])), 5, 255, -1)
                plt.subplot(4, 3, 9+idx)
                plt.imshow(cluster_img, cmap='gray')
                plt.title(f'Centroids for pose estimation {idx}')
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(img_output_dir, 'visualization.png'), dpi=100, bbox_inches='tight')
        plt.close()
    
    return {
        'image_name': img_name,
        'image_path': img_path,
        'centroids_for_pose_estimation': centroids_for_pose_estimation,
        'pattern_analyses': pattern_analyses,
        'num_round_contours': len(round_contours),
        'num_unique_centroids': len(unique_centroids_round_contours),
        'num_valid_targets': len(valid_targets),
        'num_selected_groups': len(groups_to_keep)
    }


def process_dataset(dataset_dir, output_dir, image_extensions=None, adaptive_mode=True):
    """
    Process all images in a dataset directory
    
    Args:
        dataset_dir: Path to the dataset directory containing images
        output_dir: Directory to save all output files
        image_extensions: List of image file extensions to process (default: common image formats)
        adaptive_mode: If True, retry with relaxed parameters if no circles found
    
    Returns:
        list: Results for all processed images
    """
    if image_extensions is None:
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(dataset_dir, ext)
        image_files.extend(glob.glob(pattern))
        # Also check for uppercase extensions
        pattern_upper = os.path.join(dataset_dir, ext.upper())
        image_files.extend(glob.glob(pattern_upper))
    
    # Remove duplicates
    image_files = list(set(image_files))
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {dataset_dir}")
        return []
    
    print(f"Found {len(image_files)} images to process")
    print("-" * 50)
    
    # Process all images
    all_results = []
    successful = 0
    failed = 0
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {os.path.basename(img_path)}")
        try:
            # First attempt with default parameters
            result = process_single_image(img_path, output_dir, save_visualization=True)
            
            # If no circles found and adaptive mode is on, retry with relaxed parameters
            if adaptive_mode and result and result['num_round_contours'] == 0:
                print("  ⚠ No circles found with default parameters, trying relaxed parameters...")
                
                # Try with progressively relaxed parameters
                param_sets = [
                    {'min_area': 30, 'circularity_threshold': 0.2, 'ellipse_ratio_threshold': 0.8},
                    {'min_area': 20, 'circularity_threshold': 0.25, 'ellipse_ratio_threshold': 0.75},
                    {'min_area': 15, 'circularity_threshold': 0.3, 'ellipse_ratio_threshold': 0.7}
                ]
                
                for params in param_sets:
                    result = process_single_image(img_path, output_dir, save_visualization=True, **params)
                    if result and result['num_round_contours'] > 0:
                        print(f"  ✓ Found circles with relaxed parameters: min_area={params['min_area']}, "
                              f"circularity={params['circularity_threshold']}, "
                              f"ellipse_ratio={params['ellipse_ratio_threshold']}")
                        break
            
            if result:
                all_results.append(result)
                successful += 1
                print(f"  ✓ Successfully processed")
                print(f"    - Round contours found: {result['num_round_contours']}")
                print(f"    - Unique centroids: {result['num_unique_centroids']}")
                print(f"    - Valid targets: {result['num_valid_targets']}")
                print(f"    - Selected groups: {result['num_selected_groups']}")
                
                if result['num_round_contours'] == 0:
                    print("  ⚠ WARNING: No circles detected even with relaxed parameters!")
            else:
                failed += 1
                print(f"  ✗ Failed to process")
        except Exception as e:
            failed += 1
            print(f"  ✗ Error processing: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total images: {len(image_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    
    # Save summary to file
    summary_file = os.path.join(output_dir, 'processing_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Image Processing Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset directory: {dataset_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Total images: {len(image_files)}\n")
        f.write(f"Successfully processed: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("Detailed Results:\n")
        f.write("-" * 50 + "\n")
        
        for result in all_results:
            f.write(f"\nImage: {result['image_name']}\n")
            f.write(f"  - Round contours: {result['num_round_contours']}\n")
            f.write(f"  - Unique centroids: {result['num_unique_centroids']}\n")
            f.write(f"  - Valid targets: {result['num_valid_targets']}\n")
            f.write(f"  - Selected groups: {result['num_selected_groups']}\n")
            f.write(f"  - Centroids for pose estimation: {len(result['centroids_for_pose_estimation'])} groups\n")
    
    print(f"\nSummary saved to: {summary_file}")
    
    return all_results


def main():
    """
    Main function to run the batch processing
    """
    # Configuration
    DATASET_DIR = "dataset"  # Directory containing images
    OUTPUT_DIR = "output"    # Directory to save results
    
    # You can also process a single image
    # result = process_single_image("dataset/IMG_10072025161331.png", "output_single")
    
    # Process entire dataset
    results = process_dataset(DATASET_DIR, OUTPUT_DIR)
    
    print(f"\nProcessing complete! Results saved in '{OUTPUT_DIR}' directory")
    
    return results


if __name__ == "__main__":
    # Run the batch processing
    results = main()
