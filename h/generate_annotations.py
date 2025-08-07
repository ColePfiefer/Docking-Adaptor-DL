"""
Generate YOLO annotations from preprocessing results
This script processes the dataset and creates annotation files for training
"""

import json
import os
import numpy as np
from pathlib import Path
import cv2
from test import process_single_image, classify_pattern
from tqdm import tqdm


def generate_yolo_annotations(dataset_dir, output_dir, annotation_file="annotations.json"):
    """
    Generate YOLO format annotations from thermal images
    
    Args:
        dataset_dir: Directory containing thermal images
        output_dir: Directory to save annotation files
        annotation_file: Name of the JSON annotation file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = list(Path(dataset_dir).glob("*.png")) + \
                  list(Path(dataset_dir).glob("*.jpg"))
    
    annotations = {}
    statistics = {
        'total_images': 0,
        'images_with_patterns': 0,
        'total_circles': 0,
        'pattern_types': {},
        'radius_ratios': []
    }
    
    print(f"Processing {len(image_files)} images...")
    
    for img_path in tqdm(image_files):
        img_name = img_path.stem
        
        # Process image using existing pipeline
        result = process_single_image(
            str(img_path), 
            output_dir, 
            save_visualization=False,
            debug_mode=False
        )
        
        if result and result['centroids_for_pose_estimation']:
            annotation = {
                'image_path': str(img_path),
                'image_name': img_name,
                'groups': []
            }
            
            for group_idx, group_points in enumerate(result['centroids_for_pose_estimation']):
                # Extract circle information
                circles = []
                for pt in group_points:
                    x, y = pt[0], pt[1]
                    # Estimate radius (would be better with actual contour data)
                    radius = estimate_circle_radius(str(img_path), x, y)
                    circles.append({
                        'center': [float(x), float(y)],
                        'radius': float(radius),
                        'confidence': 1.0
                    })
                
                # Calculate pattern features
                centers = np.array([c['center'] for c in circles])
                pattern_info = classify_pattern(centers)
                
                # Calculate radius ratios
                radii = [c['radius'] for c in circles]
                ratios = calculate_radius_ratios(radii)
                
                # Ensure all values are JSON serializable
                centroid_value = None
                if 'centroid' in pattern_info and pattern_info['centroid'] is not None:
                    if isinstance(pattern_info['centroid'], np.ndarray):
                        centroid_value = pattern_info['centroid'].tolist()
                    else:
                        centroid_value = list(pattern_info['centroid'])
                
                group_annotation = {
                    'circles': circles,
                    'pattern_type': pattern_info['type'],
                    'pattern_confidence': float(pattern_info['confidence']),
                    'radius_ratios': ratios.tolist() if isinstance(ratios, np.ndarray) else ratios,
                    'centroid': centroid_value,
                    'area': float(pattern_info['area']) if 'area' in pattern_info else None
                }
                
                annotation['groups'].append(group_annotation)
                
                # Update statistics
                statistics['total_circles'] += len(circles)
                statistics['pattern_types'][pattern_info['type']] = \
                    statistics['pattern_types'].get(pattern_info['type'], 0) + 1
                if ratios is not None:
                    # Convert numpy array to list for JSON serialization
                    if isinstance(ratios, np.ndarray):
                        statistics['radius_ratios'].append(ratios.tolist())
                    else:
                        statistics['radius_ratios'].append(ratios)
            
            annotations[img_name] = annotation
            statistics['images_with_patterns'] += 1
        
        statistics['total_images'] += 1
    
    # Calculate average radius ratios for the dataset
    if statistics['radius_ratios']:
        # Convert all numpy arrays to lists for JSON serialization
        statistics['radius_ratios'] = [r.tolist() if isinstance(r, np.ndarray) else r for r in statistics['radius_ratios']]
        avg_ratios = np.mean(statistics['radius_ratios'], axis=0)
        std_ratios = np.std(statistics['radius_ratios'], axis=0)
        statistics['avg_radius_ratios'] = avg_ratios.tolist() if isinstance(avg_ratios, np.ndarray) else []
        statistics['std_radius_ratios'] = std_ratios.tolist() if isinstance(std_ratios, np.ndarray) else []
    
    # Save annotations
    annotation_path = os.path.join(output_dir, annotation_file)
    with open(annotation_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\nAnnotations saved to: {annotation_path}")
    
    # Save statistics
    stats_path = os.path.join(output_dir, "dataset_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"Statistics saved to: {stats_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("ANNOTATION GENERATION SUMMARY")
    print("="*50)
    print(f"Total images processed: {statistics['total_images']}")
    print(f"Images with patterns: {statistics['images_with_patterns']}")
    print(f"Total circles detected: {statistics['total_circles']}")
    print(f"Pattern types distribution: {statistics['pattern_types']}")
    if 'avg_radius_ratios' in statistics:
        print(f"Average radius ratios: {statistics['avg_radius_ratios']}")
    
    return annotations, statistics


def estimate_circle_radius(img_path, center_x, center_y, search_radius=50):
    """
    Estimate the radius of a circle given its center
    
    Args:
        img_path: Path to the image
        center_x, center_y: Center coordinates
        search_radius: Maximum radius to search
    
    Returns:
        Estimated radius
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply preprocessing
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Search for edge points around the center
    best_radius = 10  # Default radius
    max_edge_count = 0
    
    for r in range(5, search_radius):
        edge_count = 0
        num_points = int(2 * np.pi * r)
        
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                if img_thresh[y, x] > 128:  # Edge point
                    edge_count += 1
        
        if edge_count > max_edge_count:
            max_edge_count = edge_count
            best_radius = r
    
    return best_radius


def calculate_radius_ratios(radii):
    """
    Calculate all pairwise radius ratios
    
    Args:
        radii: List of radius values
    
    Returns:
        Array of radius ratios
    """
    if len(radii) < 2:
        return None
    
    ratios = []
    for i in range(len(radii)):
        for j in range(i+1, len(radii)):
            if radii[j] > 0:
                ratio = radii[i] / radii[j]
                ratios.append(ratio)
    
    return np.array(ratios)


def convert_to_yolo_format(annotations, img_width, img_height, output_dir):
    """
    Convert annotations to YOLO format text files
    
    Args:
        annotations: Dictionary of annotations
        img_width, img_height: Image dimensions
        output_dir: Directory to save YOLO format files
    """
    yolo_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(yolo_labels_dir, exist_ok=True)
    
    for img_name, annotation in annotations.items():
        yolo_labels = []
        
        for group in annotation['groups']:
            for circle in group['circles']:
                # Convert to YOLO format: class x_center y_center width height
                x_center = circle['center'][0] / img_width
                y_center = circle['center'][1] / img_height
                # Use diameter as width and height (circles are square bounding boxes)
                width = (2 * circle['radius']) / img_width
                height = (2 * circle['radius']) / img_height
                
                # Class 0 for circle
                yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Save YOLO format file
        label_path = os.path.join(yolo_labels_dir, f"{img_name}.txt")
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_labels))
    
    print(f"YOLO format labels saved to: {yolo_labels_dir}")


def create_yolo_yaml_config(dataset_dir, output_dir):
    """
    Create YOLO configuration YAML file
    
    Args:
        dataset_dir: Path to dataset directory
        output_dir: Path to output directory
    """
    import yaml
    
    config = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        
        # Number of classes
        'nc': 1,
        
        # Class names
        'names': ['circle'],
        
        # Additional information
        'description': 'ISS Docking Concentric Circle Pattern Dataset',
        'pattern_info': {
            'num_circles': 4,
            'pattern_type': 'triangular',
            'application': 'ISS docking system'
        }
    }
    
    yaml_path = os.path.join(output_dir, 'dataset_config.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"YOLO configuration saved to: {yaml_path}")
    return yaml_path


def split_dataset(annotations, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        annotations: Dictionary of annotations
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
    
    Returns:
        Dictionary with train, val, and test splits
    """
    img_names = list(annotations.keys())
    np.random.shuffle(img_names)
    
    n_total = len(img_names)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    splits = {
        'train': img_names[:n_train],
        'val': img_names[n_train:n_train+n_val],
        'test': img_names[n_train+n_val:]
    }
    
    print(f"Dataset split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    return splits


def main():
    """
    Main function to generate annotations
    """
    # Configuration
    DATASET_DIR = "dataset"
    OUTPUT_DIR = "yolo_annotations"
    
    print("Generating YOLO annotations from thermal image dataset...")
    print("="*50)
    
    # Generate annotations
    annotations, statistics = generate_yolo_annotations(DATASET_DIR, OUTPUT_DIR)
    
    # Convert to YOLO format (assuming 640x480 images, adjust as needed)
    print("\nConverting to YOLO format...")
    convert_to_yolo_format(annotations, img_width=640, img_height=480, output_dir=OUTPUT_DIR)
    
    # Create YOLO configuration file
    print("\nCreating YOLO configuration...")
    config_path = create_yolo_yaml_config(DATASET_DIR, OUTPUT_DIR)
    
    # Split dataset
    print("\nSplitting dataset...")
    splits = split_dataset(annotations)
    
    # Save splits information
    splits_path = os.path.join(OUTPUT_DIR, "dataset_splits.json")
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Dataset splits saved to: {splits_path}")
    
    print("\n" + "="*50)
    print("ANNOTATION GENERATION COMPLETE!")
    print("="*50)
    print(f"Annotations directory: {OUTPUT_DIR}")
    print(f"YOLO config: {config_path}")
    print(f"Labels directory: {os.path.join(OUTPUT_DIR, 'labels')}")
    print("\nNext steps:")
    print("1. Review the generated annotations")
    print("2. Run: python yolo_pattern_detector.py")
    print("3. Monitor training progress in tensorboard")


if __name__ == "__main__":
    main()
