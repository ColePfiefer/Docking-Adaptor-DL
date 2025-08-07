"""
YOLO-based Concentric Circle Pattern Detection System for ISS Docking
Integrates with existing preprocessing pipeline from test.py
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing functions from test.py
from test import (
    process_single_image,
    classify_pattern,
    analyze_pattern_orientation,
    order_points_clockwise
)


class CirclePatternDataset(Dataset):
    """
    Custom dataset for loading thermal images with circle pattern annotations
    """
    
    def __init__(self, image_dir: str, annotations_file: str = None, transform=None, augment=True):
        """
        Args:
            image_dir: Directory containing thermal images
            annotations_file: Path to JSON file with annotations
            transform: Optional transforms to apply
            augment: Whether to apply data augmentation
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.augment = augment
        
        # Get all image files
        self.image_files = list(self.image_dir.glob("*.png")) + \
                          list(self.image_dir.glob("*.jpg"))
        
        # Load annotations if provided
        self.annotations = {}
        if annotations_file and os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                self.annotations = json.load(f)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Check if image was loaded successfully
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            # Return a dummy image
            image = np.zeros((640, 640), dtype=np.uint8)
        
        # Apply preprocessing pipeline from test.py
        processed_data = self._preprocess_image(image)
        
        # Get annotations if available
        img_name = img_path.stem
        annotation = self.annotations.get(img_name, {})
        
        # Apply augmentation if enabled
        if self.augment and annotation:
            image, annotation = self._augment(image, annotation)
        
        # Resize image to expected size
        image = cv2.resize(image, (640, 640))
        
        # Normalize and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
        
        # Ensure we don't return None values
        return {
            'image': image,
            'path': str(img_path)
        }
    
    def _preprocess_image(self, img):
        """Apply preprocessing steps from test.py"""
        # Gaussian blur
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(11, 11))
        img_clahe = clahe.apply(img)
        
        # Adaptive thresholding
        th_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 63, 1)
        th_mean = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 63, 1)
        
        # Inverted image
        img_inverted = cv2.bitwise_not(th_mean)
        
        return {
            'clahe': img_clahe,
            'threshold_gaussian': th_gaussian,
            'threshold_mean': th_mean,
            'inverted': img_inverted
        }
    
    def _augment(self, image, annotation):
        """Apply data augmentation"""
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-45, 45)
            center = (image.shape[1]//2, image.shape[0]//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            
            # Rotate annotations if present
            if annotation and 'centers' in annotation:
                # Transform center points
                centers = np.array(annotation['centers'])
                ones = np.ones((centers.shape[0], 1))
                centers_homogeneous = np.hstack([centers, ones])
                rotated_centers = M @ centers_homogeneous.T
                annotation['centers'] = rotated_centers.T.tolist()
        
        # Random scaling
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            image = cv2.resize(image, new_size)
            
            if annotation and 'centers' in annotation:
                annotation['centers'] = [[c[0]*scale, c[1]*scale] 
                                        for c in annotation['centers']]
        
        # Random translation
        if np.random.random() > 0.5:
            tx = np.random.randint(-50, 50)
            ty = np.random.randint(-50, 50)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            
            if annotation and 'centers' in annotation:
                annotation['centers'] = [[c[0]+tx, c[1]+ty] 
                                        for c in annotation['centers']]
        
        # Add thermal noise
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)
        
        return image, annotation


class CustomYOLOv8CircleDetector(nn.Module):
    """
    Custom YOLOv8 model adapted for circle pattern detection
    """
    
    def __init__(self, num_circles=4, pretrained=True):
        super().__init__()
        
        # Initialize backbone for grayscale images (1 channel input)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),  # 1 input channel for grayscale
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            # Additional layers to process features
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # If pretrained weights requested, we'll load YOLO but not use it directly
        # since YOLO expects RGB and we have grayscale
        if pretrained:
            # Download pretrained weights but don't use the model directly
            _ = YOLO('yolov8n.pt')
        
        # Custom head for circle detection
        self.circle_detection_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_circles * 5, 1)  # 5 values per circle: x, y, r, conf, class
        )
        
        # Pattern recognition head
        self.pattern_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(7 * 7 * 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)  # Pattern confidence scores
        )
        
        # Ratio prediction head for scale-invariant features
        self.ratio_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)  # 6 radius ratios for 4 circles
        )
        
        self.num_circles = num_circles
    
    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)
        
        # Circle detection
        circle_output = self.circle_detection_head(features)
        batch_size = x.shape[0]
        h, w = circle_output.shape[2:]
        circle_output = circle_output.view(batch_size, self.num_circles, 5, h*w)
        
        # Pattern recognition
        pattern_output = self.pattern_head(features)
        
        # Ratio prediction
        pooled_features = torch.mean(features, dim=[2, 3])
        ratio_output = self.ratio_head(pooled_features)
        
        return {
            'circles': circle_output,
            'pattern': pattern_output,
            'ratios': ratio_output
        }


class PatternAwareLoss(nn.Module):
    """
    Custom loss function for pattern-aware circle detection
    """
    
    def __init__(self, lambda_detect=1.0, lambda_pattern=2.0, 
                 lambda_center=1.5, lambda_radius=1.0, lambda_ratio=1.5):
        super().__init__()
        self.lambda_detect = lambda_detect
        self.lambda_pattern = lambda_pattern
        self.lambda_center = lambda_center
        self.lambda_radius = lambda_radius
        self.lambda_ratio = lambda_ratio
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
    
    def forward(self, predictions, targets):
        """
        Calculate composite loss
        
        Args:
            predictions: Dict with 'circles', 'pattern', 'ratios'
            targets: Dict with ground truth values
        """
        total_loss = 0
        losses = {}
        
        # Detection loss (whether circle exists)
        if 'circle_conf' in targets:
            detection_loss = self.bce_loss(
                predictions['circles'][:, :, 3, :],
                targets['circle_conf']
            )
            total_loss += self.lambda_detect * detection_loss
            losses['detection'] = detection_loss.item()
        
        # Center localization loss
        if 'centers' in targets:
            center_pred = predictions['circles'][:, :, :2, :]
            center_loss = self.smooth_l1_loss(center_pred, targets['centers'])
            total_loss += self.lambda_center * center_loss
            losses['center'] = center_loss.item()
        
        # Radius prediction loss
        if 'radii' in targets:
            radius_pred = predictions['circles'][:, :, 2, :]
            radius_loss = self.mse_loss(radius_pred, targets['radii'])
            total_loss += self.lambda_radius * radius_loss
            losses['radius'] = radius_loss.item()
        
        # Pattern recognition loss
        if 'pattern_label' in targets:
            pattern_loss = self.bce_loss(
                predictions['pattern'],
                targets['pattern_label']
            )
            total_loss += self.lambda_pattern * pattern_loss
            losses['pattern'] = pattern_loss.item()
        
        # Ratio consistency loss (scale-invariant)
        if 'ratios' in targets:
            ratio_loss = self.mse_loss(
                predictions['ratios'],
                targets['ratios']
            )
            total_loss += self.lambda_ratio * ratio_loss
            losses['ratio'] = ratio_loss.item()
        
        losses['total'] = total_loss.item()
        return total_loss, losses


class CirclePatternTrainer:
    """
    Trainer class for the circle pattern detection model
    """
    
    def __init__(self, model, dataset, config=None):
        self.model = model
        self.dataset = dataset
        self.config = config or self._default_config()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs']
        )
        
        # Initialize loss function
        self.criterion = PatternAwareLoss(
            lambda_detect=self.config['lambda_detect'],
            lambda_pattern=self.config['lambda_pattern'],
            lambda_center=self.config['lambda_center'],
            lambda_radius=self.config['lambda_radius'],
            lambda_ratio=self.config['lambda_ratio']
        )
        
        # Data loaders
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def _default_config(self):
        return {
            'epochs': 300,
            'batch_size': 8,
            'learning_rate': 1e-4,
            'weight_decay': 5e-4,
            'num_workers': 4,
            'lambda_detect': 1.0,
            'lambda_pattern': 2.0,
            'lambda_center': 1.5,
            'lambda_radius': 1.0,
            'lambda_ratio': 1.5,
            'save_interval': 10,
            'val_interval': 5
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()  # Set to training mode
        epoch_losses = []
        
        for batch_idx, data in enumerate(self.train_loader):
            images = data['image'].to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            
            # Prepare targets (would come from annotations in real scenario)
            targets = self._prepare_targets(data)
            
            # Calculate loss
            loss, loss_components = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: "
                      f"Loss = {loss.item():.4f}")
        
        return np.mean(epoch_losses)
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for data in self.val_loader:
                images = data['image'].to(self.device)
                predictions = self.model(images)
                targets = self._prepare_targets(data)
                loss, _ = self.criterion(predictions, targets)
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def _prepare_targets(self, data):
        """Prepare target tensors from data"""
        # This would be implemented based on actual annotation format
        # For now, returning dummy targets
        batch_size = data['image'].shape[0]
        device = data['image'].device
        
        # The circle outputs have shape [batch, num_circles, 5, h*w]
        # where h*w = 160*160 = 25600 for 640x640 input after convolutions
        num_grid_points = 160 * 160  # This should match the actual output size
        
        targets = {
            'circle_conf': torch.ones(batch_size, 4, num_grid_points).to(device),
            'centers': torch.randn(batch_size, 4, 2, num_grid_points).to(device),
            'radii': torch.rand(batch_size, 4, num_grid_points).to(device) * 50 + 10,
            'pattern_label': torch.ones(batch_size, 4).to(device),
            'ratios': torch.rand(batch_size, 6).to(device)
        }
        
        return targets
    
    def run_training(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config['epochs']}")
        print("-" * 50)
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            print(f"  Training Loss: {train_loss:.4f}")
            
            # Validation
            if (epoch + 1) % self.config['val_interval'] == 0:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                print(f"  Validation Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch + 1)
        
        print("\nTraining completed!")
        self.plot_losses()
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        checkpoint_path = f"checkpoints/model_epoch_{epoch}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    def plot_losses(self):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            val_epochs = list(range(self.config['val_interval']-1, 
                                   len(self.train_losses), 
                                   self.config['val_interval']))
            plt.plot(val_epochs, self.val_losses, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_progress.png')
        plt.show()


class PatternInference:
    """
    Inference class for real-time pattern detection
    """
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = CustomYOLOv8CircleDetector()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Kalman filter for tracking
        self.kalman_filter = self._init_kalman_filter()
        
        # Pattern validation thresholds
        self.confidence_threshold = 0.95
        self.pattern_threshold = 0.9
        self.temporal_window = 5
        self.detection_history = []
    
    def _init_kalman_filter(self):
        """Initialize Kalman filter for center tracking"""
        # State: [x1, y1, r1, x2, y2, r2, x3, y3, r3, x4, y4, r4, vx1, vy1, ...]
        kf = cv2.KalmanFilter(24, 12)  # 24 state variables, 12 measurements
        
        # State transition matrix
        kf.transitionMatrix = np.eye(24, dtype=np.float32)
        for i in range(12):
            kf.transitionMatrix[i, i+12] = 1  # Position += velocity
        
        # Measurement matrix
        kf.measurementMatrix = np.zeros((12, 24), dtype=np.float32)
        for i in range(12):
            kf.measurementMatrix[i, i] = 1
        
        # Process and measurement noise
        kf.processNoiseCov = np.eye(24, dtype=np.float32) * 0.01
        kf.measurementNoiseCov = np.eye(12, dtype=np.float32) * 0.1
        
        return kf
    
    def process_frame(self, frame):
        """
        Process a single frame for pattern detection
        
        Args:
            frame: Input frame (grayscale thermal image)
        
        Returns:
            Dict with detection results
        """
        # Preprocess frame
        processed = self._preprocess_frame(frame)
        
        # Convert to tensor
        input_tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Model inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Extract detections
        circles = self._extract_circles(predictions['circles'][0])
        pattern_conf = torch.sigmoid(predictions['pattern'][0]).cpu().numpy()
        ratios = predictions['ratios'][0].cpu().numpy()
        
        # Validate pattern
        is_valid = self._validate_pattern(circles, ratios)
        
        # Apply Kalman filtering for stability
        if is_valid and len(circles) == 4:
            filtered_circles = self._apply_kalman_filter(circles)
        else:
            filtered_circles = circles
        
        # Update detection history
        self.detection_history.append({
            'circles': filtered_circles,
            'pattern_conf': pattern_conf,
            'is_valid': is_valid
        })
        
        # Maintain temporal window
        if len(self.detection_history) > self.temporal_window:
            self.detection_history.pop(0)
        
        # Check temporal consistency
        temporal_valid = self._check_temporal_consistency()
        
        result = {
            'status': 'PATTERN_DETECTED' if is_valid and temporal_valid else 'NO_PATTERN',
            'circles': filtered_circles,
            'centers': [c[:2] for c in filtered_circles],
            'radii': [c[2] for c in filtered_circles],
            'pattern_confidence': np.max(pattern_conf),
            'ratios': ratios,
            'temporal_consistency': temporal_valid
        }
        
        return result
    
    def _preprocess_frame(self, frame):
        """Apply preprocessing to frame"""
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(11, 11))
        frame_enhanced = clahe.apply(frame)
        
        # Resize to model input size
        frame_resized = cv2.resize(frame_enhanced, (640, 640))
        
        # Normalize
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        return frame_normalized
    
    def _extract_circles(self, circle_output):
        """Extract circle detections from model output"""
        circles = []
        
        # Process each detected circle
        for i in range(circle_output.shape[0]):
            conf = torch.sigmoid(circle_output[i, 3, 0]).item()
            
            if conf > self.confidence_threshold:
                x = circle_output[i, 0, 0].item()
                y = circle_output[i, 1, 0].item()
                r = circle_output[i, 2, 0].item()
                circles.append([x, y, r, conf])
        
        return circles
    
    def _validate_pattern(self, circles, ratios):
        """Validate if detected circles form the expected pattern"""
        if len(circles) != 4:
            return False
        
        # Check radius ratios
        expected_ratios = self._get_expected_ratios()
        ratio_error = np.mean(np.abs(ratios - expected_ratios))
        
        if ratio_error > 0.1:
            return False
        
        # Check spatial arrangement (triangular pattern)
        centers = np.array([c[:2] for c in circles])
        pattern_type = classify_pattern(centers)
        
        if pattern_type['type'] != 'triangular' or pattern_type['confidence'] < 0.8:
            return False
        
        return True
    
    def _get_expected_ratios(self):
        """Get expected radius ratios for the target pattern"""
        # These would be learned from training data
        return np.array([0.5, 0.33, 0.25, 0.66, 0.5, 0.75])
    
    def _apply_kalman_filter(self, circles):
        """Apply Kalman filtering for smooth tracking"""
        # Prepare measurement vector
        measurement = np.zeros(12, dtype=np.float32)
        for i, circle in enumerate(circles[:4]):
            measurement[i*3:i*3+3] = circle[:3]
        
        # Kalman filter update
        self.kalman_filter.correct(measurement)
        prediction = self.kalman_filter.predict()
        
        # Extract filtered circles
        filtered_circles = []
        for i in range(4):
            x = prediction[i*3]
            y = prediction[i*3+1]
            r = prediction[i*3+2]
            conf = circles[i][3] if i < len(circles) else 0
            filtered_circles.append([x, y, r, conf])
        
        return filtered_circles
    
    def _check_temporal_consistency(self):
        """Check if detections are consistent over time"""
        if len(self.detection_history) < 3:
            return False
        
        valid_count = sum(1 for h in self.detection_history[-5:] if h['is_valid'])
        
        return valid_count >= 3
    
    def visualize_detection(self, frame, result):
        """Visualize detection results on frame"""
        vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        if result['status'] == 'PATTERN_DETECTED':
            # Draw circles
            for circle in result['circles']:
                x, y, r, conf = circle
                cv2.circle(vis_frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 0, 255), -1)
            
            # Draw pattern connections
            centers = np.array(result['centers'])
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    cv2.line(vis_frame, 
                            tuple(centers[i].astype(int)),
                            tuple(centers[j].astype(int)),
                            (255, 128, 0), 1)
            
            # Add text
            text = f"PATTERN DETECTED (Conf: {result['pattern_confidence']:.2f})"
            cv2.putText(vis_frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis_frame, "NO PATTERN", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_frame


def main():
    """
    Main function to train and test the YOLO-based pattern detector
    """
    # Configuration
    config = {
        'dataset_dir': 'dataset',
        'output_dir': 'yolo_output',
        'epochs': 100,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'weight_decay': 5e-4,
        'num_workers': 0,  # Set to 0 for Windows compatibility
        'lambda_detect': 1.0,
        'lambda_pattern': 2.0,
        'lambda_center': 1.5,
        'lambda_radius': 1.0,
        'lambda_ratio': 1.5,
        'save_interval': 10,
        'val_interval': 5
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize dataset
    print("Loading dataset...")
    dataset = CirclePatternDataset(
        config['dataset_dir'],
        transform=None,  # We'll handle the transform in the dataset itself
        augment=False  # Disable augmentation for initial testing
    )
    print(f"Dataset size: {len(dataset)} images")
    
    # Initialize model
    print("\nInitializing YOLO model...")
    model = CustomYOLOv8CircleDetector(num_circles=4, pretrained=True)
    
    # Initialize trainer
    trainer = CirclePatternTrainer(model, dataset, config)
    
    # Train model
    print("\nStarting training...")
    trainer.run_training()
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, os.path.join(config['output_dir'], 'final_model.pth'))
    
    print(f"\nTraining complete! Model saved to {config['output_dir']}/final_model.pth")
    
    # Test inference on a sample image
    print("\nTesting inference...")
    inference = PatternInference(os.path.join(config['output_dir'], 'final_model.pth'))
    
    # Process a test image
    test_image_path = list(Path(config['dataset_dir']).glob("*.png"))[0]
    test_image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
    
    result = inference.process_frame(test_image)
    print(f"Detection result: {result['status']}")
    print(f"Pattern confidence: {result['pattern_confidence']:.3f}")
    
    # Visualize result
    vis_frame = inference.visualize_detection(test_image, result)
    cv2.imwrite(os.path.join(config['output_dir'], 'test_detection.png'), vis_frame)
    print(f"Visualization saved to {config['output_dir']}/test_detection.png")


if __name__ == "__main__":
    main()
