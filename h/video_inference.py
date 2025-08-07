"""
Real-time Video Processing for ISS Docking Pattern Detection
Processes video streams or files to detect concentric circle patterns
"""

import cv2
import numpy as np
import torch
import time
from pathlib import Path
import argparse
from collections import deque
import json
from datetime import datetime
from yolo_pattern_detector import PatternInference, CustomYOLOv8CircleDetector
from test import classify_pattern


class VideoPatternDetector:
    """
    Real-time video processing for pattern detection
    """
    
    def __init__(self, model_path, confidence_threshold=0.95, 
                 buffer_size=30, display_mode='full'):
        """
        Initialize video pattern detector
        
        Args:
            model_path: Path to trained YOLO model
            confidence_threshold: Minimum confidence for detection
            buffer_size: Size of frame buffer for temporal analysis
            display_mode: 'full', 'minimal', or 'debug'
        """
        self.inference = PatternInference(model_path)
        self.confidence_threshold = confidence_threshold
        self.buffer_size = buffer_size
        self.display_mode = display_mode
        
        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.detection_buffer = deque(maxlen=buffer_size)
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'detected_frames': 0,
            'pattern_detections': [],
            'avg_confidence': 0,
            'avg_fps': 0
        }
        
        # Recording setup
        self.recording = False
        self.video_writer = None
    
    def process_video(self, source, output_path=None, save_detections=True):
        """
        Process video from file or camera
        
        Args:
            source: Video file path or camera index (0 for webcam)
            output_path: Optional path to save processed video
            save_detections: Whether to save detection data
        """
        # Open video source
        if isinstance(source, int):
            cap = cv2.VideoCapture(source)
            print(f"Opening camera {source}...")
        else:
            cap = cv2.VideoCapture(source)
            print(f"Opening video file: {source}")
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, fps, (width, height)
            )
            print(f"Recording to: {output_path}")
        
        # Create display window
        cv2.namedWindow('ISS Docking Pattern Detection', cv2.WINDOW_NORMAL)
        
        # Detection data storage
        detection_data = []
        
        print("\nProcessing video... Press 'q' to quit, 's' to save snapshot")
        print("-" * 50)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for processing
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Start timing
                start_time = time.time()
                
                # Process frame
                result = self.inference.process_frame(gray_frame)
                
                # Calculate FPS
                process_time = time.time() - start_time
                fps_current = 1.0 / process_time if process_time > 0 else 0
                self.fps_buffer.append(fps_current)
                
                # Update statistics
                self.stats['total_frames'] += 1
                if result['status'] == 'PATTERN_DETECTED':
                    self.stats['detected_frames'] += 1
                    
                    # Store detection data
                    detection_entry = {
                        'frame': self.stats['total_frames'],
                        'timestamp': datetime.now().isoformat(),
                        'centers': result['centers'],
                        'radii': result['radii'],
                        'confidence': result['pattern_confidence'],
                        'ratios': result['ratios'].tolist() if hasattr(result['ratios'], 'tolist') else result['ratios']
                    }
                    detection_data.append(detection_entry)
                
                # Visualize results
                display_frame = self._create_display_frame(
                    frame, result, fps_current
                )
                
                # Show frame
                cv2.imshow('ISS Docking Pattern Detection', display_frame)
                
                # Save to video if recording
                if self.video_writer:
                    self.video_writer.write(display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_snapshot(display_frame, result)
                elif key == ord('d'):
                    self._toggle_display_mode()
                elif key == ord('r'):
                    self._toggle_recording()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()
            
            # Calculate final statistics
            self.stats['avg_fps'] = np.mean(self.fps_buffer) if self.fps_buffer else 0
            self.stats['avg_confidence'] = np.mean([d['confidence'] for d in detection_data]) if detection_data else 0
            self.stats['detection_rate'] = self.stats['detected_frames'] / self.stats['total_frames'] if self.stats['total_frames'] > 0 else 0
            
            # Save detection data
            if save_detections and detection_data:
                self._save_detection_data(detection_data)
            
            # Print summary
            self._print_summary()
    
    def _create_display_frame(self, frame, result, fps):
        """
        Create annotated display frame
        
        Args:
            frame: Original frame
            result: Detection results
            fps: Current FPS
        
        Returns:
            Annotated frame
        """
        display_frame = frame.copy()
        
        # Draw detection results
        if result['status'] == 'PATTERN_DETECTED':
            # Draw circles
            for circle in result['circles']:
                if len(circle) >= 3:
                    x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
                    # Outer circle
                    cv2.circle(display_frame, (x, y), r, (0, 255, 0), 2)
                    # Center point
                    cv2.circle(display_frame, (x, y), 3, (0, 0, 255), -1)
            
            # Draw pattern connections
            if len(result['centers']) >= 4:
                centers = np.array(result['centers'][:4], dtype=np.int32)
                
                # Draw triangular pattern
                for i in range(len(centers)):
                    for j in range(i+1, len(centers)):
                        cv2.line(display_frame, 
                                tuple(centers[i]),
                                tuple(centers[j]),
                                (255, 128, 0), 1)
                
                # Draw bounding box around pattern
                rect = cv2.minAreaRect(centers)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(display_frame, [box], 0, (0, 255, 255), 2)
            
            # Pattern status indicator
            cv2.putText(display_frame, "PATTERN LOCKED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "SEARCHING...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Add information overlay
        if self.display_mode in ['full', 'debug']:
            self._add_info_overlay(display_frame, result, fps)
        
        # Add debug visualization if enabled
        if self.display_mode == 'debug':
            self._add_debug_info(display_frame, result)
        
        return display_frame
    
    def _add_info_overlay(self, frame, result, fps):
        """Add information overlay to frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Info panel background
        cv2.rectangle(overlay, (w-300, 0), (w, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add text information
        info_x = w - 290
        cv2.putText(frame, f"FPS: {fps:.1f}", (info_x, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(frame, f"Frame: {self.stats['total_frames']}", (info_x, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if result['status'] == 'PATTERN_DETECTED':
            cv2.putText(frame, f"Confidence: {result['pattern_confidence']:.2%}", 
                       (info_x, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.putText(frame, f"Circles: {len(result['circles'])}", 
                       (info_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, f"Temporal: {'STABLE' if result['temporal_consistency'] else 'UNSTABLE'}", 
                       (info_x, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0) if result['temporal_consistency'] else (0, 0, 255), 1)
        
        # Detection rate
        detection_rate = self.stats['detected_frames'] / max(1, self.stats['total_frames'])
        cv2.putText(frame, f"Detection Rate: {detection_rate:.1%}", 
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _add_debug_info(self, frame, result):
        """Add debug information to frame"""
        h, w = frame.shape[:2]
        
        # Debug panel
        debug_y = 180
        if 'ratios' in result and result['ratios'] is not None:
            cv2.putText(frame, "Radius Ratios:", (w-290, debug_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            for i, ratio in enumerate(result['ratios'][:6]):
                cv2.putText(frame, f"  R{i+1}: {ratio:.3f}", 
                           (w-290, debug_y + 20*(i+1)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Pattern analysis
        if len(result['centers']) >= 4:
            centers = np.array(result['centers'][:4])
            pattern_type = classify_pattern(centers)
            
            cv2.putText(frame, f"Pattern: {pattern_type['type'].upper()}", 
                       (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _save_snapshot(self, frame, result):
        """Save current frame snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        
        # Save detection data
        data_filename = f"snapshot_{timestamp}_data.json"
        with open(data_filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'result': {
                    'status': result['status'],
                    'centers': result['centers'],
                    'radii': result['radii'],
                    'confidence': result['pattern_confidence']
                }
            }, f, indent=2)
        
        print(f"Snapshot saved: {filename}")
    
    def _toggle_display_mode(self):
        """Toggle between display modes"""
        modes = ['minimal', 'full', 'debug']
        current_idx = modes.index(self.display_mode)
        self.display_mode = modes[(current_idx + 1) % len(modes)]
        print(f"Display mode: {self.display_mode}")
    
    def _toggle_recording(self):
        """Toggle video recording"""
        self.recording = not self.recording
        print(f"Recording: {'ON' if self.recording else 'OFF'}")
    
    def _save_detection_data(self, detection_data):
        """Save detection data to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_data_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'statistics': self.stats,
                'detections': detection_data
            }, f, indent=2)
        
        print(f"Detection data saved: {filename}")
    
    def _print_summary(self):
        """Print processing summary"""
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Total frames processed: {self.stats['total_frames']}")
        print(f"Frames with pattern detected: {self.stats['detected_frames']}")
        print(f"Detection rate: {self.stats['detection_rate']:.1%}")
        print(f"Average confidence: {self.stats['avg_confidence']:.2%}")
        print(f"Average FPS: {self.stats['avg_fps']:.1f}")
        print("="*50)


def main():
    """Main function for video processing"""
    parser = argparse.ArgumentParser(description='ISS Docking Pattern Video Detection')
    parser.add_argument('--source', type=str, default=0,
                       help='Video source (file path or camera index)')
    parser.add_argument('--model', type=str, default='yolo_output/final_model.pth',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence threshold')
    parser.add_argument('--display', type=str, default='full',
                       choices=['minimal', 'full', 'debug'],
                       help='Display mode')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a camera index
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    print("ISS Docking Pattern Detection System")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Source: {source}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Display mode: {args.display}")
    print("="*50)
    
    # Initialize detector
    detector = VideoPatternDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        display_mode=args.display
    )
    
    # Process video
    detector.process_video(
        source=source,
        output_path=args.output,
        save_detections=True
    )


if __name__ == "__main__":
    main()
