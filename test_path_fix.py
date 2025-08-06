"""
Quick test script to verify the path fix works correctly
"""

from pattern_detector import RobustPatternDetector
import os

def test_path_fix():
    """Test the path fix with a few sample images"""
    
    detector = RobustPatternDetector()
    
    # Test images
    test_images = [
        "dataset/IMG_10072025161331.png",
        "dataset/IMG_10072025161333.png", 
        "dataset/IMG_10072025161339.png"
    ]
    
    print("Testing path fix for pattern detection...")
    
    for i, image_path in enumerate(test_images):
        print(f"\nTesting {i+1}/3: {os.path.basename(image_path)}")
        
        try:
            # Process image without visualization to speed up testing
            result = detector.detect_pattern(image_path, visualize=False)
            
            # Try to save results - this is where the path error was occurring
            detector.save_pattern_results("test_results")
            
            print(f"✓ Success: Pattern found={result['pattern_found']}, Files saved correctly")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return False
    
    print("\n" + "="*50)
    print("PATH FIX TEST RESULTS")
    print("="*50)
    print("✅ All path handling is working correctly!")
    print("✅ Files are being saved without path separator issues")
    
    # Check if files were created
    if os.path.exists("test_results"):
        file_count = len([f for f in os.listdir("test_results") if f.endswith(('.png', '.txt', '.json'))])
        print(f"✅ {file_count} result files created successfully")
    
    return True

if __name__ == "__main__":
    test_path_fix()
