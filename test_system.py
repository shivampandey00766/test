#!/usr/bin/env python3
"""
System test script for the AI Architectural Converter.
Tests the basic functionality of the system without requiring trained models.
"""

import sys
import os
import logging
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent import AIAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_floor_plan():
    """Create a simple test floor plan."""
    # Create a simple floor plan image
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background
    
    # Draw walls (black lines)
    cv2.line(img, (50, 50), (550, 50), (0, 0, 0), 8)    # Top wall
    cv2.line(img, (50, 350), (550, 350), (0, 0, 0), 8)  # Bottom wall
    cv2.line(img, (50, 50), (50, 350), (0, 0, 0), 8)    # Left wall
    cv2.line(img, (550, 50), (550, 350), (0, 0, 0), 8)  # Right wall
    
    # Draw internal walls
    cv2.line(img, (300, 50), (300, 350), (0, 0, 0), 6)  # Vertical divider
    cv2.line(img, (50, 200), (300, 200), (0, 0, 0), 6)  # Horizontal divider
    
    # Draw doors (green rectangles)
    cv2.rectangle(img, (45, 195), (55, 205), (0, 255, 0), -1)  # Door 1
    cv2.rectangle(img, (295, 45), (305, 55), (0, 255, 0), -1)  # Door 2
    
    # Draw windows (blue rectangles)
    cv2.rectangle(img, (200, 45), (250, 55), (255, 0, 0), -1)  # Window 1
    cv2.rectangle(img, (400, 45), (450, 55), (255, 0, 0), -1)  # Window 2
    
    # Add room labels
    cv2.putText(img, "LIVING ROOM", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "KITCHEN", (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "BEDROOM", (320, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "BATHROOM", (320, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img


def test_ai_agent():
    """Test the AI Agent functionality."""
    try:
        logger.info("Testing AI Agent initialization...")
        
        # Initialize AI Agent
        ai_agent = AIAgent()
        
        # Check status
        status = ai_agent.get_status()
        logger.info(f"AI Agent status: {status}")
        
        # Create test floor plan
        logger.info("Creating test floor plan...")
        test_img = create_test_floor_plan()
        
        # Save test image
        test_path = "test_floor_plan.jpg"
        cv2.imwrite(test_path, test_img)
        logger.info(f"Test floor plan saved to: {test_path}")
        
        # Test conversion
        logger.info("Testing 2D to 3D conversion...")
        result = ai_agent.convert_2d_to_3d(
            image_path=test_path,
            output_dir="test_output",
            output_name="test_house"
        )
        
        if result['success']:
            logger.info("✅ Conversion successful!")
            logger.info(f"Exported models: {result['exported_models']}")
            logger.info(f"Visualization: {result['visualization']}")
            
            # Check if output files exist
            output_dir = Path("test_output")
            if output_dir.exists():
                files = list(output_dir.rglob("*"))
                logger.info(f"Generated {len(files)} files:")
                for file in files:
                    logger.info(f"  - {file}")
            
            return True
        else:
            logger.error(f"❌ Conversion failed: {result.get('error', 'Unknown error')}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            if 'ai_agent' in locals():
                ai_agent.cleanup()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


def test_individual_components():
    """Test individual components of the system."""
    try:
        logger.info("Testing individual components...")
        
        # Test image preprocessing
        from src.preprocessing import ImageProcessor
        processor = ImageProcessor()
        
        # Create test image
        test_img = create_test_floor_plan()
        
        # Test preprocessing
        processed = processor.preprocess(test_img)
        logger.info("✅ Image preprocessing works")
        
        # Test feature detection
        from src.preprocessing import FeatureDetector
        detector = FeatureDetector()
        
        features = detector.detect_features(processed['final_binary'])
        logger.info(f"✅ Feature detection works - found {len(features.get('walls', []))} walls")
        
        # Test vectorization
        from src.vectorization import Vectorizer
        vectorizer = Vectorizer()
        
        # Create simple segmentation mask
        segmentation_mask = np.zeros(test_img.shape[:2], dtype=np.uint8)
        segmentation_mask[test_img[:,:,0] < 128] = 1  # Simple thresholding
        
        vector_data = vectorizer.vectorize_segmentation(segmentation_mask, features)
        logger.info("✅ Vectorization works")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    logger.info("🧪 Starting AI Architectural Converter System Tests")
    logger.info("=" * 60)
    
    # Test individual components
    logger.info("\n1. Testing individual components...")
    component_test_passed = test_individual_components()
    
    # Test AI Agent
    logger.info("\n2. Testing AI Agent...")
    agent_test_passed = test_ai_agent()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Component tests: {'✅ PASSED' if component_test_passed else '❌ FAILED'}")
    logger.info(f"AI Agent tests: {'✅ PASSED' if agent_test_passed else '❌ FAILED'}")
    
    if component_test_passed and agent_test_passed:
        logger.info("\n🎉 All tests passed! The system is working correctly.")
        return 0
    else:
        logger.error("\n💥 Some tests failed. Please check the logs above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())