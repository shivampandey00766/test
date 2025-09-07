#!/usr/bin/env python3
"""
Example script demonstrating how to use the AI Agent for 2D to 3D conversion.
This example shows a simple floor plan conversion workflow.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agent import AIAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_floor_plan():
    """Create a simple sample floor plan for testing."""
    import cv2
    import numpy as np
    
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


def main():
    """Main example function."""
    try:
        logger.info("Creating sample floor plan...")
        
        # Create sample floor plan
        sample_img = create_sample_floor_plan()
        
        # Save sample image
        sample_path = Path("sample_floor_plan.jpg")
        cv2.imwrite(str(sample_path), sample_img)
        logger.info(f"Sample floor plan saved to: {sample_path}")
        
        # Initialize AI Agent
        logger.info("Initializing AI Agent...")
        ai_agent = AIAgent()
        
        # Convert to 3D
        logger.info("Converting to 3D...")
        result = ai_agent.convert_2d_to_3d(
            image_path=str(sample_path),
            output_dir="output",
            output_name="sample_house"
        )
        
        if result['success']:
            logger.info("Conversion successful!")
            logger.info(f"Exported models: {result['exported_models']}")
            logger.info(f"Visualization: {result['visualization']}")
        else:
            logger.error(f"Conversion failed: {result.get('error', 'Unknown error')}")
            return 1
        
        # Cleanup
        ai_agent.cleanup()
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in example: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())