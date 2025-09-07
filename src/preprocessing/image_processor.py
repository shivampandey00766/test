"""
Image preprocessing module for 2D floor plan analysis.
Handles image cleaning, enhancement, and preparation for AI processing.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Main image processing class for floor plan preprocessing.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (1024, 1024)):
        """
        Initialize the image processor.
        
        Args:
            target_size: Target size for resizing images (width, height)
        """
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and validate an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array
        """
        try:
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.logger.info(f"Successfully loaded image: {image.shape}")
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            raise
    
    def preprocess(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline for a floor plan image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing processed image and metadata
        """
        try:
            # Step 1: Resize image
            resized = self._resize_image(image)
            
            # Step 2: Convert to grayscale
            grayscale = self._to_grayscale(resized)
            
            # Step 3: Noise reduction
            denoised = self._denoise(grayscale)
            
            # Step 4: Enhance contrast
            enhanced = self._enhance_contrast(denoised)
            
            # Step 5: Correct perspective (if needed)
            corrected = self._correct_perspective(enhanced)
            
            # Step 6: Binarize for better feature detection
            binary = self._binarize(corrected)
            
            return {
                'original': image,
                'resized': resized,
                'grayscale': grayscale,
                'denoised': denoised,
                'enhanced': enhanced,
                'corrected': corrected,
                'binary': binary,
                'metadata': {
                    'original_shape': image.shape,
                    'target_shape': self.target_size,
                    'processing_steps': 6
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            raise
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size while maintaining aspect ratio."""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to target size if necessary
        if new_w != target_w or new_h != target_h:
            # Create canvas with white background
            canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
            
            # Calculate padding offsets
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            # Place resized image on canvas
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            resized = canvas
        
        return resized
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction using bilateral filter."""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        Attempt to correct perspective distortion.
        This is a simplified version - in practice, you'd want more sophisticated detection.
        """
        # For now, return the image as-is
        # In a full implementation, you'd detect corners and apply perspective correction
        return image
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Convert image to binary using adaptive thresholding."""
        # Use adaptive thresholding to handle varying lighting
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return binary
    
    def save_processed_image(self, processed_data: Dict[str, Any], output_path: str, 
                           step: str = 'binary') -> None:
        """
        Save a processed image to file.
        
        Args:
            processed_data: Dictionary from preprocess() method
            output_path: Path to save the image
            step: Which processing step to save ('binary', 'enhanced', etc.)
        """
        try:
            if step not in processed_data:
                raise ValueError(f"Step '{step}' not found in processed data")
            
            image = processed_data[step]
            
            # Convert grayscale to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(output_path, image)
            self.logger.info(f"Saved processed image to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving image: {e}")
            raise


def create_processing_pipeline(target_size: Tuple[int, int] = (1024, 1024)) -> ImageProcessor:
    """
    Factory function to create a configured image processor.
    
    Args:
        target_size: Target size for image processing
        
    Returns:
        Configured ImageProcessor instance
    """
    return ImageProcessor(target_size=target_size)