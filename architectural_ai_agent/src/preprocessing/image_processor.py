"""
Image Preprocessing Module for 2D Floor Plans

This module handles the initial computer vision tasks including:
- Image cleaning and noise removal
- Perspective correction
- Line straightening
- Feature enhancement
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Preprocesses 2D floor plan images for feature detection and AI analysis.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (1024, 1024)):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target size for processed images (width, height)
        """
        self.target_size = target_size
        self.original_size = None
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Complete preprocessing pipeline for a floor plan image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        image = self._load_image(image_path)
        self.original_size = image.shape[:2]
        
        # Apply preprocessing steps
        image = self._remove_noise(image)
        image = self._correct_perspective(image)
        image = self._enhance_lines(image)
        image = self._resize_image(image)
        
        logger.info(f"Preprocessed image from {self.original_size} to {image.shape[:2]}")
        return image
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and convert image to grayscale."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        return image
    
    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from the image using morphological operations.
        """
        # Apply Gaussian blur to reduce noise
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Apply morphological opening to remove small noise
        kernel = np.ones((2, 2), np.uint8)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        
        # Apply median filter for additional noise reduction
        denoised = cv2.medianBlur(denoised, 3)
        
        return denoised
    
    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct perspective distortion in floor plans.
        """
        # Find edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Find lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None or len(lines) < 4:
            logger.warning("Could not detect enough lines for perspective correction")
            return image
        
        # Find dominant angles
        angles = []
        for line in lines[:20]:  # Use top 20 lines
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            angles.append(angle)
        
        # Find the most common angle (should be close to 0 or 90 degrees)
        hist, bins = np.histogram(angles, bins=36, range=(0, 180))
        dominant_angle_idx = np.argmax(hist)
        dominant_angle = (bins[dominant_angle_idx] + bins[dominant_angle_idx + 1]) / 2
        
        # Calculate rotation needed
        if dominant_angle > 45:
            rotation_angle = dominant_angle - 90
        else:
            rotation_angle = dominant_angle
        
        # Apply rotation if significant
        if abs(rotation_angle) > 2:  # Only rotate if angle > 2 degrees
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            logger.info(f"Applied perspective correction: {rotation_angle:.2f} degrees")
        
        return image
    
    def _enhance_lines(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance line features in the floor plan.
        """
        # Apply adaptive threshold to create binary image
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert so lines are white on black background
        binary = cv2.bitwise_not(binary)
        
        # Apply morphological operations to connect broken lines
        # Horizontal kernel
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Vertical kernel
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, vertical_kernel)
        
        # Combine horizontal and vertical lines
        enhanced = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
        
        # Add back to original
        result = cv2.addWeighted(image, 0.7, enhanced, 0.3, 0)
        
        return result
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio.
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scale to fit within target size
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas of target size and center the image
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return canvas
    
    def get_scale_factor(self) -> float:
        """
        Get the scale factor applied during preprocessing.
        
        Returns:
            Scale factor from original to processed image
        """
        if self.original_size is None:
            return 1.0
        
        original_max = max(self.original_size)
        target_max = max(self.target_size)
        return target_max / original_max
    
    def denormalize_coordinates(self, coordinates: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Convert coordinates from processed image back to original image space.
        
        Args:
            coordinates: List of (x, y) coordinates in processed image space
            
        Returns:
            List of (x, y) coordinates in original image space
        """
        if self.original_size is None:
            return coordinates
        
        scale = 1.0 / self.get_scale_factor()
        target_w, target_h = self.target_size
        orig_h, orig_w = self.original_size
        
        # Calculate offsets used during resizing
        scaled_w = int(orig_w * self.get_scale_factor())
        scaled_h = int(orig_h * self.get_scale_factor())
        x_offset = (target_w - scaled_w) // 2
        y_offset = (target_h - scaled_h) // 2
        
        denormalized = []
        for x, y in coordinates:
            # Remove offset and scale back
            orig_x = int((x - x_offset) * scale)
            orig_y = int((y - y_offset) * scale)
            denormalized.append((orig_x, orig_y))
        
        return denormalized