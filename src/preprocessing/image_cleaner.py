"""
Image cleaning and preprocessing utilities for floor plan images.

This module provides functionality to clean, denoise, and enhance
2D floor plan images for better feature detection and analysis.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from PIL import Image
import matplotlib.pyplot as plt


class ImageCleaner:
    """
    Handles image cleaning and preprocessing for 2D floor plans.
    
    This class provides methods to clean scanned or photographed
    floor plans, remove noise, enhance contrast, and prepare images
    for feature detection.
    """
    
    def __init__(self):
        """Initialize the ImageCleaner with default parameters."""
        self.kernel_sizes = {
            'small': (3, 3),
            'medium': (5, 5),
            'large': (7, 7)
        }
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            return image
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def remove_noise(self, image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
        """
        Remove noise from the image.
        
        Args:
            image: Input grayscale image
            method: Denoising method ('gaussian', 'bilateral', 'median')
            
        Returns:
            Denoised image
        """
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        else:
            raise ValueError(f"Unknown denoising method: {method}")
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Enhance image contrast for better feature detection.
        
        Args:
            image: Input grayscale image
            method: Contrast enhancement method ('clahe', 'histogram', 'gamma')
            
        Returns:
            Enhanced image
        """
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        elif method == 'histogram':
            return cv2.equalizeHist(image)
        elif method == 'gamma':
            gamma = 1.5
            lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                   for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, lookup_table)
        else:
            raise ValueError(f"Unknown contrast enhancement method: {method}")
    
    def binarize(self, image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
        """
        Convert image to binary (black and white).
        
        Args:
            image: Input grayscale image
            method: Binarization method ('adaptive', 'otsu', 'threshold')
            
        Returns:
            Binary image
        """
        if method == 'adaptive':
            return cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        elif method == 'otsu':
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif method == 'threshold':
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return binary
        else:
            raise ValueError(f"Unknown binarization method: {method}")
    
    def detect_and_remove_small_objects(self, image: np.ndarray, min_area: int = 100) -> np.ndarray:
        """
        Remove small objects and noise from binary image.
        
        Args:
            image: Binary image
            min_area: Minimum area threshold for objects to keep
            
        Returns:
            Cleaned binary image
        """
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for objects larger than min_area
        mask = np.zeros_like(image)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.fillPoly(mask, [contour], 255)
        
        return mask
    
    def straighten_lines(self, image: np.ndarray) -> np.ndarray:
        """
        Straighten and clean up lines in the floor plan.
        
        Args:
            image: Binary image
            
        Returns:
            Image with straightened lines
        """
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            image, 1, np.pi/180, threshold=100, 
            minLineLength=50, maxLineGap=10
        )
        
        if lines is not None:
            # Create a clean image with only the detected lines
            line_image = np.zeros_like(image)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
            
            return line_image
        
        return image
    
    def morphological_operations(self, image: np.ndarray, operation: str = 'close') -> np.ndarray:
        """
        Apply morphological operations to clean the image.
        
        Args:
            image: Binary image
            operation: Type of operation ('close', 'open', 'dilate', 'erode')
            
        Returns:
            Processed image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        if operation == 'close':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'open':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'dilate':
            return cv2.dilate(image, kernel, iterations=1)
        elif operation == 'erode':
            return cv2.erode(image, kernel, iterations=1)
        else:
            raise ValueError(f"Unknown morphological operation: {operation}")
    
    def preprocess_pipeline(self, image_path: str, 
                          denoise_method: str = 'bilateral',
                          contrast_method: str = 'clahe',
                          binarize_method: str = 'adaptive') -> np.ndarray:
        """
        Complete preprocessing pipeline for floor plan images.
        
        Args:
            image_path: Path to input image
            denoise_method: Method for noise removal
            contrast_method: Method for contrast enhancement
            binarize_method: Method for binarization
            
        Returns:
            Preprocessed binary image ready for feature detection
        """
        # Load image
        image = self.load_image(image_path)
        
        # Convert to grayscale
        gray = self.convert_to_grayscale(image)
        
        # Remove noise
        denoised = self.remove_noise(gray, denoise_method)
        
        # Enhance contrast
        enhanced = self.enhance_contrast(denoised, contrast_method)
        
        # Binarize
        binary = self.binarize(enhanced, binarize_method)
        
        # Remove small objects
        cleaned = self.detect_and_remove_small_objects(binary, min_area=50)
        
        # Apply morphological operations
        processed = self.morphological_operations(cleaned, 'close')
        
        return processed
    
    def visualize_preprocessing_steps(self, image_path: str, save_path: Optional[str] = None):
        """
        Visualize all preprocessing steps for debugging.
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save the visualization
        """
        # Load original image
        original = self.load_image(image_path)
        gray = self.convert_to_grayscale(original)
        
        # Apply preprocessing steps
        denoised = self.remove_noise(gray, 'bilateral')
        enhanced = self.enhance_contrast(denoised, 'clahe')
        binary = self.binarize(enhanced, 'adaptive')
        cleaned = self.detect_and_remove_small_objects(binary, min_area=50)
        processed = self.morphological_operations(cleaned, 'close')
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(denoised, cmap='gray')
        axes[0, 2].set_title('Denoised')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(enhanced, cmap='gray')
        axes[1, 0].set_title('Enhanced Contrast')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(binary, cmap='gray')
        axes[1, 1].set_title('Binarized')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(processed, cmap='gray')
        axes[1, 2].set_title('Final Processed')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()