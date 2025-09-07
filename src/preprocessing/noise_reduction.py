"""
Advanced noise reduction and image enhancement for floor plan preprocessing.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NoiseReducer:
    """
    Advanced noise reduction and image enhancement for floor plans.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def reduce_noise(self, image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
        """
        Apply noise reduction to the image.
        
        Args:
            image: Input grayscale image
            method: Noise reduction method ('adaptive', 'morphological', 'gaussian')
            
        Returns:
            Denoised image
        """
        try:
            if method == 'adaptive':
                return self._adaptive_denoise(image)
            elif method == 'morphological':
                return self._morphological_denoise(image)
            elif method == 'gaussian':
                return self._gaussian_denoise(image)
            else:
                raise ValueError(f"Unknown denoising method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error in noise reduction: {e}")
            return image
    
    def _adaptive_denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive bilateral filtering."""
        # Bilateral filter preserves edges while reducing noise
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def _morphological_denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations for noise reduction."""
        # Create kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Opening (erosion followed by dilation) removes noise
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Closing (dilation followed by erosion) fills gaps
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def _gaussian_denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur for noise reduction."""
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Enhance image contrast.
        
        Args:
            image: Input grayscale image
            method: Contrast enhancement method ('clahe', 'histogram', 'gamma')
            
        Returns:
            Enhanced image
        """
        try:
            if method == 'clahe':
                return self._clahe_enhancement(image)
            elif method == 'histogram':
                return self._histogram_equalization(image)
            elif method == 'gamma':
                return self._gamma_correction(image)
            else:
                raise ValueError(f"Unknown enhancement method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error in contrast enhancement: {e}")
            return image
    
    def _clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization."""
        return cv2.equalizeHist(image)
    
    def _gamma_correction(self, image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """Apply gamma correction."""
        # Build lookup table
        lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction
        return cv2.LUT(image, lookup_table)
    
    def remove_small_objects(self, image: np.ndarray, min_size: int = 100) -> np.ndarray:
        """
        Remove small objects from binary image.
        
        Args:
            image: Binary image
            min_size: Minimum size of objects to keep
            
        Returns:
            Cleaned binary image
        """
        try:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
            
            # Create mask for objects larger than min_size
            mask = np.zeros_like(image)
            for i in range(1, num_labels):  # Skip background (label 0)
                if stats[i, cv2.CC_STAT_AREA] >= min_size:
                    mask[labels == i] = 255
            
            return mask
            
        except Exception as e:
            self.logger.error(f"Error removing small objects: {e}")
            return image
    
    def fill_holes(self, image: np.ndarray) -> np.ndarray:
        """
        Fill holes in binary image.
        
        Args:
            image: Binary image
            
        Returns:
            Image with holes filled
        """
        try:
            # Create a copy of the image
            filled = image.copy()
            
            # Find contours
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Fill each contour
            for contour in contours:
                cv2.fillPoly(filled, [contour], 255)
            
            return filled
            
        except Exception as e:
            self.logger.error(f"Error filling holes: {e}")
            return image
    
    def smooth_contours(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Smooth contours in binary image.
        
        Args:
            image: Binary image
            kernel_size: Size of smoothing kernel
            
        Returns:
            Smoothed image
        """
        try:
            # Create kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Apply morphological closing to smooth contours
            smoothed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            
            return smoothed
            
        except Exception as e:
            self.logger.error(f"Error smoothing contours: {e}")
            return image
    
    def enhance_lines(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance line features in the image.
        
        Args:
            image: Binary image
            
        Returns:
            Image with enhanced lines
        """
        try:
            # Apply morphological operations to enhance lines
            kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            # Detect horizontal lines
            horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_horizontal)
            
            # Detect vertical lines
            vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_vertical)
            
            # Combine horizontal and vertical lines
            enhanced = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
            
            # Add original image
            enhanced = cv2.addWeighted(enhanced, 0.7, image, 0.3, 0)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing lines: {e}")
            return image
    
    def preprocess_pipeline(self, image: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for noise reduction and enhancement.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Preprocessed image
        """
        try:
            # Step 1: Reduce noise
            denoised = self.reduce_noise(image, method='adaptive')
            
            # Step 2: Enhance contrast
            enhanced = self.enhance_contrast(denoised, method='clahe')
            
            # Step 3: Binarize
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Step 4: Remove small objects
            cleaned = self.remove_small_objects(binary, min_size=50)
            
            # Step 5: Fill holes
            filled = self.fill_holes(cleaned)
            
            # Step 6: Smooth contours
            smoothed = self.smooth_contours(filled, kernel_size=3)
            
            # Step 7: Enhance lines
            final = self.enhance_lines(smoothed)
            
            return final
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {e}")
            return image