"""
Perspective correction module for floor plan images.

This module handles perspective distortion correction for photographed
or scanned floor plans to ensure accurate measurements and feature detection.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class PerspectiveCorrector:
    """
    Corrects perspective distortion in floor plan images.
    
    This class provides methods to detect and correct perspective
    distortion in photographed or scanned floor plans, ensuring
    accurate measurements and proper feature detection.
    """
    
    def __init__(self):
        """Initialize the PerspectiveCorrector with default parameters."""
        self.min_contour_area = 1000
        self.approximation_epsilon = 0.02
    
    def detect_floor_plan_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the main floor plan contour in the image.
        
        Args:
            image: Preprocessed binary image
            
        Returns:
            Floor plan contour or None if not found
        """
        # Find all contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (likely the floor plan boundary)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if the contour is large enough
        if cv2.contourArea(largest_contour) < self.min_contour_area:
            return None
        
        return largest_contour
    
    def find_corners(self, contour: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the four corners of the floor plan using contour approximation.
        
        Args:
            contour: Floor plan contour
            
        Returns:
            Array of four corner points or None if not found
        """
        # Approximate the contour to a polygon
        perimeter = cv2.arcLength(contour, True)
        epsilon = self.approximation_epsilon * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If we have exactly 4 points, use them
        if len(approx) == 4:
            return approx.reshape(4, 2)
        
        # If we have more than 4 points, try to find the best 4 corners
        if len(approx) > 4:
            return self._find_best_corners(approx)
        
        return None
    
    def _find_best_corners(self, points: np.ndarray) -> np.ndarray:
        """
        Find the best 4 corners from a set of points.
        
        Args:
            points: Array of contour points
            
        Returns:
            Array of 4 best corner points
        """
        # Convert to float32 for better precision
        points = np.float32(points)
        
        # Find the convex hull
        hull = cv2.convexHull(points)
        
        # Use Douglas-Peucker algorithm with different epsilon values
        for epsilon in [0.01, 0.02, 0.03, 0.05, 0.1]:
            approx = cv2.approxPolyDP(hull, epsilon * cv2.arcLength(hull, True), True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
        
        # If still not 4 points, manually select corners
        return self._manual_corner_selection(points)
    
    def _manual_corner_selection(self, points: np.ndarray) -> np.ndarray:
        """
        Manually select the 4 best corners from points.
        
        Args:
            points: Array of contour points
            
        Returns:
            Array of 4 selected corner points
        """
        # Find bounding rectangle
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        
        # Sort points to get consistent ordering
        box = self._sort_points(box)
        
        return box
    
    def _sort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Sort points in consistent order (top-left, top-right, bottom-right, bottom-left).
        
        Args:
            points: Array of 4 points
            
        Returns:
            Sorted array of 4 points
        """
        # Calculate center point
        center = np.mean(points, axis=0)
        
        # Sort by angle from center
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        
        # Reorder to get top-left, top-right, bottom-right, bottom-left
        ordered_points = points[sorted_indices]
        
        # Ensure proper ordering
        if ordered_points[0, 0] > ordered_points[1, 0]:  # If first point is more right than second
            ordered_points = np.roll(ordered_points, -1, axis=0)
        
        return ordered_points
    
    def correct_perspective(self, image: np.ndarray, 
                          corners: np.ndarray,
                          target_width: int = 800,
                          target_height: int = 600) -> np.ndarray:
        """
        Correct perspective distortion using the detected corners.
        
        Args:
            image: Input image
            corners: Four corner points of the floor plan
            target_width: Target width for corrected image
            target_height: Target height for corrected image
            
        Returns:
            Perspective-corrected image
        """
        # Define target rectangle
        target_corners = np.array([
            [0, 0],
            [target_width, 0],
            [target_width, target_height],
            [0, target_height]
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target_corners)
        
        # Apply perspective transformation
        corrected_image = cv2.warpPerspective(image, transform_matrix, (target_width, target_height))
        
        return corrected_image
    
    def auto_correct_perspective(self, image: np.ndarray,
                               target_width: int = 800,
                               target_height: int = 600) -> Tuple[np.ndarray, bool]:
        """
        Automatically detect and correct perspective distortion.
        
        Args:
            image: Input binary image
            target_width: Target width for corrected image
            target_height: Target height for corrected image
            
        Returns:
            Tuple of (corrected_image, success_flag)
        """
        # Detect floor plan contour
        contour = self.detect_floor_plan_contour(image)
        if contour is None:
            return image, False
        
        # Find corners
        corners = self.find_corners(contour)
        if corners is None or len(corners) != 4:
            return image, False
        
        # Correct perspective
        corrected_image = self.correct_perspective(image, corners, target_width, target_height)
        
        return corrected_image, True
    
    def interactive_corner_selection(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Interactive corner selection for manual perspective correction.
        
        Args:
            image: Input image
            
        Returns:
            Selected corner points or None if cancelled
        """
        corners = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                corners.append((x, y))
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(image, f"{len(corners)}", (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Select Corners', image)
        
        # Create window and set mouse callback
        cv2.namedWindow('Select Corners', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Select Corners', mouse_callback)
        
        # Display image and wait for corner selection
        cv2.imshow('Select Corners', image)
        print("Click on 4 corners of the floor plan, then press 'q' to confirm")
        
        while len(corners) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and len(corners) == 4:
                break
            elif key == ord('r'):  # Reset
                corners = []
                image_copy = image.copy()
                cv2.imshow('Select Corners', image_copy)
        
        cv2.destroyAllWindows()
        
        if len(corners) == 4:
            return np.array(corners, dtype=np.float32)
        
        return None
    
    def measure_accuracy(self, original_image: np.ndarray, 
                        corrected_image: np.ndarray) -> float:
        """
        Measure the accuracy of perspective correction.
        
        Args:
            original_image: Original image
            corrected_image: Perspective-corrected image
            
        Returns:
            Accuracy score (0-1)
        """
        # Calculate image similarity using structural similarity
        from skimage.metrics import structural_similarity as ssim
        
        # Resize images to same size for comparison
        h, w = min(original_image.shape[0], corrected_image.shape[0]), \
               min(original_image.shape[1], corrected_image.shape[1])
        
        orig_resized = cv2.resize(original_image, (w, h))
        corr_resized = cv2.resize(corrected_image, (w, h))
        
        # Calculate SSIM
        similarity = ssim(orig_resized, corr_resized)
        
        return max(0, similarity)  # Ensure non-negative
    
    def visualize_correction(self, original_image: np.ndarray,
                           corrected_image: np.ndarray,
                           corners: Optional[np.ndarray] = None,
                           save_path: Optional[str] = None):
        """
        Visualize the perspective correction process.
        
        Args:
            original_image: Original image
            corrected_image: Corrected image
            corners: Detected corner points (optional)
            save_path: Optional path to save the visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image with corners
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        if corners is not None:
            # Draw corners on original image
            for i, corner in enumerate(corners):
                axes[0].plot(corner[0], corner[1], 'ro', markersize=8)
                axes[0].text(corner[0] + 5, corner[1] + 5, f'{i+1}', 
                           color='red', fontsize=12, fontweight='bold')
        
        # Corrected image
        axes[1].imshow(corrected_image, cmap='gray')
        axes[1].set_title('Perspective Corrected')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()