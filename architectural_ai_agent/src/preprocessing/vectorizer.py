"""
Vectorization Module for Converting Raster to Vector Data

This module converts processed raster images into vector representations
that are easier for geometric processing and 3D reconstruction.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union
import logging

logger = logging.getLogger(__name__)


class Vectorizer:
    """
    Converts raster floor plan images to vector representations.
    """
    
    def __init__(self, line_thickness_threshold: int = 3):
        """
        Initialize the vectorizer.
        
        Args:
            line_thickness_threshold: Minimum thickness for line detection
        """
        self.line_thickness_threshold = line_thickness_threshold
        
    def vectorize_image(self, image: np.ndarray) -> Dict:
        """
        Convert a preprocessed floor plan image to vector format.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Dictionary containing vectorized elements
        """
        # Extract different types of lines
        walls = self._extract_walls(image)
        thin_lines = self._extract_thin_lines(image)
        
        # Detect geometric shapes
        rectangles = self._detect_rectangles(image)
        circles = self._detect_circles(image)
        
        # Combine into structured format
        vector_data = {
            'walls': walls,
            'thin_lines': thin_lines,
            'rectangles': rectangles,
            'circles': circles,
            'image_shape': image.shape
        }
        
        logger.info(f"Vectorized image: {len(walls)} walls, {len(thin_lines)} thin lines, "
                   f"{len(rectangles)} rectangles, {len(circles)} circles")
        
        return vector_data
    
    def _extract_walls(self, image: np.ndarray) -> List[Dict]:
        """
        Extract thick lines representing walls.
        """
        # Create binary image
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to identify thick lines (walls)
        kernel = np.ones((self.line_thickness_threshold, self.line_thickness_threshold), np.uint8)
        thick_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours of thick lines
        contours, _ = cv2.findContours(thick_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        walls = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                # Approximate contour to reduce points
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Convert to list of points
                points = [(int(point[0][0]), int(point[0][1])) for point in approx]
                
                # Calculate wall properties
                wall_info = {
                    'points': points,
                    'area': cv2.contourArea(contour),
                    'perimeter': cv2.arcLength(contour, True),
                    'type': 'wall'
                }
                walls.append(wall_info)
        
        return walls
    
    def _extract_thin_lines(self, image: np.ndarray) -> List[Dict]:
        """
        Extract thin lines (doors, windows, furniture outlines).
        """
        # Create binary image
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Remove thick lines to isolate thin lines
        kernel = np.ones((self.line_thickness_threshold, self.line_thickness_threshold), np.uint8)
        thick_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        thin_lines_mask = cv2.subtract(binary, thick_lines)
        
        # Apply Hough Line Transform to detect straight lines
        edges = cv2.Canny(thin_lines_mask, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        
        thin_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                line_info = {
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'length': length,
                    'angle': np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi,
                    'type': 'thin_line'
                }
                thin_lines.append(line_info)
        
        return thin_lines
    
    def _detect_rectangles(self, image: np.ndarray) -> List[Dict]:
        """
        Detect rectangular shapes (rooms, furniture).
        """
        # Create binary image
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        for contour in contours:
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 sides)
            if len(approx) == 4 and cv2.contourArea(contour) > 500:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if it's actually rectangular (not too skewed)
                contour_area = cv2.contourArea(contour)
                bounding_area = w * h
                if contour_area / bounding_area > 0.8:  # 80% fill ratio
                    rect_info = {
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'area': contour_area,
                        'aspect_ratio': w / h if h > 0 else 0,
                        'corners': [(int(point[0][0]), int(point[0][1])) for point in approx],
                        'type': 'rectangle'
                    }
                    rectangles.append(rect_info)
        
        return rectangles
    
    def _detect_circles(self, image: np.ndarray) -> List[Dict]:
        """
        Detect circular shapes (curved walls, fixtures).
        """
        # Apply HoughCircles to detect circles
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        circle_list = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                circle_info = {
                    'center': (x, y),
                    'radius': r,
                    'area': np.pi * r * r,
                    'type': 'circle'
                }
                circle_list.append(circle_info)
        
        return circle_list
    
    def create_shapely_geometries(self, vector_data: Dict) -> Dict:
        """
        Convert vector data to Shapely geometries for advanced geometric operations.
        
        Args:
            vector_data: Output from vectorize_image
            
        Returns:
            Dictionary of Shapely geometry objects
        """
        geometries = {
            'wall_polygons': [],
            'line_strings': [],
            'rectangles': [],
            'circles': []
        }
        
        # Convert walls to polygons
        for wall in vector_data['walls']:
            if len(wall['points']) >= 3:
                try:
                    poly = Polygon(wall['points'])
                    if poly.is_valid:
                        geometries['wall_polygons'].append(poly)
                except Exception as e:
                    logger.warning(f"Could not create polygon from wall: {e}")
        
        # Convert thin lines to LineStrings
        for line in vector_data['thin_lines']:
            try:
                line_string = LineString([line['start'], line['end']])
                geometries['line_strings'].append(line_string)
            except Exception as e:
                logger.warning(f"Could not create LineString: {e}")
        
        # Convert rectangles to polygons
        for rect in vector_data['rectangles']:
            corners = [
                (rect['x'], rect['y']),
                (rect['x'] + rect['width'], rect['y']),
                (rect['x'] + rect['width'], rect['y'] + rect['height']),
                (rect['x'], rect['y'] + rect['height'])
            ]
            try:
                poly = Polygon(corners)
                if poly.is_valid:
                    geometries['rectangles'].append(poly)
            except Exception as e:
                logger.warning(f"Could not create rectangle polygon: {e}")
        
        # Convert circles to polygons (approximated)
        for circle in vector_data['circles']:
            center = Point(circle['center'])
            circle_poly = center.buffer(circle['radius'])
            geometries['circles'].append(circle_poly)
        
        return geometries
    
    def simplify_geometries(self, geometries: Dict, tolerance: float = 2.0) -> Dict:
        """
        Simplify geometries to reduce complexity while maintaining shape.
        
        Args:
            geometries: Shapely geometries dictionary
            tolerance: Simplification tolerance
            
        Returns:
            Simplified geometries
        """
        simplified = {}
        
        for key, geom_list in geometries.items():
            simplified[key] = []
            for geom in geom_list:
                try:
                    simplified_geom = geom.simplify(tolerance, preserve_topology=True)
                    if simplified_geom.is_valid and not simplified_geom.is_empty:
                        simplified[key].append(simplified_geom)
                except Exception as e:
                    logger.warning(f"Could not simplify geometry: {e}")
                    simplified[key].append(geom)  # Keep original if simplification fails
        
        return simplified