"""
Raster to vector conversion for floor plan images.

This module converts pixel-based floor plan images into vector
representations using contour detection and line approximation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
from dataclasses import dataclass
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union
import json


@dataclass
class VectorPoint:
    """Represents a 2D point in vector space."""
    x: float
    y: float
    z: float = 0.0


@dataclass
class VectorLine:
    """Represents a line segment in vector space."""
    start: VectorPoint
    end: VectorPoint
    thickness: float = 1.0
    line_type: str = "wall"


@dataclass
class VectorPolygon:
    """Represents a polygon in vector space."""
    points: List[VectorPoint]
    polygon_type: str = "room"
    area: float = 0.0


@dataclass
class VectorData:
    """Complete vector representation of a floor plan."""
    lines: List[VectorLine]
    polygons: List[VectorPolygon]
    points: List[VectorPoint]
    metadata: Dict[str, any]


class RasterToVectorConverter:
    """
    Converts raster floor plan images to vector representations.
    
    This class handles the conversion from pixel-based images to
    vector data suitable for 3D reconstruction and CAD applications.
    """
    
    def __init__(self, 
                 min_contour_area: int = 100,
                 approximation_epsilon: float = 0.02,
                 min_line_length: float = 10.0):
        """
        Initialize the RasterToVectorConverter.
        
        Args:
            min_contour_area: Minimum area for contour detection
            approximation_epsilon: Epsilon for contour approximation
            min_line_length: Minimum length for line segments
        """
        self.min_contour_area = min_contour_area
        self.approximation_epsilon = approximation_epsilon
        self.min_line_length = min_line_length
        
        # Line detection parameters
        self.hough_params = {
            'rho': 1,
            'theta': np.pi/180,
            'threshold': 50,
            'min_line_length': 30,
            'max_line_gap': 10
        }
    
    def detect_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect contours in the binary image.
        
        Args:
            image: Binary image
            
        Returns:
            List of detected contours
        """
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area
        filtered_contours = [
            contour for contour in contours 
            if cv2.contourArea(contour) >= self.min_contour_area
        ]
        
        return filtered_contours
    
    def approximate_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        Approximate contour to a polygon with fewer vertices.
        
        Args:
            contour: Input contour
            
        Returns:
            Approximated polygon
        """
        perimeter = cv2.arcLength(contour, True)
        epsilon = self.approximation_epsilon * perimeter
        approximated = cv2.approxPolyDP(contour, epsilon, True)
        
        return approximated
    
    def contour_to_polygon(self, contour: np.ndarray, 
                          polygon_type: str = "room") -> VectorPolygon:
        """
        Convert contour to VectorPolygon.
        
        Args:
            contour: Input contour
            polygon_type: Type of polygon (room, wall, etc.)
            
        Returns:
            VectorPolygon object
        """
        # Approximate contour
        approximated = self.approximate_contour(contour)
        
        # Convert to VectorPoints
        points = []
        for point in approximated:
            x, y = point[0]
            points.append(VectorPoint(x, y))
        
        # Calculate area
        area = cv2.contourArea(contour)
        
        return VectorPolygon(
            points=points,
            polygon_type=polygon_type,
            area=area
        )
    
    def detect_lines(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect line segments using Hough transform.
        
        Args:
            image: Binary image
            
        Returns:
            List of detected line segments
        """
        lines = cv2.HoughLinesP(
            image,
            self.hough_params['rho'],
            self.hough_params['theta'],
            self.hough_params['threshold'],
            minLineLength=self.hough_params['min_line_length'],
            maxLineGap=self.hough_params['max_line_gap']
        )
        
        if lines is None:
            return []
        
        # Filter lines by length
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if length >= self.min_line_length:
                filtered_lines.append(line[0])
        
        return filtered_lines
    
    def line_to_vector(self, line: np.ndarray, 
                      line_type: str = "wall",
                      thickness: float = 1.0) -> VectorLine:
        """
        Convert line segment to VectorLine.
        
        Args:
            line: Line segment [x1, y1, x2, y2]
            line_type: Type of line (wall, door, window)
            thickness: Line thickness
            
        Returns:
            VectorLine object
        """
        x1, y1, x2, y2 = line
        
        start_point = VectorPoint(x1, y1)
        end_point = VectorPoint(x2, y2)
        
        return VectorLine(
            start=start_point,
            end=end_point,
            thickness=thickness,
            line_type=line_type
        )
    
    def merge_similar_lines(self, lines: List[VectorLine], 
                          angle_threshold: float = 10.0,
                          distance_threshold: float = 20.0) -> List[VectorLine]:
        """
        Merge similar or collinear lines.
        
        Args:
            lines: List of VectorLines
            angle_threshold: Maximum angle difference for merging (degrees)
            distance_threshold: Maximum distance for merging (pixels)
            
        Returns:
            List of merged VectorLines
        """
        if not lines:
            return []
        
        merged_lines = []
        used_lines = set()
        
        for i, line1 in enumerate(lines):
            if i in used_lines:
                continue
            
            # Find similar lines to merge
            similar_lines = [line1]
            used_lines.add(i)
            
            for j, line2 in enumerate(lines[i+1:], i+1):
                if j in used_lines:
                    continue
                
                if self._are_lines_similar(line1, line2, angle_threshold, distance_threshold):
                    similar_lines.append(line2)
                    used_lines.add(j)
            
            # Merge similar lines
            if len(similar_lines) > 1:
                merged_line = self._merge_line_group(similar_lines)
                merged_lines.append(merged_line)
            else:
                merged_lines.append(line1)
        
        return merged_lines
    
    def _are_lines_similar(self, line1: VectorLine, line2: VectorLine,
                          angle_threshold: float, distance_threshold: float) -> bool:
        """Check if two lines are similar enough to merge."""
        # Calculate angles
        angle1 = np.arctan2(line1.end.y - line1.start.y, line1.end.x - line1.start.x)
        angle2 = np.arctan2(line2.end.y - line2.start.y, line2.end.x - line2.start.x)
        
        angle_diff = abs(angle1 - angle2) * 180 / np.pi
        angle_diff = min(angle_diff, 180 - angle_diff)  # Handle angle wrapping
        
        if angle_diff > angle_threshold:
            return False
        
        # Calculate minimum distance between lines
        min_distance = self._min_distance_between_lines(line1, line2)
        
        return min_distance <= distance_threshold
    
    def _min_distance_between_lines(self, line1: VectorLine, line2: VectorLine) -> float:
        """Calculate minimum distance between two line segments."""
        # Convert to shapely LineString for distance calculation
        line1_geom = LineString([(line1.start.x, line1.start.y), (line1.end.x, line1.end.y)])
        line2_geom = LineString([(line2.start.x, line2.start.y), (line2.end.x, line2.end.y)])
        
        return line1_geom.distance(line2_geom)
    
    def _merge_line_group(self, lines: List[VectorLine]) -> VectorLine:
        """Merge a group of similar lines into one line."""
        if len(lines) == 1:
            return lines[0]
        
        # Find the extreme points
        all_points = []
        for line in lines:
            all_points.extend([(line.start.x, line.start.y), (line.end.x, line.end.y)])
        
        # Find the two points that are farthest apart
        max_distance = 0
        best_points = None
        
        for i in range(len(all_points)):
            for j in range(i + 1, len(all_points)):
                p1, p2 = all_points[i], all_points[j]
                distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                
                if distance > max_distance:
                    max_distance = distance
                    best_points = (p1, p2)
        
        if best_points is None:
            return lines[0]
        
        # Create merged line
        start_point = VectorPoint(best_points[0][0], best_points[0][1])
        end_point = VectorPoint(best_points[1][0], best_points[1][1])
        
        # Use average thickness
        avg_thickness = np.mean([line.thickness for line in lines])
        
        return VectorLine(
            start=start_point,
            end=end_point,
            thickness=avg_thickness,
            line_type=lines[0].line_type
        )
    
    def classify_geometric_elements(self, image: np.ndarray, 
                                  segmentation_mask: np.ndarray = None) -> Dict[str, List]:
        """
        Classify geometric elements based on segmentation or heuristics.
        
        Args:
            image: Binary image
            segmentation_mask: Segmentation mask (optional)
            
        Returns:
            Dictionary with classified elements
        """
        elements = {
            'walls': [],
            'doors': [],
            'windows': [],
            'rooms': []
        }
        
        if segmentation_mask is not None:
            # Use segmentation mask for classification
            elements = self._classify_from_segmentation(segmentation_mask)
        else:
            # Use heuristics for classification
            elements = self._classify_from_heuristics(image)
        
        return elements
    
    def _classify_from_segmentation(self, segmentation_mask: np.ndarray) -> Dict[str, List]:
        """Classify elements based on segmentation mask."""
        elements = {
            'walls': [],
            'doors': [],
            'windows': [],
            'rooms': []
        }
        
        # Extract walls (class 1)
        wall_mask = (segmentation_mask == 1).astype(np.uint8) * 255
        wall_lines = self.detect_lines(wall_mask)
        elements['walls'] = [self.line_to_vector(line, "wall", 5.0) for line in wall_lines]
        
        # Extract doors (class 2)
        door_mask = (segmentation_mask == 2).astype(np.uint8) * 255
        door_contours = self.detect_contours(door_mask)
        elements['doors'] = [self.contour_to_polygon(contour, "door") for contour in door_contours]
        
        # Extract windows (class 3)
        window_mask = (segmentation_mask == 3).astype(np.uint8) * 255
        window_contours = self.detect_contours(window_mask)
        elements['windows'] = [self.contour_to_polygon(contour, "window") for contour in window_contours]
        
        # Extract rooms (class 4)
        room_mask = (segmentation_mask == 4).astype(np.uint8) * 255
        room_contours = self.detect_contours(room_mask)
        elements['rooms'] = [self.contour_to_polygon(contour, "room") for contour in room_contours]
        
        return elements
    
    def _classify_from_heuristics(self, image: np.ndarray) -> Dict[str, List]:
        """Classify elements using heuristic rules."""
        elements = {
            'walls': [],
            'doors': [],
            'windows': [],
            'rooms': []
        }
        
        # Detect all lines
        lines = self.detect_lines(image)
        vector_lines = [self.line_to_vector(line) for line in lines]
        
        # Classify lines based on length and thickness
        for line in vector_lines:
            length = np.sqrt((line.end.x - line.start.x)**2 + (line.end.y - line.start.y)**2)
            
            if length > 100:  # Long lines are likely walls
                line.line_type = "wall"
                line.thickness = 5.0
                elements['walls'].append(line)
            elif 20 < length < 100:  # Medium lines might be doors
                line.line_type = "door"
                line.thickness = 2.0
                elements['doors'].append(line)
        
        # Detect rooms from contours
        contours = self.detect_contours(image)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Large contours are likely rooms
                polygon = self.contour_to_polygon(contour, "room")
                elements['rooms'].append(polygon)
        
        return elements
    
    def convert_to_vector(self, image: np.ndarray,
                         segmentation_mask: np.ndarray = None) -> VectorData:
        """
        Convert raster image to vector representation.
        
        Args:
            image: Binary image
            segmentation_mask: Segmentation mask (optional)
            
        Returns:
            VectorData object
        """
        # Classify geometric elements
        elements = self.classify_geometric_elements(image, segmentation_mask)
        
        # Merge similar lines
        if elements['walls']:
            elements['walls'] = self.merge_similar_lines(elements['walls'])
        
        # Create vector data
        vector_data = VectorData(
            lines=elements['walls'] + elements['doors'],
            polygons=elements['rooms'] + elements['windows'],
            points=[],
            metadata={
                'image_shape': image.shape,
                'conversion_timestamp': np.datetime64('now').astype(str),
                'parameters': {
                    'min_contour_area': self.min_contour_area,
                    'approximation_epsilon': self.approximation_epsilon,
                    'min_line_length': self.min_line_length
                }
            }
        )
        
        return vector_data
    
    def export_to_json(self, vector_data: VectorData, 
                      output_path: str) -> None:
        """
        Export vector data to JSON format.
        
        Args:
            vector_data: VectorData object
            output_path: Output file path
        """
        # Convert to serializable format
        json_data = {
            'lines': [
                {
                    'start': {'x': line.start.x, 'y': line.start.y, 'z': line.start.z},
                    'end': {'x': line.end.x, 'y': line.end.y, 'z': line.end.z},
                    'thickness': line.thickness,
                    'line_type': line.line_type
                }
                for line in vector_data.lines
            ],
            'polygons': [
                {
                    'points': [
                        {'x': point.x, 'y': point.y, 'z': point.z}
                        for point in polygon.points
                    ],
                    'polygon_type': polygon.polygon_type,
                    'area': polygon.area
                }
                for polygon in vector_data.polygons
            ],
            'points': [
                {'x': point.x, 'y': point.y, 'z': point.z}
                for point in vector_data.points
            ],
            'metadata': vector_data.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def visualize_vector_data(self, vector_data: VectorData,
                            image_shape: Tuple[int, int] = None,
                            save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize vector data as an image.
        
        Args:
            vector_data: VectorData object
            image_shape: Shape of the output image
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        if image_shape is None:
            # Calculate bounding box
            all_x = []
            all_y = []
            
            for line in vector_data.lines:
                all_x.extend([line.start.x, line.end.x])
                all_y.extend([line.start.y, line.end.y])
            
            for polygon in vector_data.polygons:
                for point in polygon.points:
                    all_x.append(point.x)
                    all_y.append(point.y)
            
            if all_x and all_y:
                width = int(max(all_x) - min(all_x)) + 100
                height = int(max(all_y) - min(all_y)) + 100
            else:
                width, height = 800, 600
        else:
            height, width = image_shape
        
        # Create visualization image
        vis_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw polygons
        for polygon in vector_data.polygons:
            if len(polygon.points) >= 3:
                points = np.array([(p.x, p.y) for p in polygon.points], dtype=np.int32)
                color = self._get_polygon_color(polygon.polygon_type)
                cv2.fillPoly(vis_image, [points], color)
                cv2.polylines(vis_image, [points], True, (255, 255, 255), 2)
        
        # Draw lines
        for line in vector_data.lines:
            start = (int(line.start.x), int(line.start.y))
            end = (int(line.end.x), int(line.end.y))
            color = self._get_line_color(line.line_type)
            thickness = max(1, int(line.thickness))
            cv2.line(vis_image, start, end, color, thickness)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
    
    def _get_polygon_color(self, polygon_type: str) -> Tuple[int, int, int]:
        """Get color for polygon type."""
        colors = {
            'room': (0, 255, 0),      # Green
            'door': (255, 0, 0),      # Blue
            'window': (0, 0, 255),    # Red
            'wall': (128, 128, 128)   # Gray
        }
        return colors.get(polygon_type, (255, 255, 255))
    
    def _get_line_color(self, line_type: str) -> Tuple[int, int, int]:
        """Get color for line type."""
        colors = {
            'wall': (255, 255, 255),  # White
            'door': (0, 255, 255),    # Yellow
            'window': (255, 0, 255),  # Magenta
            'default': (128, 128, 128) # Gray
        }
        return colors.get(line_type, colors['default'])