"""
Vectorization module for converting raster floor plan data to vector format.
Handles conversion from pixel-based segmentation to geometric primitives.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class VectorPoint:
    """Represents a 2D point in vector space."""
    x: float
    y: float


@dataclass
class VectorLine:
    """Represents a line segment in vector space."""
    start: VectorPoint
    end: VectorPoint
    thickness: float
    element_type: str  # 'wall', 'door', 'window'


@dataclass
class VectorPolygon:
    """Represents a polygon in vector space."""
    points: List[VectorPoint]
    element_type: str  # 'room', 'furniture'
    properties: Dict[str, Any]


@dataclass
class VectorText:
    """Represents text annotation in vector space."""
    position: VectorPoint
    text: str
    font_size: float
    rotation: float


class Vectorizer:
    """
    Converts raster floor plan data to vector format.
    """
    
    def __init__(self, scale_factor: float = 1.0):
        """
        Initialize the vectorizer.
        
        Args:
            scale_factor: Scale factor for coordinate conversion
        """
        self.scale_factor = scale_factor
        self.logger = logging.getLogger(__name__)
    
    def vectorize_segmentation(self, segmentation_mask: np.ndarray, 
                             features: Dict[str, List]) -> Dict[str, Any]:
        """
        Convert segmentation mask and features to vector format.
        
        Args:
            segmentation_mask: Segmentation mask from AI model
            features: Detected features from computer vision
            
        Returns:
            Dictionary containing vectorized data
        """
        try:
            vector_data = {
                'walls': self._vectorize_walls(segmentation_mask, features.get('walls', [])),
                'doors': self._vectorize_doors(segmentation_mask, features.get('doors', [])),
                'windows': self._vectorize_windows(segmentation_mask, features.get('windows', [])),
                'rooms': self._vectorize_rooms(segmentation_mask, features.get('rooms', [])),
                'furniture': self._vectorize_furniture(segmentation_mask),
                'text': self._vectorize_text(segmentation_mask, features.get('text_regions', [])),
                'metadata': {
                    'scale_factor': self.scale_factor,
                    'image_dimensions': segmentation_mask.shape,
                    'coordinate_system': 'image_coordinates'
                }
            }
            
            self.logger.info("Successfully vectorized segmentation data")
            return vector_data
            
        except Exception as e:
            self.logger.error(f"Error vectorizing segmentation: {e}")
            raise
    
    def _vectorize_walls(self, segmentation_mask: np.ndarray, 
                        detected_walls: List) -> List[VectorLine]:
        """Convert wall segments to vector lines."""
        try:
            walls = []
            
            # Get wall pixels from segmentation
            wall_mask = (segmentation_mask == 1).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour to lines
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Convert to line segments
                for i in range(len(approx) - 1):
                    pt1 = approx[i][0]
                    pt2 = approx[i + 1][0]
                    
                    # Calculate thickness
                    thickness = self._estimate_wall_thickness(wall_mask, pt1, pt2)
                    
                    wall = VectorLine(
                        start=VectorPoint(float(pt1[0]), float(pt1[1])),
                        end=VectorPoint(float(pt2[0]), float(pt2[1])),
                        thickness=thickness,
                        element_type='wall'
                    )
                    walls.append(wall)
            
            # Also use detected walls from computer vision
            for wall in detected_walls:
                vector_wall = VectorLine(
                    start=VectorPoint(float(wall.start[0]), float(wall.start[1])),
                    end=VectorPoint(float(wall.end[0]), float(wall.end[1])),
                    thickness=wall.thickness,
                    element_type='wall'
                )
                walls.append(vector_wall)
            
            # Merge and clean up wall segments
            walls = self._merge_wall_segments(walls)
            
            self.logger.info(f"Vectorized {len(walls)} wall segments")
            return walls
            
        except Exception as e:
            self.logger.error(f"Error vectorizing walls: {e}")
            return []
    
    def _vectorize_doors(self, segmentation_mask: np.ndarray, 
                        detected_doors: List) -> List[VectorLine]:
        """Convert doors to vector lines."""
        try:
            doors = []
            
            # Get door pixels from segmentation
            door_mask = (segmentation_mask == 2).astype(np.uint8) * 255
            
            # Find door contours
            contours, _ = cv2.findContours(door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                rect = cv2.minAreaRect(contour)
                center, (width, height), angle = rect
                
                # Create door line
                door = VectorLine(
                    start=VectorPoint(float(center[0] - width/2), float(center[1])),
                    end=VectorPoint(float(center[0] + width/2), float(center[1])),
                    thickness=height,
                    element_type='door'
                )
                doors.append(door)
            
            # Also use detected doors
            for door in detected_doors:
                vector_door = VectorLine(
                    start=VectorPoint(float(door.center[0] - door.width/2), 
                                    float(door.center[1])),
                    end=VectorPoint(float(door.center[0] + door.width/2), 
                                  float(door.center[1])),
                    thickness=door.width * 0.2,  # Estimate door thickness
                    element_type='door'
                )
                doors.append(vector_door)
            
            self.logger.info(f"Vectorized {len(doors)} doors")
            return doors
            
        except Exception as e:
            self.logger.error(f"Error vectorizing doors: {e}")
            return []
    
    def _vectorize_windows(self, segmentation_mask: np.ndarray, 
                          detected_windows: List) -> List[VectorLine]:
        """Convert windows to vector lines."""
        try:
            windows = []
            
            # Get window pixels from segmentation
            window_mask = (segmentation_mask == 3).astype(np.uint8) * 255
            
            # Find window contours
            contours, _ = cv2.findContours(window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                rect = cv2.minAreaRect(contour)
                center, (width, height), angle = rect
                
                # Create window line
                window = VectorLine(
                    start=VectorPoint(float(center[0] - width/2), float(center[1])),
                    end=VectorPoint(float(center[0] + width/2), float(center[1])),
                    thickness=height,
                    element_type='window'
                )
                windows.append(window)
            
            # Also use detected windows
            for window in detected_windows:
                vector_window = VectorLine(
                    start=VectorPoint(float(window.center[0] - window.width/2), 
                                    float(window.center[1])),
                    end=VectorPoint(float(window.center[0] + window.width/2), 
                                  float(window.center[1])),
                    thickness=window.height,
                    element_type='window'
                )
                windows.append(vector_window)
            
            self.logger.info(f"Vectorized {len(windows)} windows")
            return windows
            
        except Exception as e:
            self.logger.error(f"Error vectorizing windows: {e}")
            return []
    
    def _vectorize_rooms(self, segmentation_mask: np.ndarray, 
                        detected_rooms: List) -> List[VectorPolygon]:
        """Convert rooms to vector polygons."""
        try:
            rooms = []
            
            # Process each room type
            room_types = [4, 5, 6, 7, 8]  # living, kitchen, bedroom, bathroom, closet
            room_names = ['living_room', 'kitchen', 'bedroom', 'bathroom', 'closet']
            
            for room_type, room_name in zip(room_types, room_names):
                room_mask = (segmentation_mask == room_type).astype(np.uint8) * 255
                
                if np.any(room_mask):
                    # Find contours
                    contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        # Simplify contour
                        epsilon = 0.01 * cv2.arcLength(contour, True)
                        simplified = cv2.approxPolyDP(contour, epsilon, True)
                        
                        # Convert to vector points
                        points = [VectorPoint(float(pt[0][0]), float(pt[0][1])) 
                                for pt in simplified]
                        
                        # Calculate properties
                        area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        
                        room = VectorPolygon(
                            points=points,
                            element_type='room',
                            properties={
                                'room_type': room_name,
                                'area': area,
                                'perimeter': perimeter,
                                'center': self._calculate_center(points)
                            }
                        )
                        rooms.append(room)
            
            # Also use detected rooms
            for room in detected_rooms:
                # Convert contour to vector points
                points = [VectorPoint(float(pt[0][0]), float(pt[0][1])) 
                        for pt in room.contour]
                
                vector_room = VectorPolygon(
                    points=points,
                    element_type='room',
                    properties={
                        'room_type': room.room_type,
                        'area': room.area,
                        'perimeter': cv2.arcLength(room.contour, True),
                        'center': VectorPoint(float(room.center[0]), float(room.center[1]))
                    }
                )
                rooms.append(vector_room)
            
            self.logger.info(f"Vectorized {len(rooms)} rooms")
            return rooms
            
        except Exception as e:
            self.logger.error(f"Error vectorizing rooms: {e}")
            return []
    
    def _vectorize_furniture(self, segmentation_mask: np.ndarray) -> List[VectorPolygon]:
        """Convert furniture to vector polygons."""
        try:
            furniture = []
            
            # Get furniture pixels from segmentation
            furniture_mask = (segmentation_mask == 9).astype(np.uint8) * 255
            
            if np.any(furniture_mask):
                # Find contours
                contours, _ = cv2.findContours(furniture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Simplify contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    simplified = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Convert to vector points
                    points = [VectorPoint(float(pt[0][0]), float(pt[0][1])) 
                            for pt in simplified]
                    
                    # Calculate properties
                    area = cv2.contourArea(contour)
                    rect = cv2.minAreaRect(contour)
                    center, (width, height), angle = rect
                    
                    furniture_item = VectorPolygon(
                        points=points,
                        element_type='furniture',
                        properties={
                            'area': area,
                            'width': width,
                            'height': height,
                            'angle': angle,
                            'center': VectorPoint(float(center[0]), float(center[1]))
                        }
                    )
                    furniture.append(furniture_item)
            
            self.logger.info(f"Vectorized {len(furniture)} furniture items")
            return furniture
            
        except Exception as e:
            self.logger.error(f"Error vectorizing furniture: {e}")
            return []
    
    def _vectorize_text(self, segmentation_mask: np.ndarray, 
                       text_regions: List) -> List[VectorText]:
        """Convert text regions to vector text."""
        try:
            text_elements = []
            
            # Get text pixels from segmentation
            text_mask = (segmentation_mask == 10).astype(np.uint8) * 255
            
            if np.any(text_mask):
                # Find contours
                contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Estimate font size based on height
                    font_size = h * 0.8
                    
                    text_element = VectorText(
                        position=VectorPoint(float(x + w/2), float(y + h/2)),
                        text="TEXT",  # Placeholder - would need OCR for actual text
                        font_size=font_size,
                        rotation=0.0
                    )
                    text_elements.append(text_element)
            
            # Also use detected text regions
            for region in text_regions:
                x, y, w, h = region['bbox']
                font_size = h * 0.8
                
                text_element = VectorText(
                    position=VectorPoint(float(x + w/2), float(y + h/2)),
                    text="TEXT",  # Placeholder
                    font_size=font_size,
                    rotation=0.0
                )
                text_elements.append(text_element)
            
            self.logger.info(f"Vectorized {len(text_elements)} text elements")
            return text_elements
            
        except Exception as e:
            self.logger.error(f"Error vectorizing text: {e}")
            return []
    
    def _estimate_wall_thickness(self, wall_mask: np.ndarray, 
                               pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
        """Estimate wall thickness by scanning perpendicular to the line."""
        try:
            # Calculate perpendicular direction
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            length = np.sqrt(dx*dx + dy*dy)
            
            if length == 0:
                return 5.0
            
            # Normalize direction vector
            dx /= length
            dy /= length
            
            # Sample points along the line
            num_samples = 5
            thicknesses = []
            
            for i in range(num_samples):
                t = i / (num_samples - 1)
                x = int(pt1[0] + t * (pt2[0] - pt1[0]))
                y = int(pt1[1] + t * (pt2[1] - pt1[1]))
                
                # Scan perpendicular to find wall thickness
                thickness = self._scan_wall_thickness(wall_mask, (x, y), (-dy, dx))
                thicknesses.append(thickness)
            
            return np.median(thicknesses) if thicknesses else 5.0
            
        except Exception as e:
            self.logger.error(f"Error estimating wall thickness: {e}")
            return 5.0
    
    def _scan_wall_thickness(self, wall_mask: np.ndarray, 
                           center: Tuple[int, int], direction: Tuple[float, float]) -> float:
        """Scan perpendicular to a line to find wall thickness."""
        try:
            x, y = center
            dx, dy = direction
            
            # Scan in both directions
            thickness = 0
            for offset in range(1, 50):
                x1 = int(x + offset * dx)
                y1 = int(y + offset * dy)
                x2 = int(x - offset * dx)
                y2 = int(y - offset * dy)
                
                # Check bounds
                if (0 <= x1 < wall_mask.shape[1] and 0 <= y1 < wall_mask.shape[0] and
                    0 <= x2 < wall_mask.shape[1] and 0 <= y2 < wall_mask.shape[0]):
                    
                    # Check if we hit a wall
                    if (wall_mask[y1, x1] == 255 or wall_mask[y2, x2] == 255):
                        thickness = offset * 2
                        break
            
            return max(thickness, 5.0)
            
        except Exception as e:
            self.logger.error(f"Error scanning wall thickness: {e}")
            return 5.0
    
    def _merge_wall_segments(self, walls: List[VectorLine]) -> List[VectorLine]:
        """Merge collinear wall segments."""
        try:
            if len(walls) < 2:
                return walls
            
            merged = []
            used = set()
            
            for i, wall1 in enumerate(walls):
                if i in used:
                    continue
                
                current_wall = wall1
                used.add(i)
                
                # Look for segments to merge
                for j, wall2 in enumerate(walls[i+1:], i+1):
                    if j in used:
                        continue
                    
                    if self._can_merge_walls(current_wall, wall2):
                        current_wall = self._merge_two_walls(current_wall, wall2)
                        used.add(j)
                
                merged.append(current_wall)
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Error merging wall segments: {e}")
            return walls
    
    def _can_merge_walls(self, wall1: VectorLine, wall2: VectorLine) -> bool:
        """Check if two wall segments can be merged."""
        try:
            # Check if walls are collinear
            # This is a simplified check - in practice, you'd want more sophisticated logic
            return False  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error checking wall merge: {e}")
            return False
    
    def _merge_two_walls(self, wall1: VectorLine, wall2: VectorLine) -> VectorLine:
        """Merge two wall segments into one."""
        try:
            # This is a simplified merge - in practice, you'd want more sophisticated logic
            return wall1  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error merging walls: {e}")
            return wall1
    
    def _calculate_center(self, points: List[VectorPoint]) -> VectorPoint:
        """Calculate center point of a polygon."""
        try:
            if not points:
                return VectorPoint(0, 0)
            
            x_sum = sum(pt.x for pt in points)
            y_sum = sum(pt.y for pt in points)
            
            return VectorPoint(x_sum / len(points), y_sum / len(points))
            
        except Exception as e:
            self.logger.error(f"Error calculating center: {e}")
            return VectorPoint(0, 0)
    
    def export_to_json(self, vector_data: Dict[str, Any], filepath: str) -> None:
        """Export vector data to JSON file."""
        try:
            # Convert dataclasses to dictionaries
            json_data = self._convert_to_json_serializable(vector_data)
            
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            self.logger.info(f"Vector data exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {e}")
            raise
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert dataclasses to JSON-serializable format."""
        if hasattr(obj, '__dict__'):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        else:
            return obj