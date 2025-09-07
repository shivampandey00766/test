"""
Feature detection module for identifying architectural elements in floor plans.
Uses computer vision techniques to detect walls, doors, windows, and other features.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WallSegment:
    """Represents a wall segment with start and end points."""
    start: Tuple[int, int]
    end: Tuple[int, int]
    thickness: float
    confidence: float


@dataclass
class Door:
    """Represents a door with position and orientation."""
    center: Tuple[int, int]
    width: float
    angle: float
    confidence: float


@dataclass
class Window:
    """Represents a window with position and dimensions."""
    center: Tuple[int, int]
    width: float
    height: float
    angle: float
    confidence: float


@dataclass
class Room:
    """Represents a room with boundary and type."""
    contour: np.ndarray
    area: float
    room_type: str
    center: Tuple[int, int]
    confidence: float


class FeatureDetector:
    """
    Detects architectural features in preprocessed floor plan images.
    """
    
    def __init__(self, min_wall_length: int = 50, min_door_width: int = 20):
        """
        Initialize the feature detector.
        
        Args:
            min_wall_length: Minimum length for wall segments
            min_door_width: Minimum width for door detection
        """
        self.min_wall_length = min_wall_length
        self.min_door_width = min_door_width
        self.logger = logging.getLogger(__name__)
    
    def detect_features(self, binary_image: np.ndarray) -> Dict[str, List]:
        """
        Detect all architectural features in the binary image.
        
        Args:
            binary_image: Preprocessed binary image
            
        Returns:
            Dictionary containing detected features
        """
        try:
            # Detect walls
            walls = self._detect_walls(binary_image)
            
            # Detect doors
            doors = self._detect_doors(binary_image)
            
            # Detect windows
            windows = self._detect_windows(binary_image)
            
            # Detect rooms
            rooms = self._detect_rooms(binary_image)
            
            # Detect text labels
            text_regions = self._detect_text_regions(binary_image)
            
            return {
                'walls': walls,
                'doors': doors,
                'windows': windows,
                'rooms': rooms,
                'text_regions': text_regions
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting features: {e}")
            raise
    
    def _detect_walls(self, binary_image: np.ndarray) -> List[WallSegment]:
        """Detect wall segments using line detection."""
        try:
            # Use Hough line transform to detect lines
            lines = cv2.HoughLinesP(
                binary_image, 
                rho=1, 
                theta=np.pi/180, 
                threshold=100,
                minLineLength=self.min_wall_length,
                maxLineGap=10
            )
            
            walls = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate line properties
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = np.arctan2(y2-y1, x2-x1)
                    
                    # Estimate wall thickness (simplified)
                    thickness = self._estimate_wall_thickness(binary_image, (x1, y1), (x2, y2))
                    
                    # Calculate confidence based on line length and thickness
                    confidence = min(1.0, length / 200.0)
                    
                    wall = WallSegment(
                        start=(x1, y1),
                        end=(x2, y2),
                        thickness=thickness,
                        confidence=confidence
                    )
                    walls.append(wall)
            
            self.logger.info(f"Detected {len(walls)} wall segments")
            return walls
            
        except Exception as e:
            self.logger.error(f"Error detecting walls: {e}")
            return []
    
    def _detect_doors(self, binary_image: np.ndarray) -> List[Door]:
        """Detect doors using template matching and contour analysis."""
        try:
            doors = []
            
            # Find contours
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Skip very small or very large contours
                if area < 50 or area > 5000:
                    continue
                
                # Approximate contour to polygon
                epsilon = 0.02 * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it could be a door (rectangular shape)
                if len(approx) == 4:
                    # Calculate bounding rectangle
                    rect = cv2.minAreaRect(contour)
                    center, (width, height), angle = rect
                    
                    # Check aspect ratio (doors are typically wider than tall)
                    aspect_ratio = width / height if height > 0 else 0
                    
                    if 1.5 < aspect_ratio < 5.0 and width > self.min_door_width:
                        confidence = min(1.0, area / 1000.0)
                        
                        door = Door(
                            center=(int(center[0]), int(center[1])),
                            width=width,
                            angle=angle,
                            confidence=confidence
                        )
                        doors.append(door)
            
            self.logger.info(f"Detected {len(doors)} doors")
            return doors
            
        except Exception as e:
            self.logger.error(f"Error detecting doors: {e}")
            return []
    
    def _detect_windows(self, binary_image: np.ndarray) -> List[Window]:
        """Detect windows using contour analysis."""
        try:
            windows = []
            
            # Find contours
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Skip very small contours
                if area < 100:
                    continue
                
                # Calculate bounding rectangle
                rect = cv2.minAreaRect(contour)
                center, (width, height), angle = rect
                
                # Check if it could be a window (rectangular, moderate size)
                aspect_ratio = width / height if height > 0 else 0
                
                if 0.5 < aspect_ratio < 2.0 and 30 < width < 200 and 20 < height < 150:
                    confidence = min(1.0, area / 2000.0)
                    
                    window = Window(
                        center=(int(center[0]), int(center[1])),
                        width=width,
                        height=height,
                        angle=angle,
                        confidence=confidence
                    )
                    windows.append(window)
            
            self.logger.info(f"Detected {len(windows)} windows")
            return windows
            
        except Exception as e:
            self.logger.error(f"Error detecting windows: {e}")
            return []
    
    def _detect_rooms(self, binary_image: np.ndarray) -> List[Room]:
        """Detect room boundaries using contour analysis."""
        try:
            rooms = []
            
            # Find contours
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Skip very small contours
                if area < 1000:
                    continue
                
                # Calculate contour properties
                perimeter = cv2.arcLength(contour, True)
                
                # Calculate center point
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    continue
                
                # Determine room type based on area and shape
                room_type = self._classify_room(area, perimeter)
                
                # Calculate confidence based on area and shape regularity
                confidence = min(1.0, area / 10000.0)
                
                room = Room(
                    contour=contour,
                    area=area,
                    room_type=room_type,
                    center=(cx, cy),
                    confidence=confidence
                )
                rooms.append(room)
            
            self.logger.info(f"Detected {len(rooms)} rooms")
            return rooms
            
        except Exception as e:
            self.logger.error(f"Error detecting rooms: {e}")
            return []
    
    def _detect_text_regions(self, binary_image: np.ndarray) -> List[Dict]:
        """Detect text regions using MSER (Maximally Stable Extremal Regions)."""
        try:
            # Create MSER detector
            mser = cv2.MSER_create()
            
            # Detect regions
            regions, _ = mser.detectRegions(binary_image)
            
            text_regions = []
            for region in regions:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(region)
                
                # Filter by size (text regions are typically small)
                if 10 < w < 200 and 10 < h < 100:
                    text_regions.append({
                        'bbox': (x, y, w, h),
                        'area': w * h,
                        'confidence': 0.7  # Placeholder confidence
                    })
            
            self.logger.info(f"Detected {len(text_regions)} text regions")
            return text_regions
            
        except Exception as e:
            self.logger.error(f"Error detecting text regions: {e}")
            return []
    
    def _estimate_wall_thickness(self, binary_image: np.ndarray, 
                               start: Tuple[int, int], end: Tuple[int, int]) -> float:
        """Estimate wall thickness by analyzing perpendicular lines."""
        try:
            # Calculate perpendicular direction
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = np.sqrt(dx*dx + dy*dy)
            
            if length == 0:
                return 5.0  # Default thickness
            
            # Normalize direction vector
            dx /= length
            dy /= length
            
            # Sample points along the line
            num_samples = 10
            thicknesses = []
            
            for i in range(num_samples):
                t = i / (num_samples - 1)
                x = int(start[0] + t * (end[0] - start[0]))
                y = int(start[1] + t * (end[1] - start[1]))
                
                # Sample perpendicular to the line
                perp_x = int(x - dy * 20)
                perp_y = int(y + dx * 20)
                
                # Find wall thickness by scanning perpendicular
                thickness = self._scan_wall_thickness(binary_image, (x, y), (-dy, dx))
                thicknesses.append(thickness)
            
            return np.median(thicknesses) if thicknesses else 5.0
            
        except Exception as e:
            self.logger.error(f"Error estimating wall thickness: {e}")
            return 5.0
    
    def _scan_wall_thickness(self, binary_image: np.ndarray, 
                           center: Tuple[int, int], direction: Tuple[float, float]) -> float:
        """Scan perpendicular to a line to find wall thickness."""
        try:
            x, y = center
            dx, dy = direction
            
            # Scan in both directions
            thickness = 0
            for offset in range(1, 50):  # Max scan distance
                x1 = int(x + offset * dx)
                y1 = int(y + offset * dy)
                x2 = int(x - offset * dx)
                y2 = int(y - offset * dy)
                
                # Check bounds
                if (0 <= x1 < binary_image.shape[1] and 0 <= y1 < binary_image.shape[0] and
                    0 <= x2 < binary_image.shape[1] and 0 <= y2 < binary_image.shape[0]):
                    
                    # Check if we hit a wall (white pixel in binary image)
                    if (binary_image[y1, x1] == 255 or binary_image[y2, x2] == 255):
                        thickness = offset * 2
                        break
            
            return max(thickness, 5.0)  # Minimum thickness
            
        except Exception as e:
            self.logger.error(f"Error scanning wall thickness: {e}")
            return 5.0
    
    def _classify_room(self, area: float, perimeter: float) -> str:
        """Classify room type based on area and shape."""
        try:
            # Calculate shape factor (circularity)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # Classify based on area and shape
            if area > 50000:  # Large rooms
                if circularity > 0.7:
                    return "living_room"
                else:
                    return "kitchen"
            elif area > 20000:  # Medium rooms
                if circularity > 0.6:
                    return "bedroom"
                else:
                    return "bathroom"
            else:  # Small rooms
                return "closet"
                
        except Exception as e:
            self.logger.error(f"Error classifying room: {e}")
            return "unknown"
    
    def visualize_features(self, image: np.ndarray, features: Dict[str, List]) -> np.ndarray:
        """Create a visualization of detected features."""
        try:
            # Create a copy of the image
            vis_image = image.copy()
            if len(vis_image.shape) == 2:
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
            
            # Draw walls in red
            for wall in features['walls']:
                cv2.line(vis_image, wall.start, wall.end, (0, 0, 255), 2)
            
            # Draw doors in green
            for door in features['doors']:
                cv2.circle(vis_image, door.center, int(door.width/4), (0, 255, 0), 2)
            
            # Draw windows in blue
            for window in features['windows']:
                cv2.rectangle(vis_image, 
                            (int(window.center[0] - window.width/2), 
                             int(window.center[1] - window.height/2)),
                            (int(window.center[0] + window.width/2), 
                             int(window.center[1] + window.height/2)),
                            (255, 0, 0), 2)
            
            # Draw room centers in yellow
            for room in features['rooms']:
                cv2.circle(vis_image, room.center, 10, (0, 255, 255), -1)
                cv2.putText(vis_image, room.room_type, 
                          (room.center[0] + 15, room.center[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            return vis_image
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            return image