"""
Feature detection module for identifying architectural elements in floor plans.

This module detects walls, doors, windows, and other architectural features
using computer vision techniques and deep learning approaches.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Wall:
    """Represents a wall segment in the floor plan."""
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    thickness: float
    confidence: float


@dataclass
class Door:
    """Represents a door in the floor plan."""
    position: Tuple[int, int]
    width: float
    angle: float
    confidence: float
    wall_connection: Optional[Tuple[int, int]] = None


@dataclass
class Window:
    """Represents a window in the floor plan."""
    position: Tuple[int, int]
    width: float
    height: float
    angle: float
    confidence: float
    wall_connection: Optional[Tuple[int, int]] = None


@dataclass
class Room:
    """Represents a room in the floor plan."""
    contour: np.ndarray
    area: float
    room_type: str
    confidence: float
    center: Tuple[int, int]


class FeatureDetector:
    """
    Detects architectural features in preprocessed floor plan images.
    
    This class uses a combination of traditional computer vision techniques
    and deep learning to identify walls, doors, windows, and room boundaries.
    """
    
    def __init__(self):
        """Initialize the FeatureDetector with default parameters."""
        self.wall_thickness_range = (2, 20)  # pixels
        self.door_width_range = (20, 100)    # pixels
        self.window_width_range = (30, 150)  # pixels
        self.min_room_area = 1000  # pixels
        
    def detect_walls(self, image: np.ndarray) -> List[Wall]:
        """
        Detect wall segments in the floor plan.
        
        Args:
            image: Preprocessed binary image
            
        Returns:
            List of detected wall segments
        """
        walls = []
        
        # Use Hough line transform to detect lines
        lines = cv2.HoughLinesP(
            image, 1, np.pi/180, threshold=50,
            minLineLength=30, maxLineGap=10
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line properties
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.arctan2(y2 - y1, x2 - x1)
                
                # Estimate wall thickness by analyzing perpendicular lines
                thickness = self._estimate_wall_thickness(image, (x1, y1), (x2, y2))
                
                # Calculate confidence based on line length and thickness
                confidence = min(1.0, length / 100.0) * (0.8 if self.wall_thickness_range[0] <= thickness <= self.wall_thickness_range[1] else 0.3)
                
                if confidence > 0.3:  # Filter out low-confidence detections
                    wall = Wall(
                        start_point=(x1, y1),
                        end_point=(x2, y2),
                        thickness=thickness,
                        confidence=confidence
                    )
                    walls.append(wall)
        
        return walls
    
    def _estimate_wall_thickness(self, image: np.ndarray, 
                               start: Tuple[int, int], 
                               end: Tuple[int, int]) -> float:
        """
        Estimate wall thickness by analyzing perpendicular lines.
        
        Args:
            image: Binary image
            start: Start point of the wall line
            end: End point of the wall line
            
        Returns:
            Estimated wall thickness in pixels
        """
        x1, y1 = start
        x2, y2 = end
        
        # Calculate perpendicular direction
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return 5.0  # Default thickness
        
        # Normalize direction vector
        perp_x = -dy / length
        perp_y = dx / length
        
        # Sample points along the line and measure thickness
        thicknesses = []
        num_samples = min(10, int(length // 10))
        
        for i in range(num_samples):
            t = i / max(1, num_samples - 1)
            px = int(x1 + t * dx)
            py = int(y1 + t * dy)
            
            # Check both sides of the line
            for side in [-1, 1]:
                thickness = 0
                for offset in range(1, 20):  # Check up to 20 pixels
                    check_x = int(px + side * offset * perp_x)
                    check_y = int(py + side * offset * perp_y)
                    
                    if (0 <= check_x < image.shape[1] and 
                        0 <= check_y < image.shape[0] and 
                        image[check_y, check_x] > 0):
                        thickness = offset
                    else:
                        break
                
                if thickness > 0:
                    thicknesses.append(thickness)
        
        return np.median(thicknesses) if thicknesses else 5.0
    
    def detect_doors(self, image: np.ndarray, walls: List[Wall]) -> List[Door]:
        """
        Detect doors in the floor plan.
        
        Args:
            image: Preprocessed binary image
            walls: List of detected walls
            
        Returns:
            List of detected doors
        """
        doors = []
        
        # Find contours that might be doors
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area < 50 or area > 2000:  # Filter by area
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it looks like a door (rectangular, small area)
            if len(approx) >= 4:
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                
                # Door should be longer than it is wide
                if width > height and self.door_width_range[0] <= width <= self.door_width_range[1]:
                    center = rect[0]
                    angle = rect[2]
                    
                    # Check if door is connected to a wall
                    wall_connection = self._find_nearest_wall(center, walls)
                    
                    confidence = min(1.0, area / 200.0) * (0.9 if wall_connection else 0.5)
                    
                    door = Door(
                        position=(int(center[0]), int(center[1])),
                        width=width,
                        angle=angle,
                        confidence=confidence,
                        wall_connection=wall_connection
                    )
                    doors.append(door)
        
        return doors
    
    def detect_windows(self, image: np.ndarray, walls: List[Wall]) -> List[Window]:
        """
        Detect windows in the floor plan.
        
        Args:
            image: Preprocessed binary image
            walls: List of detected walls
            
        Returns:
            List of detected windows
        """
        windows = []
        
        # Find contours that might be windows
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 100 or area > 5000:  # Filter by area
                continue
            
            # Approximate contour to polygon
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 4:
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                
                # Window should be rectangular and within size range
                if (self.window_width_range[0] <= width <= self.window_width_range[1] and
                    self.window_width_range[0] <= height <= self.window_width_range[1]):
                    
                    center = rect[0]
                    angle = rect[2]
                    
                    # Check if window is connected to a wall
                    wall_connection = self._find_nearest_wall(center, walls)
                    
                    confidence = min(1.0, area / 500.0) * (0.9 if wall_connection else 0.5)
                    
                    window = Window(
                        position=(int(center[0]), int(center[1])),
                        width=width,
                        height=height,
                        angle=angle,
                        confidence=confidence,
                        wall_connection=wall_connection
                    )
                    windows.append(window)
        
        return windows
    
    def detect_rooms(self, image: np.ndarray) -> List[Room]:
        """
        Detect room boundaries in the floor plan.
        
        Args:
            image: Preprocessed binary image
            
        Returns:
            List of detected rooms
        """
        rooms = []
        
        # Find contours that represent rooms
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_room_area:
                continue
            
            # Calculate room properties
            perimeter = cv2.arcLength(contour, True)
            moments = cv2.moments(contour)
            
            if moments['m00'] != 0:
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
                center = (center_x, center_y)
            else:
                center = (0, 0)
            
            # Classify room type based on area and shape
            room_type = self._classify_room_type(area, contour)
            
            # Calculate confidence based on area and shape regularity
            aspect_ratio = self._calculate_aspect_ratio(contour)
            confidence = min(1.0, area / 10000.0) * (0.8 if 0.3 <= aspect_ratio <= 3.0 else 0.5)
            
            room = Room(
                contour=contour,
                area=area,
                room_type=room_type,
                confidence=confidence,
                center=center
            )
            rooms.append(room)
        
        return rooms
    
    def _find_nearest_wall(self, point: Tuple[float, float], 
                          walls: List[Wall]) -> Optional[Tuple[int, int]]:
        """
        Find the nearest wall to a given point.
        
        Args:
            point: Point to check
            walls: List of walls to search
            
        Returns:
            Nearest wall's midpoint or None
        """
        if not walls:
            return None
        
        px, py = point
        min_distance = float('inf')
        nearest_wall = None
        
        for wall in walls:
            x1, y1 = wall.start_point
            x2, y2 = wall.end_point
            
            # Calculate distance from point to line
            distance = self._point_to_line_distance(px, py, x1, y1, x2, y2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_wall = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        return nearest_wall if min_distance < 50 else None
    
    def _point_to_line_distance(self, px: float, py: float, 
                               x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate distance from point to line segment."""
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            return np.sqrt(A * A + B * B)
        
        param = dot / len_sq
        
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D
        
        dx = px - xx
        dy = py - yy
        return np.sqrt(dx * dx + dy * dy)
    
    def _classify_room_type(self, area: float, contour: np.ndarray) -> str:
        """
        Classify room type based on area and shape.
        
        Args:
            area: Room area in pixels
            contour: Room contour
            
        Returns:
            Room type classification
        """
        # Simple classification based on area
        if area < 5000:
            return "bathroom"
        elif area < 15000:
            return "bedroom"
        elif area < 25000:
            return "living_room"
        else:
            return "kitchen"
    
    def _calculate_aspect_ratio(self, contour: np.ndarray) -> float:
        """Calculate aspect ratio of a contour."""
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        return max(width, height) / max(min(width, height), 1)
    
    def visualize_detections(self, image: np.ndarray, 
                           walls: List[Wall] = None,
                           doors: List[Door] = None,
                           windows: List[Window] = None,
                           rooms: List[Room] = None,
                           save_path: Optional[str] = None):
        """
        Visualize detected features on the image.
        
        Args:
            image: Original image
            walls: List of detected walls
            doors: List of detected doors
            windows: List of detected windows
            rooms: List of detected rooms
            save_path: Optional path to save the visualization
        """
        # Create a copy of the image for visualization
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Draw walls
        if walls:
            for wall in walls:
                cv2.line(vis_image, wall.start_point, wall.end_point, (0, 255, 0), 2)
        
        # Draw doors
        if doors:
            for door in doors:
                cv2.circle(vis_image, door.position, 5, (255, 0, 0), -1)
        
        # Draw windows
        if windows:
            for window in windows:
                cv2.rectangle(vis_image, 
                             (int(window.position[0] - window.width/2), 
                              int(window.position[1] - window.height/2)),
                             (int(window.position[0] + window.width/2), 
                              int(window.position[1] + window.height/2)),
                             (0, 0, 255), 2)
        
        # Draw room boundaries
        if rooms:
            for room in rooms:
                cv2.drawContours(vis_image, [room.contour], -1, (255, 255, 0), 2)
                cv2.putText(vis_image, room.room_type, room.center, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title('Detected Features')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()