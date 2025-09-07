"""
Geometry processing module for vectorized floor plan data.
Handles geometric operations, validation, and optimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Represents a 2D bounding box."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float


class GeometryProcessor:
    """
    Processes and optimizes vectorized floor plan geometry.
    """
    
    def __init__(self, tolerance: float = 1.0):
        """
        Initialize the geometry processor.
        
        Args:
            tolerance: Tolerance for geometric operations
        """
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
    
    def process_vector_data(self, vector_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and optimize vector data.
        
        Args:
            vector_data: Raw vector data from vectorizer
            
        Returns:
            Processed vector data
        """
        try:
            processed_data = vector_data.copy()
            
            # Process walls
            if 'walls' in processed_data:
                processed_data['walls'] = self._process_walls(processed_data['walls'])
            
            # Process doors
            if 'doors' in processed_data:
                processed_data['doors'] = self._process_doors(processed_data['doors'])
            
            # Process windows
            if 'windows' in processed_data:
                processed_data['windows'] = self._process_windows(processed_data['windows'])
            
            # Process rooms
            if 'rooms' in processed_data:
                processed_data['rooms'] = self._process_rooms(processed_data['rooms'])
            
            # Process furniture
            if 'furniture' in processed_data:
                processed_data['furniture'] = self._process_furniture(processed_data['furniture'])
            
            # Calculate spatial relationships
            processed_data['spatial_relationships'] = self._calculate_spatial_relationships(processed_data)
            
            # Calculate bounding box
            processed_data['bounding_box'] = self._calculate_bounding_box(processed_data)
            
            self.logger.info("Successfully processed vector data")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing vector data: {e}")
            raise
    
    def _process_walls(self, walls: List) -> List:
        """Process and optimize wall segments."""
        try:
            processed_walls = []
            
            for wall in walls:
                # Snap to grid
                snapped_wall = self._snap_line_to_grid(wall)
                
                # Validate wall
                if self._validate_wall(snapped_wall):
                    processed_walls.append(snapped_wall)
            
            # Merge collinear walls
            merged_walls = self._merge_collinear_walls(processed_walls)
            
            # Remove duplicate walls
            unique_walls = self._remove_duplicate_walls(merged_walls)
            
            self.logger.info(f"Processed {len(unique_walls)} wall segments")
            return unique_walls
            
        except Exception as e:
            self.logger.error(f"Error processing walls: {e}")
            return walls
    
    def _process_doors(self, doors: List) -> List:
        """Process and optimize door segments."""
        try:
            processed_doors = []
            
            for door in doors:
                # Snap to grid
                snapped_door = self._snap_line_to_grid(door)
                
                # Validate door
                if self._validate_door(snapped_door):
                    processed_doors.append(snapped_door)
            
            # Remove duplicate doors
            unique_doors = self._remove_duplicate_doors(processed_doors)
            
            self.logger.info(f"Processed {len(unique_doors)} doors")
            return unique_doors
            
        except Exception as e:
            self.logger.error(f"Error processing doors: {e}")
            return doors
    
    def _process_windows(self, windows: List) -> List:
        """Process and optimize window segments."""
        try:
            processed_windows = []
            
            for window in windows:
                # Snap to grid
                snapped_window = self._snap_line_to_grid(window)
                
                # Validate window
                if self._validate_window(snapped_window):
                    processed_windows.append(snapped_window)
            
            # Remove duplicate windows
            unique_windows = self._remove_duplicate_windows(processed_windows)
            
            self.logger.info(f"Processed {len(unique_windows)} windows")
            return unique_windows
            
        except Exception as e:
            self.logger.error(f"Error processing windows: {e}")
            return windows
    
    def _process_rooms(self, rooms: List) -> List:
        """Process and optimize room polygons."""
        try:
            processed_rooms = []
            
            for room in rooms:
                # Snap points to grid
                snapped_room = self._snap_polygon_to_grid(room)
                
                # Validate room
                if self._validate_room(snapped_room):
                    processed_rooms.append(snapped_room)
            
            # Remove duplicate rooms
            unique_rooms = self._remove_duplicate_rooms(processed_rooms)
            
            self.logger.info(f"Processed {len(unique_rooms)} rooms")
            return unique_rooms
            
        except Exception as e:
            self.logger.error(f"Error processing rooms: {e}")
            return rooms
    
    def _process_furniture(self, furniture: List) -> List:
        """Process and optimize furniture polygons."""
        try:
            processed_furniture = []
            
            for item in furniture:
                # Snap points to grid
                snapped_item = self._snap_polygon_to_grid(item)
                
                # Validate furniture
                if self._validate_furniture(snapped_item):
                    processed_furniture.append(snapped_item)
            
            # Remove duplicate furniture
            unique_furniture = self._remove_duplicate_furniture(processed_furniture)
            
            self.logger.info(f"Processed {len(unique_furniture)} furniture items")
            return unique_furniture
            
        except Exception as e:
            self.logger.error(f"Error processing furniture: {e}")
            return furniture
    
    def _snap_line_to_grid(self, line) -> Any:
        """Snap line endpoints to grid."""
        try:
            # Create a copy of the line
            snapped_line = type(line)(
                start=type(line.start)(
                    x=self._snap_to_grid(line.start.x),
                    y=self._snap_to_grid(line.start.y)
                ),
                end=type(line.end)(
                    x=self._snap_to_grid(line.end.x),
                    y=self._snap_to_grid(line.end.y)
                ),
                thickness=line.thickness,
                element_type=line.element_type
            )
            
            return snapped_line
            
        except Exception as e:
            self.logger.error(f"Error snapping line to grid: {e}")
            return line
    
    def _snap_polygon_to_grid(self, polygon) -> Any:
        """Snap polygon points to grid."""
        try:
            # Create a copy of the polygon
            snapped_points = []
            for point in polygon.points:
                snapped_point = type(point)(
                    x=self._snap_to_grid(point.x),
                    y=self._snap_to_grid(point.y)
                )
                snapped_points.append(snapped_point)
            
            snapped_polygon = type(polygon)(
                points=snapped_points,
                element_type=polygon.element_type,
                properties=polygon.properties.copy()
            )
            
            return snapped_polygon
            
        except Exception as e:
            self.logger.error(f"Error snapping polygon to grid: {e}")
            return polygon
    
    def _snap_to_grid(self, value: float) -> float:
        """Snap a value to the nearest grid point."""
        return round(value / self.tolerance) * self.tolerance
    
    def _validate_wall(self, wall) -> bool:
        """Validate a wall segment."""
        try:
            # Check if wall has minimum length
            length = self._calculate_line_length(wall.start, wall.end)
            if length < self.tolerance:
                return False
            
            # Check if wall has reasonable thickness
            if wall.thickness <= 0 or wall.thickness > 100:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating wall: {e}")
            return False
    
    def _validate_door(self, door) -> bool:
        """Validate a door segment."""
        try:
            # Check if door has minimum width
            width = self._calculate_line_length(door.start, door.end)
            if width < 20:  # Minimum door width
                return False
            
            # Check if door has reasonable thickness
            if door.thickness <= 0 or door.thickness > 50:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating door: {e}")
            return False
    
    def _validate_window(self, window) -> bool:
        """Validate a window segment."""
        try:
            # Check if window has minimum width
            width = self._calculate_line_length(window.start, window.end)
            if width < 10:  # Minimum window width
                return False
            
            # Check if window has reasonable thickness
            if window.thickness <= 0 or window.thickness > 30:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating window: {e}")
            return False
    
    def _validate_room(self, room) -> bool:
        """Validate a room polygon."""
        try:
            # Check if room has minimum area
            if room.properties.get('area', 0) < 100:  # Minimum room area
                return False
            
            # Check if room has minimum number of points
            if len(room.points) < 3:
                return False
            
            # Check if room is not self-intersecting
            if self._is_polygon_self_intersecting(room.points):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating room: {e}")
            return False
    
    def _validate_furniture(self, furniture) -> bool:
        """Validate a furniture polygon."""
        try:
            # Check if furniture has minimum area
            if furniture.properties.get('area', 0) < 10:  # Minimum furniture area
                return False
            
            # Check if furniture has minimum number of points
            if len(furniture.points) < 3:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating furniture: {e}")
            return False
    
    def _calculate_line_length(self, start, end) -> float:
        """Calculate length of a line segment."""
        dx = end.x - start.x
        dy = end.y - start.y
        return math.sqrt(dx*dx + dy*dy)
    
    def _is_polygon_self_intersecting(self, points: List) -> bool:
        """Check if a polygon is self-intersecting."""
        try:
            # This is a simplified check - in practice, you'd want more sophisticated logic
            return False  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error checking polygon intersection: {e}")
            return False
    
    def _merge_collinear_walls(self, walls: List) -> List:
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
                
                # Look for collinear walls to merge
                for j, wall2 in enumerate(walls[i+1:], i+1):
                    if j in used:
                        continue
                    
                    if self._are_walls_collinear(current_wall, wall2):
                        current_wall = self._merge_two_walls(current_wall, wall2)
                        used.add(j)
                
                merged.append(current_wall)
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Error merging collinear walls: {e}")
            return walls
    
    def _are_walls_collinear(self, wall1, wall2) -> bool:
        """Check if two walls are collinear."""
        try:
            # Calculate slopes
            slope1 = self._calculate_slope(wall1.start, wall1.end)
            slope2 = self._calculate_slope(wall2.start, wall2.end)
            
            # Check if slopes are approximately equal
            return abs(slope1 - slope2) < 0.1
            
        except Exception as e:
            self.logger.error(f"Error checking collinearity: {e}")
            return False
    
    def _calculate_slope(self, start, end) -> float:
        """Calculate slope of a line segment."""
        dx = end.x - start.x
        dy = end.y - start.y
        
        if dx == 0:
            return float('inf')
        
        return dy / dx
    
    def _merge_two_walls(self, wall1, wall2) -> Any:
        """Merge two collinear walls into one."""
        try:
            # Find the endpoints that are furthest apart
            points = [wall1.start, wall1.end, wall2.start, wall2.end]
            max_dist = 0
            best_start = None
            best_end = None
            
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    dist = self._calculate_line_length(points[i], points[j])
                    if dist > max_dist:
                        max_dist = dist
                        best_start = points[i]
                        best_end = points[j]
            
            # Create merged wall
            merged_wall = type(wall1)(
                start=best_start,
                end=best_end,
                thickness=max(wall1.thickness, wall2.thickness),
                element_type=wall1.element_type
            )
            
            return merged_wall
            
        except Exception as e:
            self.logger.error(f"Error merging walls: {e}")
            return wall1
    
    def _remove_duplicate_walls(self, walls: List) -> List:
        """Remove duplicate wall segments."""
        try:
            unique_walls = []
            used = set()
            
            for i, wall1 in enumerate(walls):
                if i in used:
                    continue
                
                is_duplicate = False
                for j, wall2 in enumerate(walls[i+1:], i+1):
                    if j in used:
                        continue
                    
                    if self._are_walls_duplicate(wall1, wall2):
                        is_duplicate = True
                        used.add(j)
                        break
                
                if not is_duplicate:
                    unique_walls.append(wall1)
                    used.add(i)
            
            return unique_walls
            
        except Exception as e:
            self.logger.error(f"Error removing duplicate walls: {e}")
            return walls
    
    def _are_walls_duplicate(self, wall1, wall2) -> bool:
        """Check if two walls are duplicates."""
        try:
            # Check if endpoints are approximately equal
            start1_close = (abs(wall1.start.x - wall2.start.x) < self.tolerance and 
                           abs(wall1.start.y - wall2.start.y) < self.tolerance)
            end1_close = (abs(wall1.end.x - wall2.end.x) < self.tolerance and 
                         abs(wall1.end.y - wall2.end.y) < self.tolerance)
            
            start2_close = (abs(wall1.start.x - wall2.end.x) < self.tolerance and 
                           abs(wall1.start.y - wall2.end.y) < self.tolerance)
            end2_close = (abs(wall1.end.x - wall2.start.x) < self.tolerance and 
                         abs(wall1.end.y - wall2.start.y) < self.tolerance)
            
            return (start1_close and end1_close) or (start2_close and end2_close)
            
        except Exception as e:
            self.logger.error(f"Error checking wall duplicates: {e}")
            return False
    
    def _remove_duplicate_doors(self, doors: List) -> List:
        """Remove duplicate door segments."""
        # Similar to _remove_duplicate_walls but for doors
        return self._remove_duplicate_walls(doors)
    
    def _remove_duplicate_windows(self, windows: List) -> List:
        """Remove duplicate window segments."""
        # Similar to _remove_duplicate_walls but for windows
        return self._remove_duplicate_walls(windows)
    
    def _remove_duplicate_rooms(self, rooms: List) -> List:
        """Remove duplicate room polygons."""
        try:
            unique_rooms = []
            used = set()
            
            for i, room1 in enumerate(rooms):
                if i in used:
                    continue
                
                is_duplicate = False
                for j, room2 in enumerate(rooms[i+1:], i+1):
                    if j in used:
                        continue
                    
                    if self._are_rooms_duplicate(room1, room2):
                        is_duplicate = True
                        used.add(j)
                        break
                
                if not is_duplicate:
                    unique_rooms.append(room1)
                    used.add(i)
            
            return unique_rooms
            
        except Exception as e:
            self.logger.error(f"Error removing duplicate rooms: {e}")
            return rooms
    
    def _are_rooms_duplicate(self, room1, room2) -> bool:
        """Check if two rooms are duplicates."""
        try:
            # Check if areas are approximately equal
            area1 = room1.properties.get('area', 0)
            area2 = room2.properties.get('area', 0)
            
            if abs(area1 - area2) > self.tolerance * 100:
                return False
            
            # Check if centers are close
            center1 = room1.properties.get('center')
            center2 = room2.properties.get('center')
            
            if center1 and center2:
                dist = self._calculate_line_length(center1, center2)
                return dist < self.tolerance * 10
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking room duplicates: {e}")
            return False
    
    def _remove_duplicate_furniture(self, furniture: List) -> List:
        """Remove duplicate furniture polygons."""
        # Similar to _remove_duplicate_rooms but for furniture
        return self._remove_duplicate_rooms(furniture)
    
    def _calculate_spatial_relationships(self, vector_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate spatial relationships between elements."""
        try:
            relationships = {
                'wall_connections': self._find_wall_connections(vector_data.get('walls', [])),
                'room_adjacencies': self._find_room_adjacencies(vector_data.get('rooms', [])),
                'door_wall_connections': self._find_door_wall_connections(
                    vector_data.get('doors', []), 
                    vector_data.get('walls', [])
                ),
                'window_wall_connections': self._find_window_wall_connections(
                    vector_data.get('windows', []), 
                    vector_data.get('walls', [])
                )
            }
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error calculating spatial relationships: {e}")
            return {}
    
    def _find_wall_connections(self, walls: List) -> List[Dict[str, Any]]:
        """Find connections between wall segments."""
        try:
            connections = []
            
            for i, wall1 in enumerate(walls):
                for j, wall2 in enumerate(walls[i+1:], i+1):
                    if self._are_walls_connected(wall1, wall2):
                        connections.append({
                            'wall1_index': i,
                            'wall2_index': j,
                            'connection_type': 'endpoint'
                        })
            
            return connections
            
        except Exception as e:
            self.logger.error(f"Error finding wall connections: {e}")
            return []
    
    def _are_walls_connected(self, wall1, wall2) -> bool:
        """Check if two walls are connected at their endpoints."""
        try:
            # Check all combinations of endpoints
            endpoints1 = [wall1.start, wall1.end]
            endpoints2 = [wall2.start, wall2.end]
            
            for ep1 in endpoints1:
                for ep2 in endpoints2:
                    dist = self._calculate_line_length(ep1, ep2)
                    if dist < self.tolerance:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking wall connection: {e}")
            return False
    
    def _find_room_adjacencies(self, rooms: List) -> List[Dict[str, Any]]:
        """Find adjacent rooms."""
        try:
            adjacencies = []
            
            for i, room1 in enumerate(rooms):
                for j, room2 in enumerate(rooms[i+1:], i+1):
                    if self._are_rooms_adjacent(room1, room2):
                        adjacencies.append({
                            'room1_index': i,
                            'room2_index': j,
                            'adjacency_type': 'shared_wall'
                        })
            
            return adjacencies
            
        except Exception as e:
            self.logger.error(f"Error finding room adjacencies: {e}")
            return []
    
    def _are_rooms_adjacent(self, room1, room2) -> bool:
        """Check if two rooms are adjacent."""
        try:
            # This is a simplified check - in practice, you'd want more sophisticated logic
            # to check if rooms share a wall or are very close
            return False  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error checking room adjacency: {e}")
            return False
    
    def _find_door_wall_connections(self, doors: List, walls: List) -> List[Dict[str, Any]]:
        """Find which walls doors are connected to."""
        try:
            connections = []
            
            for i, door in enumerate(doors):
                for j, wall in enumerate(walls):
                    if self._is_door_on_wall(door, wall):
                        connections.append({
                            'door_index': i,
                            'wall_index': j,
                            'connection_type': 'door_opening'
                        })
            
            return connections
            
        except Exception as e:
            self.logger.error(f"Error finding door-wall connections: {e}")
            return []
    
    def _is_door_on_wall(self, door, wall) -> bool:
        """Check if a door is on a wall."""
        try:
            # This is a simplified check - in practice, you'd want more sophisticated logic
            # to check if the door line is collinear with the wall line
            return False  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error checking door-wall connection: {e}")
            return False
    
    def _find_window_wall_connections(self, windows: List, walls: List) -> List[Dict[str, Any]]:
        """Find which walls windows are connected to."""
        try:
            connections = []
            
            for i, window in enumerate(windows):
                for j, wall in enumerate(walls):
                    if self._is_window_on_wall(window, wall):
                        connections.append({
                            'window_index': i,
                            'wall_index': j,
                            'connection_type': 'window_opening'
                        })
            
            return connections
            
        except Exception as e:
            self.logger.error(f"Error finding window-wall connections: {e}")
            return []
    
    def _is_window_on_wall(self, window, wall) -> bool:
        """Check if a window is on a wall."""
        try:
            # This is a simplified check - in practice, you'd want more sophisticated logic
            # to check if the window line is collinear with the wall line
            return False  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Error checking window-wall connection: {e}")
            return False
    
    def _calculate_bounding_box(self, vector_data: Dict[str, Any]) -> BoundingBox:
        """Calculate bounding box for all elements."""
        try:
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
            
            # Check all elements
            for element_type in ['walls', 'doors', 'windows', 'rooms', 'furniture']:
                if element_type in vector_data:
                    for element in vector_data[element_type]:
                        bbox = self._get_element_bounding_box(element)
                        min_x = min(min_x, bbox.min_x)
                        min_y = min(min_y, bbox.min_y)
                        max_x = max(max_x, bbox.max_x)
                        max_y = max(max_y, bbox.max_y)
            
            return BoundingBox(min_x, min_y, max_x, max_y)
            
        except Exception as e:
            self.logger.error(f"Error calculating bounding box: {e}")
            return BoundingBox(0, 0, 0, 0)
    
    def _get_element_bounding_box(self, element) -> BoundingBox:
        """Get bounding box for a single element."""
        try:
            if hasattr(element, 'start') and hasattr(element, 'end'):
                # Line element
                min_x = min(element.start.x, element.end.x)
                min_y = min(element.start.y, element.end.y)
                max_x = max(element.start.x, element.end.x)
                max_y = max(element.start.y, element.end.y)
            elif hasattr(element, 'points'):
                # Polygon element
                if not element.points:
                    return BoundingBox(0, 0, 0, 0)
                
                min_x = min(pt.x for pt in element.points)
                min_y = min(pt.y for pt in element.points)
                max_x = max(pt.x for pt in element.points)
                max_y = max(pt.y for pt in element.points)
            else:
                return BoundingBox(0, 0, 0, 0)
            
            return BoundingBox(min_x, min_y, max_x, max_y)
            
        except Exception as e:
            self.logger.error(f"Error getting element bounding box: {e}")
            return BoundingBox(0, 0, 0, 0)