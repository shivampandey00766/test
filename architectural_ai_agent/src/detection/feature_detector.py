"""
Feature Detection Module for Architectural Elements

This module identifies and classifies architectural features in floor plans:
- Walls and structural elements
- Doors and openings
- Windows
- Fixtures and furniture
- Text labels and annotations
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.cluster import DBSCAN
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points

logger = logging.getLogger(__name__)


class FeatureDetector:
    """
    Detects and classifies architectural features in floor plan images.
    """
    
    def __init__(self):
        """Initialize the feature detector with default parameters."""
        self.wall_thickness_range = (3, 15)  # pixels
        self.door_width_range = (15, 40)     # pixels
        self.window_width_range = (20, 60)   # pixels
        self.min_wall_length = 30            # pixels
        
    def detect_features(self, image: np.ndarray, vector_data: Dict) -> Dict:
        """
        Detect all architectural features in the image.
        
        Args:
            image: Preprocessed floor plan image
            vector_data: Vectorized data from Vectorizer
            
        Returns:
            Dictionary containing detected features
        """
        features = {
            'walls': self._classify_walls(vector_data['walls']),
            'doors': self._detect_doors(image, vector_data),
            'windows': self._detect_windows(image, vector_data),
            'rooms': self._detect_rooms(image, vector_data),
            'fixtures': self._detect_fixtures(image, vector_data),
            'text_labels': self._detect_text(image)
        }
        
        # Post-process to resolve conflicts and improve accuracy
        features = self._post_process_features(features)
        
        logger.info(f"Detected features: {len(features['walls'])} walls, "
                   f"{len(features['doors'])} doors, {len(features['windows'])} windows, "
                   f"{len(features['rooms'])} rooms, {len(features['fixtures'])} fixtures")
        
        return features
    
    def _classify_walls(self, wall_contours: List[Dict]) -> List[Dict]:
        """
        Classify and enhance wall detection from vectorized data.
        """
        walls = []
        
        for wall_contour in wall_contours:
            # Calculate wall properties
            points = wall_contour['points']
            if len(points) < 2:
                continue
                
            # For each wall segment
            wall_segments = self._extract_wall_segments(points)
            
            for segment in wall_segments:
                start, end = segment['start'], segment['end']
                length = segment['length']
                thickness = segment['thickness']
                
                if length > self.min_wall_length:
                    wall_info = {
                        'start': start,
                        'end': end,
                        'length': length,
                        'thickness': thickness,
                        'angle': segment['angle'],
                        'type': 'wall',
                        'material': self._infer_wall_material(thickness),
                        'structural': thickness > 8  # Thicker walls are structural
                    }
                    walls.append(wall_info)
        
        return walls
    
    def _extract_wall_segments(self, points: List[Tuple[int, int]]) -> List[Dict]:
        """
        Extract individual wall segments from a wall contour.
        """
        segments = []
        
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            
            # Calculate segment properties
            length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            angle = np.arctan2(end[1] - start[1], end[0] - start[0]) * 180 / np.pi
            
            # Estimate thickness (this is simplified - in practice would need more sophisticated analysis)
            thickness = self._estimate_wall_thickness(start, end)
            
            segment = {
                'start': start,
                'end': end,
                'length': length,
                'angle': angle,
                'thickness': thickness
            }
            segments.append(segment)
        
        return segments
    
    def _estimate_wall_thickness(self, start: Tuple[int, int], end: Tuple[int, int]) -> float:
        """
        Estimate wall thickness based on local image analysis.
        This is a simplified version - real implementation would be more sophisticated.
        """
        # Default thickness based on typical architectural standards
        return 6.0  # pixels
    
    def _infer_wall_material(self, thickness: float) -> str:
        """
        Infer wall material type based on thickness.
        """
        if thickness > 12:
            return 'concrete'
        elif thickness > 8:
            return 'brick'
        elif thickness > 4:
            return 'drywall'
        else:
            return 'partition'
    
    def _detect_doors(self, image: np.ndarray, vector_data: Dict) -> List[Dict]:
        """
        Detect doors in the floor plan.
        """
        doors = []
        
        # Look for door symbols - typically represented as arcs or specific line patterns
        doors.extend(self._detect_swing_doors(image, vector_data))
        doors.extend(self._detect_sliding_doors(image, vector_data))
        doors.extend(self._detect_double_doors(image, vector_data))
        
        return doors
    
    def _detect_swing_doors(self, image: np.ndarray, vector_data: Dict) -> List[Dict]:
        """
        Detect swing doors (most common type).
        """
        doors = []
        
        # Look for arc patterns that indicate door swings
        circles = cv2.HoughCircles(
            image, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=15, maxRadius=50
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Check if this circle intersects with walls (indicating a door)
                door_candidate = self._validate_door_location(x, y, r, vector_data['walls'])
                
                if door_candidate:
                    door_info = {
                        'center': (x, y),
                        'radius': r,
                        'width': r * 2,
                        'type': 'swing_door',
                        'opening_angle': 90,  # Default swing angle
                        'wall_intersection': door_candidate['wall']
                    }
                    doors.append(door_info)
        
        return doors
    
    def _detect_sliding_doors(self, image: np.ndarray, vector_data: Dict) -> List[Dict]:
        """
        Detect sliding doors.
        """
        doors = []
        
        # Look for parallel lines that might indicate sliding door tracks
        thin_lines = vector_data.get('thin_lines', [])
        
        # Group parallel lines
        parallel_groups = self._find_parallel_lines(thin_lines)
        
        for group in parallel_groups:
            if len(group) >= 2:
                # Check if these could be sliding door tracks
                door_candidate = self._validate_sliding_door(group, vector_data['walls'])
                
                if door_candidate:
                    doors.append(door_candidate)
        
        return doors
    
    def _detect_double_doors(self, image: np.ndarray, vector_data: Dict) -> List[Dict]:
        """
        Detect double doors.
        """
        doors = []
        
        # Look for symmetric arc patterns
        # This is a simplified implementation
        swing_doors = self._detect_swing_doors(image, vector_data)
        
        # Group nearby swing doors that might be double doors
        for i, door1 in enumerate(swing_doors):
            for j, door2 in enumerate(swing_doors[i+1:], i+1):
                distance = np.sqrt(
                    (door1['center'][0] - door2['center'][0])**2 + 
                    (door1['center'][1] - door2['center'][1])**2
                )
                
                # If doors are close and similar size, they might be double doors
                if distance < door1['width'] + door2['width'] and \
                   abs(door1['width'] - door2['width']) < 5:
                    
                    double_door = {
                        'centers': [door1['center'], door2['center']],
                        'total_width': door1['width'] + door2['width'],
                        'type': 'double_door',
                        'individual_doors': [door1, door2]
                    }
                    doors.append(double_door)
        
        return doors
    
    def _detect_windows(self, image: np.ndarray, vector_data: Dict) -> List[Dict]:
        """
        Detect windows in the floor plan.
        """
        windows = []
        
        # Windows are typically represented as double lines or specific patterns
        windows.extend(self._detect_standard_windows(image, vector_data))
        windows.extend(self._detect_bay_windows(image, vector_data))
        
        return windows
    
    def _detect_standard_windows(self, image: np.ndarray, vector_data: Dict) -> List[Dict]:
        """
        Detect standard rectangular windows.
        """
        windows = []
        
        # Look for parallel line pairs that could be windows
        thin_lines = vector_data.get('thin_lines', [])
        parallel_groups = self._find_parallel_lines(thin_lines, max_distance=10)
        
        for group in parallel_groups:
            if len(group) == 2:  # Window typically has two parallel lines
                line1, line2 = group
                
                # Check if these lines are on a wall
                window_candidate = self._validate_window_location(line1, line2, vector_data['walls'])
                
                if window_candidate:
                    # Calculate window properties
                    width = self._calculate_line_distance(line1, line2)
                    length = min(line1['length'], line2['length'])
                    
                    if self.window_width_range[0] <= width <= self.window_width_range[1]:
                        window_info = {
                            'lines': [line1, line2],
                            'width': width,
                            'length': length,
                            'type': 'standard_window',
                            'wall_intersection': window_candidate['wall']
                        }
                        windows.append(window_info)
        
        return windows
    
    def _detect_bay_windows(self, image: np.ndarray, vector_data: Dict) -> List[Dict]:
        """
        Detect bay windows (protruding windows).
        """
        windows = []
        
        # Look for rectangular protrusions from walls
        rectangles = vector_data.get('rectangles', [])
        walls = vector_data.get('walls', [])
        
        for rect in rectangles:
            # Check if rectangle is adjacent to a wall and has window-like proportions
            if self._is_bay_window_candidate(rect, walls):
                window_info = {
                    'rectangle': rect,
                    'type': 'bay_window',
                    'projection': self._calculate_bay_projection(rect, walls)
                }
                windows.append(window_info)
        
        return windows
    
    def _detect_rooms(self, image: np.ndarray, vector_data: Dict) -> List[Dict]:
        """
        Detect and classify rooms.
        """
        rooms = []
        
        # Use flood fill to identify enclosed areas
        room_masks = self._find_enclosed_areas(image)
        
        for i, mask in enumerate(room_masks):
            # Calculate room properties
            area = np.sum(mask > 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours and area > 1000:  # Minimum room size
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get room boundaries
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Classify room type based on size, shape, and location
                room_type = self._classify_room_type(area, w/h if h > 0 else 1, (x, y))
                
                room_info = {
                    'id': i,
                    'contour': largest_contour,
                    'area': area,
                    'bounding_box': (x, y, w, h),
                    'aspect_ratio': w/h if h > 0 else 1,
                    'type': room_type,
                    'center': (x + w//2, y + h//2)
                }
                rooms.append(room_info)
        
        return rooms
    
    def _detect_fixtures(self, image: np.ndarray, vector_data: Dict) -> List[Dict]:
        """
        Detect fixtures and furniture symbols.
        """
        fixtures = []
        
        # Look for circular fixtures (sinks, toilets, etc.)
        circles = vector_data.get('circles', [])
        for circle in circles:
            fixture_type = self._classify_circular_fixture(circle)
            if fixture_type:
                fixture_info = {
                    'center': circle['center'],
                    'radius': circle['radius'],
                    'type': fixture_type,
                    'category': 'plumbing' if fixture_type in ['sink', 'toilet', 'bathtub'] else 'furniture'
                }
                fixtures.append(fixture_info)
        
        # Look for rectangular fixtures
        rectangles = vector_data.get('rectangles', [])
        for rect in rectangles:
            fixture_type = self._classify_rectangular_fixture(rect)
            if fixture_type:
                fixture_info = {
                    'bounding_box': (rect['x'], rect['y'], rect['width'], rect['height']),
                    'area': rect['area'],
                    'aspect_ratio': rect['aspect_ratio'],
                    'type': fixture_type,
                    'category': self._get_fixture_category(fixture_type)
                }
                fixtures.append(fixture_info)
        
        return fixtures
    
    def _detect_text(self, image: np.ndarray) -> List[Dict]:
        """
        Detect text labels in the floor plan.
        """
        text_regions = []
        
        # Use morphological operations to find text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # Find contours that might be text
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on text-like properties
            if self._is_text_like(w, h, cv2.contourArea(contour)):
                text_info = {
                    'bounding_box': (x, y, w, h),
                    'area': cv2.contourArea(contour),
                    'aspect_ratio': w/h if h > 0 else 1,
                    'type': 'text_region'
                }
                text_regions.append(text_info)
        
        return text_regions
    
    # Helper methods
    
    def _validate_door_location(self, x: int, y: int, r: int, walls: List[Dict]) -> Optional[Dict]:
        """Validate if a door candidate is actually on a wall."""
        door_point = Point(x, y)
        
        for wall in walls:
            # Check if door intersects with wall
            wall_points = wall['points']
            if len(wall_points) >= 2:
                for i in range(len(wall_points) - 1):
                    wall_segment = LineString([wall_points[i], wall_points[i+1]])
                    if door_point.distance(wall_segment) < r:
                        return {'wall': wall, 'segment': wall_segment}
        
        return None
    
    def _find_parallel_lines(self, lines: List[Dict], max_distance: float = 15) -> List[List[Dict]]:
        """Find groups of parallel lines."""
        if not lines:
            return []
        
        # Group lines by similar angles
        angle_groups = {}
        for line in lines:
            angle = line['angle']
            # Normalize angle to 0-180 range
            normalized_angle = angle % 180
            
            # Find closest angle group
            closest_group = None
            min_diff = float('inf')
            
            for group_angle in angle_groups.keys():
                diff = min(abs(normalized_angle - group_angle), 
                          180 - abs(normalized_angle - group_angle))
                if diff < min_diff and diff < 10:  # 10 degree tolerance
                    min_diff = diff
                    closest_group = group_angle
            
            if closest_group is not None:
                angle_groups[closest_group].append(line)
            else:
                angle_groups[normalized_angle] = [line]
        
        # Within each angle group, find parallel lines
        parallel_groups = []
        for angle_group in angle_groups.values():
            if len(angle_group) < 2:
                continue
                
            # Use DBSCAN to cluster lines by distance
            positions = []
            for line in angle_group:
                # Calculate perpendicular distance from origin
                x1, y1 = line['start']
                x2, y2 = line['end']
                # Distance from origin to line
                dist = abs((y2-y1)*0 - (x2-x1)*0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
                positions.append([dist])
            
            if len(positions) > 1:
                clustering = DBSCAN(eps=max_distance, min_samples=2).fit(positions)
                
                # Group lines by cluster
                clusters = {}
                for i, label in enumerate(clustering.labels_):
                    if label != -1:  # Not noise
                        if label not in clusters:
                            clusters[label] = []
                        clusters[label].append(angle_group[i])
                
                parallel_groups.extend(clusters.values())
        
        return parallel_groups
    
    def _calculate_line_distance(self, line1: Dict, line2: Dict) -> float:
        """Calculate distance between two parallel lines."""
        # Create LineString objects
        l1 = LineString([line1['start'], line1['end']])
        l2 = LineString([line2['start'], line2['end']])
        
        return l1.distance(l2)
    
    def _find_enclosed_areas(self, image: np.ndarray) -> List[np.ndarray]:
        """Find enclosed areas that could be rooms."""
        # Create binary image
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Fill holes to create solid regions
        kernel = np.ones((3, 3), np.uint8)
        filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(filled)
        
        room_masks = []
        for label in range(1, num_labels):  # Skip background (label 0)
            mask = (labels == label).astype(np.uint8) * 255
            area = np.sum(mask > 0)
            
            # Filter by size
            if area > 1000:  # Minimum room area
                room_masks.append(mask)
        
        return room_masks
    
    def _classify_room_type(self, area: float, aspect_ratio: float, position: Tuple[int, int]) -> str:
        """Classify room type based on properties."""
        # This is a simplified classification - real implementation would use ML
        if area < 2000:
            if aspect_ratio > 2:
                return 'hallway'
            else:
                return 'bathroom'
        elif area < 5000:
            if aspect_ratio > 1.5:
                return 'kitchen'
            else:
                return 'bedroom'
        else:
            return 'living_room'
    
    def _classify_circular_fixture(self, circle: Dict) -> Optional[str]:
        """Classify circular fixtures based on size."""
        radius = circle['radius']
        if 5 <= radius <= 15:
            return 'sink'
        elif 15 <= radius <= 25:
            return 'toilet'
        elif 25 <= radius <= 40:
            return 'bathtub'
        return None
    
    def _classify_rectangular_fixture(self, rect: Dict) -> Optional[str]:
        """Classify rectangular fixtures."""
        area = rect['area']
        aspect_ratio = rect['aspect_ratio']
        
        if 100 <= area <= 1000:
            if aspect_ratio > 2:
                return 'counter'
            elif aspect_ratio < 0.5:
                return 'counter'
            else:
                return 'cabinet'
        elif area > 1000:
            if aspect_ratio > 3:
                return 'kitchen_island'
        
        return None
    
    def _get_fixture_category(self, fixture_type: str) -> str:
        """Get category for fixture type."""
        plumbing_fixtures = ['sink', 'toilet', 'bathtub', 'shower']
        kitchen_fixtures = ['counter', 'cabinet', 'kitchen_island']
        
        if fixture_type in plumbing_fixtures:
            return 'plumbing'
        elif fixture_type in kitchen_fixtures:
            return 'kitchen'
        else:
            return 'furniture'
    
    def _is_text_like(self, width: int, height: int, area: float) -> bool:
        """Check if a contour has text-like properties."""
        aspect_ratio = width / height if height > 0 else 0
        
        # Text typically has certain aspect ratios and sizes
        return (50 <= area <= 2000 and 
                0.1 <= aspect_ratio <= 10 and 
                width >= 10 and height >= 5)
    
    def _post_process_features(self, features: Dict) -> Dict:
        """Post-process features to resolve conflicts and improve accuracy."""
        # Remove overlapping doors and windows
        features['doors'] = self._remove_overlapping_features(features['doors'])
        features['windows'] = self._remove_overlapping_features(features['windows'])
        
        # Validate door and window positions against walls
        features['doors'] = self._validate_openings_on_walls(features['doors'], features['walls'])
        features['windows'] = self._validate_openings_on_walls(features['windows'], features['walls'])
        
        return features
    
    def _remove_overlapping_features(self, features: List[Dict]) -> List[Dict]:
        """Remove overlapping features of the same type."""
        if not features:
            return features
        
        filtered_features = []
        for i, feature in enumerate(features):
            overlaps = False
            
            for j, other_feature in enumerate(features):
                if i != j and self._features_overlap(feature, other_feature):
                    # Keep the feature with higher confidence or larger size
                    if self._get_feature_score(feature) < self._get_feature_score(other_feature):
                        overlaps = True
                        break
            
            if not overlaps:
                filtered_features.append(feature)
        
        return filtered_features
    
    def _features_overlap(self, feature1: Dict, feature2: Dict) -> bool:
        """Check if two features overlap significantly."""
        # Get feature centers
        center1 = self._get_feature_center(feature1)
        center2 = self._get_feature_center(feature2)
        
        if center1 is None or center2 is None:
            return False
        
        # Calculate distance
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Get feature sizes
        size1 = self._get_feature_size(feature1)
        size2 = self._get_feature_size(feature2)
        
        # Consider overlapping if distance is less than average size
        return distance < (size1 + size2) / 2
    
    def _get_feature_center(self, feature: Dict) -> Optional[Tuple[int, int]]:
        """Get the center point of a feature."""
        if 'center' in feature:
            return feature['center']
        elif 'bounding_box' in feature:
            x, y, w, h = feature['bounding_box']
            return (x + w//2, y + h//2)
        elif 'start' in feature and 'end' in feature:
            start, end = feature['start'], feature['end']
            return ((start[0] + end[0])//2, (start[1] + end[1])//2)
        return None
    
    def _get_feature_size(self, feature: Dict) -> float:
        """Get the size of a feature."""
        if 'radius' in feature:
            return feature['radius'] * 2
        elif 'width' in feature:
            return feature['width']
        elif 'area' in feature:
            return np.sqrt(feature['area'])
        return 10.0  # default size
    
    def _get_feature_score(self, feature: Dict) -> float:
        """Get a score for feature quality (higher is better)."""
        score = 0.0
        
        # Size factor
        size = self._get_feature_size(feature)
        score += size / 100.0
        
        # Type-specific scoring
        if feature.get('type') == 'swing_door':
            score += 1.0
        elif feature.get('type') == 'standard_window':
            score += 1.0
        
        return score
    
    def _validate_openings_on_walls(self, openings: List[Dict], walls: List[Dict]) -> List[Dict]:
        """Validate that doors and windows are actually on walls."""
        validated_openings = []
        
        for opening in openings:
            center = self._get_feature_center(opening)
            if center is None:
                continue
            
            # Check if opening is near any wall
            near_wall = False
            for wall in walls:
                if self._is_opening_on_wall(opening, wall):
                    near_wall = True
                    break
            
            if near_wall:
                validated_openings.append(opening)
        
        return validated_openings
    
    def _is_opening_on_wall(self, opening: Dict, wall: Dict) -> bool:
        """Check if an opening is positioned on a wall."""
        center = self._get_feature_center(opening)
        if center is None:
            return False
        
        opening_point = Point(center)
        
        # Check distance to wall segments
        wall_points = wall['points']
        if len(wall_points) < 2:
            return False
        
        for i in range(len(wall_points) - 1):
            wall_segment = LineString([wall_points[i], wall_points[i+1]])
            if opening_point.distance(wall_segment) < 10:  # 10 pixel tolerance
                return True
        
        return False