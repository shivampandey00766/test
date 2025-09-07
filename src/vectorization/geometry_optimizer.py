"""
Geometry optimization module for vector data.

This module provides functionality to optimize and clean up vector
data for better 3D reconstruction and CAD compatibility.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union, snap
from shapely.validation import make_valid
import matplotlib.pyplot as plt
from dataclasses import dataclass
from .raster_to_vector import VectorLine, VectorPolygon, VectorPoint


@dataclass
class OptimizationResult:
    """Result of geometry optimization."""
    optimized_lines: List[VectorLine]
    optimized_polygons: List[VectorPolygon]
    removed_elements: int
    merged_elements: int
    optimization_metrics: Dict[str, float]


class GeometryOptimizer:
    """
    Optimizes vector geometry for better 3D reconstruction.
    
    This class provides methods to clean up, merge, and optimize
    vector data to improve quality and reduce complexity.
    """
    
    def __init__(self, 
                 snap_tolerance: float = 5.0,
                 merge_tolerance: float = 10.0,
                 min_polygon_area: float = 100.0):
        """
        Initialize the GeometryOptimizer.
        
        Args:
            snap_tolerance: Tolerance for snapping nearby points
            merge_tolerance: Tolerance for merging similar elements
            min_polygon_area: Minimum area for valid polygons
        """
        self.snap_tolerance = snap_tolerance
        self.merge_tolerance = merge_tolerance
        self.min_polygon_area = min_polygon_area
    
    def snap_points(self, lines: List[VectorLine], 
                   polygons: List[VectorPolygon]) -> Tuple[List[VectorLine], List[VectorPolygon]]:
        """
        Snap nearby points to common coordinates.
        
        Args:
            lines: List of VectorLines
            polygons: List of VectorPolygons
            
        Returns:
            Tuple of (snapped_lines, snapped_polygons)
        """
        # Collect all points
        all_points = []
        point_to_element = {}  # Map from point to element containing it
        
        for i, line in enumerate(lines):
            all_points.extend([line.start, line.end])
            point_to_element[line.start] = ('line', i, 'start')
            point_to_element[line.end] = ('line', i, 'end')
        
        for i, polygon in enumerate(polygons):
            for j, point in enumerate(polygon.points):
                all_points.append(point)
                point_to_element[point] = ('polygon', i, j)
        
        # Group nearby points
        point_groups = self._group_nearby_points(all_points)
        
        # Create mapping from old points to new points
        point_mapping = {}
        for group in point_groups:
            if len(group) > 1:
                # Use the first point as the representative
                representative = group[0]
                for point in group:
                    point_mapping[point] = representative
        
        # Apply snapping to lines
        snapped_lines = []
        for line in lines:
            new_start = point_mapping.get(line.start, line.start)
            new_end = point_mapping.get(line.end, line.end)
            
            snapped_line = VectorLine(
                start=new_start,
                end=new_end,
                thickness=line.thickness,
                line_type=line.line_type
            )
            snapped_lines.append(snapped_line)
        
        # Apply snapping to polygons
        snapped_polygons = []
        for polygon in polygons:
            new_points = [point_mapping.get(point, point) for point in polygon.points]
            
            snapped_polygon = VectorPolygon(
                points=new_points,
                polygon_type=polygon.polygon_type,
                area=polygon.area
            )
            snapped_polygons.append(snapped_polygon)
        
        return snapped_lines, snapped_polygons
    
    def _group_nearby_points(self, points: List[VectorPoint]) -> List[List[VectorPoint]]:
        """Group points that are within snap tolerance."""
        groups = []
        used_points = set()
        
        for i, point1 in enumerate(points):
            if point1 in used_points:
                continue
            
            group = [point1]
            used_points.add(point1)
            
            for j, point2 in enumerate(points[i+1:], i+1):
                if point2 in used_points:
                    continue
                
                distance = np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
                if distance <= self.snap_tolerance:
                    group.append(point2)
                    used_points.add(point2)
            
            groups.append(group)
        
        return groups
    
    def merge_collinear_lines(self, lines: List[VectorLine]) -> List[VectorLine]:
        """
        Merge collinear line segments.
        
        Args:
            lines: List of VectorLines
            
        Returns:
            List of merged VectorLines
        """
        if not lines:
            return []
        
        # Group lines by similar direction
        line_groups = self._group_collinear_lines(lines)
        
        merged_lines = []
        for group in line_groups:
            if len(group) == 1:
                merged_lines.append(group[0])
            else:
                merged_line = self._merge_collinear_group(group)
                merged_lines.append(merged_line)
        
        return merged_lines
    
    def _group_collinear_lines(self, lines: List[VectorLine]) -> List[List[VectorLine]]:
        """Group lines that are collinear."""
        groups = []
        used_lines = set()
        
        for i, line1 in enumerate(lines):
            if i in used_lines:
                continue
            
            group = [line1]
            used_lines.add(i)
            
            for j, line2 in enumerate(lines[i+1:], i+1):
                if j in used_lines:
                    continue
                
                if self._are_lines_collinear(line1, line2):
                    group.append(line2)
                    used_lines.add(j)
            
            groups.append(group)
        
        return groups
    
    def _are_lines_collinear(self, line1: VectorLine, line2: VectorLine) -> bool:
        """Check if two lines are collinear."""
        # Calculate direction vectors
        dir1 = np.array([line1.end.x - line1.start.x, line1.end.y - line1.start.y])
        dir2 = np.array([line2.end.x - line2.start.x, line2.end.y - line2.start.y])
        
        # Normalize direction vectors
        if np.linalg.norm(dir1) == 0 or np.linalg.norm(dir2) == 0:
            return False
        
        dir1_norm = dir1 / np.linalg.norm(dir1)
        dir2_norm = dir2 / np.linalg.norm(dir2)
        
        # Check if directions are parallel (dot product close to 1 or -1)
        dot_product = abs(np.dot(dir1_norm, dir2_norm))
        if dot_product < 0.99:  # Not parallel
            return False
        
        # Check if lines are on the same line (distance from one line to the other)
        # Convert to shapely LineString for distance calculation
        line1_geom = LineString([(line1.start.x, line1.start.y), (line1.end.x, line1.end.y)])
        line2_geom = LineString([(line2.start.x, line2.start.y), (line2.end.x, line2.end.y)])
        
        distance = line1_geom.distance(line2_geom)
        return distance <= self.merge_tolerance
    
    def _merge_collinear_group(self, lines: List[VectorLine]) -> VectorLine:
        """Merge a group of collinear lines."""
        if len(lines) == 1:
            return lines[0]
        
        # Find all endpoints
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
        
        # Use average thickness and most common line type
        avg_thickness = np.mean([line.thickness for line in lines])
        line_types = [line.line_type for line in lines]
        most_common_type = max(set(line_types), key=line_types.count)
        
        return VectorLine(
            start=start_point,
            end=end_point,
            thickness=avg_thickness,
            line_type=most_common_type
        )
    
    def clean_polygons(self, polygons: List[VectorPolygon]) -> List[VectorPolygon]:
        """
        Clean and validate polygons.
        
        Args:
            polygons: List of VectorPolygons
            
        Returns:
            List of cleaned VectorPolygons
        """
        cleaned_polygons = []
        
        for polygon in polygons:
            # Skip if too few points
            if len(polygon.points) < 3:
                continue
            
            # Create shapely polygon for validation
            points = [(p.x, p.y) for p in polygon.points]
            shapely_polygon = Polygon(points)
            
            # Make polygon valid if needed
            if not shapely_polygon.is_valid:
                shapely_polygon = make_valid(shapely_polygon)
            
            # Skip if area is too small
            if shapely_polygon.area < self.min_polygon_area:
                continue
            
            # Simplify polygon if it has too many points
            if len(polygon.points) > 20:
                shapely_polygon = shapely_polygon.simplify(2.0, preserve_topology=True)
            
            # Convert back to VectorPolygon
            if hasattr(shapely_polygon, 'exterior'):
                exterior_coords = list(shapely_polygon.exterior.coords[:-1])  # Remove duplicate last point
                new_points = [VectorPoint(x, y) for x, y in exterior_coords]
                
                cleaned_polygon = VectorPolygon(
                    points=new_points,
                    polygon_type=polygon.polygon_type,
                    area=shapely_polygon.area
                )
                cleaned_polygons.append(cleaned_polygon)
        
        return cleaned_polygons
    
    def remove_duplicate_elements(self, lines: List[VectorLine], 
                                polygons: List[VectorPolygon]) -> Tuple[List[VectorLine], List[VectorPolygon]]:
        """
        Remove duplicate lines and polygons.
        
        Args:
            lines: List of VectorLines
            polygons: List of VectorPolygons
            
        Returns:
            Tuple of (unique_lines, unique_polygons)
        """
        # Remove duplicate lines
        unique_lines = []
        for line in lines:
            is_duplicate = False
            for existing_line in unique_lines:
                if self._are_lines_identical(line, existing_line):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_lines.append(line)
        
        # Remove duplicate polygons
        unique_polygons = []
        for polygon in polygons:
            is_duplicate = False
            for existing_polygon in unique_polygons:
                if self._are_polygons_identical(polygon, existing_polygon):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_polygons.append(polygon)
        
        return unique_lines, unique_polygons
    
    def _are_lines_identical(self, line1: VectorLine, line2: VectorLine) -> bool:
        """Check if two lines are identical."""
        # Check if endpoints are the same (in either order)
        same_direction = (
            abs(line1.start.x - line2.start.x) < 1e-6 and
            abs(line1.start.y - line2.start.y) < 1e-6 and
            abs(line1.end.x - line2.end.x) < 1e-6 and
            abs(line1.end.y - line2.end.y) < 1e-6
        )
        
        opposite_direction = (
            abs(line1.start.x - line2.end.x) < 1e-6 and
            abs(line1.start.y - line2.end.y) < 1e-6 and
            abs(line1.end.x - line2.start.x) < 1e-6 and
            abs(line1.end.y - line2.start.y) < 1e-6
        )
        
        return same_direction or opposite_direction
    
    def _are_polygons_identical(self, poly1: VectorPolygon, poly2: VectorPolygon) -> bool:
        """Check if two polygons are identical."""
        if len(poly1.points) != len(poly2.points):
            return False
        
        # Check if all points match (allowing for different starting points)
        points1 = [(p.x, p.y) for p in poly1.points]
        points2 = [(p.x, p.y) for p in poly2.points]
        
        # Try all possible starting points
        for i in range(len(points1)):
            rotated_points2 = points2[i:] + points2[:i]
            if all(abs(p1[0] - p2[0]) < 1e-6 and abs(p1[1] - p2[1]) < 1e-6 
                   for p1, p2 in zip(points1, rotated_points2)):
                return True
        
        return False
    
    def optimize_geometry(self, lines: List[VectorLine], 
                         polygons: List[VectorPolygon]) -> OptimizationResult:
        """
        Perform comprehensive geometry optimization.
        
        Args:
            lines: List of VectorLines
            polygons: List of VectorPolygons
            
        Returns:
            OptimizationResult object
        """
        original_line_count = len(lines)
        original_polygon_count = len(polygons)
        
        # Step 1: Snap nearby points
        lines, polygons = self.snap_points(lines, polygons)
        
        # Step 2: Merge collinear lines
        lines = self.merge_collinear_lines(lines)
        
        # Step 3: Clean polygons
        polygons = self.clean_polygons(polygons)
        
        # Step 4: Remove duplicates
        lines, polygons = self.remove_duplicate_elements(lines, polygons)
        
        # Calculate metrics
        removed_lines = original_line_count - len(lines)
        removed_polygons = original_polygon_count - len(polygons)
        merged_elements = removed_lines + removed_polygons
        
        # Calculate quality metrics
        total_line_length = sum(
            np.sqrt((line.end.x - line.start.x)**2 + (line.end.y - line.start.y)**2)
            for line in lines
        )
        
        total_polygon_area = sum(polygon.area for polygon in polygons)
        
        optimization_metrics = {
            'total_line_length': total_line_length,
            'total_polygon_area': total_polygon_area,
            'line_count': len(lines),
            'polygon_count': len(polygons),
            'compression_ratio': (original_line_count + original_polygon_count) / max(len(lines) + len(polygons), 1)
        }
        
        return OptimizationResult(
            optimized_lines=lines,
            optimized_polygons=polygons,
            removed_elements=removed_lines + removed_polygons,
            merged_elements=merged_elements,
            optimization_metrics=optimization_metrics
        )
    
    def visualize_optimization(self, original_lines: List[VectorLine],
                             original_polygons: List[VectorPolygon],
                             optimized_lines: List[VectorLine],
                             optimized_polygons: List[VectorPolygon],
                             save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize optimization results.
        
        Args:
            original_lines: Original lines
            original_polygons: Original polygons
            optimized_lines: Optimized lines
            optimized_polygons: Optimized polygons
            save_path: Optional path to save visualization
            
        Returns:
            Comparison visualization
        """
        # Calculate bounding box
        all_x = []
        all_y = []
        
        for line in original_lines + optimized_lines:
            all_x.extend([line.start.x, line.end.x])
            all_y.extend([line.start.y, line.end.y])
        
        for polygon in original_polygons + optimized_polygons:
            for point in polygon.points:
                all_x.append(point.x)
                all_y.append(point.y)
        
        if not all_x:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        width = int(max(all_x) - min(all_x)) + 100
        height = int(max(all_y) - min(all_y)) + 100
        
        # Create comparison visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original
        ax1.set_title(f'Original ({len(original_lines)} lines, {len(original_polygons)} polygons)')
        ax1.set_xlim(min(all_x) - 50, max(all_x) + 50)
        ax1.set_ylim(min(all_y) - 50, max(all_y) + 50)
        
        for line in original_lines:
            ax1.plot([line.start.x, line.end.x], [line.start.y, line.end.y], 'b-', linewidth=2)
        
        for polygon in original_polygons:
            if len(polygon.points) >= 3:
                points = [(p.x, p.y) for p in polygon.points]
                points.append(points[0])  # Close polygon
                xs, ys = zip(*points)
                ax1.plot(xs, ys, 'r-', linewidth=2)
                ax1.fill(xs, ys, alpha=0.3)
        
        # Optimized
        ax2.set_title(f'Optimized ({len(optimized_lines)} lines, {len(optimized_polygons)} polygons)')
        ax2.set_xlim(min(all_x) - 50, max(all_x) + 50)
        ax2.set_ylim(min(all_y) - 50, max(all_y) + 50)
        
        for line in optimized_lines:
            ax2.plot([line.start.x, line.end.x], [line.start.y, line.end.y], 'g-', linewidth=2)
        
        for polygon in optimized_polygons:
            if len(polygon.points) >= 3:
                points = [(p.x, p.y) for p in polygon.points]
                points.append(points[0])  # Close polygon
                xs, ys = zip(*points)
                ax2.plot(xs, ys, 'm-', linewidth=2)
                ax2.fill(xs, ys, alpha=0.3)
        
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Convert to numpy array for return
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        return img