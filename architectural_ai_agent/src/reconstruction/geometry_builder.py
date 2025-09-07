"""
Geometry Builder Module

This module provides utilities for building complex 3D geometries
and handling geometric operations for architectural reconstruction.
"""

import numpy as np
import trimesh
from typing import List, Tuple, Dict, Optional, Union
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import unary_union, triangulate
import logging

logger = logging.getLogger(__name__)


class GeometryBuilder:
    """
    Utility class for building complex 3D geometries from 2D floor plan data.
    """
    
    def __init__(self):
        """Initialize the geometry builder."""
        self.tolerance = 1e-6
        
    def create_extruded_polygon(self, polygon: Polygon, height: float, 
                              base_z: float = 0.0) -> trimesh.Trimesh:
        """
        Create a 3D mesh by extruding a 2D polygon.
        
        Args:
            polygon: 2D polygon to extrude
            height: Extrusion height
            base_z: Base Z coordinate
            
        Returns:
            Extruded 3D mesh
        """
        if not polygon.is_valid or polygon.is_empty:
            raise ValueError("Invalid or empty polygon")
        
        # Handle MultiPolygon
        if isinstance(polygon, MultiPolygon):
            meshes = []
            for poly in polygon.geoms:
                if poly.area > self.tolerance:
                    meshes.append(self.create_extruded_polygon(poly, height, base_z))
            
            if meshes:
                return trimesh.util.concatenate(meshes)
            else:
                raise ValueError("No valid polygons in MultiPolygon")
        
        # Get exterior coordinates
        exterior_coords = list(polygon.exterior.coords[:-1])  # Remove duplicate last point
        
        if len(exterior_coords) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        
        # Create vertices
        vertices = []
        
        # Bottom vertices
        for x, y in exterior_coords:
            vertices.append([x, y, base_z])
        
        # Top vertices
        for x, y in exterior_coords:
            vertices.append([x, y, base_z + height])
        
        # Handle holes
        hole_vertices_start = len(vertices)
        hole_starts = []
        
        for interior in polygon.interiors:
            hole_starts.append(len(vertices))
            hole_coords = list(interior.coords[:-1])
            
            # Bottom hole vertices
            for x, y in hole_coords:
                vertices.append([x, y, base_z])
            
            # Top hole vertices
            for x, y in hole_coords:
                vertices.append([x, y, base_z + height])
        
        # Create faces
        faces = []
        n_exterior = len(exterior_coords)
        
        # Bottom face
        bottom_faces = self._triangulate_polygon_with_holes(
            exterior_coords, [list(interior.coords[:-1]) for interior in polygon.interiors]
        )
        faces.extend(bottom_faces)
        
        # Top face (reverse order for correct normal)
        top_faces = []
        for face in bottom_faces:
            top_face = [v + n_exterior for v in reversed(face)]
            top_faces.append(top_face)
        faces.extend(top_faces)
        
        # Exterior side faces
        for i in range(n_exterior):
            next_i = (i + 1) % n_exterior
            faces.extend([
                [i, next_i, n_exterior + next_i],
                [i, n_exterior + next_i, n_exterior + i]
            ])
        
        # Interior side faces (holes)
        vertex_offset = 2 * n_exterior
        for hole_idx, interior in enumerate(polygon.interiors):
            hole_coords = list(interior.coords[:-1])
            n_hole = len(hole_coords)
            
            for i in range(n_hole):
                next_i = (i + 1) % n_hole
                v1 = vertex_offset + i
                v2 = vertex_offset + next_i
                v3 = vertex_offset + n_hole + next_i
                v4 = vertex_offset + n_hole + i
                
                # Reverse order for holes (inward-facing normals)
                faces.extend([
                    [v1, v4, v3],
                    [v1, v3, v2]
                ])
            
            vertex_offset += 2 * n_hole
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def _triangulate_polygon_with_holes(self, exterior: List[Tuple[float, float]], 
                                      holes: List[List[Tuple[float, float]]]) -> List[List[int]]:
        """
        Triangulate a polygon with holes using earcut algorithm.
        
        Args:
            exterior: Exterior boundary vertices
            holes: List of hole boundaries
            
        Returns:
            List of triangle faces as vertex indices
        """
        try:
            # Use trimesh's triangulation
            polygon = Polygon(exterior, holes)
            
            # Get triangulation
            triangles = triangulate(polygon)
            
            # Convert to face indices
            faces = []
            vertex_map = {}
            current_idx = 0
            
            # Map exterior vertices
            for i, (x, y) in enumerate(exterior):
                vertex_map[(x, y)] = i
            
            # Map hole vertices
            for hole in holes:
                for x, y in hole:
                    if (x, y) not in vertex_map:
                        vertex_map[(x, y)] = len(exterior) + current_idx
                        current_idx += 1
            
            # Convert triangles to faces
            for triangle in triangles.geoms:
                if hasattr(triangle, 'exterior'):
                    coords = list(triangle.exterior.coords[:-1])
                    if len(coords) == 3:
                        face = [vertex_map[coord] for coord in coords]
                        faces.append(face)
            
            return faces
            
        except Exception as e:
            logger.warning(f"Triangulation failed, using simple fan triangulation: {e}")
            # Fallback to simple fan triangulation (only works for convex polygons)
            faces = []
            for i in range(1, len(exterior) - 1):
                faces.append([0, i, i + 1])
            return faces
    
    def create_wall_with_openings(self, wall_line: LineString, thickness: float, 
                                height: float, openings: List[Dict]) -> trimesh.Trimesh:
        """
        Create a wall mesh with door and window openings.
        
        Args:
            wall_line: Wall centerline
            thickness: Wall thickness
            height: Wall height
            openings: List of opening specifications
            
        Returns:
            Wall mesh with openings cut out
        """
        # Create wall polygon
        wall_polygon = wall_line.buffer(thickness / 2, cap_style=2)  # Square caps
        
        # Create opening polygons to subtract
        opening_polygons = []
        
        for opening in openings:
            opening_type = opening.get('type', 'door')
            position = opening.get('position', 0.5)  # Position along wall (0-1)
            width = opening.get('width', 0.8)
            opening_height = opening.get('height', 2.1)
            sill_height = opening.get('sill_height', 0.0)
            
            # Calculate opening position on wall
            wall_point = wall_line.interpolate(position, normalized=True)
            
            # Create opening polygon
            opening_polygon = self._create_opening_polygon(
                wall_point, wall_line, width, thickness
            )
            
            if opening_polygon and opening_polygon.is_valid:
                opening_polygons.append(opening_polygon)
        
        # Subtract openings from wall
        if opening_polygons:
            openings_union = unary_union(opening_polygons)
            wall_with_openings = wall_polygon.difference(openings_union)
        else:
            wall_with_openings = wall_polygon
        
        # Extrude to create 3D mesh
        if wall_with_openings.is_empty:
            # Wall completely removed by openings
            return trimesh.Trimesh()
        
        return self.create_extruded_polygon(wall_with_openings, height)
    
    def _create_opening_polygon(self, point: Point, wall_line: LineString, 
                              width: float, wall_thickness: float) -> Optional[Polygon]:
        """Create a polygon representing an opening in a wall."""
        try:
            # Get wall direction at this point
            # Find closest point on line to get direction
            distance = wall_line.project(point)
            
            # Get a small segment around this point to determine direction
            epsilon = 0.01
            p1 = wall_line.interpolate(max(0, distance - epsilon))
            p2 = wall_line.interpolate(min(wall_line.length, distance + epsilon))
            
            # Calculate direction vector
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            length = np.sqrt(dx*dx + dy*dy)
            
            if length < self.tolerance:
                return None
            
            # Normalize direction
            dx /= length
            dy /= length
            
            # Perpendicular vector
            perp_x = -dy
            perp_y = dx
            
            # Create opening rectangle
            half_width = width / 2
            half_thickness = wall_thickness / 2 + 0.01  # Slightly larger to ensure cut-through
            
            corners = [
                (point.x - dx * half_width - perp_x * half_thickness,
                 point.y - dy * half_width - perp_y * half_thickness),
                (point.x + dx * half_width - perp_x * half_thickness,
                 point.y + dy * half_width - perp_y * half_thickness),
                (point.x + dx * half_width + perp_x * half_thickness,
                 point.y + dy * half_width + perp_y * half_thickness),
                (point.x - dx * half_width + perp_x * half_thickness,
                 point.y - dy * half_width + perp_y * half_thickness)
            ]
            
            return Polygon(corners)
            
        except Exception as e:
            logger.warning(f"Failed to create opening polygon: {e}")
            return None
    
    def create_room_mesh(self, room_polygon: Polygon, floor_height: float = 0.0,
                        ceiling_height: float = 2.4) -> Dict[str, trimesh.Trimesh]:
        """
        Create floor and ceiling meshes for a room.
        
        Args:
            room_polygon: Room boundary polygon
            floor_height: Floor Z coordinate
            ceiling_height: Ceiling Z coordinate
            
        Returns:
            Dictionary with 'floor' and 'ceiling' meshes
        """
        meshes = {}
        
        try:
            # Floor mesh
            floor_mesh = self.create_extruded_polygon(
                room_polygon, 0.01, floor_height  # Thin floor
            )
            meshes['floor'] = floor_mesh
            
            # Ceiling mesh
            ceiling_mesh = self.create_extruded_polygon(
                room_polygon, 0.01, ceiling_height - 0.01  # Thin ceiling
            )
            meshes['ceiling'] = ceiling_mesh
            
        except Exception as e:
            logger.error(f"Failed to create room meshes: {e}")
        
        return meshes
    
    def create_stairs_mesh(self, stair_polygon: Polygon, num_steps: int = 10,
                          total_height: float = 2.7, step_depth: float = 0.25) -> trimesh.Trimesh:
        """
        Create a stairs mesh from a stair region polygon.
        
        Args:
            stair_polygon: Stair region boundary
            num_steps: Number of steps
            total_height: Total height of stairs
            step_depth: Depth of each step
            
        Returns:
            Stairs mesh
        """
        try:
            # Get stair direction (longest side of bounding box)
            bounds = stair_polygon.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            
            if width > height:
                # Steps run along width
                step_width = width / num_steps
                is_horizontal = True
            else:
                # Steps run along height
                step_width = height / num_steps
                is_horizontal = False
            
            step_height = total_height / num_steps
            
            # Create individual step meshes
            step_meshes = []
            
            for i in range(num_steps):
                z_base = i * step_height
                z_top = (i + 1) * step_height
                
                if is_horizontal:
                    # Create step polygon
                    x_start = bounds[0] + i * step_width
                    x_end = bounds[0] + (i + 1) * step_width
                    
                    step_polygon = Polygon([
                        (x_start, bounds[1]),
                        (x_end, bounds[1]),
                        (x_end, bounds[3]),
                        (x_start, bounds[3])
                    ])
                else:
                    # Create step polygon
                    y_start = bounds[1] + i * step_width
                    y_end = bounds[1] + (i + 1) * step_width
                    
                    step_polygon = Polygon([
                        (bounds[0], y_start),
                        (bounds[2], y_start),
                        (bounds[2], y_end),
                        (bounds[0], y_end)
                    ])
                
                # Intersect with original stair polygon
                step_polygon = step_polygon.intersection(stair_polygon)
                
                if not step_polygon.is_empty and step_polygon.area > self.tolerance:
                    step_mesh = self.create_extruded_polygon(step_polygon, z_top, 0)
                    step_meshes.append(step_mesh)
            
            # Combine all steps
            if step_meshes:
                return trimesh.util.concatenate(step_meshes)
            else:
                # Fallback: create simple ramp
                return self.create_extruded_polygon(stair_polygon, total_height)
                
        except Exception as e:
            logger.error(f"Failed to create stairs mesh: {e}")
            # Fallback: create simple extruded polygon
            return self.create_extruded_polygon(stair_polygon, total_height)
    
    def create_curved_wall(self, center: Tuple[float, float], radius: float,
                          start_angle: float, end_angle: float, thickness: float,
                          height: float, segments: int = 16) -> trimesh.Trimesh:
        """
        Create a curved wall mesh.
        
        Args:
            center: Center point of curve
            radius: Radius of curve
            start_angle: Start angle in radians
            end_angle: End angle in radians
            thickness: Wall thickness
            height: Wall height
            segments: Number of segments for curve approximation
            
        Returns:
            Curved wall mesh
        """
        try:
            # Generate points along the arc
            angles = np.linspace(start_angle, end_angle, segments + 1)
            
            # Inner and outer radii
            inner_radius = radius - thickness / 2
            outer_radius = radius + thickness / 2
            
            # Generate vertices
            vertices = []
            
            # Bottom vertices
            for angle in angles:
                # Inner arc
                x_inner = center[0] + inner_radius * np.cos(angle)
                y_inner = center[1] + inner_radius * np.sin(angle)
                vertices.append([x_inner, y_inner, 0])
                
                # Outer arc
                x_outer = center[0] + outer_radius * np.cos(angle)
                y_outer = center[1] + outer_radius * np.sin(angle)
                vertices.append([x_outer, y_outer, 0])
            
            # Top vertices (same pattern)
            for angle in angles:
                # Inner arc
                x_inner = center[0] + inner_radius * np.cos(angle)
                y_inner = center[1] + inner_radius * np.sin(angle)
                vertices.append([x_inner, y_inner, height])
                
                # Outer arc
                x_outer = center[0] + outer_radius * np.cos(angle)
                y_outer = center[1] + outer_radius * np.sin(angle)
                vertices.append([x_outer, y_outer, height])
            
            # Generate faces
            faces = []
            n_points = len(angles)
            
            # Side faces
            for i in range(n_points - 1):
                # Inner side
                v1 = i * 2
                v2 = (i + 1) * 2
                v3 = 2 * n_points + (i + 1) * 2
                v4 = 2 * n_points + i * 2
                
                faces.extend([[v1, v2, v3], [v1, v3, v4]])
                
                # Outer side
                v1 = i * 2 + 1
                v2 = (i + 1) * 2 + 1
                v3 = 2 * n_points + (i + 1) * 2 + 1
                v4 = 2 * n_points + i * 2 + 1
                
                faces.extend([[v1, v4, v3], [v1, v3, v2]])
                
                # Top and bottom connecting faces
                # Bottom
                faces.extend([[i * 2, i * 2 + 1, (i + 1) * 2 + 1], 
                             [i * 2, (i + 1) * 2 + 1, (i + 1) * 2]])
                
                # Top
                top_base = 2 * n_points
                faces.extend([[top_base + i * 2, top_base + (i + 1) * 2 + 1, top_base + i * 2 + 1],
                             [top_base + i * 2, top_base + (i + 1) * 2, top_base + (i + 1) * 2 + 1]])
            
            # End caps
            if abs(end_angle - start_angle) < 2 * np.pi - 0.1:  # Not a full circle
                # Start cap
                faces.extend([
                    [0, 2 * n_points, 2 * n_points + 1],
                    [0, 2 * n_points + 1, 1]
                ])
                
                # End cap
                last_idx = (n_points - 1) * 2
                faces.extend([
                    [last_idx, last_idx + 1, 2 * n_points + last_idx + 1],
                    [last_idx, 2 * n_points + last_idx + 1, 2 * n_points + last_idx]
                ])
            
            return trimesh.Trimesh(vertices=vertices, faces=faces)
            
        except Exception as e:
            logger.error(f"Failed to create curved wall: {e}")
            return trimesh.Trimesh()
    
    def boolean_difference(self, mesh_a: trimesh.Trimesh, 
                          mesh_b: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Perform boolean difference operation (A - B).
        
        Args:
            mesh_a: Main mesh
            mesh_b: Mesh to subtract
            
        Returns:
            Result of boolean difference
        """
        try:
            result = mesh_a.difference(mesh_b)
            return result
        except Exception as e:
            logger.warning(f"Boolean difference failed: {e}")
            return mesh_a  # Return original if operation fails
    
    def boolean_union(self, meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
        """
        Perform boolean union of multiple meshes.
        
        Args:
            meshes: List of meshes to union
            
        Returns:
            Union of all meshes
        """
        if not meshes:
            return trimesh.Trimesh()
        
        if len(meshes) == 1:
            return meshes[0]
        
        try:
            result = meshes[0]
            for mesh in meshes[1:]:
                result = result.union(mesh)
            return result
        except Exception as e:
            logger.warning(f"Boolean union failed: {e}")
            # Fallback: concatenate meshes
            return trimesh.util.concatenate(meshes)
    
    def smooth_mesh(self, mesh: trimesh.Trimesh, iterations: int = 1) -> trimesh.Trimesh:
        """
        Apply Laplacian smoothing to mesh.
        
        Args:
            mesh: Input mesh
            iterations: Number of smoothing iterations
            
        Returns:
            Smoothed mesh
        """
        try:
            smoothed = mesh.copy()
            for _ in range(iterations):
                smoothed = smoothed.smoothed()
            return smoothed
        except Exception as e:
            logger.warning(f"Mesh smoothing failed: {e}")
            return mesh
    
    def optimize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Optimize mesh by removing duplicate vertices and fixing normals.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Optimized mesh
        """
        try:
            # Remove duplicate vertices
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()
            
            # Fix normals
            mesh.fix_normals()
            
            return mesh
        except Exception as e:
            logger.warning(f"Mesh optimization failed: {e}")
            return mesh