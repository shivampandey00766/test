"""
Mesh generation utilities for 3D reconstruction.

This module provides utility functions for generating and manipulating
3D meshes from 2D floor plan data.
"""

import numpy as np
import trimesh
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from ..vectorization.raster_to_vector import VectorLine, VectorPolygon, VectorPoint, VectorData


class MeshGenerator:
    """
    Generates and manipulates 3D meshes from floor plan data.
    
    This class provides utilities for creating, combining, and optimizing
    3D meshes for architectural visualization.
    """
    
    def __init__(self, 
                 wall_height: float = 2.4,
                 floor_thickness: float = 0.2,
                 ceiling_height: float = 2.7):
        """
        Initialize the MeshGenerator.
        
        Args:
            wall_height: Standard wall height in meters
            floor_thickness: Floor thickness in meters
            ceiling_height: Ceiling height in meters
        """
        self.wall_height = wall_height
        self.floor_thickness = floor_thickness
        self.ceiling_height = ceiling_height
        
        # Material properties
        self.materials = {
            'wall': {'color': [200, 200, 200, 255], 'roughness': 0.7},
            'floor': {'color': [150, 100, 50, 255], 'roughness': 0.3},
            'ceiling': {'color': [230, 230, 230, 255], 'roughness': 0.8},
            'door': {'color': [100, 50, 25, 255], 'roughness': 0.5},
            'window': {'color': [180, 200, 255, 128], 'roughness': 0.1}
        }
    
    def create_wall_mesh(self, start_point: VectorPoint, end_point: VectorPoint, 
                        height: float = None, thickness: float = 0.2) -> trimesh.Trimesh:
        """
        Create a 3D wall mesh from two points.
        
        Args:
            start_point: Start point of the wall
            end_point: End point of the wall
            height: Wall height (uses default if None)
            thickness: Wall thickness
            
        Returns:
            Wall mesh
        """
        if height is None:
            height = self.wall_height
        
        # Calculate wall dimensions
        length = np.sqrt((end_point.x - start_point.x)**2 + (end_point.y - start_point.y)**2)
        center_x = (start_point.x + end_point.x) / 2
        center_y = (start_point.y + end_point.y) / 2
        angle = np.arctan2(end_point.y - start_point.y, end_point.x - start_point.x)
        
        # Create wall box
        wall_mesh = trimesh.creation.box(
            extents=[length, thickness, height]
        )
        
        # Position wall
        wall_mesh.apply_translation([center_x, center_y, height / 2])
        
        # Rotate wall to match direction
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle, [0, 0, 1], [center_x, center_y, height / 2]
        )
        wall_mesh.apply_transform(rotation_matrix)
        
        # Set visual properties
        wall_mesh.visual.face_colors = self.materials['wall']['color']
        
        return wall_mesh
    
    def create_floor_mesh(self, polygons: List[VectorPolygon]) -> trimesh.Trimesh:
        """
        Create floor mesh from room polygons.
        
        Args:
            polygons: List of room polygons
            
        Returns:
            Floor mesh
        """
        # Combine all room polygons
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        for polygon in polygons:
            if polygon.polygon_type == 'room' and len(polygon.points) >= 3:
                # Create vertices
                vertices = np.array([[p.x, p.y, 0] for p in polygon.points])
                all_vertices.append(vertices)
                
                # Triangulate polygon
                faces = self._triangulate_polygon(vertices)
                faces += vertex_offset
                all_faces.append(faces)
                
                vertex_offset += len(vertices)
        
        if not all_vertices:
            return trimesh.Trimesh()
        
        # Combine vertices and faces
        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.vstack(all_faces)
        
        # Create floor mesh
        floor_mesh = trimesh.Trimesh(
            vertices=combined_vertices,
            faces=combined_faces
        )
        
        # Extrude to create thickness
        floor_mesh = self._extrude_mesh(floor_mesh, self.floor_thickness)
        
        # Set visual properties
        floor_mesh.visual.face_colors = self.materials['floor']['color']
        
        return floor_mesh
    
    def create_ceiling_mesh(self, polygons: List[VectorPolygon]) -> trimesh.Trimesh:
        """
        Create ceiling mesh from room polygons.
        
        Args:
            polygons: List of room polygons
            
        Returns:
            Ceiling mesh
        """
        # Similar to floor but at ceiling height
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        for polygon in polygons:
            if polygon.polygon_type == 'room' and len(polygon.points) >= 3:
                # Create vertices at ceiling height
                vertices = np.array([[p.x, p.y, self.ceiling_height] for p in polygon.points])
                all_vertices.append(vertices)
                
                # Triangulate polygon (reverse order for correct normal)
                faces = self._triangulate_polygon(vertices)
                faces = faces[:, [0, 2, 1]]  # Reverse triangle order
                faces += vertex_offset
                all_faces.append(faces)
                
                vertex_offset += len(vertices)
        
        if not all_vertices:
            return trimesh.Trimesh()
        
        # Combine vertices and faces
        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.vstack(all_faces)
        
        # Create ceiling mesh
        ceiling_mesh = trimesh.Trimesh(
            vertices=combined_vertices,
            faces=combined_faces
        )
        
        # Set visual properties
        ceiling_mesh.visual.face_colors = self.materials['ceiling']['color']
        
        return ceiling_mesh
    
    def create_door_mesh(self, polygon: VectorPolygon, wall_height: float = None) -> trimesh.Trimesh:
        """
        Create a 3D door mesh from polygon.
        
        Args:
            polygon: Door polygon
            wall_height: Wall height for door sizing
            
        Returns:
            Door mesh
        """
        if wall_height is None:
            wall_height = self.wall_height
        
        if len(polygon.points) < 3:
            return trimesh.Trimesh()
        
        # Calculate door dimensions
        xs = [p.x for p in polygon.points]
        ys = [p.y for p in polygon.points]
        
        width = max(xs) - min(xs)
        depth = max(ys) - min(ys)
        height = wall_height * 0.8  # Door is 80% of wall height
        
        # Create door box
        door_mesh = trimesh.creation.box(
            extents=[width, depth, height]
        )
        
        # Position door
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        door_mesh.apply_translation([center_x, center_y, height / 2])
        
        # Set visual properties
        door_mesh.visual.face_colors = self.materials['door']['color']
        
        return door_mesh
    
    def create_window_mesh(self, polygon: VectorPolygon, wall_height: float = None) -> trimesh.Trimesh:
        """
        Create a 3D window mesh from polygon.
        
        Args:
            polygon: Window polygon
            wall_height: Wall height for window sizing
            
        Returns:
            Window mesh
        """
        if wall_height is None:
            wall_height = self.wall_height
        
        if len(polygon.points) < 3:
            return trimesh.Trimesh()
        
        # Calculate window dimensions
        xs = [p.x for p in polygon.points]
        ys = [p.y for p in polygon.points]
        
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        depth = 0.1  # Thin window
        
        # Create window box
        window_mesh = trimesh.creation.box(
            extents=[width, depth, height]
        )
        
        # Position window
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        window_mesh.apply_translation([center_x, center_y, wall_height * 0.6])
        
        # Set visual properties
        window_mesh.visual.face_colors = self.materials['window']['color']
        
        return window_mesh
    
    def create_room_volume_mesh(self, polygon: VectorPolygon, 
                               room_type: str = "room") -> trimesh.Trimesh:
        """
        Create a 3D room volume mesh from polygon.
        
        Args:
            polygon: Room polygon
            room_type: Type of room
            
        Returns:
            Room volume mesh
        """
        if len(polygon.points) < 3:
            return trimesh.Trimesh()
        
        # Create floor and ceiling vertices
        floor_vertices = np.array([[p.x, p.y, 0] for p in polygon.points])
        ceiling_vertices = np.array([[p.x, p.y, self.ceiling_height] for p in polygon.points])
        
        # Triangulate floor
        floor_faces = self._triangulate_polygon(floor_vertices)
        
        # Triangulate ceiling (reverse order for correct normal)
        ceiling_faces = self._triangulate_polygon(ceiling_vertices)
        ceiling_faces = ceiling_faces[:, [0, 2, 1]]  # Reverse triangle order
        
        # Create wall faces
        wall_faces = []
        num_points = len(polygon.points)
        
        for i in range(num_points):
            next_i = (i + 1) % num_points
            
            # Create two triangles for each wall segment
            # Triangle 1
            wall_faces.append([i, next_i, i + num_points])
            # Triangle 2
            wall_faces.append([next_i, next_i + num_points, i + num_points])
        
        # Combine all vertices and faces
        all_vertices = np.vstack([floor_vertices, ceiling_vertices])
        all_faces = np.vstack([
            floor_faces,
            ceiling_faces,
            np.array(wall_faces) + num_points  # Offset for ceiling vertices
        ])
        
        # Create mesh
        room_mesh = trimesh.Trimesh(
            vertices=all_vertices,
            faces=all_faces
        )
        
        # Set color based on room type
        room_colors = {
            'living_room': [200, 150, 100, 255],
            'bedroom': [150, 150, 200, 255],
            'kitchen': [200, 200, 150, 255],
            'bathroom': [150, 200, 200, 255],
            'default': [180, 180, 180, 255]
        }
        
        color = room_colors.get(room_type, room_colors['default'])
        room_mesh.visual.face_colors = color
        
        return room_mesh
    
    def _triangulate_polygon(self, vertices: np.ndarray) -> np.ndarray:
        """
        Triangulate a polygon using ear clipping algorithm.
        
        Args:
            vertices: Polygon vertices
            
        Returns:
            Triangle indices
        """
        if len(vertices) < 3:
            return np.array([])
        
        # Simple triangulation for convex polygons
        triangles = []
        for i in range(1, len(vertices) - 1):
            triangles.append([0, i, i + 1])
        
        return np.array(triangles)
    
    def _extrude_mesh(self, mesh: trimesh.Trimesh, height: float) -> trimesh.Trimesh:
        """
        Extrude a mesh to create thickness.
        
        Args:
            mesh: Input mesh
            height: Extrusion height
            
        Returns:
            Extruded mesh
        """
        # Get vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Create top vertices
        top_vertices = vertices.copy()
        top_vertices[:, 2] += height
        
        # Create side faces
        side_faces = []
        for i in range(len(vertices)):
            next_i = (i + 1) % len(vertices)
            
            # Create two triangles for each side
            side_faces.append([i, next_i, i + len(vertices)])
            side_faces.append([next_i, next_i + len(vertices), i + len(vertices)])
        
        # Combine vertices and faces
        all_vertices = np.vstack([vertices, top_vertices])
        all_faces = np.vstack([faces, np.array(side_faces) + len(vertices)])
        
        # Create new mesh
        extruded_mesh = trimesh.Trimesh(
            vertices=all_vertices,
            faces=all_faces
        )
        
        return extruded_mesh
    
    def combine_meshes(self, meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
        """
        Combine multiple meshes into one.
        
        Args:
            meshes: List of meshes to combine
            
        Returns:
            Combined mesh
        """
        if not meshes:
            return trimesh.Trimesh()
        
        # Filter out empty meshes
        valid_meshes = [mesh for mesh in meshes if len(mesh.vertices) > 0]
        
        if not valid_meshes:
            return trimesh.Trimesh()
        
        # Combine meshes
        combined_mesh = trimesh.util.concatenate(valid_meshes)
        
        return combined_mesh
    
    def optimize_mesh(self, mesh: trimesh.Trimesh, 
                     target_faces: int = 10000) -> trimesh.Trimesh:
        """
        Optimize mesh by reducing polygon count.
        
        Args:
            mesh: Input mesh
            target_faces: Target number of faces
            
        Returns:
            Optimized mesh
        """
        if len(mesh.faces) <= target_faces:
            return mesh
        
        # Simplify mesh
        simplified_mesh = mesh.simplify_quadric_decimation(target_faces)
        
        return simplified_mesh
    
    def repair_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Repair mesh by fixing common issues.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Repaired mesh
        """
        # Fill holes
        mesh.fill_holes()
        
        # Remove degenerate faces
        mesh.remove_degenerate_faces()
        
        # Remove duplicate faces
        mesh.remove_duplicate_faces()
        
        # Remove unused vertices
        mesh.remove_unreferenced_vertices()
        
        return mesh
    
    def create_floor_plan_2d(self, vector_data: VectorData) -> trimesh.Trimesh:
        """
        Create a 2D floor plan mesh for visualization.
        
        Args:
            vector_data: VectorData object
            
        Returns:
            2D floor plan mesh
        """
        # Create line segments
        line_segments = []
        
        for line in vector_data.lines:
            line_segments.append([
                [line.start.x, line.start.y, 0],
                [line.end.x, line.end.y, 0]
            ])
        
        # Create 2D mesh from line segments
        if line_segments:
            # Convert to trimesh line set
            line_set = trimesh.load_path(line_segments)
            return line_set
        else:
            return trimesh.Trimesh()
    
    def export_mesh(self, mesh: trimesh.Trimesh, 
                   output_path: str, format: str = 'obj') -> None:
        """
        Export mesh to file.
        
        Args:
            mesh: Mesh to export
            output_path: Output file path
            format: Export format ('obj', 'ply', 'stl', 'gltf')
        """
        if format == 'obj':
            mesh.export(output_path)
        elif format == 'ply':
            mesh.export(output_path)
        elif format == 'stl':
            mesh.export(output_path)
        elif format == 'gltf':
            mesh.export(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def visualize_mesh(self, mesh: trimesh.Trimesh, 
                      show_wireframe: bool = False) -> None:
        """
        Visualize mesh using matplotlib.
        
        Args:
            mesh: Mesh to visualize
            show_wireframe: Whether to show wireframe
        """
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot mesh
        if len(mesh.vertices) > 0:
            ax.plot_trisurf(
                mesh.vertices[:, 0],
                mesh.vertices[:, 1],
                mesh.vertices[:, 2],
                triangles=mesh.faces,
                alpha=0.7,
                edgecolor='black' if show_wireframe else 'none'
            )
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Floor Plan Model')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        plt.show()
    
    def get_mesh_statistics(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """
        Get statistics about the mesh.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Dictionary with mesh statistics
        """
        return {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'volume': mesh.volume,
            'surface_area': mesh.surface_area,
            'bounds': mesh.bounds.tolist(),
            'is_watertight': mesh.is_watertight,
            'is_winding_consistent': mesh.is_winding_consistent
        }