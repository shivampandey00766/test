"""
Open3D-based 3D reconstruction for architectural models.

This module provides functionality to create 3D models from 2D floor plans
using Open3D for point clouds, meshes, and visualization.
"""

import open3d as o3d
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from ..vectorization.raster_to_vector import VectorLine, VectorPolygon, VectorPoint, VectorData


class Open3DReconstructor:
    """
    Creates 3D architectural models using Open3D.
    
    This class provides methods to convert 2D floor plan data
    into 3D point clouds, meshes, and visualizations.
    """
    
    def __init__(self, 
                 wall_height: float = 2.4,
                 floor_thickness: float = 0.2,
                 ceiling_height: float = 2.7):
        """
        Initialize the Open3DReconstructor.
        
        Args:
            wall_height: Standard wall height in meters
            floor_thickness: Floor thickness in meters
            ceiling_height: Ceiling height in meters
        """
        self.wall_height = wall_height
        self.floor_thickness = floor_thickness
        self.ceiling_height = ceiling_height
        
        # Color scheme for different elements
        self.colors = {
            'wall': [0.8, 0.8, 0.8],
            'floor': [0.6, 0.4, 0.2],
            'ceiling': [0.9, 0.9, 0.9],
            'door': [0.4, 0.2, 0.1],
            'window': [0.7, 0.8, 1.0],
            'room': [0.8, 0.6, 0.4]
        }
    
    def create_wall_mesh(self, start_point: VectorPoint, end_point: VectorPoint, 
                        height: float = None, thickness: float = 0.2) -> o3d.geometry.TriangleMesh:
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
        
        # Calculate wall dimensions and position
        length = np.sqrt((end_point.x - start_point.x)**2 + (end_point.y - start_point.y)**2)
        center_x = (start_point.x + end_point.x) / 2
        center_y = (start_point.y + end_point.y) / 2
        angle = np.arctan2(end_point.y - start_point.y, end_point.x - start_point.x)
        
        # Create wall box
        wall_mesh = o3d.geometry.TriangleMesh.create_box(
            width=length,
            height=thickness,
            depth=height
        )
        
        # Position wall
        wall_mesh.translate([center_x, center_y, height / 2])
        
        # Rotate wall to match direction
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        wall_mesh.rotate(rotation_matrix, center=[center_x, center_y, height / 2])
        
        # Set color
        wall_mesh.paint_uniform_color(self.colors['wall'])
        
        return wall_mesh
    
    def create_floor_mesh(self, polygons: List[VectorPolygon]) -> o3d.geometry.TriangleMesh:
        """
        Create floor mesh from room polygons.
        
        Args:
            polygons: List of room polygons
            
        Returns:
            Floor mesh
        """
        # Combine all room polygons
        all_vertices = []
        all_triangles = []
        vertex_offset = 0
        
        for polygon in polygons:
            if polygon.polygon_type == 'room' and len(polygon.points) >= 3:
                # Create vertices
                vertices = np.array([[p.x, p.y, 0] for p in polygon.points])
                all_vertices.append(vertices)
                
                # Create triangles using ear clipping
                triangles = self._triangulate_polygon(vertices)
                triangles += vertex_offset
                all_triangles.append(triangles)
                
                vertex_offset += len(vertices)
        
        if not all_vertices:
            return o3d.geometry.TriangleMesh()
        
        # Combine vertices and triangles
        combined_vertices = np.vstack(all_vertices)
        combined_triangles = np.vstack(all_triangles)
        
        # Create mesh
        floor_mesh = o3d.geometry.TriangleMesh()
        floor_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
        floor_mesh.triangles = o3d.utility.Vector3iVector(combined_triangles)
        
        # Extrude to create thickness
        floor_mesh = self._extrude_mesh(floor_mesh, self.floor_thickness)
        
        # Set color
        floor_mesh.paint_uniform_color(self.colors['floor'])
        
        return floor_mesh
    
    def create_ceiling_mesh(self, polygons: List[VectorPolygon]) -> o3d.geometry.TriangleMesh:
        """
        Create ceiling mesh from room polygons.
        
        Args:
            polygons: List of room polygons
            
        Returns:
            Ceiling mesh
        """
        # Similar to floor but at ceiling height
        all_vertices = []
        all_triangles = []
        vertex_offset = 0
        
        for polygon in polygons:
            if polygon.polygon_type == 'room' and len(polygon.points) >= 3:
                # Create vertices at ceiling height
                vertices = np.array([[p.x, p.y, self.ceiling_height] for p in polygon.points])
                all_vertices.append(vertices)
                
                # Create triangles
                triangles = self._triangulate_polygon(vertices)
                triangles += vertex_offset
                all_triangles.append(triangles)
                
                vertex_offset += len(vertices)
        
        if not all_vertices:
            return o3d.geometry.TriangleMesh()
        
        # Combine vertices and triangles
        combined_vertices = np.vstack(all_vertices)
        combined_triangles = np.vstack(all_triangles)
        
        # Create mesh
        ceiling_mesh = o3d.geometry.TriangleMesh()
        ceiling_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
        ceiling_mesh.triangles = o3d.utility.Vector3iVector(combined_triangles)
        
        # Set color
        ceiling_mesh.paint_uniform_color(self.colors['ceiling'])
        
        return ceiling_mesh
    
    def create_door_mesh(self, polygon: VectorPolygon, wall_height: float = None) -> o3d.geometry.TriangleMesh:
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
            return o3d.geometry.TriangleMesh()
        
        # Calculate door dimensions
        xs = [p.x for p in polygon.points]
        ys = [p.y for p in polygon.points]
        
        width = max(xs) - min(xs)
        depth = max(ys) - min(ys)
        height = wall_height * 0.8  # Door is 80% of wall height
        
        # Create door box
        door_mesh = o3d.geometry.TriangleMesh.create_box(
            width=width,
            height=depth,
            depth=height
        )
        
        # Position door
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        door_mesh.translate([center_x, center_y, height / 2])
        
        # Set color
        door_mesh.paint_uniform_color(self.colors['door'])
        
        return door_mesh
    
    def create_window_mesh(self, polygon: VectorPolygon, wall_height: float = None) -> o3d.geometry.TriangleMesh:
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
            return o3d.geometry.TriangleMesh()
        
        # Calculate window dimensions
        xs = [p.x for p in polygon.points]
        ys = [p.y for p in polygon.points]
        
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        depth = 0.1  # Thin window
        
        # Create window box
        window_mesh = o3d.geometry.TriangleMesh.create_box(
            width=width,
            height=depth,
            depth=height
        )
        
        # Position window
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        window_mesh.translate([center_x, center_y, wall_height * 0.6])
        
        # Set color
        window_mesh.paint_uniform_color(self.colors['window'])
        
        return window_mesh
    
    def create_room_volume_mesh(self, polygon: VectorPolygon, 
                               room_type: str = "room") -> o3d.geometry.TriangleMesh:
        """
        Create a 3D room volume mesh from polygon.
        
        Args:
            polygon: Room polygon
            room_type: Type of room
            
        Returns:
            Room volume mesh
        """
        if len(polygon.points) < 3:
            return o3d.geometry.TriangleMesh()
        
        # Create floor vertices
        floor_vertices = np.array([[p.x, p.y, 0] for p in polygon.points])
        ceiling_vertices = np.array([[p.x, p.y, self.ceiling_height] for p in polygon.points])
        
        # Triangulate floor
        floor_triangles = self._triangulate_polygon(floor_vertices)
        
        # Triangulate ceiling (reverse order for correct normal)
        ceiling_triangles = self._triangulate_polygon(ceiling_vertices)
        ceiling_triangles = ceiling_triangles[:, [0, 2, 1]]  # Reverse triangle order
        
        # Create wall triangles
        wall_triangles = []
        num_points = len(polygon.points)
        
        for i in range(num_points):
            next_i = (i + 1) % num_points
            
            # Create two triangles for each wall segment
            # Triangle 1
            wall_triangles.append([i, next_i, i + num_points])
            # Triangle 2
            wall_triangles.append([next_i, next_i + num_points, i + num_points])
        
        # Combine all vertices and triangles
        all_vertices = np.vstack([floor_vertices, ceiling_vertices])
        all_triangles = np.vstack([
            floor_triangles,
            ceiling_triangles,
            np.array(wall_triangles) + num_points  # Offset for ceiling vertices
        ])
        
        # Create mesh
        room_mesh = o3d.geometry.TriangleMesh()
        room_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
        room_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
        
        # Set color based on room type
        room_colors = {
            'living_room': [0.8, 0.6, 0.4],
            'bedroom': [0.6, 0.6, 0.8],
            'kitchen': [0.8, 0.8, 0.6],
            'bathroom': [0.6, 0.8, 0.8],
            'default': [0.7, 0.7, 0.7]
        }
        
        color = room_colors.get(room_type, room_colors['default'])
        room_mesh.paint_uniform_color(color)
        
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
    
    def _extrude_mesh(self, mesh: o3d.geometry.TriangleMesh, 
                     height: float) -> o3d.geometry.TriangleMesh:
        """
        Extrude a mesh to create thickness.
        
        Args:
            mesh: Input mesh
            height: Extrusion height
            
        Returns:
            Extruded mesh
        """
        # Get vertices and triangles
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # Create top vertices
        top_vertices = vertices.copy()
        top_vertices[:, 2] += height
        
        # Create side triangles
        side_triangles = []
        for i in range(len(vertices)):
            next_i = (i + 1) % len(vertices)
            
            # Create two triangles for each side
            side_triangles.append([i, next_i, i + len(vertices)])
            side_triangles.append([next_i, next_i + len(vertices), i + len(vertices)])
        
        # Combine vertices and triangles
        all_vertices = np.vstack([vertices, top_vertices])
        all_triangles = np.vstack([triangles, np.array(side_triangles) + len(vertices)])
        
        # Create new mesh
        extruded_mesh = o3d.geometry.TriangleMesh()
        extruded_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
        extruded_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
        
        return extruded_mesh
    
    def reconstruct_from_vector_data(self, vector_data: VectorData, 
                                   room_types: List[str] = None) -> o3d.geometry.TriangleMesh:
        """
        Reconstruct 3D model from vector data.
        
        Args:
            vector_data: VectorData object
            room_types: List of room types for each room
            
        Returns:
            Combined 3D mesh
        """
        meshes = []
        
        # Create walls from lines
        for line in vector_data.lines:
            if line.line_type == 'wall':
                wall_mesh = self.create_wall_mesh(line.start, line.end)
                meshes.append(wall_mesh)
        
        # Create doors from polygons
        door_polygons = [p for p in vector_data.polygons if p.polygon_type == 'door']
        for polygon in door_polygons:
            door_mesh = self.create_door_mesh(polygon)
            meshes.append(door_mesh)
        
        # Create windows from polygons
        window_polygons = [p for p in vector_data.polygons if p.polygon_type == 'window']
        for polygon in window_polygons:
            window_mesh = self.create_window_mesh(polygon)
            meshes.append(window_mesh)
        
        # Create rooms
        room_polygons = [p for p in vector_data.polygons if p.polygon_type == 'room']
        for i, polygon in enumerate(room_polygons):
            room_type = room_types[i] if room_types and i < len(room_types) else 'room'
            room_mesh = self.create_room_volume_mesh(polygon, room_type)
            meshes.append(room_mesh)
        
        # Create floor and ceiling
        if room_polygons:
            floor_mesh = self.create_floor_mesh(room_polygons)
            ceiling_mesh = self.create_ceiling_mesh(room_polygons)
            meshes.extend([floor_mesh, ceiling_mesh])
        
        # Combine all meshes
        if meshes:
            combined_mesh = meshes[0]
            for mesh in meshes[1:]:
                combined_mesh += mesh
            return combined_mesh
        else:
            return o3d.geometry.TriangleMesh()
    
    def create_point_cloud(self, vector_data: VectorData, 
                          density: float = 0.1) -> o3d.geometry.PointCloud:
        """
        Create a point cloud from vector data.
        
        Args:
            vector_data: VectorData object
            density: Point density (points per unit)
            
        Returns:
            Point cloud
        """
        points = []
        colors = []
        
        # Sample points from lines
        for line in vector_data.lines:
            if line.line_type == 'wall':
                # Sample points along the line
                num_points = int(np.sqrt((line.end.x - line.start.x)**2 + 
                                       (line.end.y - line.start.y)**2) * density)
                for i in range(num_points + 1):
                    t = i / max(1, num_points)
                    x = line.start.x + t * (line.end.x - line.start.x)
                    y = line.start.y + t * (line.end.y - line.start.y)
                    z = np.random.uniform(0, self.wall_height)
                    points.append([x, y, z])
                    colors.append(self.colors['wall'])
        
        # Sample points from polygons
        for polygon in vector_data.polygons:
            if polygon.polygon_type == 'room':
                # Sample points within the polygon
                # This is a simplified approach - in practice, you'd use proper polygon sampling
                for point in polygon.points:
                    z = np.random.uniform(0, self.ceiling_height)
                    points.append([point.x, point.y, z])
                    colors.append(self.colors['room'])
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        if points:
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return pcd
    
    def visualize_model(self, mesh: o3d.geometry.TriangleMesh, 
                       show_wireframe: bool = False) -> None:
        """
        Visualize the 3D model.
        
        Args:
            mesh: 3D mesh to visualize
            show_wireframe: Whether to show wireframe
        """
        # Create visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh)
        
        # Set render options
        render_option = vis.get_render_option()
        render_option.show_coordinate_frame = True
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        
        if show_wireframe:
            render_option.mesh_show_wireframe = True
        
        # Run visualization
        vis.run()
        vis.destroy_window()
    
    def save_model(self, mesh: o3d.geometry.TriangleMesh, 
                  output_path: str, format: str = 'obj') -> None:
        """
        Save 3D model to file.
        
        Args:
            mesh: 3D mesh to save
            output_path: Output file path
            format: Export format ('obj', 'ply', 'stl', 'gltf')
        """
        if format == 'obj':
            o3d.io.write_triangle_mesh(output_path, mesh)
        elif format == 'ply':
            o3d.io.write_triangle_mesh(output_path, mesh)
        elif format == 'stl':
            o3d.io.write_triangle_mesh(output_path, mesh)
        elif format == 'gltf':
            o3d.io.write_triangle_mesh(output_path, mesh)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def create_floor_plan_3d(self, vector_data: VectorData) -> o3d.geometry.LineSet:
        """
        Create a 3D line set representing the floor plan.
        
        Args:
            vector_data: VectorData object
            
        Returns:
            3D line set
        """
        points = []
        lines = []
        colors = []
        
        # Add line endpoints
        point_map = {}
        point_index = 0
        
        for line in vector_data.lines:
            # Add start point
            start_key = (line.start.x, line.start.y, line.start.z)
            if start_key not in point_map:
                point_map[start_key] = point_index
                points.append([line.start.x, line.start.y, line.start.z])
                point_index += 1
            
            # Add end point
            end_key = (line.end.x, line.end.y, line.end.z)
            if end_key not in point_map:
                point_map[end_key] = point_index
                points.append([line.end.x, line.end.y, line.end.z])
                point_index += 1
            
            # Add line
            lines.append([point_map[start_key], point_map[end_key]])
            
            # Add color based on line type
            if line.line_type == 'wall':
                colors.append(self.colors['wall'])
            elif line.line_type == 'door':
                colors.append(self.colors['door'])
            elif line.line_type == 'window':
                colors.append(self.colors['window'])
            else:
                colors.append([0.5, 0.5, 0.5])
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(points))
        line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
        line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return line_set