"""
Mesh Generator Module

This module provides specialized mesh generation functions for different
architectural elements with proper materials, textures, and optimization.
"""

import numpy as np
import trimesh
from typing import Dict, List, Tuple, Optional, Any
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MaterialType(Enum):
    """Enumeration of material types."""
    WALL_INTERIOR = "wall_interior"
    WALL_EXTERIOR = "wall_exterior"
    DOOR_WOOD = "door_wood"
    DOOR_GLASS = "door_glass"
    WINDOW_FRAME = "window_frame"
    WINDOW_GLASS = "window_glass"
    FLOOR_WOOD = "floor_wood"
    FLOOR_TILE = "floor_tile"
    FLOOR_CARPET = "floor_carpet"
    CEILING = "ceiling"
    FIXTURE_METAL = "fixture_metal"
    FIXTURE_CERAMIC = "fixture_ceramic"
    FURNITURE_WOOD = "furniture_wood"


class MeshGenerator:
    """
    Specialized mesh generator for architectural elements.
    """
    
    def __init__(self):
        """Initialize the mesh generator."""
        self.materials = self._create_material_library()
        self.texture_scale = 1.0  # Scale factor for texture coordinates
        
    def _create_material_library(self) -> Dict[MaterialType, Dict]:
        """Create a library of architectural materials."""
        materials = {
            MaterialType.WALL_INTERIOR: {
                'name': 'Interior Wall',
                'color': [0.95, 0.95, 0.95, 1.0],
                'roughness': 0.8,
                'metallic': 0.0,
                'specular': 0.1
            },
            MaterialType.WALL_EXTERIOR: {
                'name': 'Exterior Wall',
                'color': [0.85, 0.85, 0.80, 1.0],
                'roughness': 0.9,
                'metallic': 0.0,
                'specular': 0.05
            },
            MaterialType.DOOR_WOOD: {
                'name': 'Wood Door',
                'color': [0.6, 0.4, 0.2, 1.0],
                'roughness': 0.7,
                'metallic': 0.0,
                'specular': 0.3
            },
            MaterialType.DOOR_GLASS: {
                'name': 'Glass Door',
                'color': [0.8, 0.9, 1.0, 0.3],
                'roughness': 0.0,
                'metallic': 0.0,
                'specular': 0.9,
                'transmission': 0.9
            },
            MaterialType.WINDOW_FRAME: {
                'name': 'Window Frame',
                'color': [0.9, 0.9, 0.9, 1.0],
                'roughness': 0.4,
                'metallic': 0.1,
                'specular': 0.5
            },
            MaterialType.WINDOW_GLASS: {
                'name': 'Window Glass',
                'color': [0.7, 0.9, 1.0, 0.2],
                'roughness': 0.0,
                'metallic': 0.0,
                'specular': 0.95,
                'transmission': 0.95
            },
            MaterialType.FLOOR_WOOD: {
                'name': 'Wood Floor',
                'color': [0.7, 0.5, 0.3, 1.0],
                'roughness': 0.6,
                'metallic': 0.0,
                'specular': 0.4
            },
            MaterialType.FLOOR_TILE: {
                'name': 'Tile Floor',
                'color': [0.8, 0.8, 0.75, 1.0],
                'roughness': 0.2,
                'metallic': 0.0,
                'specular': 0.8
            },
            MaterialType.FLOOR_CARPET: {
                'name': 'Carpet Floor',
                'color': [0.6, 0.5, 0.4, 1.0],
                'roughness': 0.95,
                'metallic': 0.0,
                'specular': 0.0
            },
            MaterialType.CEILING: {
                'name': 'Ceiling',
                'color': [1.0, 1.0, 1.0, 1.0],
                'roughness': 0.9,
                'metallic': 0.0,
                'specular': 0.1
            },
            MaterialType.FIXTURE_METAL: {
                'name': 'Metal Fixture',
                'color': [0.7, 0.7, 0.7, 1.0],
                'roughness': 0.3,
                'metallic': 0.8,
                'specular': 0.9
            },
            MaterialType.FIXTURE_CERAMIC: {
                'name': 'Ceramic Fixture',
                'color': [1.0, 1.0, 1.0, 1.0],
                'roughness': 0.1,
                'metallic': 0.0,
                'specular': 0.9
            },
            MaterialType.FURNITURE_WOOD: {
                'name': 'Wood Furniture',
                'color': [0.5, 0.35, 0.2, 1.0],
                'roughness': 0.5,
                'metallic': 0.0,
                'specular': 0.6
            }
        }
        
        return materials
    
    def generate_wall_mesh(self, start: Tuple[float, float], end: Tuple[float, float],
                          thickness: float, height: float, 
                          wall_type: str = 'interior') -> trimesh.Trimesh:
        """
        Generate a detailed wall mesh with proper materials.
        
        Args:
            start: Wall start point (x, y)
            end: Wall end point (x, y)
            thickness: Wall thickness
            height: Wall height
            wall_type: 'interior' or 'exterior'
            
        Returns:
            Wall mesh with materials
        """
        # Calculate wall geometry
        direction = np.array([end[0] - start[0], end[1] - start[1]])
        length = np.linalg.norm(direction)
        
        if length < 0.01:
            return trimesh.Trimesh()
        
        direction = direction / length
        perpendicular = np.array([-direction[1], direction[0]]) * thickness / 2
        
        # Create vertices
        vertices = [
            # Bottom face
            [start[0] - perpendicular[0], start[1] - perpendicular[1], 0],
            [start[0] + perpendicular[0], start[1] + perpendicular[1], 0],
            [end[0] + perpendicular[0], end[1] + perpendicular[1], 0],
            [end[0] - perpendicular[0], end[1] - perpendicular[1], 0],
            # Top face
            [start[0] - perpendicular[0], start[1] - perpendicular[1], height],
            [start[0] + perpendicular[0], start[1] + perpendicular[1], height],
            [end[0] + perpendicular[0], end[1] + perpendicular[1], height],
            [end[0] - perpendicular[0], end[1] - perpendicular[1], height]
        ]
        
        # Create faces with proper normals
        faces = [
            # Bottom (facing down)
            [0, 3, 2], [0, 2, 1],
            # Top (facing up)
            [4, 5, 6], [4, 6, 7],
            # Sides
            [0, 1, 5], [0, 5, 4],  # Front
            [2, 3, 7], [2, 7, 6],  # Back
            [1, 2, 6], [1, 6, 5],  # Right
            [3, 0, 4], [3, 4, 7]   # Left
        ]
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Add texture coordinates
        mesh = self._add_wall_texture_coordinates(mesh, length, height, thickness)
        
        # Apply material
        material_type = MaterialType.WALL_EXTERIOR if wall_type == 'exterior' else MaterialType.WALL_INTERIOR
        mesh.visual.material = self._create_pbr_material(material_type)
        
        return mesh
    
    def generate_door_mesh(self, center: Tuple[float, float], width: float, height: float,
                          thickness: float, door_type: str = 'wood') -> trimesh.Trimesh:
        """
        Generate a detailed door mesh.
        
        Args:
            center: Door center point
            width: Door width
            height: Door height
            thickness: Door thickness
            door_type: 'wood', 'glass', or 'metal'
            
        Returns:
            Door mesh with materials
        """
        half_width = width / 2
        half_thickness = thickness / 2
        
        # Door panel vertices
        vertices = [
            # Front face
            [center[0] - half_width, center[1] - half_thickness, 0],
            [center[0] + half_width, center[1] - half_thickness, 0],
            [center[0] + half_width, center[1] - half_thickness, height],
            [center[0] - half_width, center[1] - half_thickness, height],
            # Back face
            [center[0] - half_width, center[1] + half_thickness, 0],
            [center[0] + half_width, center[1] + half_thickness, 0],
            [center[0] + half_width, center[1] + half_thickness, height],
            [center[0] - half_width, center[1] + half_thickness, height]
        ]
        
        faces = [
            # Front face
            [0, 1, 2], [0, 2, 3],
            # Back face
            [4, 7, 6], [4, 6, 5],
            # Top
            [3, 2, 6], [3, 6, 7],
            # Bottom
            [0, 4, 5], [0, 5, 1],
            # Sides
            [1, 5, 6], [1, 6, 2],  # Right
            [4, 0, 3], [4, 3, 7]   # Left
        ]
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Add door details (handle, panels, etc.)
        mesh = self._add_door_details(mesh, width, height, thickness, door_type)
        
        # Apply material
        if door_type == 'glass':
            material_type = MaterialType.DOOR_GLASS
        else:
            material_type = MaterialType.DOOR_WOOD
        
        mesh.visual.material = self._create_pbr_material(material_type)
        
        return mesh
    
    def generate_window_mesh(self, center: Tuple[float, float], width: float, height: float,
                           sill_height: float, frame_thickness: float = 0.05) -> trimesh.Trimesh:
        """
        Generate a detailed window mesh with frame and glass.
        
        Args:
            center: Window center point
            width: Window width
            height: Window height
            sill_height: Height from floor to window sill
            frame_thickness: Frame thickness
            
        Returns:
            Window mesh with materials
        """
        meshes = []
        
        # Window frame
        frame_mesh = self._create_window_frame(center, width, height, sill_height, frame_thickness)
        if frame_mesh:
            frame_mesh.visual.material = self._create_pbr_material(MaterialType.WINDOW_FRAME)
            meshes.append(frame_mesh)
        
        # Window glass
        glass_mesh = self._create_window_glass(center, width, height, sill_height, frame_thickness)
        if glass_mesh:
            glass_mesh.visual.material = self._create_pbr_material(MaterialType.WINDOW_GLASS)
            meshes.append(glass_mesh)
        
        if meshes:
            return trimesh.util.concatenate(meshes)
        else:
            return trimesh.Trimesh()
    
    def generate_floor_mesh(self, polygon_coords: List[Tuple[float, float]], 
                          thickness: float = 0.1, floor_type: str = 'wood') -> trimesh.Trimesh:
        """
        Generate a floor mesh from polygon coordinates.
        
        Args:
            polygon_coords: Floor boundary coordinates
            thickness: Floor thickness
            floor_type: 'wood', 'tile', or 'carpet'
            
        Returns:
            Floor mesh with materials
        """
        if len(polygon_coords) < 3:
            return trimesh.Trimesh()
        
        # Create floor polygon
        from shapely.geometry import Polygon
        floor_polygon = Polygon(polygon_coords)
        
        if not floor_polygon.is_valid:
            return trimesh.Trimesh()
        
        # Extrude polygon to create floor
        from .geometry_builder import GeometryBuilder
        builder = GeometryBuilder()
        mesh = builder.create_extruded_polygon(floor_polygon, thickness, 0)
        
        # Add texture coordinates
        mesh = self._add_floor_texture_coordinates(mesh, floor_polygon.bounds)
        
        # Apply material
        if floor_type == 'tile':
            material_type = MaterialType.FLOOR_TILE
        elif floor_type == 'carpet':
            material_type = MaterialType.FLOOR_CARPET
        else:
            material_type = MaterialType.FLOOR_WOOD
        
        mesh.visual.material = self._create_pbr_material(material_type)
        
        return mesh
    
    def generate_fixture_mesh(self, fixture_type: str, center: Tuple[float, float],
                            dimensions: Dict[str, float]) -> trimesh.Trimesh:
        """
        Generate fixture meshes based on type.
        
        Args:
            fixture_type: Type of fixture ('sink', 'toilet', 'bathtub', etc.)
            center: Fixture center point
            dimensions: Fixture dimensions
            
        Returns:
            Fixture mesh with materials
        """
        if fixture_type == 'sink':
            return self._create_sink_mesh(center, dimensions)
        elif fixture_type == 'toilet':
            return self._create_toilet_mesh(center, dimensions)
        elif fixture_type == 'bathtub':
            return self._create_bathtub_mesh(center, dimensions)
        elif fixture_type == 'counter':
            return self._create_counter_mesh(center, dimensions)
        elif fixture_type == 'cabinet':
            return self._create_cabinet_mesh(center, dimensions)
        else:
            return self._create_generic_fixture(center, dimensions)
    
    def _add_wall_texture_coordinates(self, mesh: trimesh.Trimesh, length: float, 
                                    height: float, thickness: float) -> trimesh.Trimesh:
        """Add texture coordinates to wall mesh."""
        # Calculate UV coordinates based on face positions
        uv_coordinates = []
        
        # This is a simplified UV mapping - in practice, you'd want more sophisticated mapping
        for vertex in mesh.vertices:
            u = (vertex[0] % 1.0) * self.texture_scale
            v = (vertex[2] / height) * self.texture_scale
            uv_coordinates.append([u, v])
        
        mesh.visual.uv = np.array(uv_coordinates)
        return mesh
    
    def _add_floor_texture_coordinates(self, mesh: trimesh.Trimesh, bounds: Tuple) -> trimesh.Trimesh:
        """Add texture coordinates to floor mesh."""
        min_x, min_y, max_x, max_y = bounds
        width = max_x - min_x
        height = max_y - min_y
        
        uv_coordinates = []
        for vertex in mesh.vertices:
            u = ((vertex[0] - min_x) / width) * self.texture_scale
            v = ((vertex[1] - min_y) / height) * self.texture_scale
            uv_coordinates.append([u, v])
        
        mesh.visual.uv = np.array(uv_coordinates)
        return mesh
    
    def _add_door_details(self, mesh: trimesh.Trimesh, width: float, height: float,
                         thickness: float, door_type: str) -> trimesh.Trimesh:
        """Add details like handles and panels to door mesh."""
        # This is a simplified version - in practice, you'd add actual geometry for handles, panels, etc.
        
        if door_type == 'wood':
            # Could add raised panels, handle geometry, etc.
            pass
        elif door_type == 'glass':
            # Could add frame divisions, handle geometry, etc.
            pass
        
        return mesh
    
    def _create_window_frame(self, center: Tuple[float, float], width: float, height: float,
                           sill_height: float, thickness: float) -> trimesh.Trimesh:
        """Create window frame mesh."""
        frame_width = thickness
        
        # Create frame as hollow rectangle
        outer_width = width + 2 * frame_width
        outer_height = height + 2 * frame_width
        
        # Outer frame vertices
        half_outer_w = outer_width / 2
        half_outer_h = outer_height / 2
        half_inner_w = width / 2
        half_inner_h = height / 2
        
        vertices = []
        faces = []
        
        # Bottom frame
        bottom_vertices = [
            [center[0] - half_outer_w, center[1], sill_height - frame_width],
            [center[0] + half_outer_w, center[1], sill_height - frame_width],
            [center[0] + half_outer_w, center[1], sill_height],
            [center[0] - half_outer_w, center[1], sill_height],
            [center[0] - half_inner_w, center[1], sill_height - frame_width],
            [center[0] + half_inner_w, center[1], sill_height - frame_width],
            [center[0] + half_inner_w, center[1], sill_height],
            [center[0] - half_inner_w, center[1], sill_height]
        ]
        
        vertices.extend(bottom_vertices)
        
        # Add faces for bottom frame (simplified)
        base_idx = 0
        faces.extend([
            [base_idx, base_idx+1, base_idx+2], [base_idx, base_idx+2, base_idx+3],
            [base_idx+4, base_idx+7, base_idx+6], [base_idx+4, base_idx+6, base_idx+5]
        ])
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def _create_window_glass(self, center: Tuple[float, float], width: float, height: float,
                           sill_height: float, frame_thickness: float) -> trimesh.Trimesh:
        """Create window glass mesh."""
        glass_thickness = 0.005  # 5mm glass
        half_width = width / 2
        half_height = height / 2
        half_thickness = glass_thickness / 2
        
        vertices = [
            [center[0] - half_width, center[1] - half_thickness, sill_height],
            [center[0] + half_width, center[1] - half_thickness, sill_height],
            [center[0] + half_width, center[1] - half_thickness, sill_height + height],
            [center[0] - half_width, center[1] - half_thickness, sill_height + height],
            [center[0] - half_width, center[1] + half_thickness, sill_height],
            [center[0] + half_width, center[1] + half_thickness, sill_height],
            [center[0] + half_width, center[1] + half_thickness, sill_height + height],
            [center[0] - half_width, center[1] + half_thickness, sill_height + height]
        ]
        
        faces = [
            [0, 1, 2], [0, 2, 3],  # Front
            [4, 7, 6], [4, 6, 5],  # Back
            [3, 2, 6], [3, 6, 7],  # Top
            [0, 4, 5], [0, 5, 1],  # Bottom
            [1, 5, 6], [1, 6, 2],  # Right
            [4, 0, 3], [4, 3, 7]   # Left
        ]
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def _create_sink_mesh(self, center: Tuple[float, float], dimensions: Dict) -> trimesh.Trimesh:
        """Create a sink mesh."""
        width = dimensions.get('width', 0.6)
        depth = dimensions.get('depth', 0.4)
        height = dimensions.get('height', 0.15)
        
        # Create simple rectangular sink basin
        sink_mesh = trimesh.creation.box(extents=[width, depth, height])
        
        # Position at center
        transform = trimesh.transformations.translation_matrix([center[0], center[1], height/2])
        sink_mesh.apply_transform(transform)
        
        # Apply ceramic material
        sink_mesh.visual.material = self._create_pbr_material(MaterialType.FIXTURE_CERAMIC)
        
        return sink_mesh
    
    def _create_toilet_mesh(self, center: Tuple[float, float], dimensions: Dict) -> trimesh.Trimesh:
        """Create a toilet mesh."""
        width = dimensions.get('width', 0.4)
        depth = dimensions.get('depth', 0.6)
        height = dimensions.get('height', 0.8)
        
        # Create simplified toilet shape (bowl + tank)
        bowl = trimesh.creation.cylinder(radius=width/2, height=0.4)
        tank = trimesh.creation.box(extents=[width*0.8, depth*0.3, 0.4])
        
        # Position components
        bowl_transform = trimesh.transformations.translation_matrix([center[0], center[1], 0.2])
        bowl.apply_transform(bowl_transform)
        
        tank_transform = trimesh.transformations.translation_matrix([center[0], center[1] - depth*0.35, 0.6])
        tank.apply_transform(tank_transform)
        
        # Combine
        toilet_mesh = trimesh.util.concatenate([bowl, tank])
        toilet_mesh.visual.material = self._create_pbr_material(MaterialType.FIXTURE_CERAMIC)
        
        return toilet_mesh
    
    def _create_bathtub_mesh(self, center: Tuple[float, float], dimensions: Dict) -> trimesh.Trimesh:
        """Create a bathtub mesh."""
        width = dimensions.get('width', 0.8)
        depth = dimensions.get('depth', 1.6)
        height = dimensions.get('height', 0.6)
        
        # Create bathtub as rounded rectangle
        bathtub_mesh = trimesh.creation.box(extents=[width, depth, height])
        
        # Position at center
        transform = trimesh.transformations.translation_matrix([center[0], center[1], height/2])
        bathtub_mesh.apply_transform(transform)
        
        # Apply ceramic material
        bathtub_mesh.visual.material = self._create_pbr_material(MaterialType.FIXTURE_CERAMIC)
        
        return bathtub_mesh
    
    def _create_counter_mesh(self, center: Tuple[float, float], dimensions: Dict) -> trimesh.Trimesh:
        """Create a counter mesh."""
        width = dimensions.get('width', 2.0)
        depth = dimensions.get('depth', 0.6)
        height = dimensions.get('height', 0.05)  # Counter top thickness
        counter_height = dimensions.get('counter_height', 0.9)
        
        # Counter top
        counter_top = trimesh.creation.box(extents=[width, depth, height])
        transform = trimesh.transformations.translation_matrix(
            [center[0], center[1], counter_height + height/2]
        )
        counter_top.apply_transform(transform)
        
        # Counter base/cabinets
        base_height = counter_height - height
        counter_base = trimesh.creation.box(extents=[width, depth*0.9, base_height])
        base_transform = trimesh.transformations.translation_matrix(
            [center[0], center[1], base_height/2]
        )
        counter_base.apply_transform(base_transform)
        
        # Combine
        counter_mesh = trimesh.util.concatenate([counter_top, counter_base])
        counter_mesh.visual.material = self._create_pbr_material(MaterialType.FURNITURE_WOOD)
        
        return counter_mesh
    
    def _create_cabinet_mesh(self, center: Tuple[float, float], dimensions: Dict) -> trimesh.Trimesh:
        """Create a cabinet mesh."""
        width = dimensions.get('width', 0.6)
        depth = dimensions.get('depth', 0.3)
        height = dimensions.get('height', 2.1)
        
        # Create cabinet box
        cabinet_mesh = trimesh.creation.box(extents=[width, depth, height])
        
        # Position at center
        transform = trimesh.transformations.translation_matrix([center[0], center[1], height/2])
        cabinet_mesh.apply_transform(transform)
        
        # Apply wood material
        cabinet_mesh.visual.material = self._create_pbr_material(MaterialType.FURNITURE_WOOD)
        
        return cabinet_mesh
    
    def _create_generic_fixture(self, center: Tuple[float, float], dimensions: Dict) -> trimesh.Trimesh:
        """Create a generic fixture mesh."""
        width = dimensions.get('width', 0.5)
        depth = dimensions.get('depth', 0.5)
        height = dimensions.get('height', 0.8)
        
        fixture_mesh = trimesh.creation.box(extents=[width, depth, height])
        
        # Position at center
        transform = trimesh.transformations.translation_matrix([center[0], center[1], height/2])
        fixture_mesh.apply_transform(transform)
        
        # Apply generic material
        fixture_mesh.visual.material = self._create_pbr_material(MaterialType.FIXTURE_METAL)
        
        return fixture_mesh
    
    def _create_pbr_material(self, material_type: MaterialType) -> trimesh.visual.material.PBRMaterial:
        """Create a PBR material from material type."""
        material_props = self.materials[material_type]
        
        # Create PBR material
        material = trimesh.visual.material.PBRMaterial(
            name=material_props['name'],
            baseColorFactor=material_props['color'],
            roughnessFactor=material_props['roughness'],
            metallicFactor=material_props['metallic']
        )
        
        # Add additional properties if available
        if 'transmission' in material_props:
            # For glass materials
            material.transmissionFactor = material_props['transmission']
        
        return material
    
    def optimize_mesh_for_export(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Optimize mesh for export by reducing complexity while maintaining quality.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Optimized mesh
        """
        try:
            # Remove duplicate vertices and faces
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()
            
            # Fix normals
            mesh.fix_normals()
            
            # Simplify if too complex
            if len(mesh.vertices) > 10000:
                mesh = mesh.simplify_quadric_decimation(face_count=5000)
            
            return mesh
            
        except Exception as e:
            logger.warning(f"Mesh optimization failed: {e}")
            return mesh