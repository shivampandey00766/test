"""
Blender-based 3D reconstruction for architectural models.

This module provides functionality to create 3D models from 2D floor plans
using Blender's Python API (bpy).
"""

import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import os
import json
from pathlib import Path
from ..vectorization.raster_to_vector import VectorLine, VectorPolygon, VectorPoint, VectorData


class BlenderReconstructor:
    """
    Creates 3D architectural models using Blender.
    
    This class provides methods to convert 2D floor plan data
    into 3D models with proper proportions, materials, and lighting.
    """
    
    def __init__(self, 
                 wall_height: float = 2.4,
                 floor_thickness: float = 0.2,
                 ceiling_height: float = 2.7):
        """
        Initialize the BlenderReconstructor.
        
        Args:
            wall_height: Standard wall height in meters
            floor_thickness: Floor thickness in meters
            ceiling_height: Ceiling height in meters
        """
        self.wall_height = wall_height
        self.floor_thickness = floor_thickness
        self.ceiling_height = ceiling_height
        
        # Material definitions
        self.materials = {
            'wall': {'color': (0.8, 0.8, 0.8, 1.0), 'roughness': 0.7},
            'floor': {'color': (0.6, 0.4, 0.2, 1.0), 'roughness': 0.3},
            'ceiling': {'color': (0.9, 0.9, 0.9, 1.0), 'roughness': 0.8},
            'door': {'color': (0.4, 0.2, 0.1, 1.0), 'roughness': 0.5},
            'window': {'color': (0.7, 0.8, 1.0, 0.3), 'roughness': 0.1}
        }
        
        # Clear existing scene
        self._clear_scene()
    
    def _clear_scene(self):
        """Clear the current Blender scene."""
        # Select all objects
        bpy.ops.object.select_all(action='SELECT')
        
        # Delete all objects
        bpy.ops.object.delete(use_global=False)
        
        # Clear materials
        for material in bpy.data.materials:
            bpy.data.materials.remove(material)
    
    def create_material(self, name: str, properties: Dict[str, Any]) -> bpy.types.Material:
        """
        Create a material with specified properties.
        
        Args:
            name: Material name
            properties: Material properties (color, roughness, etc.)
            
        Returns:
            Created material
        """
        material = bpy.data.materials.new(name=name)
        material.use_nodes = True
        
        # Get the principled BSDF node
        bsdf = material.node_tree.nodes["Principled BSDF"]
        
        # Set color
        if 'color' in properties:
            bsdf.inputs[0].default_value = properties['color']
        
        # Set roughness
        if 'roughness' in properties:
            bsdf.inputs[9].default_value = properties['roughness']
        
        # Set metallic
        if 'metallic' in properties:
            bsdf.inputs[6].default_value = properties['metallic']
        
        return material
    
    def create_wall(self, start_point: VectorPoint, end_point: VectorPoint, 
                   height: float = None, thickness: float = 0.2) -> bpy.types.Object:
        """
        Create a 3D wall from two points.
        
        Args:
            start_point: Start point of the wall
            end_point: End point of the wall
            height: Wall height (uses default if None)
            thickness: Wall thickness
            
        Returns:
            Created wall object
        """
        if height is None:
            height = self.wall_height
        
        # Calculate wall dimensions
        length = np.sqrt((end_point.x - start_point.x)**2 + (end_point.y - start_point.y)**2)
        
        # Create wall mesh
        bpy.ops.mesh.primitive_cube_add(size=1)
        wall = bpy.context.active_object
        wall.name = "Wall"
        
        # Scale to correct dimensions
        wall.scale = (length, thickness, height)
        
        # Position wall
        center_x = (start_point.x + end_point.x) / 2
        center_y = (start_point.y + end_point.y) / 2
        wall.location = (center_x, center_y, height / 2)
        
        # Rotate wall to match direction
        angle = np.arctan2(end_point.y - start_point.y, end_point.x - start_point.x)
        wall.rotation_euler = (0, 0, angle)
        
        # Add material
        material = self.create_material("WallMaterial", self.materials['wall'])
        wall.data.materials.append(material)
        
        return wall
    
    def create_floor(self, polygons: List[VectorPolygon]) -> bpy.types.Object:
        """
        Create floor from room polygons.
        
        Args:
            polygons: List of room polygons
            
        Returns:
            Created floor object
        """
        # Create floor mesh
        bpy.ops.mesh.primitive_plane_add(size=1)
        floor = bpy.context.active_object
        floor.name = "Floor"
        
        # Enter edit mode
        bpy.context.view_layer.objects.active = floor
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Clear existing geometry
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.delete(type='VERT')
        
        # Create floor geometry from polygons
        bm = bmesh.new()
        
        for polygon in polygons:
            if polygon.polygon_type == 'room' and len(polygon.points) >= 3:
                # Create vertices
                verts = []
                for point in polygon.points:
                    vert = bm.verts.new((point.x, point.y, 0))
                    verts.append(vert)
                
                # Create face
                try:
                    face = bm.faces.new(verts)
                    face.normal_update()
                except ValueError:
                    # Skip invalid faces
                    continue
        
        # Update mesh
        bm.to_mesh(floor.data)
        bm.free()
        
        # Exit edit mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Scale floor thickness
        floor.scale = (1, 1, self.floor_thickness)
        
        # Add material
        material = self.create_material("FloorMaterial", self.materials['floor'])
        floor.data.materials.append(material)
        
        return floor
    
    def create_ceiling(self, polygons: List[VectorPolygon]) -> bpy.types.Object:
        """
        Create ceiling from room polygons.
        
        Args:
            polygons: List of room polygons
            
        Returns:
            Created ceiling object
        """
        # Create ceiling mesh
        bpy.ops.mesh.primitive_plane_add(size=1)
        ceiling = bpy.context.active_object
        ceiling.name = "Ceiling"
        
        # Enter edit mode
        bpy.context.view_layer.objects.active = ceiling
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Clear existing geometry
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.delete(type='VERT')
        
        # Create ceiling geometry from polygons
        bm = bmesh.new()
        
        for polygon in polygons:
            if polygon.polygon_type == 'room' and len(polygon.points) >= 3:
                # Create vertices at ceiling height
                verts = []
                for point in polygon.points:
                    vert = bm.verts.new((point.x, point.y, self.ceiling_height))
                    verts.append(vert)
                
                # Create face (reverse order for correct normal)
                try:
                    face = bm.faces.new(verts[::-1])
                    face.normal_update()
                except ValueError:
                    # Skip invalid faces
                    continue
        
        # Update mesh
        bm.to_mesh(ceiling.data)
        bm.free()
        
        # Exit edit mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Add material
        material = self.create_material("CeilingMaterial", self.materials['ceiling'])
        ceiling.data.materials.append(material)
        
        return ceiling
    
    def create_door(self, polygon: VectorPolygon, wall_height: float = None) -> bpy.types.Object:
        """
        Create a 3D door from polygon.
        
        Args:
            polygon: Door polygon
            wall_height: Wall height for door sizing
            
        Returns:
            Created door object
        """
        if wall_height is None:
            wall_height = self.wall_height
        
        # Calculate door dimensions
        if len(polygon.points) < 3:
            return None
        
        # Get bounding box
        xs = [p.x for p in polygon.points]
        ys = [p.y for p in polygon.points]
        
        width = max(xs) - min(xs)
        depth = max(ys) - min(ys)
        
        # Create door mesh
        bpy.ops.mesh.primitive_cube_add(size=1)
        door = bpy.context.active_object
        door.name = "Door"
        
        # Scale to correct dimensions
        door.scale = (width, depth, wall_height * 0.8)  # Door is 80% of wall height
        
        # Position door
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        door.location = (center_x, center_y, wall_height * 0.4)
        
        # Add material
        material = self.create_material("DoorMaterial", self.materials['door'])
        door.data.materials.append(material)
        
        return door
    
    def create_window(self, polygon: VectorPolygon, wall_height: float = None) -> bpy.types.Object:
        """
        Create a 3D window from polygon.
        
        Args:
            polygon: Window polygon
            wall_height: Wall height for window sizing
            
        Returns:
            Created window object
        """
        if wall_height is None:
            wall_height = self.wall_height
        
        # Calculate window dimensions
        if len(polygon.points) < 3:
            return None
        
        # Get bounding box
        xs = [p.x for p in polygon.points]
        ys = [p.y for p in polygon.points]
        
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        
        # Create window mesh
        bpy.ops.mesh.primitive_cube_add(size=1)
        window = bpy.context.active_object
        window.name = "Window"
        
        # Scale to correct dimensions
        window.scale = (width, 0.1, height)  # Thin window
        
        # Position window
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        window.location = (center_x, center_y, wall_height * 0.6)  # Windows at 60% height
        
        # Add material
        material = self.create_material("WindowMaterial", self.materials['window'])
        window.data.materials.append(material)
        
        return window
    
    def create_room_volume(self, polygon: VectorPolygon, 
                          room_type: str = "room") -> bpy.types.Object:
        """
        Create a 3D room volume from polygon.
        
        Args:
            polygon: Room polygon
            room_type: Type of room
            
        Returns:
            Created room volume object
        """
        if len(polygon.points) < 3:
            return None
        
        # Create room mesh
        bpy.ops.mesh.primitive_plane_add(size=1)
        room = bpy.context.active_object
        room.name = f"Room_{room_type}"
        
        # Enter edit mode
        bpy.context.view_layer.objects.active = room
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Clear existing geometry
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.delete(type='VERT')
        
        # Create room geometry
        bm = bmesh.new()
        
        # Create vertices
        verts = []
        for point in polygon.points:
            vert = bm.verts.new((point.x, point.y, 0))
            verts.append(vert)
        
        # Create face
        try:
            face = bm.faces.new(verts)
            face.normal_update()
        except ValueError:
            bm.free()
            return None
        
        # Extrude to create volume
        bpy.ops.mesh.extrude_region_move(
            TRANSFORM_OT_translate={
                "value": (0, 0, self.ceiling_height)
            }
        )
        
        # Update mesh
        bm.to_mesh(room.data)
        bm.free()
        
        # Exit edit mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Add material based on room type
        room_materials = {
            'living_room': {'color': (0.8, 0.6, 0.4, 0.1), 'roughness': 0.5},
            'bedroom': {'color': (0.6, 0.6, 0.8, 0.1), 'roughness': 0.5},
            'kitchen': {'color': (0.8, 0.8, 0.6, 0.1), 'roughness': 0.3},
            'bathroom': {'color': (0.6, 0.8, 0.8, 0.1), 'roughness': 0.2},
            'default': {'color': (0.7, 0.7, 0.7, 0.1), 'roughness': 0.5}
        }
        
        material_props = room_materials.get(room_type, room_materials['default'])
        material = self.create_material(f"RoomMaterial_{room_type}", material_props)
        room.data.materials.append(material)
        
        return room
    
    def reconstruct_from_vector_data(self, vector_data: VectorData, 
                                   room_types: List[str] = None) -> Dict[str, bpy.types.Object]:
        """
        Reconstruct 3D model from vector data.
        
        Args:
            vector_data: VectorData object
            room_types: List of room types for each room
            
        Returns:
            Dictionary of created objects
        """
        created_objects = {
            'walls': [],
            'doors': [],
            'windows': [],
            'rooms': [],
            'floor': None,
            'ceiling': None
        }
        
        # Create walls from lines
        for line in vector_data.lines:
            if line.line_type == 'wall':
                wall = self.create_wall(line.start, line.end)
                if wall:
                    created_objects['walls'].append(wall)
        
        # Create doors from polygons
        door_polygons = [p for p in vector_data.polygons if p.polygon_type == 'door']
        for polygon in door_polygons:
            door = self.create_door(polygon)
            if door:
                created_objects['doors'].append(door)
        
        # Create windows from polygons
        window_polygons = [p for p in vector_data.polygons if p.polygon_type == 'window']
        for polygon in window_polygons:
            window = self.create_window(polygon)
            if window:
                created_objects['windows'].append(window)
        
        # Create rooms
        room_polygons = [p for p in vector_data.polygons if p.polygon_type == 'room']
        for i, polygon in enumerate(room_polygons):
            room_type = room_types[i] if room_types and i < len(room_types) else 'room'
            room = self.create_room_volume(polygon, room_type)
            if room:
                created_objects['rooms'].append(room)
        
        # Create floor and ceiling
        if room_polygons:
            created_objects['floor'] = self.create_floor(room_polygons)
            created_objects['ceiling'] = self.create_ceiling(room_polygons)
        
        return created_objects
    
    def add_lighting(self, lighting_type: str = 'basic') -> None:
        """
        Add lighting to the scene.
        
        Args:
            lighting_type: Type of lighting ('basic', 'architectural', 'studio')
        """
        if lighting_type == 'basic':
            # Add basic sun light
            bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
            sun = bpy.context.active_object
            sun.data.energy = 3
            sun.rotation_euler = (0.785, 0, 0.785)  # 45 degrees
            
            # Add area light
            bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
            area_light = bpy.context.active_object
            area_light.data.energy = 100
            area_light.data.size = 10
        
        elif lighting_type == 'architectural':
            # Add multiple lights for architectural visualization
            lights = [
                {'location': (0, 0, 8), 'energy': 5, 'type': 'SUN'},
                {'location': (5, 5, 3), 'energy': 200, 'type': 'AREA'},
                {'location': (-5, -5, 3), 'energy': 200, 'type': 'AREA'},
            ]
            
            for light_data in lights:
                bpy.ops.object.light_add(type=light_data['type'], location=light_data['location'])
                light = bpy.context.active_object
                light.data.energy = light_data['energy']
                if light_data['type'] == 'AREA':
                    light.data.size = 5
    
    def setup_camera(self, camera_type: str = 'overview') -> None:
        """
        Setup camera for rendering.
        
        Args:
            camera_type: Type of camera setup ('overview', 'interior', 'exterior')
        """
        # Add camera
        bpy.ops.object.camera_add(location=(0, 0, 0))
        camera = bpy.context.active_object
        
        if camera_type == 'overview':
            camera.location = (10, -10, 8)
            camera.rotation_euler = (1.1, 0, 0.785)
        elif camera_type == 'interior':
            camera.location = (0, 0, 1.5)
            camera.rotation_euler = (1.57, 0, 0)
        elif camera_type == 'exterior':
            camera.location = (15, -15, 10)
            camera.rotation_euler = (1.0, 0, 0.785)
        
        # Set as active camera
        bpy.context.scene.camera = camera
    
    def export_model(self, output_path: str, format: str = 'obj') -> None:
        """
        Export 3D model to file.
        
        Args:
            output_path: Output file path
            format: Export format ('obj', 'fbx', 'gltf', 'blend')
        """
        if format == 'obj':
            bpy.ops.export_scene.obj(filepath=output_path)
        elif format == 'fbx':
            bpy.ops.export_scene.fbx(filepath=output_path)
        elif format == 'gltf':
            bpy.ops.export_scene.gltf(filepath=output_path)
        elif format == 'blend':
            bpy.ops.wm.save_as_mainfile(filepath=output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def render_image(self, output_path: str, 
                    resolution: Tuple[int, int] = (1920, 1080),
                    samples: int = 128) -> None:
        """
        Render the 3D model to an image.
        
        Args:
            output_path: Output image path
            resolution: Image resolution (width, height)
            samples: Number of render samples
        """
        # Set render settings
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.render.resolution_x = resolution[0]
        scene.render.resolution_y = resolution[1]
        scene.render.filepath = output_path
        
        # Set cycles settings
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        
        # Render
        bpy.ops.render.render(write_still=True)
    
    def cleanup(self) -> None:
        """Clean up Blender scene and free memory."""
        # Clear scene
        self._clear_scene()
        
        # Clear unused data
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)