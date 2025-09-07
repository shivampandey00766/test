"""
3D model reconstruction module for converting vectorized floor plans to 3D models.
Uses Blender Python API for 3D model generation.
"""

import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)


class ModelBuilder:
    """
    Builds 3D models from vectorized floor plan data using Blender.
    """
    
    def __init__(self, scale_factor: float = 0.01):
        """
        Initialize the model builder.
        
        Args:
            scale_factor: Scale factor for converting coordinates to Blender units
        """
        self.scale_factor = scale_factor
        self.logger = logging.getLogger(__name__)
        
        # Initialize Blender scene
        self._setup_blender_scene()
    
    def _setup_blender_scene(self):
        """Set up Blender scene for 3D model generation."""
        try:
            # Clear existing mesh objects
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)
            
            # Set up scene properties
            bpy.context.scene.unit_settings.length_unit = 'METERS'
            bpy.context.scene.unit_settings.scale_length = 1.0
            
            self.logger.info("Blender scene initialized")
            
        except Exception as e:
            self.logger.error(f"Error setting up Blender scene: {e}")
            raise
    
    def build_3d_model(self, vector_data: Dict[str, Any], 
                      depth_predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        Build complete 3D model from vector data and depth predictions.
        
        Args:
            vector_data: Processed vector data
            depth_predictions: Depth and height predictions from AI model
            
        Returns:
            Dictionary containing 3D model information
        """
        try:
            model_info = {
                'walls': self._build_walls(vector_data.get('walls', []), depth_predictions),
                'floors': self._build_floors(vector_data.get('rooms', []), depth_predictions),
                'doors': self._build_doors(vector_data.get('doors', []), depth_predictions),
                'windows': self._build_windows(vector_data.get('windows', []), depth_predictions),
                'furniture': self._build_furniture(vector_data.get('furniture', []), depth_predictions),
                'materials': self._setup_materials(),
                'lighting': self._setup_lighting(),
                'metadata': {
                    'scale_factor': self.scale_factor,
                    'total_objects': len(bpy.data.objects),
                    'total_materials': len(bpy.data.materials)
                }
            }
            
            self.logger.info("Successfully built 3D model")
            return model_info
            
        except Exception as e:
            self.logger.error(f"Error building 3D model: {e}")
            raise
    
    def _build_walls(self, walls: List, depth_predictions: Dict[str, float]) -> List[str]:
        """Build 3D wall objects."""
        try:
            wall_objects = []
            wall_height = depth_predictions.get('wall_height', 2.5)  # Default 2.5m height
            
            for i, wall in enumerate(walls):
                # Create wall mesh
                wall_mesh = self._create_wall_mesh(wall, wall_height)
                
                # Create object
                wall_obj = bpy.data.objects.new(f"Wall_{i:03d}", wall_mesh)
                bpy.context.collection.objects.link(wall_obj)
                
                # Set material
                wall_obj.data.materials.append(self._get_wall_material())
                
                wall_objects.append(wall_obj.name)
            
            self.logger.info(f"Built {len(wall_objects)} wall objects")
            return wall_objects
            
        except Exception as e:
            self.logger.error(f"Error building walls: {e}")
            return []
    
    def _create_wall_mesh(self, wall, height: float) -> bpy.types.Mesh:
        """Create mesh for a single wall."""
        try:
            # Convert coordinates to Blender units
            start = Vector((wall.start.x * self.scale_factor, wall.start.y * self.scale_factor, 0))
            end = Vector((wall.end.x * self.scale_factor, wall.end.y * self.scale_factor, 0))
            thickness = wall.thickness * self.scale_factor
            
            # Calculate wall direction and perpendicular
            direction = (end - start).normalized()
            perpendicular = Vector((-direction.y, direction.x, 0))
            
            # Create wall vertices
            vertices = [
                start + perpendicular * thickness / 2,  # Bottom left
                start - perpendicular * thickness / 2,  # Bottom right
                end - perpendicular * thickness / 2,    # Top right
                end + perpendicular * thickness / 2,    # Top left
                start + perpendicular * thickness / 2 + Vector((0, 0, height)),  # Top left
                start - perpendicular * thickness / 2 + Vector((0, 0, height)),  # Top right
                end - perpendicular * thickness / 2 + Vector((0, 0, height)),    # Bottom right
                end + perpendicular * thickness / 2 + Vector((0, 0, height))     # Bottom left
            ]
            
            # Create faces
            faces = [
                (0, 1, 2, 3),  # Bottom face
                (4, 7, 6, 5),  # Top face
                (0, 4, 5, 1),  # Front face
                (2, 6, 7, 3),  # Back face
                (0, 3, 7, 4),  # Left face
                (1, 5, 6, 2)   # Right face
            ]
            
            # Create mesh
            mesh = bpy.data.meshes.new(f"Wall_Mesh_{len(bpy.data.meshes)}")
            mesh.from_pydata(vertices, [], faces)
            mesh.update()
            
            return mesh
            
        except Exception as e:
            self.logger.error(f"Error creating wall mesh: {e}")
            raise
    
    def _build_floors(self, rooms: List, depth_predictions: Dict[str, float]) -> List[str]:
        """Build 3D floor objects for each room."""
        try:
            floor_objects = []
            
            for i, room in enumerate(rooms):
                # Create floor mesh
                floor_mesh = self._create_floor_mesh(room)
                
                # Create object
                floor_obj = bpy.data.objects.new(f"Floor_{i:03d}", floor_mesh)
                bpy.context.collection.objects.link(floor_obj)
                
                # Set material based on room type
                room_type = room.properties.get('room_type', 'unknown')
                floor_obj.data.materials.append(self._get_floor_material(room_type))
                
                floor_objects.append(floor_obj.name)
            
            self.logger.info(f"Built {len(floor_objects)} floor objects")
            return floor_objects
            
        except Exception as e:
            self.logger.error(f"Error building floors: {e}")
            return []
    
    def _create_floor_mesh(self, room) -> bpy.types.Mesh:
        """Create mesh for a room floor."""
        try:
            # Convert room points to vertices
            vertices = []
            for point in room.points:
                vertex = Vector((
                    point.x * self.scale_factor,
                    point.y * self.scale_factor,
                    0
                ))
                vertices.append(vertex)
            
            # Create faces (triangulate the polygon)
            faces = []
            if len(vertices) >= 3:
                # Simple triangulation (fan triangulation)
                for i in range(1, len(vertices) - 1):
                    faces.append((0, i, i + 1))
            
            # Create mesh
            mesh = bpy.data.meshes.new(f"Floor_Mesh_{len(bpy.data.meshes)}")
            mesh.from_pydata(vertices, [], faces)
            mesh.update()
            
            return mesh
            
        except Exception as e:
            self.logger.error(f"Error creating floor mesh: {e}")
            raise
    
    def _build_doors(self, doors: List, depth_predictions: Dict[str, float]) -> List[str]:
        """Build 3D door objects."""
        try:
            door_objects = []
            door_height = depth_predictions.get('wall_height', 2.5) * 0.8  # 80% of wall height
            
            for i, door in enumerate(doors):
                # Create door mesh
                door_mesh = self._create_door_mesh(door, door_height)
                
                # Create object
                door_obj = bpy.data.objects.new(f"Door_{i:03d}", door_mesh)
                bpy.context.collection.objects.link(door_obj)
                
                # Set material
                door_obj.data.materials.append(self._get_door_material())
                
                door_objects.append(door_obj.name)
            
            self.logger.info(f"Built {len(door_objects)} door objects")
            return door_objects
            
        except Exception as e:
            self.logger.error(f"Error building doors: {e}")
            return []
    
    def _create_door_mesh(self, door, height: float) -> bpy.types.Mesh:
        """Create mesh for a single door."""
        try:
            # Convert coordinates to Blender units
            start = Vector((door.start.x * self.scale_factor, door.start.y * self.scale_factor, 0))
            end = Vector((door.end.x * self.scale_factor, door.end.y * self.scale_factor, 0))
            thickness = door.thickness * self.scale_factor
            
            # Calculate door direction and perpendicular
            direction = (end - start).normalized()
            perpendicular = Vector((-direction.y, direction.x, 0))
            
            # Create door vertices
            vertices = [
                start + perpendicular * thickness / 2,  # Bottom left
                start - perpendicular * thickness / 2,  # Bottom right
                end - perpendicular * thickness / 2,    # Top right
                end + perpendicular * thickness / 2,    # Top left
                start + perpendicular * thickness / 2 + Vector((0, 0, height)),  # Top left
                start - perpendicular * thickness / 2 + Vector((0, 0, height)),  # Top right
                end - perpendicular * thickness / 2 + Vector((0, 0, height)),    # Bottom right
                end + perpendicular * thickness / 2 + Vector((0, 0, height))     # Bottom left
            ]
            
            # Create faces
            faces = [
                (0, 1, 2, 3),  # Bottom face
                (4, 7, 6, 5),  # Top face
                (0, 4, 5, 1),  # Front face
                (2, 6, 7, 3),  # Back face
                (0, 3, 7, 4),  # Left face
                (1, 5, 6, 2)   # Right face
            ]
            
            # Create mesh
            mesh = bpy.data.meshes.new(f"Door_Mesh_{len(bpy.data.meshes)}")
            mesh.from_pydata(vertices, [], faces)
            mesh.update()
            
            return mesh
            
        except Exception as e:
            self.logger.error(f"Error creating door mesh: {e}")
            raise
    
    def _build_windows(self, windows: List, depth_predictions: Dict[str, float]) -> List[str]:
        """Build 3D window objects."""
        try:
            window_objects = []
            window_height = depth_predictions.get('wall_height', 2.5) * 0.6  # 60% of wall height
            
            for i, window in enumerate(windows):
                # Create window mesh
                window_mesh = self._create_window_mesh(window, window_height)
                
                # Create object
                window_obj = bpy.data.objects.new(f"Window_{i:03d}", window_mesh)
                bpy.context.collection.objects.link(window_obj)
                
                # Set material
                window_obj.data.materials.append(self._get_window_material())
                
                window_objects.append(window_obj.name)
            
            self.logger.info(f"Built {len(window_objects)} window objects")
            return window_objects
            
        except Exception as e:
            self.logger.error(f"Error building windows: {e}")
            return []
    
    def _create_window_mesh(self, window, height: float) -> bpy.types.Mesh:
        """Create mesh for a single window."""
        try:
            # Convert coordinates to Blender units
            start = Vector((window.start.x * self.scale_factor, window.start.y * self.scale_factor, 0))
            end = Vector((window.end.x * self.scale_factor, window.end.y * self.scale_factor, 0))
            thickness = window.thickness * self.scale_factor
            
            # Calculate window direction and perpendicular
            direction = (end - start).normalized()
            perpendicular = Vector((-direction.y, direction.x, 0))
            
            # Create window vertices
            vertices = [
                start + perpendicular * thickness / 2,  # Bottom left
                start - perpendicular * thickness / 2,  # Bottom right
                end - perpendicular * thickness / 2,    # Top right
                end + perpendicular * thickness / 2,    # Top left
                start + perpendicular * thickness / 2 + Vector((0, 0, height)),  # Top left
                start - perpendicular * thickness / 2 + Vector((0, 0, height)),  # Top right
                end - perpendicular * thickness / 2 + Vector((0, 0, height)),    # Bottom right
                end + perpendicular * thickness / 2 + Vector((0, 0, height))     # Bottom left
            ]
            
            # Create faces
            faces = [
                (0, 1, 2, 3),  # Bottom face
                (4, 7, 6, 5),  # Top face
                (0, 4, 5, 1),  # Front face
                (2, 6, 7, 3),  # Back face
                (0, 3, 7, 4),  # Left face
                (1, 5, 6, 2)   # Right face
            ]
            
            # Create mesh
            mesh = bpy.data.meshes.new(f"Window_Mesh_{len(bpy.data.meshes)}")
            mesh.from_pydata(vertices, [], faces)
            mesh.update()
            
            return mesh
            
        except Exception as e:
            self.logger.error(f"Error creating window mesh: {e}")
            raise
    
    def _build_furniture(self, furniture: List, depth_predictions: Dict[str, float]) -> List[str]:
        """Build 3D furniture objects."""
        try:
            furniture_objects = []
            furniture_height = 0.8  # Default furniture height
            
            for i, item in enumerate(furniture):
                # Create furniture mesh
                furniture_mesh = self._create_furniture_mesh(item, furniture_height)
                
                # Create object
                furniture_obj = bpy.data.objects.new(f"Furniture_{i:03d}", furniture_mesh)
                bpy.context.collection.objects.link(furniture_obj)
                
                # Set material
                furniture_obj.data.materials.append(self._get_furniture_material())
                
                furniture_objects.append(furniture_obj.name)
            
            self.logger.info(f"Built {len(furniture_objects)} furniture objects")
            return furniture_objects
            
        except Exception as e:
            self.logger.error(f"Error building furniture: {e}")
            return []
    
    def _create_furniture_mesh(self, item, height: float) -> bpy.types.Mesh:
        """Create mesh for a furniture item."""
        try:
            # Convert furniture points to vertices
            vertices = []
            for point in item.points:
                vertex = Vector((
                    point.x * self.scale_factor,
                    point.y * self.scale_factor,
                    0
                ))
                vertices.append(vertex)
            
            # Add top vertices
            top_vertices = []
            for vertex in vertices:
                top_vertex = vertex + Vector((0, 0, height))
                top_vertices.append(top_vertex)
            
            all_vertices = vertices + top_vertices
            
            # Create faces
            faces = []
            
            # Bottom face
            if len(vertices) >= 3:
                for i in range(1, len(vertices) - 1):
                    faces.append((0, i, i + 1))
            
            # Top face
            if len(top_vertices) >= 3:
                offset = len(vertices)
                for i in range(1, len(top_vertices) - 1):
                    faces.append((offset, offset + i, offset + i + 1))
            
            # Side faces
            for i in range(len(vertices)):
                next_i = (i + 1) % len(vertices)
                faces.append((i, next_i, next_i + len(vertices), i + len(vertices)))
            
            # Create mesh
            mesh = bpy.data.meshes.new(f"Furniture_Mesh_{len(bpy.data.meshes)}")
            mesh.from_pydata(all_vertices, [], faces)
            mesh.update()
            
            return mesh
            
        except Exception as e:
            self.logger.error(f"Error creating furniture mesh: {e}")
            raise
    
    def _setup_materials(self) -> Dict[str, str]:
        """Set up materials for different elements."""
        try:
            materials = {}
            
            # Wall material
            wall_mat = bpy.data.materials.new(name="Wall_Material")
            wall_mat.use_nodes = True
            wall_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.8, 0.8, 1.0)  # Light gray
            materials['wall'] = wall_mat.name
            
            # Floor material
            floor_mat = bpy.data.materials.new(name="Floor_Material")
            floor_mat.use_nodes = True
            floor_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.6, 0.4, 0.2, 1.0)  # Brown
            materials['floor'] = floor_mat.name
            
            # Door material
            door_mat = bpy.data.materials.new(name="Door_Material")
            door_mat.use_nodes = True
            door_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.4, 0.2, 0.1, 1.0)  # Dark brown
            materials['door'] = door_mat.name
            
            # Window material
            window_mat = bpy.data.materials.new(name="Window_Material")
            window_mat.use_nodes = True
            window_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.7, 0.9, 1.0, 1.0)  # Light blue
            window_mat.node_tree.nodes["Principled BSDF"].inputs[19].default_value = 0.1  # Transmission
            materials['window'] = window_mat.name
            
            # Furniture material
            furniture_mat = bpy.data.materials.new(name="Furniture_Material")
            furniture_mat.use_nodes = True
            furniture_mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.5, 0.3, 0.1, 1.0)  # Dark brown
            materials['furniture'] = furniture_mat.name
            
            self.logger.info(f"Set up {len(materials)} materials")
            return materials
            
        except Exception as e:
            self.logger.error(f"Error setting up materials: {e}")
            return {}
    
    def _get_wall_material(self) -> bpy.types.Material:
        """Get wall material."""
        return bpy.data.materials.get("Wall_Material", bpy.data.materials.new("Wall_Material"))
    
    def _get_floor_material(self, room_type: str) -> bpy.types.Material:
        """Get floor material based on room type."""
        # This could be expanded to have different materials for different room types
        return bpy.data.materials.get("Floor_Material", bpy.data.materials.new("Floor_Material"))
    
    def _get_door_material(self) -> bpy.types.Material:
        """Get door material."""
        return bpy.data.materials.get("Door_Material", bpy.data.materials.new("Door_Material"))
    
    def _get_window_material(self) -> bpy.types.Material:
        """Get window material."""
        return bpy.data.materials.get("Window_Material", bpy.data.materials.new("Window_Material"))
    
    def _get_furniture_material(self) -> bpy.types.Material:
        """Get furniture material."""
        return bpy.data.materials.get("Furniture_Material", bpy.data.materials.new("Furniture_Material"))
    
    def _setup_lighting(self) -> List[str]:
        """Set up basic lighting for the scene."""
        try:
            lights = []
            
            # Add sun light
            sun_data = bpy.data.lights.new(name="Sun", type='SUN')
            sun_obj = bpy.data.objects.new("Sun", sun_data)
            bpy.context.collection.objects.link(sun_obj)
            sun_obj.location = (5, 5, 10)
            sun_obj.rotation_euler = (0.785, 0, 0.785)  # 45 degrees
            sun_data.energy = 3
            lights.append(sun_obj.name)
            
            # Add area light
            area_data = bpy.data.lights.new(name="Area_Light", type='AREA')
            area_obj = bpy.data.objects.new("Area_Light", area_data)
            bpy.context.collection.objects.link(area_obj)
            area_obj.location = (0, 0, 3)
            area_obj.rotation_euler = (0, 0, 0)
            area_data.energy = 2
            area_data.size = 5
            lights.append(area_obj.name)
            
            self.logger.info(f"Set up {len(lights)} lights")
            return lights
            
        except Exception as e:
            self.logger.error(f"Error setting up lighting: {e}")
            return []
    
    def export_model(self, filepath: str, format: str = 'obj') -> None:
        """
        Export 3D model to file.
        
        Args:
            filepath: Path to save the model
            format: Export format ('obj', 'gltf', 'fbx')
        """
        try:
            if format.lower() == 'obj':
                bpy.ops.export_scene.obj(
                    filepath=filepath,
                    use_selection=False,
                    use_animation=False,
                    use_materials=True,
                    use_edges=True,
                    use_smooth_groups=False,
                    use_normals=True,
                    use_uvs=True,
                    use_mtllib=True
                )
            elif format.lower() == 'gltf':
                bpy.ops.export_scene.gltf(
                    filepath=filepath,
                    export_format='GLTF_SEPARATE',
                    export_materials='EXPORT',
                    export_colors=True,
                    export_cameras=False,
                    export_lights=True
                )
            elif format.lower() == 'fbx':
                bpy.ops.export_scene.fbx(
                    filepath=filepath,
                    use_selection=False,
                    use_animation=False,
                    use_materials=True,
                    use_armature_deform_only=False,
                    bake_anim_use_all_bones=False,
                    bake_anim_use_nla_strips=False,
                    bake_anim_use_all_actions=False,
                    add_leaf_bones=False,
                    use_armature=False,
                    use_mesh_modifiers=True,
                    use_mesh_modifiers_render=True,
                    use_mesh_edges=False,
                    use_tspace=False,
                    use_custom_props=False,
                    use_anim=True,
                    use_anim_action_all=True,
                    use_default_take=True,
                    use_anim_optimize=True,
                    anim_optimize_precision=6,
                    path_mode='AUTO',
                    embed_textures=False,
                    batch_mode='OFF',
                    use_batch_own_dir=True,
                    use_metadata=True
                )
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported 3D model to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            raise
    
    def render_image(self, filepath: str, resolution: Tuple[int, int] = (1920, 1080)) -> None:
        """
        Render the 3D model to an image.
        
        Args:
            filepath: Path to save the rendered image
            resolution: Image resolution (width, height)
        """
        try:
            # Set up camera
            camera_data = bpy.data.cameras.new("Camera")
            camera_obj = bpy.data.objects.new("Camera", camera_data)
            bpy.context.collection.objects.link(camera_obj)
            
            # Position camera
            camera_obj.location = (10, -10, 8)
            camera_obj.rotation_euler = (1.1, 0, 0.785)
            
            # Set as active camera
            bpy.context.scene.camera = camera_obj
            
            # Set render settings
            bpy.context.scene.render.engine = 'CYCLES'
            bpy.context.scene.render.resolution_x = resolution[0]
            bpy.context.scene.render.resolution_y = resolution[1]
            bpy.context.scene.render.filepath = filepath
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            
            # Render
            bpy.ops.render.render(write_still=True)
            
            self.logger.info(f"Rendered image to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error rendering image: {e}")
            raise
    
    def cleanup(self):
        """Clean up Blender scene."""
        try:
            # Clear all mesh objects
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)
            
            # Clear materials
            for material in bpy.data.materials:
                bpy.data.materials.remove(material)
            
            # Clear meshes
            for mesh in bpy.data.meshes:
                bpy.data.meshes.remove(mesh)
            
            # Clear lights
            for light in bpy.data.lights:
                bpy.data.lights.remove(light)
            
            self.logger.info("Cleaned up Blender scene")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up scene: {e}")
            raise