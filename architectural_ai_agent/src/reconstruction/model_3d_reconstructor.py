"""
3D Model Reconstruction Module

This module handles the conversion from 2D floor plan analysis to 3D architectural models,
including wall extrusion, opening placement, and object instantiation.
"""

import numpy as np
import trimesh
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Any
import logging
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import json

logger = logging.getLogger(__name__)


class Model3DReconstructor:
    """
    Main class for reconstructing 3D models from 2D floor plan analysis.
    """
    
    def __init__(self, output_format: str = 'obj'):
        """
        Initialize the 3D reconstructor.
        
        Args:
            output_format: Output format ('obj', 'gltf', 'stl', 'ply')
        """
        self.output_format = output_format.lower()
        self.scene = trimesh.Scene()
        
        # Material definitions
        self.materials = self._create_materials()
        
        # Standard dimensions (in meters)
        self.standard_dimensions = {
            'wall_thickness': 0.15,
            'door_thickness': 0.05,
            'window_thickness': 0.02,
            'floor_thickness': 0.2,
            'ceiling_thickness': 0.1
        }
        
    def reconstruct_3d_model(self, segmentation_data: Dict, features_data: Dict, 
                           depth_data: Dict, scale_factor: float = 1.0) -> Dict:
        """
        Reconstruct a complete 3D model from 2D analysis data.
        
        Args:
            segmentation_data: Results from semantic segmentation
            features_data: Results from feature detection
            depth_data: Results from depth estimation
            scale_factor: Scale factor for converting pixels to real-world units
            
        Returns:
            Dictionary containing 3D model data and metadata
        """
        logger.info("Starting 3D reconstruction...")
        
        # Clear previous scene
        self.scene = trimesh.Scene()
        
        # Build floor plan base
        floor_mesh = self._build_floor(segmentation_data, scale_factor)
        if floor_mesh:
            self.scene.add_geometry(floor_mesh, node_name='floor')
        
        # Build walls
        wall_meshes = self._build_walls(features_data, depth_data, scale_factor)
        for i, wall_mesh in enumerate(wall_meshes):
            self.scene.add_geometry(wall_mesh, node_name=f'wall_{i}')
        
        # Build openings (doors and windows)
        door_meshes = self._build_doors(features_data, depth_data, scale_factor)
        for i, door_mesh in enumerate(door_meshes):
            self.scene.add_geometry(door_mesh, node_name=f'door_{i}')
        
        window_meshes = self._build_windows(features_data, depth_data, scale_factor)
        for i, window_mesh in enumerate(window_meshes):
            self.scene.add_geometry(window_mesh, node_name=f'window_{i}')
        
        # Build ceiling
        ceiling_mesh = self._build_ceiling(segmentation_data, depth_data, scale_factor)
        if ceiling_mesh:
            self.scene.add_geometry(ceiling_mesh, node_name='ceiling')
        
        # Build fixtures and furniture
        fixture_meshes = self._build_fixtures(features_data, depth_data, scale_factor)
        for i, fixture_mesh in enumerate(fixture_meshes):
            self.scene.add_geometry(fixture_mesh, node_name=f'fixture_{i}')
        
        # Build stairs if detected
        stair_meshes = self._build_stairs(segmentation_data, depth_data, scale_factor)
        for i, stair_mesh in enumerate(stair_meshes):
            self.scene.add_geometry(stair_mesh, node_name=f'stairs_{i}')
        
        # Generate metadata
        metadata = self._generate_metadata(segmentation_data, features_data, depth_data, scale_factor)
        
        logger.info(f"3D reconstruction completed with {len(self.scene.geometry)} objects")
        
        return {
            'scene': self.scene,
            'metadata': metadata,
            'bounding_box': self.scene.bounds,
            'total_vertices': sum(len(mesh.vertices) for mesh in self.scene.geometry.values()),
            'total_faces': sum(len(mesh.faces) for mesh in self.scene.geometry.values())
        }
    
    def _create_materials(self) -> Dict:
        """Create material definitions for different architectural elements."""
        materials = {
            'wall': {
                'color': [0.9, 0.9, 0.9, 1.0],  # Light gray
                'roughness': 0.8,
                'metallic': 0.0
            },
            'door': {
                'color': [0.6, 0.4, 0.2, 1.0],  # Brown wood
                'roughness': 0.6,
                'metallic': 0.0
            },
            'window': {
                'color': [0.7, 0.9, 1.0, 0.6],  # Light blue, transparent
                'roughness': 0.1,
                'metallic': 0.0
            },
            'floor': {
                'color': [0.8, 0.7, 0.6, 1.0],  # Beige
                'roughness': 0.7,
                'metallic': 0.0
            },
            'ceiling': {
                'color': [1.0, 1.0, 1.0, 1.0],  # White
                'roughness': 0.9,
                'metallic': 0.0
            },
            'fixture': {
                'color': [0.5, 0.5, 0.5, 1.0],  # Gray
                'roughness': 0.5,
                'metallic': 0.3
            }
        }
        return materials
    
    def _build_floor(self, segmentation_data: Dict, scale_factor: float) -> Optional[trimesh.Trimesh]:
        """Build the floor mesh from room segmentation."""
        room_analysis = segmentation_data.get('room_analysis', {})
        
        if not room_analysis:
            return None
        
        # Combine all room areas to create floor
        floor_polygons = []
        
        for room_type, room_info in room_analysis.items():
            for room in room_info.get('rooms', []):
                # Convert contour to polygon
                contour = room['contour']
                if len(contour) >= 3:
                    # Convert contour points to 2D coordinates
                    points = [(point[0][0] * scale_factor, point[0][1] * scale_factor) 
                             for point in contour]
                    
                    try:
                        polygon = Polygon(points)
                        if polygon.is_valid and polygon.area > 0:
                            floor_polygons.append(polygon)
                    except Exception as e:
                        logger.warning(f"Could not create polygon for room: {e}")
        
        if not floor_polygons:
            return None
        
        # Union all room polygons
        try:
            combined_floor = unary_union(floor_polygons)
            
            # Convert to mesh
            floor_mesh = self._polygon_to_mesh(combined_floor, 0, self.standard_dimensions['floor_thickness'])
            
            # Apply floor material
            floor_mesh.visual.material = trimesh.visual.material.PBRMaterial(**self.materials['floor'])
            
            return floor_mesh
            
        except Exception as e:
            logger.error(f"Failed to create floor mesh: {e}")
            return None
    
    def _build_walls(self, features_data: Dict, depth_data: Dict, scale_factor: float) -> List[trimesh.Trimesh]:
        """Build wall meshes from feature detection."""
        wall_meshes = []
        
        walls = features_data.get('walls', [])
        wall_heights = depth_data.get('wall_heights', {})
        
        for wall in walls:
            try:
                # Get wall coordinates
                start = (wall['start'][0] * scale_factor, wall['start'][1] * scale_factor)
                end = (wall['end'][0] * scale_factor, wall['end'][1] * scale_factor)
                
                # Get wall properties
                thickness = wall.get('thickness', 6) * scale_factor
                if thickness < self.standard_dimensions['wall_thickness']:
                    thickness = self.standard_dimensions['wall_thickness']
                
                # Get height
                height = self.standard_dimensions.get('wall_height', 2.4)
                individual_walls = wall_heights.get('individual_walls', [])
                for wall_info in individual_walls:
                    if (wall_info['start'] == wall['start'] and wall_info['end'] == wall['end']):
                        height = wall_info['height']
                        break
                
                # Create wall mesh
                wall_mesh = self._create_wall_segment(start, end, thickness, height)
                
                if wall_mesh:
                    # Apply wall material
                    wall_mesh.visual.material = trimesh.visual.material.PBRMaterial(**self.materials['wall'])
                    wall_meshes.append(wall_mesh)
                    
            except Exception as e:
                logger.warning(f"Failed to create wall mesh: {e}")
        
        return wall_meshes
    
    def _create_wall_segment(self, start: Tuple[float, float], end: Tuple[float, float], 
                           thickness: float, height: float) -> Optional[trimesh.Trimesh]:
        """Create a single wall segment mesh."""
        # Calculate wall direction and perpendicular
        direction = np.array([end[0] - start[0], end[1] - start[1]])
        length = np.linalg.norm(direction)
        
        if length < 0.01:  # Too short
            return None
        
        direction = direction / length
        perpendicular = np.array([-direction[1], direction[0]]) * thickness / 2
        
        # Create wall rectangle vertices
        vertices = [
            [start[0] - perpendicular[0], start[1] - perpendicular[1], 0],
            [start[0] + perpendicular[0], start[1] + perpendicular[1], 0],
            [end[0] + perpendicular[0], end[1] + perpendicular[1], 0],
            [end[0] - perpendicular[0], end[1] - perpendicular[1], 0],
            [start[0] - perpendicular[0], start[1] - perpendicular[1], height],
            [start[0] + perpendicular[0], start[1] + perpendicular[1], height],
            [end[0] + perpendicular[0], end[1] + perpendicular[1], height],
            [end[0] - perpendicular[0], end[1] - perpendicular[1], height]
        ]
        
        # Create faces (box topology)
        faces = [
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 7, 6], [4, 6, 5],  # Top
            [0, 4, 5], [0, 5, 1],  # Side 1
            [2, 6, 7], [2, 7, 3],  # Side 2
            [1, 5, 6], [1, 6, 2],  # Front
            [0, 3, 7], [0, 7, 4]   # Back
        ]
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def _build_doors(self, features_data: Dict, depth_data: Dict, scale_factor: float) -> List[trimesh.Trimesh]:
        """Build door meshes."""
        door_meshes = []
        
        doors = features_data.get('doors', [])
        opening_dimensions = depth_data.get('opening_dimensions', {})
        door_specs = opening_dimensions.get('doors', [])
        
        for i, door in enumerate(doors):
            try:
                # Get door specifications
                door_spec = door_specs[i] if i < len(door_specs) else {}
                
                width = door_spec.get('width', 0.8)
                height = door_spec.get('height', 2.1)
                thickness = self.standard_dimensions['door_thickness']
                
                # Get door position
                if 'center' in door:
                    center = (door['center'][0] * scale_factor, door['center'][1] * scale_factor)
                else:
                    center = (0, 0)
                
                # Create door mesh (simplified as a rectangular panel)
                door_mesh = self._create_door_mesh(center, width, height, thickness)
                
                if door_mesh:
                    door_mesh.visual.material = trimesh.visual.material.PBRMaterial(**self.materials['door'])
                    door_meshes.append(door_mesh)
                    
            except Exception as e:
                logger.warning(f"Failed to create door mesh: {e}")
        
        return door_meshes
    
    def _create_door_mesh(self, center: Tuple[float, float], width: float, 
                         height: float, thickness: float) -> trimesh.Trimesh:
        """Create a door mesh."""
        # Create door as a rectangular panel
        half_width = width / 2
        half_thickness = thickness / 2
        
        vertices = [
            [center[0] - half_width, center[1] - half_thickness, 0],
            [center[0] + half_width, center[1] - half_thickness, 0],
            [center[0] + half_width, center[1] + half_thickness, 0],
            [center[0] - half_width, center[1] + half_thickness, 0],
            [center[0] - half_width, center[1] - half_thickness, height],
            [center[0] + half_width, center[1] - half_thickness, height],
            [center[0] + half_width, center[1] + half_thickness, height],
            [center[0] - half_width, center[1] + half_thickness, height]
        ]
        
        faces = [
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 7, 6], [4, 6, 5],  # Top
            [0, 4, 5], [0, 5, 1],  # Side 1
            [2, 6, 7], [2, 7, 3],  # Side 2
            [1, 5, 6], [1, 6, 2],  # Front
            [0, 3, 7], [0, 7, 4]   # Back
        ]
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def _build_windows(self, features_data: Dict, depth_data: Dict, scale_factor: float) -> List[trimesh.Trimesh]:
        """Build window meshes."""
        window_meshes = []
        
        windows = features_data.get('windows', [])
        opening_dimensions = depth_data.get('opening_dimensions', {})
        window_specs = opening_dimensions.get('windows', [])
        
        for i, window in enumerate(windows):
            try:
                # Get window specifications
                window_spec = window_specs[i] if i < len(window_specs) else {}
                
                width = window_spec.get('width', 1.0)
                height = window_spec.get('height', 1.2)
                sill_height = window_spec.get('sill_height', 0.9)
                thickness = self.standard_dimensions['window_thickness']
                
                # Get window position
                if 'center' in window:
                    center = (window['center'][0] * scale_factor, window['center'][1] * scale_factor)
                else:
                    center = (0, 0)
                
                # Create window mesh
                window_mesh = self._create_window_mesh(center, width, height, sill_height, thickness)
                
                if window_mesh:
                    window_mesh.visual.material = trimesh.visual.material.PBRMaterial(**self.materials['window'])
                    window_meshes.append(window_mesh)
                    
            except Exception as e:
                logger.warning(f"Failed to create window mesh: {e}")
        
        return window_meshes
    
    def _create_window_mesh(self, center: Tuple[float, float], width: float, 
                          height: float, sill_height: float, thickness: float) -> trimesh.Trimesh:
        """Create a window mesh."""
        half_width = width / 2
        half_thickness = thickness / 2
        
        vertices = [
            [center[0] - half_width, center[1] - half_thickness, sill_height],
            [center[0] + half_width, center[1] - half_thickness, sill_height],
            [center[0] + half_width, center[1] + half_thickness, sill_height],
            [center[0] - half_width, center[1] + half_thickness, sill_height],
            [center[0] - half_width, center[1] - half_thickness, sill_height + height],
            [center[0] + half_width, center[1] - half_thickness, sill_height + height],
            [center[0] + half_width, center[1] + half_thickness, sill_height + height],
            [center[0] - half_width, center[1] + half_thickness, sill_height + height]
        ]
        
        faces = [
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 7, 6], [4, 6, 5],  # Top
            [0, 4, 5], [0, 5, 1],  # Side 1
            [2, 6, 7], [2, 7, 3],  # Side 2
            [1, 5, 6], [1, 6, 2],  # Front
            [0, 3, 7], [0, 7, 4]   # Back
        ]
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def _build_ceiling(self, segmentation_data: Dict, depth_data: Dict, scale_factor: float) -> Optional[trimesh.Trimesh]:
        """Build ceiling mesh."""
        # Use the same floor polygon but at ceiling height
        room_analysis = segmentation_data.get('room_analysis', {})
        ceiling_heights = depth_data.get('ceiling_heights', {})
        
        if not room_analysis:
            return None
        
        # Get average ceiling height
        avg_height = np.mean(list(ceiling_heights.values())) if ceiling_heights else 2.4
        
        # Reuse floor polygon logic but at ceiling height
        floor_polygons = []
        
        for room_type, room_info in room_analysis.items():
            for room in room_info.get('rooms', []):
                contour = room['contour']
                if len(contour) >= 3:
                    points = [(point[0][0] * scale_factor, point[0][1] * scale_factor) 
                             for point in contour]
                    
                    try:
                        polygon = Polygon(points)
                        if polygon.is_valid and polygon.area > 0:
                            floor_polygons.append(polygon)
                    except Exception as e:
                        logger.warning(f"Could not create ceiling polygon: {e}")
        
        if not floor_polygons:
            return None
        
        try:
            combined_ceiling = unary_union(floor_polygons)
            ceiling_mesh = self._polygon_to_mesh(combined_ceiling, avg_height, 
                                               self.standard_dimensions['ceiling_thickness'])
            
            ceiling_mesh.visual.material = trimesh.visual.material.PBRMaterial(**self.materials['ceiling'])
            return ceiling_mesh
            
        except Exception as e:
            logger.error(f"Failed to create ceiling mesh: {e}")
            return None
    
    def _build_fixtures(self, features_data: Dict, depth_data: Dict, scale_factor: float) -> List[trimesh.Trimesh]:
        """Build fixture and furniture meshes."""
        fixture_meshes = []
        
        fixtures = features_data.get('fixtures', [])
        fixture_heights = depth_data.get('fixture_heights', {})
        
        for fixture in fixtures:
            try:
                fixture_type = fixture.get('type', 'unknown')
                category = fixture.get('category', 'furniture')
                
                # Get fixture height
                height = 0.8  # Default height
                for cat_fixtures in fixture_heights.values():
                    for fh in cat_fixtures:
                        if fh.get('type') == fixture_type:
                            height = fh.get('height', 0.8)
                            break
                
                # Create fixture mesh based on type
                if 'bounding_box' in fixture:
                    x, y, w, h = fixture['bounding_box']
                    center = (x * scale_factor + w * scale_factor / 2, 
                             y * scale_factor + h * scale_factor / 2)
                    width = w * scale_factor
                    depth = h * scale_factor
                    
                    fixture_mesh = self._create_fixture_mesh(center, width, depth, height, fixture_type)
                    
                elif 'center' in fixture:
                    center = (fixture['center'][0] * scale_factor, fixture['center'][1] * scale_factor)
                    radius = fixture.get('radius', 0.3) * scale_factor
                    
                    fixture_mesh = self._create_cylindrical_fixture(center, radius, height, fixture_type)
                
                else:
                    continue
                
                if fixture_mesh:
                    fixture_mesh.visual.material = trimesh.visual.material.PBRMaterial(**self.materials['fixture'])
                    fixture_meshes.append(fixture_mesh)
                    
            except Exception as e:
                logger.warning(f"Failed to create fixture mesh: {e}")
        
        return fixture_meshes
    
    def _create_fixture_mesh(self, center: Tuple[float, float], width: float, 
                           depth: float, height: float, fixture_type: str) -> trimesh.Trimesh:
        """Create a rectangular fixture mesh."""
        half_width = width / 2
        half_depth = depth / 2
        
        vertices = [
            [center[0] - half_width, center[1] - half_depth, 0],
            [center[0] + half_width, center[1] - half_depth, 0],
            [center[0] + half_width, center[1] + half_depth, 0],
            [center[0] - half_width, center[1] + half_depth, 0],
            [center[0] - half_width, center[1] - half_depth, height],
            [center[0] + half_width, center[1] - half_depth, height],
            [center[0] + half_width, center[1] + half_depth, height],
            [center[0] - half_width, center[1] + half_depth, height]
        ]
        
        faces = [
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 7, 6], [4, 6, 5],  # Top
            [0, 4, 5], [0, 5, 1],  # Side 1
            [2, 6, 7], [2, 7, 3],  # Side 2
            [1, 5, 6], [1, 6, 2],  # Front
            [0, 3, 7], [0, 7, 4]   # Back
        ]
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def _create_cylindrical_fixture(self, center: Tuple[float, float], radius: float, 
                                  height: float, fixture_type: str) -> trimesh.Trimesh:
        """Create a cylindrical fixture mesh."""
        return trimesh.creation.cylinder(radius=radius, height=height, 
                                       transform=trimesh.transformations.translation_matrix(
                                           [center[0], center[1], height/2]))
    
    def _build_stairs(self, segmentation_data: Dict, depth_data: Dict, scale_factor: float) -> List[trimesh.Trimesh]:
        """Build stair meshes if stairs are detected."""
        stair_meshes = []
        
        # Check if stairs class exists in segmentation
        segmentation_map = segmentation_data.get('segmentation_map')
        if segmentation_map is None:
            return stair_meshes
        
        stairs_class = 9  # Stairs class ID
        stairs_mask = segmentation_map == stairs_class
        
        if not np.any(stairs_mask):
            return stair_meshes
        
        # Find stair regions
        import cv2
        num_labels, labels = cv2.connectedComponents(stairs_mask.astype(np.uint8))
        
        for label in range(1, num_labels):
            try:
                stair_mask = labels == label
                
                # Find contour of stair region
                contours, _ = cv2.findContours(stair_mask.astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Create simple stair mesh (simplified as steps)
                    stair_mesh = self._create_stairs_mesh(largest_contour, scale_factor)
                    
                    if stair_mesh:
                        stair_mesh.visual.material = trimesh.visual.material.PBRMaterial(**self.materials['wall'])
                        stair_meshes.append(stair_mesh)
                        
            except Exception as e:
                logger.warning(f"Failed to create stairs mesh: {e}")
        
        return stair_meshes
    
    def _create_stairs_mesh(self, contour: np.ndarray, scale_factor: float) -> Optional[trimesh.Trimesh]:
        """Create a simplified stairs mesh."""
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Convert to real coordinates
        x *= scale_factor
        y *= scale_factor
        w *= scale_factor
        h *= scale_factor
        
        # Create simple stepped geometry
        num_steps = max(3, int(max(w, h) / 0.3))  # Roughly 30cm per step
        step_height = 0.18  # 18cm step height
        step_depth = max(w, h) / num_steps
        
        vertices = []
        faces = []
        
        # Create steps
        for i in range(num_steps):
            z_base = i * step_height
            z_top = (i + 1) * step_height
            
            if w > h:  # Steps along width
                x_start = x + i * step_depth
                x_end = x + (i + 1) * step_depth
                
                # Add step vertices
                base_idx = len(vertices)
                vertices.extend([
                    [x_start, y, z_base], [x_end, y, z_base],
                    [x_end, y + h, z_base], [x_start, y + h, z_base],
                    [x_start, y, z_top], [x_end, y, z_top],
                    [x_end, y + h, z_top], [x_start, y + h, z_top]
                ])
            else:  # Steps along height
                y_start = y + i * step_depth
                y_end = y + (i + 1) * step_depth
                
                base_idx = len(vertices)
                vertices.extend([
                    [x, y_start, z_base], [x + w, y_start, z_base],
                    [x + w, y_end, z_base], [x, y_end, z_base],
                    [x, y_start, z_top], [x + w, y_start, z_top],
                    [x + w, y_end, z_top], [x, y_end, z_top]
                ])
            
            # Add step faces
            step_faces = [
                [base_idx, base_idx+1, base_idx+2], [base_idx, base_idx+2, base_idx+3],  # Bottom
                [base_idx+4, base_idx+7, base_idx+6], [base_idx+4, base_idx+6, base_idx+5],  # Top
                [base_idx, base_idx+4, base_idx+5], [base_idx, base_idx+5, base_idx+1],  # Side 1
                [base_idx+2, base_idx+6, base_idx+7], [base_idx+2, base_idx+7, base_idx+3],  # Side 2
                [base_idx+1, base_idx+5, base_idx+6], [base_idx+1, base_idx+6, base_idx+2],  # Front
                [base_idx, base_idx+3, base_idx+7], [base_idx, base_idx+7, base_idx+4]   # Back
            ]
            faces.extend(step_faces)
        
        if vertices and faces:
            return trimesh.Trimesh(vertices=vertices, faces=faces)
        
        return None
    
    def _polygon_to_mesh(self, polygon: Polygon, z_base: float, thickness: float) -> trimesh.Trimesh:
        """Convert a 2D polygon to a 3D mesh with thickness."""
        # Get exterior coordinates
        coords = list(polygon.exterior.coords[:-1])  # Remove duplicate last point
        
        # Create vertices for bottom and top
        vertices = []
        
        # Bottom vertices
        for x, y in coords:
            vertices.append([x, y, z_base])
        
        # Top vertices
        for x, y in coords:
            vertices.append([x, y, z_base + thickness])
        
        # Create faces
        faces = []
        n = len(coords)
        
        # Bottom face (triangulate)
        for i in range(1, n - 1):
            faces.append([0, i, i + 1])
        
        # Top face (triangulate, reverse order for correct normal)
        for i in range(1, n - 1):
            faces.append([n, n + i + 1, n + i])
        
        # Side faces
        for i in range(n):
            next_i = (i + 1) % n
            # Two triangles per side
            faces.append([i, next_i, n + next_i])
            faces.append([i, n + next_i, n + i])
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def _generate_metadata(self, segmentation_data: Dict, features_data: Dict, 
                          depth_data: Dict, scale_factor: float) -> Dict:
        """Generate metadata for the 3D model."""
        metadata = {
            'scale_factor': scale_factor,
            'units': 'meters',
            'creation_timestamp': np.datetime64('now').astype(str),
            'model_info': {
                'total_rooms': len(segmentation_data.get('room_analysis', {})),
                'total_walls': len(features_data.get('walls', [])),
                'total_doors': len(features_data.get('doors', [])),
                'total_windows': len(features_data.get('windows', [])),
                'total_fixtures': len(features_data.get('fixtures', []))
            },
            'dimensions': {
                'bounding_box': self.scene.bounds.tolist() if hasattr(self.scene, 'bounds') else None,
                'floor_area': self._calculate_total_floor_area(segmentation_data, scale_factor),
                'wall_heights': depth_data.get('ceiling_heights', {})
            },
            'materials': self.materials
        }
        
        return metadata
    
    def _calculate_total_floor_area(self, segmentation_data: Dict, scale_factor: float) -> float:
        """Calculate total floor area."""
        room_analysis = segmentation_data.get('room_analysis', {})
        total_area = 0
        
        for room_type, room_info in room_analysis.items():
            total_area += room_info.get('total_area', 0)
        
        # Convert from pixels to square meters
        return total_area * (scale_factor ** 2)
    
    def export_model(self, output_path: str, model_data: Dict) -> bool:
        """
        Export the 3D model to file.
        
        Args:
            output_path: Path to save the model
            model_data: 3D model data from reconstruct_3d_model
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            scene = model_data['scene']
            
            if self.output_format == 'obj':
                scene.export(output_path)
            elif self.output_format == 'gltf':
                scene.export(output_path)
            elif self.output_format == 'stl':
                # STL doesn't support scenes, so merge all meshes
                combined_mesh = trimesh.util.concatenate([mesh for mesh in scene.geometry.values()])
                combined_mesh.export(output_path)
            elif self.output_format == 'ply':
                combined_mesh = trimesh.util.concatenate([mesh for mesh in scene.geometry.values()])
                combined_mesh.export(output_path)
            else:
                logger.error(f"Unsupported export format: {self.output_format}")
                return False
            
            # Export metadata
            metadata_path = output_path.rsplit('.', 1)[0] + '_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(model_data['metadata'], f, indent=2)
            
            logger.info(f"Model exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            return False