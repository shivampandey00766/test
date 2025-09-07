"""
Depth Estimation Module for 2D to 3D Conversion

This module estimates depth information and height parameters from 2D floor plans
to enable 3D reconstruction with realistic proportions and spatial relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    Estimates depth and height information from 2D floor plans for 3D reconstruction.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize the depth estimator.
        
        Args:
            device: Device to run inference on
        """
        self.device = self._setup_device(device)
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Standard architectural heights (in relative units)
        self.standard_heights = {
            'wall_height': 2.4,      # Standard ceiling height
            'door_height': 2.1,      # Standard door height
            'window_height': 1.2,    # Standard window height
            'window_sill': 0.9,      # Standard window sill height
            'counter_height': 0.9,   # Kitchen counter height
            'cabinet_height': 2.1,   # Upper cabinet height
            'fixture_height': 0.8    # General fixture height
        }
        
    def estimate_depths(self, image: np.ndarray, segmentation_data: Dict, 
                       features_data: Dict) -> Dict:
        """
        Estimate depth and height information from floor plan analysis.
        
        Args:
            image: Preprocessed floor plan image
            segmentation_data: Results from semantic segmentation
            features_data: Results from feature detection
            
        Returns:
            Dictionary containing depth and height estimates
        """
        # Estimate wall heights based on room types and context
        wall_heights = self._estimate_wall_heights(segmentation_data, features_data)
        
        # Estimate door and window dimensions
        opening_dimensions = self._estimate_opening_dimensions(features_data)
        
        # Estimate room ceiling heights
        ceiling_heights = self._estimate_ceiling_heights(segmentation_data)
        
        # Estimate fixture and furniture heights
        fixture_heights = self._estimate_fixture_heights(features_data)
        
        # Generate depth map using neural network
        depth_map = self._generate_depth_map(image)
        
        # Combine all depth information
        depth_data = {
            'wall_heights': wall_heights,
            'opening_dimensions': opening_dimensions,
            'ceiling_heights': ceiling_heights,
            'fixture_heights': fixture_heights,
            'depth_map': depth_map,
            'scale_factor': self._estimate_scale_factor(image, features_data)
        }
        
        logger.info("Depth estimation completed")
        return depth_data
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _create_model(self) -> nn.Module:
        """Create depth estimation neural network."""
        return DepthEstimationNet()
    
    def _estimate_wall_heights(self, segmentation_data: Dict, features_data: Dict) -> Dict:
        """
        Estimate wall heights based on room types and architectural context.
        """
        wall_heights = {}
        
        # Get room analysis from segmentation
        room_analysis = segmentation_data.get('room_analysis', {})
        
        # Default wall height
        default_height = self.standard_heights['wall_height']
        
        for room_type, room_info in room_analysis.items():
            # Adjust height based on room type
            if 'bathroom' in room_type.lower():
                height = default_height  # Standard height for bathrooms
            elif 'kitchen' in room_type.lower():
                height = default_height  # Standard height for kitchens
            elif 'living' in room_type.lower():
                height = default_height * 1.2  # Higher ceilings for living areas
            elif 'bedroom' in room_type.lower():
                height = default_height  # Standard height for bedrooms
            elif 'hallway' in room_type.lower():
                height = default_height  # Standard height for hallways
            else:
                height = default_height
            
            wall_heights[room_type] = {
                'height': height,
                'rooms': room_info['rooms']
            }
        
        # Process individual walls from feature detection
        walls = features_data.get('walls', [])
        individual_walls = []
        
        for wall in walls:
            # Determine wall type and adjust height
            wall_height = default_height
            
            # Structural walls might be thicker and potentially higher
            if wall.get('structural', False):
                wall_height *= 1.1
            
            # Exterior walls (detected by position or thickness) might be higher
            if wall.get('thickness', 0) > 10:  # Thick walls are likely exterior
                wall_height *= 1.05
            
            wall_info = {
                'start': wall['start'],
                'end': wall['end'],
                'height': wall_height,
                'thickness': wall.get('thickness', 6),
                'material': wall.get('material', 'drywall')
            }
            individual_walls.append(wall_info)
        
        wall_heights['individual_walls'] = individual_walls
        
        return wall_heights
    
    def _estimate_opening_dimensions(self, features_data: Dict) -> Dict:
        """
        Estimate dimensions for doors and windows.
        """
        opening_dimensions = {
            'doors': [],
            'windows': []
        }
        
        # Process doors
        doors = features_data.get('doors', [])
        for door in doors:
            door_type = door.get('type', 'swing_door')
            
            # Estimate door dimensions based on type
            if door_type == 'swing_door':
                width = door.get('width', 0.8)  # Standard door width
                height = self.standard_heights['door_height']
            elif door_type == 'double_door':
                width = door.get('total_width', 1.6)  # Double door width
                height = self.standard_heights['door_height']
            elif door_type == 'sliding_door':
                width = door.get('width', 1.2)  # Sliding door width
                height = self.standard_heights['door_height']
            else:
                width = 0.8
                height = self.standard_heights['door_height']
            
            door_info = {
                'type': door_type,
                'width': width,
                'height': height,
                'position': door.get('center', (0, 0)),
                'opening_angle': door.get('opening_angle', 90)
            }
            opening_dimensions['doors'].append(door_info)
        
        # Process windows
        windows = features_data.get('windows', [])
        for window in windows:
            window_type = window.get('type', 'standard_window')
            
            # Estimate window dimensions
            if window_type == 'standard_window':
                width = window.get('width', 1.0)
                height = self.standard_heights['window_height']
                sill_height = self.standard_heights['window_sill']
            elif window_type == 'bay_window':
                width = window.get('width', 1.5)
                height = self.standard_heights['window_height']
                sill_height = self.standard_heights['window_sill']
                projection = window.get('projection', 0.3)
            else:
                width = 1.0
                height = self.standard_heights['window_height']
                sill_height = self.standard_heights['window_sill']
            
            window_info = {
                'type': window_type,
                'width': width,
                'height': height,
                'sill_height': sill_height,
                'position': window.get('center', (0, 0))
            }
            
            if window_type == 'bay_window':
                window_info['projection'] = window.get('projection', 0.3)
            
            opening_dimensions['windows'].append(window_info)
        
        return opening_dimensions
    
    def _estimate_ceiling_heights(self, segmentation_data: Dict) -> Dict:
        """
        Estimate ceiling heights for different rooms.
        """
        ceiling_heights = {}
        
        room_analysis = segmentation_data.get('room_analysis', {})
        
        for room_type, room_info in room_analysis.items():
            # Base ceiling height
            base_height = self.standard_heights['wall_height']
            
            # Adjust based on room type
            if 'living' in room_type.lower():
                ceiling_height = base_height * 1.2  # Higher ceilings for living areas
            elif 'kitchen' in room_type.lower():
                ceiling_height = base_height  # Standard height
            elif 'bathroom' in room_type.lower():
                ceiling_height = base_height * 0.95  # Slightly lower for bathrooms
            elif 'bedroom' in room_type.lower():
                ceiling_height = base_height  # Standard height
            elif 'hallway' in room_type.lower():
                ceiling_height = base_height * 0.9  # Lower for hallways
            else:
                ceiling_height = base_height
            
            # Consider room size - larger rooms might have higher ceilings
            if room_info.get('total_area', 0) > 5000:  # Large rooms
                ceiling_height *= 1.1
            
            ceiling_heights[room_type] = ceiling_height
        
        return ceiling_heights
    
    def _estimate_fixture_heights(self, features_data: Dict) -> Dict:
        """
        Estimate heights for fixtures and furniture.
        """
        fixture_heights = {
            'plumbing': [],
            'kitchen': [],
            'furniture': []
        }
        
        fixtures = features_data.get('fixtures', [])
        
        for fixture in fixtures:
            fixture_type = fixture.get('type', 'unknown')
            category = fixture.get('category', 'furniture')
            
            # Estimate height based on fixture type
            if fixture_type == 'sink':
                height = self.standard_heights['counter_height']
            elif fixture_type == 'toilet':
                height = 0.4  # Toilet height
            elif fixture_type == 'bathtub':
                height = 0.6  # Bathtub height
            elif fixture_type == 'counter':
                height = self.standard_heights['counter_height']
            elif fixture_type == 'cabinet':
                height = self.standard_heights['cabinet_height']
            elif fixture_type == 'kitchen_island':
                height = self.standard_heights['counter_height']
            else:
                height = self.standard_heights['fixture_height']
            
            fixture_info = {
                'type': fixture_type,
                'height': height,
                'position': fixture.get('center', (0, 0)),
                'area': fixture.get('area', 0)
            }
            
            fixture_heights[category].append(fixture_info)
        
        return fixture_heights
    
    def _generate_depth_map(self, image: np.ndarray) -> np.ndarray:
        """
        Generate depth map using neural network.
        """
        # Prepare image for neural network
        input_tensor = self._prepare_image_for_depth(image)
        
        # Run depth estimation
        with torch.no_grad():
            depth_output = self.model(input_tensor)
            depth_map = depth_output[0, 0].cpu().numpy()
        
        # Normalize depth map
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        # Apply architectural constraints
        depth_map = self._apply_architectural_constraints(depth_map, image)
        
        return depth_map
    
    def _prepare_image_for_depth(self, image: np.ndarray) -> torch.Tensor:
        """Prepare image for depth estimation network."""
        # Resize to network input size
        resized = cv2.resize(image, (256, 256))
        
        # Convert to tensor
        if len(resized.shape) == 2:
            resized = np.stack([resized] * 3, axis=-1)
        
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def _apply_architectural_constraints(self, depth_map: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Apply architectural constraints to improve depth estimation.
        """
        # Smooth depth transitions
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 1.0)
        
        # Ensure walls have consistent depth
        edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
        wall_mask = edges > 0
        
        # Set wall areas to consistent depth
        if np.any(wall_mask):
            wall_depth = np.median(depth_map[wall_mask])
            depth_map[wall_mask] = wall_depth
        
        return depth_map
    
    def _estimate_scale_factor(self, image: np.ndarray, features_data: Dict) -> float:
        """
        Estimate scale factor to convert pixel measurements to real-world units.
        """
        # Use door width as reference (standard door is ~0.8m wide)
        doors = features_data.get('doors', [])
        
        if doors:
            # Find average door width in pixels
            door_widths = []
            for door in doors:
                if 'width' in door:
                    door_widths.append(door['width'])
            
            if door_widths:
                avg_door_width_pixels = np.mean(door_widths)
                standard_door_width_meters = 0.8
                scale_factor = standard_door_width_meters / avg_door_width_pixels
                return scale_factor
        
        # Fallback: use image dimensions and assume typical floor plan size
        image_width = image.shape[1]
        assumed_real_width = 10.0  # Assume 10 meters wide
        scale_factor = assumed_real_width / image_width
        
        return scale_factor
    
    def create_height_map(self, segmentation_map: np.ndarray, depth_data: Dict) -> np.ndarray:
        """
        Create a height map where each pixel represents the height at that location.
        
        Args:
            segmentation_map: Semantic segmentation result
            depth_data: Depth estimation results
            
        Returns:
            Height map as numpy array
        """
        height_map = np.zeros_like(segmentation_map, dtype=np.float32)
        
        # Apply heights based on segmentation classes
        wall_class = 1
        door_class = 2
        window_class = 3
        
        # Set wall heights
        wall_mask = segmentation_map == wall_class
        if np.any(wall_mask):
            height_map[wall_mask] = self.standard_heights['wall_height']
        
        # Set door heights
        door_mask = segmentation_map == door_class
        if np.any(door_mask):
            height_map[door_mask] = self.standard_heights['door_height']
        
        # Set window areas (window + sill)
        window_mask = segmentation_map == window_class
        if np.any(window_mask):
            height_map[window_mask] = self.standard_heights['window_height'] + self.standard_heights['window_sill']
        
        # Apply room-specific ceiling heights
        room_classes = [4, 5, 6, 7, 8, 10, 11]  # Room classes
        ceiling_heights = depth_data.get('ceiling_heights', {})
        
        for room_class in room_classes:
            room_mask = segmentation_map == room_class
            if np.any(room_mask):
                # Find corresponding ceiling height
                room_name = None
                for name, class_id in {'room_living': 4, 'room_bedroom': 5, 'room_kitchen': 6,
                                     'room_bathroom': 7, 'room_hallway': 8, 'closet': 10, 'balcony': 11}.items():
                    if class_id == room_class:
                        room_name = name
                        break
                
                if room_name and room_name in ceiling_heights:
                    height_map[room_mask] = ceiling_heights[room_name]
                else:
                    height_map[room_mask] = self.standard_heights['wall_height']
        
        return height_map


class DepthEstimationNet(nn.Module):
    """
    Neural network for depth estimation from floor plan images.
    """
    
    def __init__(self):
        super(DepthEstimationNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Final depth prediction
        self.depth_head = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()  # Normalize depth to [0, 1]
        )
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        # Predict depth
        depth = self.depth_head(decoded)
        
        return depth