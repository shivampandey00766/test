"""
Depth estimation module for 2D to 3D conversion.

This module estimates depth information from 2D floor plans to enable
3D reconstruction with proper proportions and heights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class DepthEstimationCNN(nn.Module):
    """
    CNN for depth estimation from 2D floor plan images.
    
    This model predicts depth maps that can be used for 3D reconstruction
    with proper wall heights and architectural proportions.
    """
    
    def __init__(self, input_channels: int = 1, output_channels: int = 1):
        super(DepthEstimationCNN, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Block 1
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Block 2
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Block 3
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Block 4
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Output
            nn.Conv2d(32, output_channels, kernel_size=1),
            nn.Sigmoid()  # Normalize to [0, 1]
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DepthEstimator:
    """
    Main class for depth estimation from floor plans.
    
    This class combines deep learning with architectural heuristics
    to estimate depth information for 3D reconstruction.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 device: str = 'auto'):
        """
        Initialize the DepthEstimator.
        
        Args:
            model_path: Path to pre-trained model weights
            device: Device to run inference on
        """
        self.device = self._get_device(device)
        
        # Initialize model
        self.model = DepthEstimationCNN(input_channels=1, output_channels=1)
        self.model.to(self.device)
        
        # Load pre-trained weights if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        # Architectural constants (in meters)
        self.architectural_constants = {
            'standard_wall_height': 2.4,      # 2.4m standard wall height
            'door_height': 2.1,               # 2.1m door height
            'window_height': 1.2,             # 1.2m window height
            'ceiling_height': 2.7,            # 2.7m ceiling height
            'floor_thickness': 0.2,           # 0.2m floor thickness
        }
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for computation."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def load_model(self, model_path: str):
        """Load model weights from file."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Depth estimation model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading depth estimation model: {e}")
    
    def estimate_depth_cnn(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth using CNN model.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Depth map as numpy array
        """
        self.model.eval()
        
        # Preprocess image
        image_resized = cv2.resize(image, (256, 256))
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            depth_tensor = self.model(image_tensor)
            depth_map = depth_tensor.cpu().numpy()[0, 0]
        
        return depth_map
    
    def estimate_depth_heuristic(self, segmentation_mask: np.ndarray) -> np.ndarray:
        """
        Estimate depth using architectural heuristics.
        
        Args:
            segmentation_mask: Segmentation mask with different classes
            
        Returns:
            Depth map based on architectural rules
        """
        height, width = segmentation_mask.shape
        depth_map = np.zeros((height, width), dtype=np.float32)
        
        # Define depth values for different architectural elements
        depth_values = {
            0: 0.0,      # Background - ground level
            1: 0.0,      # Walls - ground level
            2: 0.0,      # Doors - ground level
            3: 0.0,      # Windows - ground level
            4: 0.0,      # Rooms - ground level
        }
        
        # Apply depth values
        for class_id, depth_value in depth_values.items():
            mask = segmentation_mask == class_id
            depth_map[mask] = depth_value
        
        return depth_map
    
    def estimate_wall_heights(self, segmentation_mask: np.ndarray) -> np.ndarray:
        """
        Estimate wall heights based on architectural elements.
        
        Args:
            segmentation_mask: Segmentation mask
            
        Returns:
            Height map for walls
        """
        height, width = segmentation_mask.shape
        height_map = np.zeros((height, width), dtype=np.float32)
        
        # Standard wall height
        wall_height = self.architectural_constants['standard_wall_height']
        
        # Find wall regions
        wall_mask = segmentation_mask == 1
        
        # Apply wall height
        height_map[wall_mask] = wall_height
        
        # Adjust heights based on nearby elements
        # Windows typically reduce wall height
        window_mask = segmentation_mask == 3
        if np.any(window_mask):
            # Dilate window mask to affect nearby walls
            kernel = np.ones((5, 5), np.uint8)
            window_dilated = cv2.dilate(window_mask.astype(np.uint8), kernel, iterations=2)
            
            # Reduce wall height near windows
            wall_near_window = wall_mask & window_dilated.astype(bool)
            height_map[wall_near_window] = wall_height * 0.7
        
        return height_map
    
    def estimate_room_heights(self, segmentation_mask: np.ndarray, 
                            room_types: List[str] = None) -> np.ndarray:
        """
        Estimate room heights based on room types and architectural standards.
        
        Args:
            segmentation_mask: Segmentation mask
            room_types: List of room types for each room region
            
        Returns:
            Height map for rooms
        """
        height, width = segmentation_mask.shape
        height_map = np.zeros((height, width), dtype=np.float32)
        
        # Standard ceiling heights for different room types
        room_heights = {
            'living_room': 2.7,
            'bedroom': 2.4,
            'kitchen': 2.4,
            'bathroom': 2.1,
            'dining_room': 2.7,
            'hallway': 2.4,
            'default': 2.4
        }
        
        # Find room regions
        room_mask = segmentation_mask == 4
        
        if room_types:
            # Use specific room type heights
            for i, room_type in enumerate(room_types):
                room_height = room_heights.get(room_type, room_heights['default'])
                height_map[room_mask] = room_height
        else:
            # Use default height
            height_map[room_mask] = room_heights['default']
        
        return height_map
    
    def estimate_depth(self, image: np.ndarray, 
                      segmentation_mask: np.ndarray = None,
                      room_types: List[str] = None,
                      use_cnn: bool = True) -> Dict[str, np.ndarray]:
        """
        Estimate comprehensive depth information for 3D reconstruction.
        
        Args:
            image: Input floor plan image
            segmentation_mask: Segmentation mask (optional)
            room_types: List of room types (optional)
            use_cnn: Whether to use CNN for depth estimation
            
        Returns:
            Dictionary containing various depth maps
        """
        results = {}
        
        # Estimate base depth using CNN or heuristics
        if use_cnn:
            base_depth = self.estimate_depth_cnn(image)
        else:
            base_depth = self.estimate_depth_heuristic(segmentation_mask)
        
        results['base_depth'] = base_depth
        
        # Estimate wall heights
        if segmentation_mask is not None:
            wall_heights = self.estimate_wall_heights(segmentation_mask)
            results['wall_heights'] = wall_heights
            
            # Estimate room heights
            room_heights = self.estimate_room_heights(segmentation_mask, room_types)
            results['room_heights'] = room_heights
            
            # Combine depth information
            combined_depth = self._combine_depth_maps(base_depth, wall_heights, room_heights)
            results['combined_depth'] = combined_depth
        
        return results
    
    def _combine_depth_maps(self, base_depth: np.ndarray,
                          wall_heights: np.ndarray,
                          room_heights: np.ndarray) -> np.ndarray:
        """
        Combine different depth maps into a comprehensive depth map.
        
        Args:
            base_depth: Base depth map
            wall_heights: Wall height map
            room_heights: Room height map
            
        Returns:
            Combined depth map
        """
        # Start with base depth
        combined = base_depth.copy()
        
        # Add wall heights
        wall_mask = wall_heights > 0
        combined[wall_mask] = wall_heights[wall_mask]
        
        # Add room heights
        room_mask = room_heights > 0
        combined[room_mask] = room_heights[room_mask]
        
        return combined
    
    def normalize_depth_map(self, depth_map: np.ndarray, 
                          min_depth: float = 0.0,
                          max_depth: float = 3.0) -> np.ndarray:
        """
        Normalize depth map to specified range.
        
        Args:
            depth_map: Input depth map
            min_depth: Minimum depth value
            max_depth: Maximum depth value
            
        Returns:
            Normalized depth map
        """
        # Normalize to [0, 1]
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        
        if depth_max > depth_min:
            normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(depth_map)
        
        # Scale to desired range
        scaled = normalized * (max_depth - min_depth) + min_depth
        
        return scaled
    
    def visualize_depth_map(self, depth_map: np.ndarray,
                          title: str = "Depth Map",
                          save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize depth map with color coding.
        
        Args:
            depth_map: Depth map to visualize
            title: Title for the plot
            save_path: Optional path to save visualization
            
        Returns:
            Colored depth map
        """
        # Normalize depth map for visualization
        depth_normalized = self.normalize_depth_map(depth_map, 0, 1)
        
        # Apply colormap
        colored_depth = plt.cm.viridis(depth_normalized)[:, :, :3]
        colored_depth = (colored_depth * 255).astype(np.uint8)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(colored_depth)
        plt.title(title)
        plt.colorbar(label='Depth (normalized)')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return colored_depth
    
    def export_depth_data(self, depth_results: Dict[str, np.ndarray],
                         output_path: str):
        """
        Export depth data for 3D reconstruction.
        
        Args:
            depth_results: Dictionary of depth maps
            output_path: Path to save depth data
        """
        # Save depth maps as numpy arrays
        for key, depth_map in depth_results.items():
            np.save(f"{output_path}_{key}.npy", depth_map)
        
        # Save metadata
        metadata = {
            'architectural_constants': self.architectural_constants,
            'depth_maps': list(depth_results.keys()),
            'image_shape': depth_results['base_depth'].shape
        }
        
        import json
        with open(f"{output_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Depth data exported to {output_path}")
    
    def get_depth_statistics(self, depth_map: np.ndarray) -> Dict[str, float]:
        """
        Get statistics about the depth map.
        
        Args:
            depth_map: Input depth map
            
        Returns:
            Dictionary with depth statistics
        """
        return {
            'min_depth': float(np.min(depth_map)),
            'max_depth': float(np.max(depth_map)),
            'mean_depth': float(np.mean(depth_map)),
            'std_depth': float(np.std(depth_map)),
            'median_depth': float(np.median(depth_map))
        }