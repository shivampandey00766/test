"""
Depth estimation model for inferring 3D structure from 2D floor plans.
Uses deep learning to predict wall heights and spatial relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DepthEstimationModel(nn.Module):
    """
    CNN-based model for depth estimation from floor plan segmentation.
    Predicts wall heights and spatial relationships.
    """
    
    def __init__(self, input_channels: int = 11, hidden_dim: int = 256):
        """
        Initialize the depth estimation model.
        
        Args:
            input_channels: Number of input channels (segmentation classes)
            hidden_dim: Hidden dimension size
        """
        super(DepthEstimationModel, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Depth prediction head
        self.depth_head = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),  # Single depth value
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
        # Wall height prediction head
        self.height_head = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),  # Wall height
            nn.Sigmoid()
        )
        
        # Spatial relationship head
        self.spatial_head = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 6),  # 6 spatial features
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, segmentation_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            segmentation_mask: One-hot encoded segmentation mask
            
        Returns:
            Dictionary containing depth predictions
        """
        # Encode features
        features = self.encoder(segmentation_mask)
        
        # Global pooling
        global_features = self.global_pool(features)
        global_features = global_features.view(global_features.size(0), -1)
        
        # Predict depth, height, and spatial features
        depth = self.depth_head(global_features)
        wall_height = self.height_head(global_features)
        spatial_features = self.spatial_head(global_features)
        
        return {
            'depth': depth,
            'wall_height': wall_height,
            'spatial_features': spatial_features
        }
    
    def predict_depth(self, segmentation_mask: np.ndarray, device: str = 'cpu') -> Dict[str, float]:
        """
        Predict depth information from segmentation mask.
        
        Args:
            segmentation_mask: Segmentation mask as numpy array
            device: Device to run inference on
            
        Returns:
            Dictionary containing depth predictions
        """
        try:
            self.eval()
            
            # Convert segmentation mask to one-hot encoding
            one_hot = self._to_one_hot(segmentation_mask)
            
            # Convert to tensor
            mask_tensor = torch.from_numpy(one_hot).float().unsqueeze(0)
            mask_tensor = mask_tensor.to(device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.forward(mask_tensor)
            
            # Convert to numpy and extract values
            depth = predictions['depth'].cpu().numpy()[0, 0]
            wall_height = predictions['wall_height'].cpu().numpy()[0, 0]
            spatial_features = predictions['spatial_features'].cpu().numpy()[0]
            
            # Scale predictions to realistic values
            depth_scaled = depth * 10.0  # Scale to 0-10 meters
            wall_height_scaled = wall_height * 3.0  # Scale to 0-3 meters
            
            return {
                'depth': float(depth_scaled),
                'wall_height': float(wall_height_scaled),
                'spatial_features': spatial_features.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error in depth prediction: {e}")
            raise
    
    def _to_one_hot(self, mask: np.ndarray) -> np.ndarray:
        """Convert segmentation mask to one-hot encoding."""
        try:
            # Get unique classes
            unique_classes = np.unique(mask)
            num_classes = len(unique_classes)
            
            # Create one-hot encoding
            one_hot = np.zeros((num_classes, mask.shape[0], mask.shape[1]), dtype=np.float32)
            
            for i, class_id in enumerate(unique_classes):
                one_hot[i, mask == class_id] = 1.0
            
            return one_hot
            
        except Exception as e:
            self.logger.error(f"Error creating one-hot encoding: {e}")
            raise
    
    def estimate_room_heights(self, segmentation_mask: np.ndarray, 
                            room_centers: List[Tuple[int, int]], 
                            device: str = 'cpu') -> Dict[Tuple[int, int], float]:
        """
        Estimate heights for different rooms based on their type and position.
        
        Args:
            segmentation_mask: Segmentation mask
            room_centers: List of room center coordinates
            device: Device to run inference on
            
        Returns:
            Dictionary mapping room centers to estimated heights
        """
        try:
            room_heights = {}
            
            # Get base predictions
            predictions = self.predict_depth(segmentation_mask, device)
            base_height = predictions['wall_height']
            
            # Estimate heights for each room
            for center in room_centers:
                y, x = center
                
                # Get room type from segmentation
                room_type = self._get_room_type_at_position(segmentation_mask, x, y)
                
                # Adjust height based on room type
                height_multiplier = self._get_room_height_multiplier(room_type)
                estimated_height = base_height * height_multiplier
                
                room_heights[center] = estimated_height
            
            return room_heights
            
        except Exception as e:
            self.logger.error(f"Error estimating room heights: {e}")
            return {}
    
    def _get_room_type_at_position(self, mask: np.ndarray, x: int, y: int) -> str:
        """Get room type at specific position."""
        try:
            # Check if coordinates are within bounds
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                class_id = mask[y, x]
                
                # Map class IDs to room types
                room_type_map = {
                    4: 'living_room',
                    5: 'kitchen',
                    6: 'bedroom',
                    7: 'bathroom',
                    8: 'closet'
                }
                
                return room_type_map.get(class_id, 'unknown')
            else:
                return 'unknown'
                
        except Exception as e:
            self.logger.error(f"Error getting room type: {e}")
            return 'unknown'
    
    def _get_room_height_multiplier(self, room_type: str) -> float:
        """Get height multiplier based on room type."""
        height_multipliers = {
            'living_room': 1.0,
            'kitchen': 0.9,
            'bedroom': 0.95,
            'bathroom': 0.8,
            'closet': 0.7,
            'unknown': 1.0
        }
        
        return height_multipliers.get(room_type, 1.0)
    
    def predict_opening_heights(self, segmentation_mask: np.ndarray, 
                              doors: List[Tuple[int, int]], 
                              windows: List[Tuple[int, int]], 
                              device: str = 'cpu') -> Dict[str, List[float]]:
        """
        Predict heights for doors and windows.
        
        Args:
            segmentation_mask: Segmentation mask
            doors: List of door positions
            windows: List of window positions
            device: Device to run inference on
            
        Returns:
            Dictionary containing door and window heights
        """
        try:
            predictions = self.predict_depth(segmentation_mask, device)
            base_height = predictions['wall_height']
            
            # Standard door and window heights (as multipliers of wall height)
            door_height = base_height * 0.8  # Doors are typically 80% of wall height
            window_height = base_height * 0.6  # Windows are typically 60% of wall height
            
            return {
                'door_heights': [door_height] * len(doors),
                'window_heights': [window_height] * len(windows)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting opening heights: {e}")
            return {'door_heights': [], 'window_heights': []}
    
    def save_model(self, filepath: str) -> None:
        """Save the model to file."""
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'input_channels': self.input_channels,
                'hidden_dim': self.hidden_dim
            }, filepath)
            self.logger.info(f"Depth estimation model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu') -> 'DepthEstimationModel':
        """Load a saved model from file."""
        try:
            checkpoint = torch.load(filepath, map_location=device)
            
            model = cls(
                input_channels=checkpoint['input_channels'],
                hidden_dim=checkpoint['hidden_dim']
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            logger.info(f"Depth estimation model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


def create_depth_estimation_model(input_channels: int = 11, 
                                hidden_dim: int = 256) -> DepthEstimationModel:
    """
    Factory function to create a depth estimation model.
    
    Args:
        input_channels: Number of input channels
        hidden_dim: Hidden dimension size
        
    Returns:
        Configured depth estimation model
    """
    return DepthEstimationModel(input_channels, hidden_dim)