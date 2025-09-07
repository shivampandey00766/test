"""
Room classification model for identifying room types and their properties.
Uses deep learning to classify rooms and predict their characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RoomClassifier(nn.Module):
    """
    CNN-based model for room classification and property prediction.
    """
    
    # Room type labels
    ROOM_TYPES = {
        0: 'living_room',
        1: 'kitchen',
        2: 'bedroom',
        3: 'bathroom',
        4: 'closet',
        5: 'dining_room',
        6: 'office',
        7: 'hallway',
        8: 'unknown'
    }
    
    def __init__(self, input_channels: int = 1, num_room_types: int = 9, 
                 hidden_dim: int = 512):
        """
        Initialize the room classifier.
        
        Args:
            input_channels: Number of input channels
            num_room_types: Number of room types to classify
            hidden_dim: Hidden dimension size
        """
        super(RoomClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.num_room_types = num_room_types
        self.hidden_dim = hidden_dim
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Room type classification head
        self.room_classifier = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_room_types)
        )
        
        # Room property prediction heads
        self.area_predictor = nn.Sequential(
            nn.Linear(512, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # Area should be positive
        )
        
        self.aspect_ratio_predictor = nn.Sequential(
            nn.Linear(512, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Aspect ratio between 0 and 1
        )
        
        self.furniture_density_predictor = nn.Sequential(
            nn.Linear(512, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Density between 0 and 1
        )
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary containing predictions
        """
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Predict room type
        room_type_logits = self.room_classifier(features)
        
        # Predict room properties
        area = self.area_predictor(features)
        aspect_ratio = self.aspect_ratio_predictor(features)
        furniture_density = self.furniture_density_predictor(features)
        
        return {
            'room_type_logits': room_type_logits,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'furniture_density': furniture_density
        }
    
    def predict_room_properties(self, room_patch: np.ndarray, 
                              device: str = 'cpu') -> Dict[str, float]:
        """
        Predict properties for a room patch.
        
        Args:
            room_patch: Room image patch as numpy array
            device: Device to run inference on
            
        Returns:
            Dictionary containing room properties
        """
        try:
            self.eval()
            
            # Preprocess image
            if len(room_patch.shape) == 3:
                room_patch = cv2.cvtColor(room_patch, cv2.COLOR_RGB2GRAY)
            
            # Normalize to [0, 1]
            room_patch = room_patch.astype(np.float32) / 255.0
            
            # Resize to standard size
            room_patch = cv2.resize(room_patch, (224, 224))
            
            # Convert to tensor
            patch_tensor = torch.from_numpy(room_patch).float().unsqueeze(0).unsqueeze(0)
            patch_tensor = patch_tensor.to(device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.forward(patch_tensor)
            
            # Extract predictions
            room_type_probs = F.softmax(predictions['room_type_logits'], dim=1)
            room_type_id = torch.argmax(room_type_probs, dim=1).item()
            room_type_confidence = room_type_probs[0, room_type_id].item()
            
            area = predictions['area'].item()
            aspect_ratio = predictions['aspect_ratio'].item()
            furniture_density = predictions['furniture_density'].item()
            
            return {
                'room_type': self.ROOM_TYPES[room_type_id],
                'room_type_confidence': room_type_confidence,
                'area': area,
                'aspect_ratio': aspect_ratio,
                'furniture_density': furniture_density
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting room properties: {e}")
            raise
    
    def classify_rooms(self, segmentation_mask: np.ndarray, 
                      room_contours: List[np.ndarray], 
                      device: str = 'cpu') -> List[Dict[str, any]]:
        """
        Classify multiple rooms from segmentation mask.
        
        Args:
            segmentation_mask: Segmentation mask
            room_contours: List of room contours
            device: Device to run inference on
            
        Returns:
            List of room classification results
        """
        try:
            results = []
            
            for contour in room_contours:
                # Extract room patch
                room_patch = self._extract_room_patch(segmentation_mask, contour)
                
                if room_patch is not None:
                    # Predict properties
                    properties = self.predict_room_properties(room_patch, device)
                    
                    # Add contour information
                    properties['contour'] = contour
                    properties['area_pixels'] = cv2.contourArea(contour)
                    
                    results.append(properties)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error classifying rooms: {e}")
            return []
    
    def _extract_room_patch(self, segmentation_mask: np.ndarray, 
                          contour: np.ndarray) -> Optional[np.ndarray]:
        """Extract room patch from segmentation mask."""
        try:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(segmentation_mask.shape[1] - x, w + 2 * padding)
            h = min(segmentation_mask.shape[0] - y, h + 2 * padding)
            
            # Extract patch
            patch = segmentation_mask[y:y+h, x:x+w]
            
            # Check if patch is valid
            if patch.size == 0:
                return None
            
            return patch
            
        except Exception as e:
            self.logger.error(f"Error extracting room patch: {e}")
            return None
    
    def get_room_recommendations(self, room_properties: Dict[str, any]) -> Dict[str, List[str]]:
        """
        Get recommendations for room based on its properties.
        
        Args:
            room_properties: Room properties dictionary
            
        Returns:
            Dictionary containing recommendations
        """
        try:
            room_type = room_properties.get('room_type', 'unknown')
            area = room_properties.get('area', 0)
            aspect_ratio = room_properties.get('aspect_ratio', 0.5)
            furniture_density = room_properties.get('furniture_density', 0.5)
            
            recommendations = {
                'furniture': [],
                'lighting': [],
                'materials': [],
                'colors': []
            }
            
            # Furniture recommendations based on room type
            if room_type == 'living_room':
                recommendations['furniture'] = ['sofa', 'coffee_table', 'tv_stand', 'bookshelf']
                recommendations['lighting'] = ['ceiling_light', 'floor_lamp', 'table_lamp']
                recommendations['materials'] = ['wood_flooring', 'carpet', 'painted_walls']
                recommendations['colors'] = ['neutral_tones', 'warm_colors']
                
            elif room_type == 'kitchen':
                recommendations['furniture'] = ['kitchen_cabinet', 'dining_table', 'stools']
                recommendations['lighting'] = ['under_cabinet_lighting', 'pendant_lights']
                recommendations['materials'] = ['tile_flooring', 'granite_countertop', 'stainless_steel']
                recommendations['colors'] = ['white', 'light_gray', 'wood_tones']
                
            elif room_type == 'bedroom':
                recommendations['furniture'] = ['bed', 'nightstand', 'dresser', 'wardrobe']
                recommendations['lighting'] = ['bedside_lamp', 'ceiling_fan']
                recommendations['materials'] = ['carpet', 'wood_flooring', 'painted_walls']
                recommendations['colors'] = ['calm_colors', 'pastels', 'neutral_tones']
                
            elif room_type == 'bathroom':
                recommendations['furniture'] = ['vanity', 'toilet', 'bathtub', 'shower']
                recommendations['lighting'] = ['vanity_lighting', 'recessed_lighting']
                recommendations['materials'] = ['ceramic_tile', 'marble', 'chrome_fixtures']
                recommendations['colors'] = ['white', 'light_blue', 'gray']
                
            # Adjust recommendations based on room size
            if area > 0.7:  # Large room
                recommendations['furniture'].append('additional_seating')
                recommendations['lighting'].append('accent_lighting')
            elif area < 0.3:  # Small room
                recommendations['furniture'] = [item for item in recommendations['furniture'] 
                                             if item not in ['additional_seating', 'large_furniture']]
                recommendations['lighting'].append('mirror_lighting')
            
            # Adjust based on aspect ratio
            if aspect_ratio > 0.8:  # Square room
                recommendations['furniture'].append('centered_layout')
            else:  # Rectangular room
                recommendations['furniture'].append('linear_layout')
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return {'furniture': [], 'lighting': [], 'materials': [], 'colors': []}
    
    def save_model(self, filepath: str) -> None:
        """Save the model to file."""
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'input_channels': self.input_channels,
                'num_room_types': self.num_room_types,
                'hidden_dim': self.hidden_dim
            }, filepath)
            self.logger.info(f"Room classifier model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu') -> 'RoomClassifier':
        """Load a saved model from file."""
        try:
            checkpoint = torch.load(filepath, map_location=device)
            
            model = cls(
                input_channels=checkpoint['input_channels'],
                num_room_types=checkpoint['num_room_types'],
                hidden_dim=checkpoint['hidden_dim']
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            logger.info(f"Room classifier model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


def create_room_classifier(input_channels: int = 1, num_room_types: int = 9, 
                         hidden_dim: int = 512) -> RoomClassifier:
    """
    Factory function to create a room classifier.
    
    Args:
        input_channels: Number of input channels
        num_room_types: Number of room types
        hidden_dim: Hidden dimension size
        
    Returns:
        Configured room classifier
    """
    return RoomClassifier(input_channels, num_room_types, hidden_dim)