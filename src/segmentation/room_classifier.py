"""
Room classification module for floor plan analysis.

This module provides functionality to classify and analyze different
room types in segmented floor plans using deep learning and heuristics.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class RoomInfo:
    """Information about a detected room."""
    contour: np.ndarray
    area: float
    perimeter: float
    center: Tuple[float, float]
    room_type: str
    confidence: float
    features: Dict[str, float]


class RoomClassifierCNN(nn.Module):
    """
    CNN for room type classification based on room features.
    
    This model classifies rooms into different types based on
    geometric and contextual features.
    """
    
    def __init__(self, input_size: int = 64, num_classes: int = 6):
        super(RoomClassifierCNN, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


class RoomClassifier:
    """
    Main class for room classification in floor plans.
    
    This class combines deep learning with heuristic rules to
    classify different room types in floor plans.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 device: str = 'auto'):
        """
        Initialize the RoomClassifier.
        
        Args:
            model_path: Path to pre-trained model weights
            device: Device to run inference on
        """
        self.device = self._get_device(device)
        
        # Room type definitions
        self.room_types = [
            'living_room',
            'bedroom', 
            'kitchen',
            'bathroom',
            'dining_room',
            'hallway'
        ]
        
        # Initialize CNN model
        self.cnn_model = RoomClassifierCNN(num_classes=len(self.room_types))
        self.cnn_model.to(self.device)
        
        # Load pre-trained weights if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        # Heuristic rules for room classification
        self.heuristic_rules = {
            'bathroom': {
                'min_area': 2000,
                'max_area': 15000,
                'aspect_ratio_range': (0.5, 2.0),
                'features': ['small', 'rectangular']
            },
            'kitchen': {
                'min_area': 8000,
                'max_area': 30000,
                'aspect_ratio_range': (0.6, 2.5),
                'features': ['medium', 'rectangular', 'appliances']
            },
            'bedroom': {
                'min_area': 10000,
                'max_area': 40000,
                'aspect_ratio_range': (0.7, 2.0),
                'features': ['medium', 'rectangular', 'bedroom_furniture']
            },
            'living_room': {
                'min_area': 15000,
                'max_area': 60000,
                'aspect_ratio_range': (0.8, 3.0),
                'features': ['large', 'flexible_shape', 'entertainment']
            },
            'dining_room': {
                'min_area': 8000,
                'max_area': 25000,
                'aspect_ratio_range': (0.8, 2.0),
                'features': ['medium', 'rectangular', 'dining_furniture']
            },
            'hallway': {
                'min_area': 2000,
                'max_area': 20000,
                'aspect_ratio_range': (0.2, 5.0),
                'features': ['narrow', 'long', 'connector']
            }
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
            self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Room classifier model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading room classifier model: {e}")
    
    def extract_room_features(self, room_mask: np.ndarray) -> Dict[str, float]:
        """
        Extract geometric and contextual features from room mask.
        
        Args:
            room_mask: Binary mask of the room
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic geometric features
        area = np.sum(room_mask > 0)
        features['area'] = float(area)
        
        # Find contours
        contours, _ = cv2.findContours(room_mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return features
        
        contour = max(contours, key=cv2.contourArea)
        
        # Perimeter
        perimeter = cv2.arcLength(contour, True)
        features['perimeter'] = float(perimeter)
        
        # Aspect ratio
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        features['aspect_ratio'] = float(aspect_ratio)
        
        # Compactness (4π * area / perimeter²)
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        features['compactness'] = float(compactness)
        
        # Solidity (area / convex_hull_area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        features['solidity'] = float(solidity)
        
        # Bounding box features
        x, y, w, h = cv2.boundingRect(contour)
        features['bounding_box_area'] = float(w * h)
        features['bounding_box_aspect_ratio'] = float(max(w, h) / max(min(w, h), 1))
        
        # Center coordinates
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            features['center_x'] = float(cx)
            features['center_y'] = float(cy)
        
        # Shape complexity (perimeter² / area)
        complexity = (perimeter ** 2) / area if area > 0 else 0
        features['complexity'] = float(complexity)
        
        return features
    
    def classify_room_cnn(self, room_mask: np.ndarray) -> Tuple[str, float]:
        """
        Classify room using CNN model.
        
        Args:
            room_mask: Binary mask of the room
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        self.cnn_model.eval()
        
        # Preprocess room mask
        room_resized = cv2.resize(room_mask, (64, 64))
        room_normalized = room_resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        room_tensor = torch.from_numpy(room_normalized).unsqueeze(0).unsqueeze(0)
        room_tensor = room_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.cnn_model(room_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.room_types[predicted.item()]
            confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
    def classify_room_heuristic(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify room using heuristic rules.
        
        Args:
            features: Extracted room features
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        area = features.get('area', 0)
        aspect_ratio = features.get('aspect_ratio', 1.0)
        
        best_match = None
        best_score = 0.0
        
        for room_type, rules in self.heuristic_rules.items():
            score = 0.0
            
            # Area matching
            if rules['min_area'] <= area <= rules['max_area']:
                score += 0.4
            elif area < rules['min_area']:
                score += 0.2 * (area / rules['min_area'])
            elif area > rules['max_area']:
                score += 0.2 * (rules['max_area'] / area)
            
            # Aspect ratio matching
            min_ar, max_ar = rules['aspect_ratio_range']
            if min_ar <= aspect_ratio <= max_ar:
                score += 0.3
            else:
                # Penalty for aspect ratio mismatch
                if aspect_ratio < min_ar:
                    score += 0.1 * (aspect_ratio / min_ar)
                else:
                    score += 0.1 * (max_ar / aspect_ratio)
            
            # Additional feature matching
            compactness = features.get('compactness', 0)
            if 'rectangular' in rules['features'] and compactness > 0.7:
                score += 0.2
            elif 'flexible_shape' in rules['features'] and compactness < 0.7:
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_match = room_type
        
        return best_match or 'unknown', best_score
    
    def classify_room(self, room_mask: np.ndarray, 
                     use_cnn: bool = True, 
                     use_heuristic: bool = True) -> RoomInfo:
        """
        Classify a room using both CNN and heuristic methods.
        
        Args:
            room_mask: Binary mask of the room
            use_cnn: Whether to use CNN classification
            use_heuristic: Whether to use heuristic classification
            
        Returns:
            RoomInfo object with classification results
        """
        # Extract features
        features = self.extract_room_features(room_mask)
        
        # Get contour for additional info
        contours, _ = cv2.findContours(room_mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else np.array([])
        
        # Calculate basic properties
        area = features.get('area', 0)
        perimeter = features.get('perimeter', 0)
        center = (features.get('center_x', 0), features.get('center_y', 0))
        
        # Classification
        cnn_class = "unknown"
        cnn_confidence = 0.0
        heuristic_class = "unknown"
        heuristic_confidence = 0.0
        
        if use_cnn:
            cnn_class, cnn_confidence = self.classify_room_cnn(room_mask)
        
        if use_heuristic:
            heuristic_class, heuristic_confidence = self.classify_room_heuristic(features)
        
        # Combine results
        if use_cnn and use_heuristic:
            # Weighted combination
            if cnn_confidence > 0.7:
                final_class = cnn_class
                final_confidence = cnn_confidence * 0.7 + heuristic_confidence * 0.3
            else:
                final_class = heuristic_class
                final_confidence = heuristic_confidence * 0.7 + cnn_confidence * 0.3
        elif use_cnn:
            final_class = cnn_class
            final_confidence = cnn_confidence
        else:
            final_class = heuristic_class
            final_confidence = heuristic_confidence
        
        return RoomInfo(
            contour=contour,
            area=area,
            perimeter=perimeter,
            center=center,
            room_type=final_class,
            confidence=final_confidence,
            features=features
        )
    
    def classify_rooms_in_plan(self, segmentation_mask: np.ndarray) -> List[RoomInfo]:
        """
        Classify all rooms in a floor plan segmentation.
        
        Args:
            segmentation_mask: Segmentation mask with room regions
            
        Returns:
            List of RoomInfo objects for each detected room
        """
        # Extract room regions (assuming class 4 is rooms)
        room_mask = (segmentation_mask == 4).astype(np.uint8)
        
        # Find individual room contours
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rooms = []
        for contour in contours:
            # Create mask for this room
            room_mask_individual = np.zeros_like(room_mask)
            cv2.fillPoly(room_mask_individual, [contour], 1)
            
            # Skip very small rooms
            if np.sum(room_mask_individual) < 1000:
                continue
            
            # Classify room
            room_info = self.classify_room(room_mask_individual)
            rooms.append(room_info)
        
        return rooms
    
    def visualize_room_classification(self, image: np.ndarray, 
                                    rooms: List[RoomInfo],
                                    save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize room classification results.
        
        Args:
            image: Original floor plan image
            rooms: List of classified rooms
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        # Create visualization
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # Color map for different room types
        room_colors = {
            'living_room': (255, 0, 0),      # Red
            'bedroom': (0, 255, 0),          # Green
            'kitchen': (0, 0, 255),          # Blue
            'bathroom': (255, 255, 0),       # Cyan
            'dining_room': (255, 0, 255),    # Magenta
            'hallway': (0, 255, 255),        # Yellow
            'unknown': (128, 128, 128)       # Gray
        }
        
        # Draw room boundaries and labels
        for room in rooms:
            color = room_colors.get(room.room_type, (128, 128, 128))
            
            # Draw contour
            cv2.drawContours(vis_image, [room.contour], -1, color, 2)
            
            # Draw label
            label = f"{room.room_type} ({room.confidence:.2f})"
            cv2.putText(vis_image, label, 
                       (int(room.center[0]), int(room.center[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
    
    def get_room_statistics(self, rooms: List[RoomInfo]) -> Dict[str, any]:
        """
        Get statistics about classified rooms.
        
        Args:
            rooms: List of classified rooms
            
        Returns:
            Dictionary with room statistics
        """
        if not rooms:
            return {}
        
        # Count by room type
        type_counts = {}
        total_area = 0
        
        for room in rooms:
            room_type = room.room_type
            if room_type not in type_counts:
                type_counts[room_type] = {'count': 0, 'total_area': 0}
            
            type_counts[room_type]['count'] += 1
            type_counts[room_type]['total_area'] += room.area
            total_area += room.area
        
        # Calculate percentages
        for room_type in type_counts:
            type_counts[room_type]['percentage'] = (
                type_counts[room_type]['total_area'] / total_area * 100
            )
        
        return {
            'total_rooms': len(rooms),
            'total_area': total_area,
            'room_types': type_counts,
            'average_room_area': total_area / len(rooms)
        }