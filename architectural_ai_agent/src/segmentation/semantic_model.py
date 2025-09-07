"""
Semantic Segmentation Model for Floor Plan Analysis

This module implements deep learning models for semantic segmentation of floor plans,
identifying different room types, architectural elements, and spatial relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet101
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class SemanticSegmentationModel:
    """
    Semantic segmentation model for floor plan analysis using deep learning.
    """
    
    def __init__(self, model_name: str = 'unet', encoder: str = 'resnet50', 
                 num_classes: int = 15, device: str = 'auto'):
        """
        Initialize the semantic segmentation model.
        
        Args:
            model_name: Architecture name ('unet', 'deeplabv3+', 'fpn', 'pspnet')
            encoder: Encoder backbone ('resnet50', 'resnet101', 'efficientnet-b4')
            num_classes: Number of segmentation classes
            device: Device to run inference on
        """
        self.device = self._setup_device(device)
        self.num_classes = num_classes
        self.model_name = model_name
        self.encoder = encoder
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Setup transforms
        self.transform = self._setup_transforms()
        
        # Class definitions for architectural elements
        self.class_names = {
            0: 'background',
            1: 'wall',
            2: 'door',
            3: 'window',
            4: 'room_living',
            5: 'room_bedroom',
            6: 'room_kitchen',
            7: 'room_bathroom',
            8: 'room_hallway',
            9: 'stairs',
            10: 'closet',
            11: 'balcony',
            12: 'furniture',
            13: 'fixture',
            14: 'text'
        }
        
        # Color map for visualization
        self.color_map = self._create_color_map()
        
    def segment_image(self, image: np.ndarray, return_probabilities: bool = False) -> Dict:
        """
        Perform semantic segmentation on a floor plan image.
        
        Args:
            image: Input floor plan image
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary containing segmentation results
        """
        # Prepare image
        input_tensor = self._prepare_image(image)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        # Post-process results
        segmentation_map = predictions[0].cpu().numpy().astype(np.uint8)
        
        # Create results dictionary
        results = {
            'segmentation_map': segmentation_map,
            'class_names': self.class_names,
            'room_analysis': self._analyze_rooms(segmentation_map),
            'element_analysis': self._analyze_elements(segmentation_map),
            'spatial_relationships': self._analyze_spatial_relationships(segmentation_map)
        }
        
        if return_probabilities:
            results['probabilities'] = probabilities[0].cpu().numpy()
        
        logger.info(f"Segmentation completed: {len(np.unique(segmentation_map))} classes detected")
        
        return results
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _create_model(self) -> nn.Module:
        """Create the segmentation model."""
        if self.model_name.lower() == 'unet':
            model = smp.Unet(
                encoder_name=self.encoder,
                encoder_weights='imagenet',
                classes=self.num_classes,
                activation=None
            )
        elif self.model_name.lower() == 'deeplabv3+':
            model = smp.DeepLabV3Plus(
                encoder_name=self.encoder,
                encoder_weights='imagenet',
                classes=self.num_classes,
                activation=None
            )
        elif self.model_name.lower() == 'fpn':
            model = smp.FPN(
                encoder_name=self.encoder,
                encoder_weights='imagenet',
                classes=self.num_classes,
                activation=None
            )
        elif self.model_name.lower() == 'pspnet':
            model = smp.PSPNet(
                encoder_name=self.encoder,
                encoder_weights='imagenet',
                classes=self.num_classes,
                activation=None
            )
        else:
            # Custom architectural segmentation model
            model = ArchitecturalSegmentationNet(self.num_classes)
        
        logger.info(f"Created {self.model_name} model with {self.encoder} encoder")
        return model
    
    def _setup_transforms(self) -> transforms.Compose:
        """Setup image preprocessing transforms."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """Prepare image for model inference."""
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply transforms
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def _create_color_map(self) -> np.ndarray:
        """Create color map for visualization."""
        colors = [
            [0, 0, 0],        # background - black
            [128, 128, 128],  # wall - gray
            [255, 0, 0],      # door - red
            [0, 255, 0],      # window - green
            [255, 255, 0],    # living room - yellow
            [0, 0, 255],      # bedroom - blue
            [255, 0, 255],    # kitchen - magenta
            [0, 255, 255],    # bathroom - cyan
            [128, 0, 0],      # hallway - maroon
            [0, 128, 0],      # stairs - dark green
            [128, 0, 128],    # closet - purple
            [255, 128, 0],    # balcony - orange
            [128, 128, 0],    # furniture - olive
            [0, 128, 128],    # fixture - teal
            [192, 192, 192]   # text - light gray
        ]
        
        return np.array(colors, dtype=np.uint8)
    
    def _analyze_rooms(self, segmentation_map: np.ndarray) -> Dict:
        """Analyze room information from segmentation map."""
        room_classes = [4, 5, 6, 7, 8, 10, 11]  # Room-related classes
        room_analysis = {}
        
        for room_class in room_classes:
            mask = segmentation_map == room_class
            if np.any(mask):
                # Find connected components for this room type
                num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
                
                rooms_of_type = []
                for label in range(1, num_labels):
                    room_mask = labels == label
                    area = np.sum(room_mask)
                    
                    if area > 100:  # Minimum room area
                        # Calculate room properties
                        contours, _ = cv2.findContours(
                            room_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(largest_contour)
                            
                            room_info = {
                                'area': area,
                                'bounding_box': (x, y, w, h),
                                'center': (x + w//2, y + h//2),
                                'aspect_ratio': w/h if h > 0 else 1,
                                'contour': largest_contour
                            }
                            rooms_of_type.append(room_info)
                
                if rooms_of_type:
                    room_analysis[self.class_names[room_class]] = {
                        'count': len(rooms_of_type),
                        'total_area': sum(room['area'] for room in rooms_of_type),
                        'rooms': rooms_of_type
                    }
        
        return room_analysis
    
    def _analyze_elements(self, segmentation_map: np.ndarray) -> Dict:
        """Analyze architectural elements from segmentation map."""
        element_classes = [1, 2, 3, 9, 12, 13]  # Wall, door, window, stairs, furniture, fixture
        element_analysis = {}
        
        for element_class in element_classes:
            mask = segmentation_map == element_class
            if np.any(mask):
                # Count occurrences and measure total area
                num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
                
                elements = []
                for label in range(1, num_labels):
                    element_mask = labels == label
                    area = np.sum(element_mask)
                    
                    if area > 10:  # Minimum element area
                        contours, _ = cv2.findContours(
                            element_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(largest_contour)
                            
                            element_info = {
                                'area': area,
                                'bounding_box': (x, y, w, h),
                                'center': (x + w//2, y + h//2),
                                'contour': largest_contour
                            }
                            elements.append(element_info)
                
                if elements:
                    element_analysis[self.class_names[element_class]] = {
                        'count': len(elements),
                        'total_area': sum(elem['area'] for elem in elements),
                        'elements': elements
                    }
        
        return element_analysis
    
    def _analyze_spatial_relationships(self, segmentation_map: np.ndarray) -> Dict:
        """Analyze spatial relationships between rooms and elements."""
        relationships = {
            'adjacency_matrix': self._compute_adjacency_matrix(segmentation_map),
            'connectivity_graph': self._compute_connectivity_graph(segmentation_map),
            'access_analysis': self._analyze_access_patterns(segmentation_map)
        }
        
        return relationships
    
    def _compute_adjacency_matrix(self, segmentation_map: np.ndarray) -> np.ndarray:
        """Compute adjacency matrix between different classes."""
        num_classes = self.num_classes
        adjacency = np.zeros((num_classes, num_classes), dtype=int)
        
        # Check horizontal adjacency
        for i in range(segmentation_map.shape[0]):
            for j in range(segmentation_map.shape[1] - 1):
                class1 = segmentation_map[i, j]
                class2 = segmentation_map[i, j + 1]
                if class1 != class2:
                    adjacency[class1, class2] += 1
                    adjacency[class2, class1] += 1
        
        # Check vertical adjacency
        for i in range(segmentation_map.shape[0] - 1):
            for j in range(segmentation_map.shape[1]):
                class1 = segmentation_map[i, j]
                class2 = segmentation_map[i + 1, j]
                if class1 != class2:
                    adjacency[class1, class2] += 1
                    adjacency[class2, class1] += 1
        
        return adjacency
    
    def _compute_connectivity_graph(self, segmentation_map: np.ndarray) -> Dict:
        """Compute connectivity graph between rooms."""
        room_classes = [4, 5, 6, 7, 8, 10, 11]
        door_class = 2
        
        connectivity = {}
        
        # Find doors
        door_mask = segmentation_map == door_class
        door_labels, door_components = cv2.connectedComponents(door_mask.astype(np.uint8))
        
        # For each door, find which rooms it connects
        for door_label in range(1, door_labels):
            door_component = door_components == door_label
            
            # Dilate door to find adjacent areas
            kernel = np.ones((5, 5), np.uint8)
            dilated_door = cv2.dilate(door_component.astype(np.uint8), kernel, iterations=1)
            
            # Find room classes in dilated area
            adjacent_classes = []
            for room_class in room_classes:
                room_mask = segmentation_map == room_class
                if np.any(room_mask & dilated_door):
                    adjacent_classes.append(room_class)
            
            # Record connections
            if len(adjacent_classes) >= 2:
                for i, class1 in enumerate(adjacent_classes):
                    for class2 in adjacent_classes[i+1:]:
                        room1_name = self.class_names[class1]
                        room2_name = self.class_names[class2]
                        
                        if room1_name not in connectivity:
                            connectivity[room1_name] = set()
                        if room2_name not in connectivity:
                            connectivity[room2_name] = set()
                        
                        connectivity[room1_name].add(room2_name)
                        connectivity[room2_name].add(room1_name)
        
        # Convert sets to lists for JSON serialization
        for room in connectivity:
            connectivity[room] = list(connectivity[room])
        
        return connectivity
    
    def _analyze_access_patterns(self, segmentation_map: np.ndarray) -> Dict:
        """Analyze access patterns and circulation."""
        hallway_class = 8
        door_class = 2
        
        access_analysis = {
            'circulation_area': 0,
            'access_points': [],
            'dead_ends': [],
            'main_circulation': None
        }
        
        # Calculate circulation area
        hallway_mask = segmentation_map == hallway_class
        access_analysis['circulation_area'] = np.sum(hallway_mask)
        
        # Find access points (doors)
        door_mask = segmentation_map == door_class
        door_labels, door_components = cv2.connectedComponents(door_mask.astype(np.uint8))
        
        for door_label in range(1, door_labels):
            door_component = door_components == door_label
            
            # Find centroid of door
            moments = cv2.moments(door_component.astype(np.uint8))
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                access_analysis['access_points'].append((cx, cy))
        
        return access_analysis
    
    def visualize_segmentation(self, segmentation_map: np.ndarray) -> np.ndarray:
        """
        Create a colored visualization of the segmentation map.
        
        Args:
            segmentation_map: Segmentation result
            
        Returns:
            RGB image with colored segments
        """
        h, w = segmentation_map.shape
        colored_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in range(self.num_classes):
            mask = segmentation_map == class_id
            if class_id < len(self.color_map):
                colored_image[mask] = self.color_map[class_id]
        
        return colored_image
    
    def train_model(self, train_loader, val_loader, epochs: int = 100, learning_rate: float = 0.001):
        """
        Train the segmentation model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        # Setup training components
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            val_loss = self._validate_model(val_loader, criterion)
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f'best_segmentation_model_{self.model_name}.pth')
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    def _validate_model(self, val_loader, criterion) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'encoder': self.encoder,
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info(f"Model loaded from {path}")


class ArchitecturalSegmentationNet(nn.Module):
    """
    Custom segmentation network designed specifically for architectural floor plans.
    """
    
    def __init__(self, num_classes: int = 15):
        super(ArchitecturalSegmentationNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Encoder (feature extraction)
        self.encoder = self._create_encoder()
        
        # Decoder (upsampling and classification)
        self.decoder = self._create_decoder()
        
        # Final classification layer
        self.classifier = nn.Conv2d(64, num_classes, 1)
        
    def _create_encoder(self) -> nn.Module:
        """Create encoder for feature extraction."""
        return nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
    
    def _create_decoder(self) -> nn.Module:
        """Create decoder for upsampling."""
        return nn.Sequential(
            # Upsampling block 1
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Upsampling block 2
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Upsampling block 3
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        
        # Decoder
        decoded = self.decoder(encoded)
        
        # Final classification
        output = self.classifier(decoded)
        
        return output