"""
Object Detection Module using Deep Learning

This module provides deep learning-based object detection for architectural elements
using pre-trained models and custom architectures for floor plan analysis.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Deep learning-based object detector for architectural elements.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to pre-trained model weights
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._setup_transforms()
        
        # Class mappings for architectural elements
        self.class_names = {
            0: 'wall',
            1: 'door',
            2: 'window', 
            3: 'room',
            4: 'stairs',
            5: 'bathroom',
            6: 'kitchen',
            7: 'bedroom',
            8: 'living_room',
            9: 'hallway',
            10: 'furniture',
            11: 'fixture',
            12: 'text'
        }
        
    def detect_objects(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
        """
        Detect architectural objects in the image.
        
        Args:
            image: Input floor plan image
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dictionary containing detected objects with bounding boxes and classes
        """
        # Prepare image for inference
        input_tensor = self._prepare_image(image)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Post-process predictions
        detections = self._post_process_predictions(predictions, confidence_threshold)
        
        # Convert to output format
        results = self._format_results(detections, image.shape)
        
        logger.info(f"Detected {len(results['objects'])} objects with confidence > {confidence_threshold}")
        
        return results
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        torch_device = torch.device(device)
        logger.info(f"Using device: {torch_device}")
        return torch_device
    
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load the object detection model."""
        if model_path and os.path.exists(model_path):
            # Load custom trained model
            model = self._create_custom_model()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded custom model from {model_path}")
        else:
            # Create and initialize a new model
            model = self._create_custom_model()
            logger.info("Initialized new model (no pre-trained weights)")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _create_custom_model(self) -> nn.Module:
        """Create custom object detection model for architectural elements."""
        return ArchitecturalObjectDetector(num_classes=len(self.class_names))
    
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
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def _post_process_predictions(self, predictions: torch.Tensor, threshold: float) -> List[Dict]:
        """Post-process model predictions."""
        # This is a simplified version - real implementation would depend on model architecture
        detections = []
        
        # Assuming predictions contains [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, conf, class]
        if isinstance(predictions, dict):
            # Handle different model output formats
            boxes = predictions.get('boxes', torch.tensor([]))
            scores = predictions.get('scores', torch.tensor([]))
            labels = predictions.get('labels', torch.tensor([]))
        else:
            # Simple tensor output
            pred = predictions[0]  # Remove batch dimension
            boxes = pred[:, :4]
            scores = pred[:, 4]
            labels = pred[:, 5].long()
        
        # Filter by confidence threshold
        valid_indices = scores > threshold
        
        if valid_indices.sum() > 0:
            valid_boxes = boxes[valid_indices]
            valid_scores = scores[valid_indices]
            valid_labels = labels[valid_indices]
            
            for i in range(len(valid_boxes)):
                detection = {
                    'bbox': valid_boxes[i].cpu().numpy(),
                    'confidence': valid_scores[i].item(),
                    'class_id': valid_labels[i].item(),
                    'class_name': self.class_names.get(valid_labels[i].item(), 'unknown')
                }
                detections.append(detection)
        
        return detections
    
    def _format_results(self, detections: List[Dict], image_shape: Tuple[int, int]) -> Dict:
        """Format detection results."""
        h, w = image_shape[:2]
        
        formatted_objects = []
        for detection in detections:
            # Convert normalized coordinates to pixel coordinates
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Scale to image dimensions
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
            
            formatted_object = {
                'bounding_box': (x1, y1, x2 - x1, y2 - y1),  # (x, y, width, height)
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'confidence': detection['confidence'],
                'class_id': detection['class_id'],
                'class_name': detection['class_name'],
                'area': (x2 - x1) * (y2 - y1)
            }
            formatted_objects.append(formatted_object)
        
        return {
            'objects': formatted_objects,
            'num_detections': len(formatted_objects),
            'image_shape': image_shape
        }
    
    def train_model(self, train_dataset, val_dataset, epochs: int = 100, learning_rate: float = 0.001):
        """
        Train the object detection model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        # Setup training components
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (images, targets) in enumerate(train_dataset):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            if val_dataset:
                val_loss = self._validate_model(val_dataset, criterion)
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
    
    def _validate_model(self, val_dataset, criterion) -> float:
        """Validate the model on validation dataset."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_dataset:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        return val_loss / len(val_dataset)
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info(f"Model loaded from {path}")


class ArchitecturalObjectDetector(nn.Module):
    """
    Custom CNN architecture for detecting architectural elements in floor plans.
    """
    
    def __init__(self, num_classes: int = 13):
        super(ArchitecturalObjectDetector, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = resnet50(pretrained=True)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Feature pyramid network for multi-scale detection
        self.fpn = FeaturePyramidNetwork()
        
        # Detection head
        self.detection_head = DetectionHead(num_classes)
        
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply FPN for multi-scale features
        pyramid_features = self.fpn(features)
        
        # Generate detections
        detections = self.detection_head(pyramid_features)
        
        return detections


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction."""
    
    def __init__(self):
        super(FeaturePyramidNetwork, self).__init__()
        
        # Lateral connections
        self.lateral_conv1 = nn.Conv2d(2048, 256, 1)
        self.lateral_conv2 = nn.Conv2d(1024, 256, 1)
        self.lateral_conv3 = nn.Conv2d(512, 256, 1)
        
        # Output convolutions
        self.output_conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.output_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.output_conv3 = nn.Conv2d(256, 256, 3, padding=1)
        
    def forward(self, x):
        # This is a simplified FPN - real implementation would handle multiple feature levels
        # For now, just apply some convolutions to transform features
        x = self.lateral_conv1(x)
        x = self.output_conv1(x)
        return x


class DetectionHead(nn.Module):
    """Detection head for generating object predictions."""
    
    def __init__(self, num_classes: int):
        super(DetectionHead, self).__init__()
        
        self.num_classes = num_classes
        
        # Shared convolutions
        self.shared_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.cls_head = nn.Conv2d(256, num_classes, 3, padding=1)
        
        # Regression head (bounding box coordinates)
        self.reg_head = nn.Conv2d(256, 4, 3, padding=1)
        
        # Objectness head
        self.obj_head = nn.Conv2d(256, 1, 3, padding=1)
        
    def forward(self, x):
        # Shared features
        shared_features = self.shared_conv(x)
        
        # Classification predictions
        cls_pred = self.cls_head(shared_features)
        
        # Regression predictions
        reg_pred = self.reg_head(shared_features)
        
        # Objectness predictions
        obj_pred = torch.sigmoid(self.obj_head(shared_features))
        
        # Combine predictions
        # In a real implementation, this would be more sophisticated
        batch_size, _, h, w = x.shape
        
        # Flatten spatial dimensions
        cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
        reg_pred = reg_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        obj_pred = obj_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
        
        # Combine all predictions
        predictions = torch.cat([reg_pred, obj_pred, cls_pred], dim=-1)
        
        return predictions


# Additional utility functions for object detection

def non_maximum_suppression(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Filtered list of detections
    """
    if not detections:
        return []
    
    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    filtered_detections = []
    
    while detections:
        # Take the detection with highest confidence
        best_detection = detections.pop(0)
        filtered_detections.append(best_detection)
        
        # Remove overlapping detections
        remaining_detections = []
        for detection in detections:
            if calculate_iou(best_detection['bounding_box'], detection['bounding_box']) < iou_threshold:
                remaining_detections.append(detection)
        
        detections = remaining_detections
    
    return filtered_detections


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: (x, y, width, height) of first box
        box2: (x, y, width, height) of second box
        
    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0