"""
Deep learning model for semantic segmentation of floor plans.
Uses U-Net architecture for pixel-level classification of architectural elements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net architecture."""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad x1 to match x2 size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class FloorPlanSegmentationModel(nn.Module):
    """
    U-Net based model for semantic segmentation of floor plans.
    """
    
    # Define class labels for floor plan elements
    CLASS_LABELS = {
        0: 'background',
        1: 'wall',
        2: 'door',
        3: 'window',
        4: 'room_living',
        5: 'room_kitchen',
        6: 'room_bedroom',
        7: 'room_bathroom',
        8: 'room_closet',
        9: 'furniture',
        10: 'text'
    }
    
    def __init__(self, n_channels: int = 1, n_classes: int = 11, bilinear: bool = False):
        """
        Initialize the segmentation model.
        
        Args:
            n_channels: Number of input channels (1 for grayscale)
            n_classes: Number of output classes
            bilinear: Use bilinear upsampling instead of transposed convolution
        """
        super(FloorPlanSegmentationModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder (downsampling path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (upsampling path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, x):
        """Forward pass through the network."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def predict(self, image: np.ndarray, device: str = 'cpu') -> np.ndarray:
        """
        Predict segmentation for a single image.
        
        Args:
            image: Input image as numpy array
            device: Device to run inference on ('cpu' or 'cuda')
            
        Returns:
            Segmentation mask as numpy array
        """
        try:
            self.eval()
            
            # Preprocess image
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            image_tensor = image_tensor.to(device)
            
            # Run inference
            with torch.no_grad():
                logits = self.forward(image_tensor)
                predictions = torch.argmax(logits, dim=1)
            
            # Convert back to numpy
            predictions = predictions.squeeze().cpu().numpy()
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            raise
    
    def predict_batch(self, images: List[np.ndarray], device: str = 'cpu') -> List[np.ndarray]:
        """
        Predict segmentation for a batch of images.
        
        Args:
            images: List of input images
            device: Device to run inference on
            
        Returns:
            List of segmentation masks
        """
        try:
            self.eval()
            results = []
            
            for image in images:
                result = self.predict(image, device)
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {e}")
            raise
    
    def get_class_probabilities(self, image: np.ndarray, device: str = 'cpu') -> np.ndarray:
        """
        Get class probabilities for each pixel.
        
        Args:
            image: Input image as numpy array
            device: Device to run inference on
            
        Returns:
            Probability tensor of shape (height, width, n_classes)
        """
        try:
            self.eval()
            
            # Preprocess image
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            image = image.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            image_tensor = image_tensor.to(device)
            
            # Run inference
            with torch.no_grad():
                logits = self.forward(image_tensor)
                probabilities = F.softmax(logits, dim=1)
            
            # Convert to numpy and remove batch dimension
            probabilities = probabilities.squeeze().cpu().numpy()
            probabilities = np.transpose(probabilities, (1, 2, 0))  # (H, W, C)
            
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Error getting class probabilities: {e}")
            raise
    
    def visualize_segmentation(self, image: np.ndarray, mask: np.ndarray, 
                             alpha: float = 0.6) -> np.ndarray:
        """
        Create a visualization of the segmentation results.
        
        Args:
            image: Original image
            mask: Segmentation mask
            alpha: Transparency for overlay
            
        Returns:
            Visualization image
        """
        try:
            # Create color map for different classes
            colors = np.array([
                [0, 0, 0],        # background - black
                [128, 128, 128],  # wall - gray
                [0, 255, 0],      # door - green
                [0, 0, 255],      # window - blue
                [255, 0, 0],      # living room - red
                [255, 165, 0],    # kitchen - orange
                [255, 192, 203],  # bedroom - pink
                [0, 255, 255],    # bathroom - cyan
                [128, 0, 128],    # closet - purple
                [255, 255, 0],    # furniture - yellow
                [255, 255, 255]   # text - white
            ])
            
            # Convert mask to RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Create colored mask
            colored_mask = colors[mask]
            
            # Blend with original image
            result = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            return image
    
    def get_class_statistics(self, mask: np.ndarray) -> Dict[str, int]:
        """
        Get statistics about detected classes in the mask.
        
        Args:
            mask: Segmentation mask
            
        Returns:
            Dictionary with class counts
        """
        try:
            unique, counts = np.unique(mask, return_counts=True)
            stats = {}
            
            for class_id, count in zip(unique, counts):
                class_name = self.CLASS_LABELS.get(class_id, f'unknown_{class_id}')
                stats[class_name] = int(count)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting class statistics: {e}")
            return {}
    
    def save_model(self, filepath: str) -> None:
        """Save the model to file."""
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'n_channels': self.n_channels,
                'n_classes': self.n_classes,
                'bilinear': self.bilinear
            }, filepath)
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu') -> 'FloorPlanSegmentationModel':
        """Load a saved model from file."""
        try:
            checkpoint = torch.load(filepath, map_location=device)
            
            model = cls(
                n_channels=checkpoint['n_channels'],
                n_classes=checkpoint['n_classes'],
                bilinear=checkpoint['bilinear']
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            logger.info(f"Model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


def create_segmentation_model(n_channels: int = 1, n_classes: int = 11, 
                            bilinear: bool = False) -> FloorPlanSegmentationModel:
    """
    Factory function to create a segmentation model.
    
    Args:
        n_channels: Number of input channels
        n_classes: Number of output classes
        bilinear: Use bilinear upsampling
        
    Returns:
        Configured segmentation model
    """
    return FloorPlanSegmentationModel(n_channels, n_classes, bilinear)