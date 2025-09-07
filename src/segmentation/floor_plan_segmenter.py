"""
Deep learning-based semantic segmentation for floor plans.

This module provides a U-Net based model for segmenting architectural
elements in 2D floor plans, including walls, doors, windows, and rooms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from pathlib import Path


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
        
        # Input is CHW
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


class FloorPlanUNet(nn.Module):
    """
    U-Net architecture for floor plan semantic segmentation.
    
    This model segments floor plans into different architectural elements:
    - Background (0)
    - Walls (1)
    - Doors (2)
    - Windows (3)
    - Rooms (4)
    """
    
    def __init__(self, n_channels: int = 1, n_classes: int = 5, bilinear: bool = False):
        super(FloorPlanUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
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


class FloorPlanDataset(Dataset):
    """Dataset class for floor plan segmentation training."""
    
    def __init__(self, image_paths: List[str], mask_paths: List[str], 
                 transform: Optional[A.Compose] = None, 
                 image_size: Tuple[int, int] = (512, 512)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.image_size = image_size
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {self.image_paths[idx]}")
        
        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {self.mask_paths[idx]}")
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.long()


class FloorPlanSegmenter:
    """
    Main class for floor plan semantic segmentation.
    
    This class handles training, inference, and model management
    for floor plan segmentation using a U-Net architecture.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 device: str = 'auto', num_classes: int = 5):
        """
        Initialize the FloorPlanSegmenter.
        
        Args:
            model_path: Path to pre-trained model weights
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            num_classes: Number of segmentation classes
        """
        self.num_classes = num_classes
        self.device = self._get_device(device)
        
        # Initialize model
        self.model = FloorPlanUNet(n_channels=1, n_classes=num_classes)
        self.model.to(self.device)
        
        # Load pre-trained weights if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        # Class names and colors for visualization
        self.class_names = ['Background', 'Walls', 'Doors', 'Windows', 'Rooms']
        self.class_colors = [
            [0, 0, 0],        # Background - Black
            [255, 0, 0],      # Walls - Red
            [0, 255, 0],      # Doors - Green
            [0, 0, 255],      # Windows - Blue
            [255, 255, 0]     # Rooms - Yellow
        ]
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for computation."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def load_model(self, model_path: str):
        """Load model weights from file."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def save_model(self, model_path: str, epoch: int = None, 
                   optimizer_state: dict = None, loss: float = None):
        """Save model weights to file."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if loss is not None:
            checkpoint['loss'] = loss
        
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")
    
    def preprocess_image(self, image: np.ndarray, 
                        target_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image (grayscale)
            target_size: Target size for resizing
            
        Returns:
            Preprocessed image tensor
        """
        # Resize image
        image_resized = cv2.resize(image, target_size)
        
        # Normalize
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(self, image: np.ndarray, 
                return_probabilities: bool = False) -> np.ndarray:
        """
        Perform semantic segmentation on input image.
        
        Args:
            image: Input grayscale image
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Segmentation mask or probability tensor
        """
        self.model.eval()
        
        with torch.no_grad():
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Run inference
            logits = self.model(image_tensor)
            
            if return_probabilities:
                probabilities = F.softmax(logits, dim=1)
                return probabilities.cpu().numpy()[0]
            else:
                predictions = torch.argmax(logits, dim=1)
                return predictions.cpu().numpy()[0]
    
    def predict_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Perform batch prediction on multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of segmentation masks
        """
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for image in images:
                mask = self.predict(image)
                results.append(mask)
        
        return results
    
    def visualize_segmentation(self, image: np.ndarray, 
                             mask: np.ndarray,
                             save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize segmentation results.
        
        Args:
            image: Original image
            mask: Segmentation mask
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        # Create colored mask
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        for class_id in range(self.num_classes):
            class_mask = mask == class_id
            colored_mask[class_mask] = self.class_colors[class_id]
        
        # Blend with original image
        if len(image.shape) == 2:
            image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_colored = image.copy()
        
        # Resize colored mask to match image
        if colored_mask.shape[:2] != image_colored.shape[:2]:
            colored_mask = cv2.resize(colored_mask, 
                                    (image_colored.shape[1], image_colored.shape[0]))
        
        # Blend images
        alpha = 0.6
        blended = cv2.addWeighted(image_colored, 1 - alpha, colored_mask, alpha, 0)
        
        # Add legend
        legend_height = 50
        legend = np.zeros((legend_height, 300, 3), dtype=np.uint8)
        
        for i, (name, color) in enumerate(zip(self.class_names, self.class_colors)):
            y_start = 10
            y_end = 40
            x_start = 10 + i * 60
            x_end = x_start + 50
            
            cv2.rectangle(legend, (x_start, y_start), (x_end, y_end), color, -1)
            cv2.putText(legend, name, (x_start, y_end + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Combine image and legend
        result = np.vstack([blended, legend])
        
        if save_path:
            cv2.imwrite(save_path, result)
        
        return result
    
    def get_class_statistics(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics for each class in the segmentation mask.
        
        Args:
            mask: Segmentation mask
            
        Returns:
            Dictionary with class statistics
        """
        total_pixels = mask.size
        stats = {}
        
        for class_id, class_name in enumerate(self.class_names):
            class_pixels = np.sum(mask == class_id)
            percentage = (class_pixels / total_pixels) * 100
            stats[class_name] = {
                'pixel_count': int(class_pixels),
                'percentage': round(percentage, 2)
            }
        
        return stats
    
    def extract_room_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Extract room contours from segmentation mask.
        
        Args:
            mask: Segmentation mask
            
        Returns:
            List of room contours
        """
        # Get room mask (class 4)
        room_mask = (mask == 4).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        min_area = 1000
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        return filtered_contours
    
    def extract_wall_lines(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Extract wall lines from segmentation mask.
        
        Args:
            mask: Segmentation mask
            
        Returns:
            List of wall line segments
        """
        # Get wall mask (class 1)
        wall_mask = (mask == 1).astype(np.uint8) * 255
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            wall_mask, 1, np.pi/180, threshold=50,
            minLineLength=30, maxLineGap=10
        )
        
        if lines is not None:
            return [line[0] for line in lines]
        
        return []
    
    def extract_doors_windows(self, mask: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract door and window locations from segmentation mask.
        
        Args:
            mask: Segmentation mask
            
        Returns:
            Tuple of (door_contours, window_contours)
        """
        # Get door mask (class 2)
        door_mask = (mask == 2).astype(np.uint8) * 255
        door_contours, _ = cv2.findContours(door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get window mask (class 3)
        window_mask = (mask == 3).astype(np.uint8) * 255
        window_contours, _ = cv2.findContours(window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        min_door_area = 100
        min_window_area = 200
        
        door_contours = [c for c in door_contours if cv2.contourArea(c) > min_door_area]
        window_contours = [c for c in window_contours if cv2.contourArea(c) > min_window_area]
        
        return door_contours, window_contours