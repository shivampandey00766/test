# Developer Guide

This guide is for developers who want to extend, modify, or contribute to the AI Agent for 2D to 3D Architectural Conversion.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Adding New Features](#adding-new-features)
5. [Testing](#testing)
6. [Performance Optimization](#performance-optimization)
7. [Contributing](#contributing)

## Architecture Overview

The AI Agent follows a modular pipeline architecture with the following main components:

```
Input Image → Preprocessing → Segmentation → Vectorization → 3D Reconstruction → Output
```

### Pipeline Stages

1. **Preprocessing**: Image cleaning, noise reduction, perspective correction
2. **Segmentation**: Deep learning-based semantic segmentation
3. **Room Classification**: Room type classification and analysis
4. **Depth Estimation**: 3D depth information estimation
5. **Vectorization**: Raster to vector conversion
6. **3D Reconstruction**: 3D model generation
7. **Export**: Multi-format output generation

## Project Structure

```
workspace/
├── src/                          # Source code
│   ├── preprocessing/            # Image preprocessing modules
│   │   ├── __init__.py
│   │   ├── image_cleaner.py
│   │   ├── feature_detector.py
│   │   └── perspective_corrector.py
│   ├── segmentation/             # Deep learning modules
│   │   ├── __init__.py
│   │   ├── floor_plan_segmenter.py
│   │   ├── room_classifier.py
│   │   └── depth_estimator.py
│   ├── vectorization/            # Vector conversion modules
│   │   ├── __init__.py
│   │   ├── raster_to_vector.py
│   │   ├── geometry_optimizer.py
│   │   └── svg_exporter.py
│   ├── reconstruction/           # 3D reconstruction modules
│   │   ├── __init__.py
│   │   ├── blender_reconstructor.py
│   │   ├── open3d_reconstructor.py
│   │   └── mesh_generator.py
│   └── agent/                    # Main agent and pipeline management
│       ├── __init__.py
│       ├── architectural_agent.py
│       └── pipeline_manager.py
├── examples/                     # Example scripts
├── docs/                         # Documentation
├── tests/                        # Test files
├── models/                       # Pre-trained models
├── data/                         # Sample data
├── app.py                        # Flask web application
├── main.py                       # Command-line interface
└── requirements.txt              # Dependencies
```

## Core Components

### 1. Preprocessing Module

**Location**: `src/preprocessing/`

**Purpose**: Clean and prepare 2D floor plan images for analysis.

**Key Classes**:
- `ImageCleaner`: Image cleaning and enhancement
- `FeatureDetector`: Architectural feature detection
- `PerspectiveCorrector`: Perspective distortion correction

**Example Usage**:
```python
from src.preprocessing import ImageCleaner, FeatureDetector

# Clean image
cleaner = ImageCleaner()
cleaned_image = cleaner.preprocess_pipeline('floor_plan.jpg')

# Detect features
detector = FeatureDetector()
walls = detector.detect_walls(cleaned_image)
doors = detector.detect_doors(cleaned_image, walls)
```

### 2. Segmentation Module

**Location**: `src/segmentation/`

**Purpose**: Semantic segmentation of floor plan elements.

**Key Classes**:
- `FloorPlanSegmenter`: U-Net based segmentation
- `RoomClassifier`: Room type classification
- `DepthEstimator`: 3D depth estimation

**Example Usage**:
```python
from src.segmentation import FloorPlanSegmenter, RoomClassifier

# Segment image
segmenter = FloorPlanSegmenter()
segmentation_mask = segmenter.predict(image)

# Classify rooms
classifier = RoomClassifier()
rooms = classifier.classify_rooms_in_plan(segmentation_mask)
```

### 3. Vectorization Module

**Location**: `src/vectorization/`

**Purpose**: Convert raster images to vector data.

**Key Classes**:
- `RasterToVectorConverter`: Raster to vector conversion
- `GeometryOptimizer`: Vector geometry optimization
- `SVGExporter`: SVG export functionality

**Example Usage**:
```python
from src.vectorization import RasterToVectorConverter

# Convert to vector
converter = RasterToVectorConverter()
vector_data = converter.convert_to_vector(segmentation_mask)
```

### 4. Reconstruction Module

**Location**: `src/reconstruction/`

**Purpose**: Generate 3D models from vector data.

**Key Classes**:
- `BlenderReconstructor`: Blender-based reconstruction
- `Open3DReconstructor`: Open3D-based reconstruction
- `MeshGenerator`: Mesh generation utilities

**Example Usage**:
```python
from src.reconstruction import BlenderReconstructor

# Reconstruct 3D model
reconstructor = BlenderReconstructor()
objects = reconstructor.reconstruct_from_vector_data(vector_data, room_types)
```

### 5. Agent Module

**Location**: `src/agent/`

**Purpose**: Main orchestration and pipeline management.

**Key Classes**:
- `ArchitecturalAgent`: Main AI agent
- `PipelineManager`: Pipeline job management

**Example Usage**:
```python
from src.agent import ArchitecturalAgent

# Initialize agent
agent = ArchitecturalAgent(config=config)

# Process floor plan
result = agent.process_floor_plan(
    image_path='floor_plan.jpg',
    output_formats=['obj', 'gltf'],
    room_types=['living_room', 'bedroom']
)
```

## Adding New Features

### 1. Adding a New Preprocessing Step

Create a new preprocessing class:

```python
# src/preprocessing/new_preprocessor.py
import cv2
import numpy as np

class NewPreprocessor:
    def __init__(self, param1=default_value):
        self.param1 = param1
    
    def process(self, image):
        # Your preprocessing logic here
        processed_image = self._apply_processing(image)
        return processed_image
    
    def _apply_processing(self, image):
        # Implementation details
        pass
```

Update the `__init__.py`:

```python
# src/preprocessing/__init__.py
from .new_preprocessor import NewPreprocessor

__all__ = ['ImageCleaner', 'FeatureDetector', 'PerspectiveCorrector', 'NewPreprocessor']
```

### 2. Adding a New Segmentation Model

Create a new segmentation model:

```python
# src/segmentation/new_segmenter.py
import torch
import torch.nn as nn

class NewSegmenter(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Define your model architecture
        self.model = self._build_model(num_classes)
    
    def _build_model(self, num_classes):
        # Model definition
        pass
    
    def forward(self, x):
        return self.model(x)
```

### 3. Adding a New Export Format

Create a new exporter:

```python
# src/vectorization/new_exporter.py
class NewExporter:
    def __init__(self):
        pass
    
    def export(self, vector_data, output_path):
        # Export logic
        pass
```

### 4. Adding a New 3D Reconstruction Method

Create a new reconstructor:

```python
# src/reconstruction/new_reconstructor.py
class NewReconstructor:
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def reconstruct_from_vector_data(self, vector_data, room_types=None):
        # Reconstruction logic
        pass
```

## Testing

### Unit Tests

Create unit tests for individual components:

```python
# tests/test_preprocessing.py
import unittest
import numpy as np
from src.preprocessing import ImageCleaner

class TestImageCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = ImageCleaner()
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_preprocess_pipeline(self):
        result = self.cleaner.preprocess_pipeline('test_image.jpg')
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result.shape), 2)  # Should be grayscale
```

### Integration Tests

Test the complete pipeline:

```python
# tests/test_pipeline.py
import unittest
from src.agent import ArchitecturalAgent

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.agent = ArchitecturalAgent()
    
    def test_complete_pipeline(self):
        result = self.agent.process_floor_plan(
            image_path='test_data/sample_floor_plan.jpg',
            output_formats=['obj']
        )
        self.assertEqual(result['status'], 'success')
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_preprocessing.py

# Run with coverage
python -m pytest --cov=src tests/
```

## Performance Optimization

### 1. GPU Acceleration

Ensure CUDA is properly configured:

```python
import torch

# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device('cpu')
    print("Using CPU")
```

### 2. Memory Optimization

Optimize memory usage:

```python
# Use smaller batch sizes
batch_size = 1 if torch.cuda.get_device_properties(0).total_memory < 8e9 else 4

# Clear cache
torch.cuda.empty_cache()

# Use mixed precision
from torch.cuda.amp import autocast
```

### 3. Parallel Processing

Use parallel processing for batch operations:

```python
from concurrent.futures import ThreadPoolExecutor

def process_batch(images):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single, img) for img in images]
        results = [future.result() for future in futures]
    return results
```

### 4. Caching

Implement caching for expensive operations:

```python
import functools
import hashlib

@functools.lru_cache(maxsize=128)
def cached_segmentation(image_hash, model_params):
    # Expensive segmentation operation
    pass

def get_image_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()
```

## Contributing

### 1. Code Style

Follow PEP 8 guidelines:

```bash
# Install linting tools
pip install flake8 black isort

# Format code
black src/
isort src/

# Check style
flake8 src/
```

### 2. Documentation

Document all public functions and classes:

```python
def process_image(self, image: np.ndarray, 
                 method: str = 'default') -> np.ndarray:
    """
    Process an image using the specified method.
    
    Args:
        image: Input image as numpy array
        method: Processing method ('default', 'advanced', 'fast')
        
    Returns:
        Processed image as numpy array
        
    Raises:
        ValueError: If method is not supported
    """
    pass
```

### 3. Type Hints

Use type hints for better code clarity:

```python
from typing import List, Dict, Optional, Tuple
import numpy as np

def classify_rooms(self, 
                  segmentation_mask: np.ndarray,
                  room_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    pass
```

### 4. Error Handling

Implement proper error handling:

```python
try:
    result = expensive_operation()
except SpecificException as e:
    logger.error(f"Specific error occurred: {e}")
    raise ProcessingError(f"Failed to process: {e}") from e
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

### 5. Logging

Use structured logging:

```python
import logging

logger = logging.getLogger(__name__)

def process_data(self, data):
    logger.info("Starting data processing", extra={
        'data_size': len(data),
        'method': 'advanced'
    })
    
    try:
        result = self._process(data)
        logger.info("Data processing completed successfully")
        return result
    except Exception as e:
        logger.error("Data processing failed", extra={
            'error': str(e),
            'data_size': len(data)
        })
        raise
```

### 6. Configuration Management

Use configuration files for settings:

```python
import json
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'preprocessing': {
                'denoise_method': 'bilateral',
                'contrast_method': 'clahe'
            }
        }
```

## Advanced Topics

### 1. Custom Model Training

Train custom models for better accuracy:

```python
# Training script
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_model(model, train_loader, val_loader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # Training loop
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['image'])
            loss = criterion(outputs, batch['mask'])
            loss.backward()
            optimizer.step()
        
        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                outputs = model(batch['image'])
                val_loss += criterion(outputs, batch['mask']).item()
        
        print(f"Epoch {epoch}: Val Loss = {val_loss/len(val_loader):.4f}")
```

### 2. Plugin Architecture

Implement a plugin system for extensibility:

```python
# Plugin base class
class Plugin:
    def __init__(self, name: str):
        self.name = name
    
    def process(self, data: Any) -> Any:
        raise NotImplementedError

# Plugin manager
class PluginManager:
    def __init__(self):
        self.plugins = {}
    
    def register_plugin(self, plugin: Plugin):
        self.plugins[plugin.name] = plugin
    
    def process_with_plugin(self, plugin_name: str, data: Any) -> Any:
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin {plugin_name} not found")
        return self.plugins[plugin_name].process(data)
```

### 3. API Extensions

Extend the API with new endpoints:

```python
# In app.py
@app.route('/api/custom/endpoint', methods=['POST'])
def custom_endpoint():
    data = request.get_json()
    # Custom processing logic
    return jsonify({'result': 'success'})
```

### 4. Database Integration

Add database support for job persistence:

```python
# Database models
from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Job(Base):
    __tablename__ = 'jobs'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String(50), unique=True)
    status = Column(String(20))
    created_at = Column(DateTime)
    # ... other fields
```

This developer guide provides a comprehensive overview of how to extend and contribute to the AI Agent for 2D to 3D Architectural Conversion. Follow these guidelines to ensure your contributions are consistent with the project's architecture and coding standards.