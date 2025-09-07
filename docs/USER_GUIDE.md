# User Guide

This guide will help you get started with the AI Agent for 2D to 3D Architectural Conversion.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Command Line Interface](#command-line-interface)
4. [Web Interface](#web-interface)
5. [Configuration](#configuration)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)
- 8GB+ RAM
- OpenGL support for 3D rendering

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: Install Blender

For 3D reconstruction using Blender:

1. Download Blender from https://www.blender.org/
2. Install Blender and note the installation path
3. Install the bpy package:
   ```bash
   pip install bpy
   ```

## Quick Start

### 1. Command Line Processing

Process a single floor plan image:

```bash
python main.py --input floor_plan.jpg --output model.obj
```

### 2. Web Interface

Start the web interface:

```bash
python main.py --web
```

Then open your browser to `http://localhost:5000`

### 3. Batch Processing

Process multiple images:

```bash
python examples/batch_processing.py
```

## Command Line Interface

### Basic Usage

```bash
python main.py --input <input_file> --output <output_file_or_directory>
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input`, `-i` | Input floor plan image file | Required |
| `--output`, `-o` | Output file or directory | Required |
| `--formats`, `-f` | Output formats (obj, gltf, svg, etc.) | obj |
| `--config`, `-c` | Configuration file (JSON) | None |
| `--room-types` | Room types for classification | None |
| `--wall-height` | Wall height in meters | 2.4 |
| `--ceiling-height` | Ceiling height in meters | 2.7 |
| `--reconstruction-method` | 3D reconstruction method | blender |
| `--use-cnn` | Use CNN for segmentation | True |
| `--optimize-geometry` | Optimize vector geometry | True |
| `--web` | Start web interface | False |
| `--port` | Web server port | 5000 |
| `--verbose`, `-v` | Verbose output | False |

### Examples

#### Process with multiple output formats

```bash
python main.py --input floor_plan.jpg --output model --formats obj gltf svg
```

#### Process with custom configuration

```bash
python main.py --input floor_plan.jpg --output model.obj --config config.json
```

#### Process with specific room types

```bash
python main.py --input floor_plan.jpg --output model.obj --room-types living_room bedroom kitchen
```

#### Start web interface

```bash
python main.py --web --port 8080
```

## Web Interface

The web interface provides a user-friendly way to process floor plans without using the command line.

### Features

- **Drag & Drop Upload**: Simply drag and drop your floor plan image
- **Real-time Progress**: See processing progress in real-time
- **Multiple Formats**: Download results in various formats
- **Job Management**: View and manage processing jobs
- **Visualization**: Preview generated visualizations

### Usage

1. **Upload**: Drag and drop or click to select a floor plan image
2. **Process**: Click "Process Floor Plan" to start processing
3. **Monitor**: Watch the progress bar and status updates
4. **Download**: Download generated files when processing is complete

### Supported File Types

- PNG
- JPEG/JPG
- BMP
- TIFF

## Configuration

### Configuration File

Create a JSON configuration file to customize the processing pipeline:

```json
{
  "preprocessing": {
    "denoise_method": "bilateral",
    "contrast_method": "clahe",
    "binarize_method": "adaptive",
    "perspective_correction": true
  },
  "segmentation": {
    "use_cnn": true,
    "use_heuristic": true,
    "confidence_threshold": 0.5
  },
  "vectorization": {
    "min_contour_area": 100,
    "approximation_epsilon": 0.02,
    "min_line_length": 10.0,
    "optimize_geometry": true
  },
  "reconstruction": {
    "wall_height": 2.4,
    "floor_thickness": 0.2,
    "ceiling_height": 2.7,
    "reconstruction_method": "blender"
  },
  "export": {
    "formats": ["obj", "gltf", "svg"],
    "include_metadata": true,
    "generate_visualizations": true
  }
}
```

### Configuration Options

#### Preprocessing Options

- `denoise_method`: Noise removal method (`gaussian`, `bilateral`, `median`)
- `contrast_method`: Contrast enhancement method (`clahe`, `histogram`, `gamma`)
- `binarize_method`: Binarization method (`adaptive`, `otsu`, `threshold`)
- `perspective_correction`: Enable automatic perspective correction

#### Segmentation Options

- `use_cnn`: Use deep learning model for segmentation
- `use_heuristic`: Use heuristic rules for classification
- `confidence_threshold`: Minimum confidence for detections

#### Vectorization Options

- `min_contour_area`: Minimum area for contour detection
- `approximation_epsilon`: Epsilon for contour approximation
- `min_line_length`: Minimum length for line segments
- `optimize_geometry`: Enable geometry optimization

#### Reconstruction Options

- `wall_height`: Standard wall height in meters
- `floor_thickness`: Floor thickness in meters
- `ceiling_height`: Ceiling height in meters
- `reconstruction_method`: 3D reconstruction method (`blender`, `open3d`, `mesh`)

#### Export Options

- `formats`: List of output formats
- `include_metadata`: Include metadata in output
- `generate_visualizations`: Generate visualization images

## Examples

### Example 1: Basic Processing

```bash
# Process a single floor plan
python main.py --input sample_floor_plan.jpg --output my_model.obj
```

### Example 2: Multiple Formats

```bash
# Generate multiple output formats
python main.py --input sample_floor_plan.jpg --output my_model --formats obj gltf svg
```

### Example 3: Custom Configuration

```bash
# Use custom configuration
python main.py --input sample_floor_plan.jpg --output my_model.obj --config my_config.json
```

### Example 4: Web Interface

```bash
# Start web interface
python main.py --web --port 8080
```

### Example 5: Batch Processing

```python
# Process multiple images
python examples/batch_processing.py
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` when running the script.

**Solution**: Make sure you're running from the project root directory and all dependencies are installed:

```bash
pip install -r requirements.txt
```

#### 2. CUDA/GPU Issues

**Problem**: CUDA-related errors or slow processing.

**Solution**: 
- Install CUDA-compatible PyTorch
- Check GPU availability
- Fall back to CPU processing if needed

#### 3. Blender Not Found

**Problem**: Blender-related errors during 3D reconstruction.

**Solution**:
- Install Blender
- Use alternative reconstruction method: `--reconstruction-method open3d`
- Or use mesh-based reconstruction: `--reconstruction-method mesh`

#### 4. Memory Issues

**Problem**: Out of memory errors during processing.

**Solution**:
- Reduce image resolution
- Use CPU instead of GPU
- Process smaller images
- Increase system RAM

#### 5. File Format Issues

**Problem**: Unsupported file format errors.

**Solution**:
- Convert image to supported format (PNG, JPEG, BMP, TIFF)
- Check file extension
- Verify file is not corrupted

### Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster processing
2. **Optimize Images**: Use appropriately sized images (512x512 to 1024x1024)
3. **Batch Processing**: Process multiple images together for efficiency
4. **Memory Management**: Close other applications to free up RAM

### Getting Help

1. **Check Logs**: Enable verbose mode with `--verbose`
2. **Review Configuration**: Ensure configuration file is valid JSON
3. **Test with Sample Data**: Use provided sample images first
4. **Check Dependencies**: Verify all required packages are installed

### Debug Mode

Enable debug mode for detailed logging:

```bash
python main.py --input floor_plan.jpg --output model.obj --verbose
```

### Log Files

Log files are created in the output directory and contain detailed information about the processing pipeline.

## Advanced Usage

### Custom Room Types

Define custom room types for better classification:

```python
room_types = [
    "living_room",
    "bedroom", 
    "kitchen",
    "bathroom",
    "dining_room",
    "office",
    "storage"
]
```

### API Integration

Use the REST API for programmatic access:

```python
import requests

# Upload file
with open('floor_plan.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/upload', files=files)

# Process file
data = {'filename': 'uploaded_file.jpg', 'output_formats': ['obj', 'gltf']}
response = requests.post('http://localhost:5000/api/process', json=data)
```

### Custom Models

Train custom models for better accuracy:

1. Prepare training data
2. Modify the segmentation models
3. Update configuration to use custom models

## Support

For additional help and support:

1. Check the documentation
2. Review example scripts
3. Check the GitHub issues page
4. Contact the development team