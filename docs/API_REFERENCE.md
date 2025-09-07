# API Reference

## Overview

The AI Architectural Converter provides both a RESTful API and a command-line interface for converting 2D floor plans to 3D models.

## REST API

### Base URL
```
http://localhost:5000
```

### Endpoints

#### GET /
Serves the web interface for uploading and converting floor plans.

#### GET /api/status
Get the current status of the AI Agent and API.

**Response:**
```json
{
  "status": "online",
  "ai_agent": {
    "models_loaded": {
      "segmentation_model": true,
      "depth_model": true,
      "room_classifier": true
    },
    "config": {...},
    "components_initialized": true
  },
  "timestamp": 1234567890.123
}
```

#### POST /api/convert
Convert a 2D floor plan to 3D model.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response:**
```json
{
  "success": true,
  "input_image": "/path/to/input.jpg",
  "output_directory": "/path/to/output",
  "exported_models": {
    "obj": "/path/to/model.obj",
    "gltf": "/path/to/model.gltf"
  },
  "visualization": "/path/to/visualization.png",
  "model_info": {
    "walls": ["Wall_001", "Wall_002"],
    "floors": ["Floor_001", "Floor_002"],
    "doors": ["Door_001"],
    "windows": ["Window_001"],
    "furniture": ["Furniture_001"],
    "materials": {...},
    "lighting": [...],
    "metadata": {...}
  },
  "depth_predictions": {
    "depth": 5.0,
    "wall_height": 2.5,
    "spatial_features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  },
  "room_classifications": [...],
  "vector_data": {...},
  "metadata": {
    "total_processing_time": 45.2,
    "models_used": ["segmentation_model", "depth_model"],
    "processing_steps": 10
  }
}
```

#### GET /api/download/<filename>
Download generated files.

**Parameters:**
- `filename`: Name of the file to download

**Response:**
- File download

#### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1234567890.123,
  "version": "1.0.0"
}
```

### Error Responses

All error responses follow this format:
```json
{
  "error": "Error message describing what went wrong"
}
```

Common HTTP status codes:
- `400`: Bad Request (invalid file, missing parameters)
- `413`: Payload Too Large (file too big)
- `500`: Internal Server Error

## Command Line Interface

### Usage
```bash
python main.py <input> -o <output> [options]
```

### Arguments

#### Required
- `input`: Path to the input 2D floor plan image

#### Optional
- `-o, --output`: Output directory for the 3D model (required)
- `-n, --name`: Name for the output files (default: converted_model)
- `--formats`: Output formats - obj, gltf, fbx (default: obj)
- `--render`: Generate a rendered image of the 3D model
- `--resolution WIDTH HEIGHT`: Render resolution (default: 1920 1080)
- `--config`: Path to custom configuration file
- `--verbose, -v`: Enable verbose logging
- `--quiet, -q`: Suppress output except errors
- `--status`: Show AI Agent status and exit

### Examples

#### Basic conversion
```bash
python main.py floor_plan.jpg -o output/
```

#### Multiple formats with custom name
```bash
python main.py floor_plan.png -o output/ -n my_house --formats obj gltf
```

#### With 3D rendering
```bash
python main.py floor_plan.jpg -o output/ --render --resolution 2560 1440
```

#### Using custom configuration
```bash
python main.py floor_plan.jpg -o output/ --config custom_config.json
```

#### Batch processing
```bash
python examples/batch_processing.py input_images/ -o batch_output/ --workers 4
```

## Configuration

### Default Configuration
```json
{
  "image_processing": {
    "target_size": [1024, 1024],
    "noise_reduction_method": "adaptive",
    "contrast_enhancement_method": "clahe"
  },
  "models": {
    "segmentation_model_path": "models/segmentation_model.pth",
    "depth_model_path": "models/depth_model.pth",
    "room_classifier_path": "models/room_classifier.pth",
    "device": "cpu"
  },
  "vectorization": {
    "scale_factor": 1.0,
    "tolerance": 1.0
  },
  "3d_reconstruction": {
    "scale_factor": 0.01,
    "default_wall_height": 2.5,
    "default_door_height": 2.0,
    "default_window_height": 1.5
  },
  "output": {
    "formats": ["obj", "gltf"],
    "render_resolution": [1920, 1080],
    "export_materials": true,
    "export_lighting": true
  }
}
```

### Custom Configuration
You can provide a custom configuration file using the `--config` option. The file should be in JSON format and can override any of the default settings.

## Supported File Formats

### Input Formats
- PNG
- JPEG/JPG
- BMP
- TIFF

### Output Formats
- **OBJ**: Wavefront OBJ format (recommended for most 3D software)
- **GLTF**: glTF 2.0 format (recommended for web and modern 3D applications)
- **FBX**: Autodesk FBX format (for professional 3D software)

## Error Handling

### Common Issues

1. **File not found**: Ensure the input file exists and the path is correct
2. **Invalid file format**: Use supported image formats (PNG, JPG, BMP, TIFF)
3. **File too large**: Maximum file size is 16MB
4. **Model loading errors**: Ensure model files are present and accessible
5. **Memory errors**: Large images may require more RAM or GPU memory

### Debugging

Enable verbose logging to see detailed information:
```bash
python main.py input.jpg -o output/ --verbose
```

Check the AI Agent status:
```bash
python main.py --status
```

## Performance Tips

1. **Use appropriate image size**: Very large images will take longer to process
2. **Enable GPU acceleration**: Set `device: "cuda"` in configuration if GPU is available
3. **Batch processing**: Use the batch processing example for multiple images
4. **Parallel processing**: Use multiple workers for batch operations

## Rate Limits

Currently, there are no rate limits implemented. However, for production use, consider implementing rate limiting based on your requirements.