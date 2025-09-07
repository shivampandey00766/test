# User Guide

## Getting Started

The AI Architectural Converter is a powerful tool that transforms 2D floor plans into detailed 3D models using artificial intelligence. This guide will help you get started with the tool.

## Installation

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- 2GB+ free disk space
- CUDA-capable GPU (optional, for faster processing)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python main.py --status
```

## Quick Start

### 1. Prepare Your Floor Plan

Your floor plan should be:
- A clear, high-contrast image
- In PNG, JPG, JPEG, BMP, or TIFF format
- Maximum 16MB in size
- Preferably with black lines on white background

### 2. Basic Conversion

#### Using the Web Interface
1. Start the web server:
   ```bash
   python app.py
   ```
2. Open your browser to `http://localhost:5000`
3. Upload your floor plan image
4. Click "Convert to 3D"
5. Download the generated 3D model

#### Using the Command Line
```bash
python main.py your_floor_plan.jpg -o output_directory/
```

### 3. View Your 3D Model

The tool generates 3D models in multiple formats:
- **OBJ**: Use with Blender, Maya, 3ds Max, or other 3D software
- **GLTF**: Use with web viewers, Unity, Unreal Engine
- **FBX**: Use with professional 3D software

## Understanding the Output

### Generated Files

After conversion, you'll find several files in your output directory:

1. **3D Model Files** (`.obj`, `.gltf`, `.fbx`)
   - The main 3D model with geometry, materials, and lighting

2. **Visualization** (`_visualization.png`)
   - Shows the detected features and segmentation results

3. **3D Render** (`_render.png`, if requested)
   - A rendered image of the 3D model

### Model Components

The generated 3D model includes:

- **Walls**: Extruded wall segments with proper thickness
- **Floors**: Room floor surfaces
- **Doors**: 3D door objects with realistic proportions
- **Windows**: 3D window objects
- **Furniture**: Basic furniture representations
- **Materials**: Applied textures and materials
- **Lighting**: Basic scene lighting setup

## Advanced Usage

### Custom Configuration

Create a custom configuration file to adjust the conversion process:

```json
{
  "image_processing": {
    "target_size": [2048, 2048],
    "noise_reduction_method": "morphological"
  },
  "3d_reconstruction": {
    "default_wall_height": 3.0,
    "default_door_height": 2.1
  },
  "output": {
    "formats": ["obj", "gltf"],
    "render_resolution": [2560, 1440]
  }
}
```

Use the configuration:
```bash
python main.py input.jpg -o output/ --config my_config.json
```

### Batch Processing

Process multiple floor plans at once:

```bash
python examples/batch_processing.py input_folder/ -o output_folder/ --workers 4
```

### High-Quality Rendering

Generate a high-resolution render of your 3D model:

```bash
python main.py input.jpg -o output/ --render --resolution 3840 2160
```

## Tips for Best Results

### Floor Plan Preparation

1. **Use High Contrast**: Black lines on white background work best
2. **Clean Images**: Remove noise, shadows, and unnecessary elements
3. **Proper Scale**: Ensure the floor plan is not too small or too large
4. **Clear Labels**: Room labels should be readable
5. **Complete Walls**: Ensure all walls are clearly drawn

### Common Issues and Solutions

#### Poor Segmentation Results
- **Problem**: AI doesn't recognize walls or rooms correctly
- **Solution**: Improve image contrast, remove noise, ensure clear line separation

#### Missing Elements
- **Problem**: Doors or windows not detected
- **Solution**: Make sure openings are clearly marked and have sufficient size

#### Incorrect Room Classification
- **Problem**: Rooms are classified as wrong types
- **Solution**: Add clear room labels or improve the room classifier model

#### Low-Quality 3D Model
- **Problem**: 3D model looks unrealistic
- **Solution**: Check depth predictions, adjust wall heights, improve materials

## Troubleshooting

### Common Error Messages

#### "File not found"
- Check that the input file path is correct
- Ensure the file exists and is readable

#### "Invalid file format"
- Use supported formats: PNG, JPG, JPEG, BMP, TIFF
- Check file extension and format

#### "File too large"
- Reduce image size or resolution
- Maximum supported size is 16MB

#### "Model loading error"
- Check that model files are present
- Verify file permissions
- Try reinstalling dependencies

### Performance Issues

#### Slow Processing
- Use smaller images for faster processing
- Enable GPU acceleration if available
- Close other applications to free up memory

#### Memory Errors
- Reduce image size
- Increase system RAM
- Use CPU-only mode if GPU memory is insufficient

### Getting Help

1. **Check Logs**: Enable verbose logging with `--verbose`
2. **Status Check**: Run `python main.py --status`
3. **Test with Sample**: Use the provided sample floor plan
4. **Community Support**: Check GitHub issues and discussions

## Examples

### Example 1: Simple House
```bash
# Create a sample floor plan
python examples/simple_floor_plan.py

# Convert to 3D
python main.py sample_floor_plan.jpg -o output/ -n simple_house --formats obj gltf --render
```

### Example 2: Batch Processing
```bash
# Process multiple floor plans
python examples/batch_processing.py floor_plans/ -o 3d_models/ --workers 2 --formats obj
```

### Example 3: High-Quality Output
```bash
# Generate high-resolution 3D model with custom settings
python main.py floor_plan.jpg -o output/ --render --resolution 3840 2160 --formats obj gltf fbx
```

## Best Practices

### For Architects
- Use standard architectural drawing conventions
- Ensure all rooms are clearly labeled
- Maintain consistent line weights
- Include dimensions when possible

### For Developers
- Implement proper error handling
- Use batch processing for large datasets
- Monitor memory usage
- Implement logging and monitoring

### For Researchers
- Experiment with different model configurations
- Collect feedback for model improvement
- Document edge cases and limitations
- Contribute to model training datasets

## Next Steps

1. **Explore Advanced Features**: Try different configuration options
2. **Integrate with Workflows**: Use the API in your applications
3. **Customize Models**: Train custom models for specific use cases
4. **Contribute**: Help improve the tool by reporting issues and contributing code

## Support

For additional help and support:
- Check the documentation in the `docs/` folder
- Review example scripts in the `examples/` folder
- Report issues on the project repository
- Join the community discussions