# AI Agent for 2D to 3D Architectural Conversion

An intelligent AI agent that converts 2D floor plans into detailed 3D architectural models using computer vision, deep learning, and 3D reconstruction techniques.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 🚀 Features

- **🖼️ Image Preprocessing**: Advanced image cleaning, noise reduction, and perspective correction
- **🧠 Semantic Segmentation**: Deep learning-based room and object detection using U-Net architecture
- **📐 Vectorization**: Conversion from raster to vector data for precise geometry
- **🏗️ 3D Reconstruction**: Automatic generation of 3D models with proper proportions
- **🌐 Web Interface**: Easy-to-use API and web UI for uploading and processing floor plans
- **📊 Room Classification**: Intelligent room type detection and classification
- **📏 Depth Estimation**: 3D depth information estimation for realistic models
- **💾 Multi-format Export**: Support for OBJ, GLTF, SVG, BLEND, PLY, and STL formats

## 🏗️ Architecture

The system follows a modular pipeline approach:

```
Input Image → Preprocessing → Segmentation → Vectorization → 3D Reconstruction → Export
     ↓              ↓              ↓              ↓              ↓              ↓
  Floor Plan    Image Clean    Deep Learning   Raster→Vector   3D Generation   Multi-format
    Image       & Enhance      Segmentation    Conversion      (Blender/Open3D)   Output
```

### Pipeline Stages

1. **Image Preprocessing** → Clean and prepare 2D floor plan images
2. **Feature Detection** → Identify walls, doors, windows, and room boundaries
3. **Semantic Segmentation** → Classify different architectural elements
4. **Room Classification** → Determine room types and properties
5. **Depth Estimation** → Estimate 3D depth information
6. **Vectorization** → Convert pixel data to geometric primitives
7. **3D Reconstruction** → Generate 3D models with proper dimensions
8. **Export** → Output in standard 3D formats

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)
- 8GB+ RAM
- OpenGL support for 3D rendering

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-username/ai-architectural-agent.git
cd ai-architectural-agent

# Install dependencies
pip install -r requirements.txt

# Optional: Install Blender for 3D reconstruction
# Download from https://www.blender.org/ and install bpy
pip install bpy
```

### Docker Installation

```bash
# Build Docker image
docker build -t ai-architectural-agent .

# Run container
docker run -p 5000:5000 -v $(pwd)/data:/app/data ai-architectural-agent
```

## 🚀 Quick Start

### Command Line Interface

```bash
# Process a single floor plan
python main.py --input floor_plan.jpg --output model.obj

# Process with multiple output formats
python main.py --input floor_plan.jpg --output model --formats obj gltf svg

# Process with custom configuration
python main.py --input floor_plan.jpg --output model.obj --config config.json
```

### Web Interface

```bash
# Start web interface
python main.py --web

# Or use the dedicated script
python examples/web_interface.py
```

Then open your browser to `http://localhost:5000`

### Python API

```python
from src.agent import ArchitecturalAgent

# Initialize agent
agent = ArchitecturalAgent()

# Process floor plan
result = agent.process_floor_plan(
    image_path='floor_plan.jpg',
    output_formats=['obj', 'gltf', 'svg'],
    room_types=['living_room', 'bedroom', 'kitchen']
)

print(f"Status: {result['status']}")
print(f"Output files: {result['output_files']}")
```

## 📁 Project Structure

```
workspace/
├── src/                          # Source code
│   ├── preprocessing/            # Image preprocessing modules
│   │   ├── image_cleaner.py     # Image cleaning and enhancement
│   │   ├── feature_detector.py  # Architectural feature detection
│   │   └── perspective_corrector.py # Perspective correction
│   ├── segmentation/             # Deep learning modules
│   │   ├── floor_plan_segmenter.py # U-Net segmentation
│   │   ├── room_classifier.py   # Room type classification
│   │   └── depth_estimator.py   # 3D depth estimation
│   ├── vectorization/            # Vector conversion modules
│   │   ├── raster_to_vector.py  # Raster to vector conversion
│   │   ├── geometry_optimizer.py # Vector geometry optimization
│   │   └── svg_exporter.py      # SVG export functionality
│   ├── reconstruction/           # 3D reconstruction modules
│   │   ├── blender_reconstructor.py # Blender-based reconstruction
│   │   ├── open3d_reconstructor.py  # Open3D-based reconstruction
│   │   └── mesh_generator.py    # Mesh generation utilities
│   └── agent/                    # Main agent and pipeline management
│       ├── architectural_agent.py # Main AI agent
│       └── pipeline_manager.py  # Pipeline job management
├── examples/                     # Example scripts
│   ├── process_single.py        # Single image processing
│   ├── batch_processing.py      # Batch processing
│   └── web_interface.py         # Web interface launcher
├── docs/                         # Documentation
│   ├── API_REFERENCE.md         # API documentation
│   ├── USER_GUIDE.md            # User guide
│   └── DEVELOPER_GUIDE.md       # Developer guide
├── tests/                        # Test files
├── models/                       # Pre-trained models
├── data/                         # Sample data
├── app.py                        # Flask web application
├── main.py                       # Command-line interface
└── requirements.txt              # Dependencies
```

## 🎯 Usage Examples

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
python main.py --input sample_floor_plan.jpg --output my_model.obj --config examples/example_config.json
```

### Example 4: Batch Processing

```python
# Process multiple images
python examples/batch_processing.py
```

### Example 5: Web Interface

```bash
# Start web interface
python main.py --web --port 8080
```

## 🔧 Configuration

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

## 🌐 API Reference

### REST API Endpoints

- `POST /api/upload` - Upload floor plan image
- `POST /api/process` - Start processing
- `GET /api/job/{job_id}` - Get job status
- `GET /api/job/{job_id}/download/{format}` - Download results
- `GET /api/health` - Health check

### Python API

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

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/

# Run specific test
python -m pytest tests/test_preprocessing.py
```

## 📊 Performance

### Benchmarks

- **Processing Time**: 30-60 seconds per floor plan (GPU)
- **Memory Usage**: 2-4GB RAM
- **Supported Formats**: PNG, JPEG, BMP, TIFF
- **Output Formats**: OBJ, GLTF, SVG, BLEND, PLY, STL

### Optimization Tips

1. Use GPU acceleration for faster processing
2. Optimize image resolution (512x512 to 1024x1024)
3. Use batch processing for multiple images
4. Enable geometry optimization for cleaner models

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/ai-architectural-agent.git
cd ai-architectural-agent

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/
```

## 📚 Documentation

- [User Guide](docs/USER_GUIDE.md) - Complete user documentation
- [API Reference](docs/API_REFERENCE.md) - API documentation
- [Developer Guide](docs/DEVELOPER_GUIDE.md) - Developer documentation

## 🐛 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Install CUDA-compatible PyTorch
2. **Blender Not Found**: Install Blender or use alternative reconstruction method
3. **Memory Issues**: Reduce image resolution or use CPU processing
4. **File Format Issues**: Convert to supported formats (PNG, JPEG, BMP, TIFF)

### Getting Help

1. Check the [User Guide](docs/USER_GUIDE.md)
2. Review example scripts in `examples/`
3. Check GitHub issues
4. Contact the development team

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV for computer vision capabilities
- PyTorch for deep learning framework
- Blender for 3D reconstruction
- Open3D for 3D processing
- The open-source community for inspiration and tools

## 📈 Roadmap

- [ ] Support for more architectural styles
- [ ] Real-time processing capabilities
- [ ] Mobile app integration
- [ ] Cloud deployment options
- [ ] Advanced material and texture support
- [ ] Integration with CAD software

## 📞 Support

For support and questions:

- 📧 Email: support@ai-architectural-agent.com
- 💬 Discord: [Join our community](https://discord.gg/ai-architectural-agent)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/ai-architectural-agent/issues)
- 📖 Wiki: [Project Wiki](https://github.com/your-username/ai-architectural-agent/wiki)

---

**Made with ❤️ by the AI Architectural Agent Team**