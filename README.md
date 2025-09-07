# AI Agent for 2D to 3D Architectural Conversion

An intelligent AI agent that converts 2D floor plans into detailed 3D architectural models using computer vision, deep learning, and 3D reconstruction techniques.

## Features

- **Image Preprocessing**: Advanced image cleaning, noise reduction, and perspective correction
- **Semantic Segmentation**: Deep learning-based room and object detection
- **Vectorization**: Conversion from raster to vector data for precise geometry
- **3D Reconstruction**: Automated 3D model generation with walls, openings, and objects
- **Web API**: RESTful interface for easy integration
- **Multiple Export Formats**: Support for OBJ, GLTF, and STL formats

## Architecture

The system follows a modular pipeline approach:

1. **Input Processing**: 2D floor plan image upload and validation
2. **Computer Vision**: Image preprocessing and feature detection
3. **AI Inference**: Deep learning model for semantic understanding
4. **Vectorization**: Conversion to structured geometric data
5. **3D Generation**: Automated 3D model construction
6. **Export**: Multi-format 3D model output

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface
```bash
python main.py --input floor_plan.jpg --output model.obj
```

### Web API
```bash
python app.py
```

Then visit `http://localhost:5000` and upload your floor plan.

## Project Structure

```
├── src/
│   ├── preprocessing/     # Image preprocessing modules
│   ├── models/           # Deep learning models
│   ├── vectorization/    # Vector conversion tools
│   ├── reconstruction/   # 3D model generation
│   └── agent/           # Main AI agent logic
├── data/                # Training data and examples
├── tests/               # Unit tests
├── app.py              # Web application
└── main.py             # Command line interface
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 2GB+ free disk space

## License

MIT License