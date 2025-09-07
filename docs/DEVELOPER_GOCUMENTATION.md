# Developer Documentation

## Architecture Overview

The AI Architectural Converter is built as a modular system with clear separation of concerns. The architecture follows a pipeline pattern where data flows through distinct processing stages.

```
Input Image → Preprocessing → AI Models → Vectorization → 3D Reconstruction → Output
```

### Core Components

1. **Image Preprocessing** (`src/preprocessing/`)
   - Image cleaning and enhancement
   - Feature detection using computer vision
   - Noise reduction and contrast enhancement

2. **AI Models** (`src/models/`)
   - Semantic segmentation for floor plan understanding
   - Depth estimation for 3D reconstruction
   - Room classification for space understanding

3. **Vectorization** (`src/vectorization/`)
   - Conversion from raster to vector data
   - Geometry processing and optimization
   - SVG generation for 2D output

4. **3D Reconstruction** (`src/reconstruction/`)
   - 3D model generation using Blender API
   - Material and lighting setup
   - Export to various 3D formats

5. **AI Agent** (`src/agent/`)
   - Orchestrates the entire pipeline
   - Manages component lifecycle
   - Handles error recovery and logging

## Code Structure

```
src/
├── preprocessing/          # Image processing modules
│   ├── image_processor.py  # Main image processing class
│   ├── feature_detector.py # Computer vision feature detection
│   └── noise_reduction.py  # Advanced noise reduction
├── models/                 # Deep learning models
│   ├── segmentation_model.py # U-Net for semantic segmentation
│   ├── depth_estimation.py   # CNN for depth prediction
│   └── room_classifier.py    # CNN for room classification
├── vectorization/          # Vector processing
│   ├── vectorizer.py       # Raster to vector conversion
│   ├── geometry_processor.py # Geometry optimization
│   └── svg_generator.py    # SVG output generation
├── reconstruction/         # 3D model generation
│   ├── model_builder.py    # Blender-based 3D generation
│   ├── texture_manager.py  # Material and texture management
│   └── lighting_setup.py   # Scene lighting configuration
└── agent/                  # Main AI agent
    ├── ai_agent.py         # Main orchestration class
    └── pipeline.py         # Pipeline management
```

## Extending the System

### Adding New AI Models

To add a new AI model:

1. Create a new model class in `src/models/`
2. Inherit from `torch.nn.Module`
3. Implement required methods:
   - `forward()`: Forward pass
   - `predict()`: Inference method
   - `save_model()`: Model serialization
   - `load_model()`: Model deserialization

Example:
```python
class NewModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize model layers
        
    def forward(self, x):
        # Forward pass implementation
        pass
    
    def predict(self, input_data, device='cpu'):
        # Inference implementation
        pass
    
    def save_model(self, filepath):
        # Save model to file
        pass
    
    @classmethod
    def load_model(cls, filepath, device='cpu'):
        # Load model from file
        pass
```

### Adding New Preprocessing Steps

To add new preprocessing functionality:

1. Create a new class in `src/preprocessing/`
2. Implement the required interface
3. Add to the preprocessing pipeline

Example:
```python
class NewPreprocessor:
    def __init__(self, config):
        self.config = config
    
    def process(self, image):
        # Processing implementation
        return processed_image
```

### Adding New 3D Export Formats

To add support for new 3D formats:

1. Extend the `ModelBuilder` class
2. Add export method for the new format
3. Update the configuration schema

Example:
```python
def export_to_new_format(self, filepath):
    """Export 3D model to new format."""
    # Implementation for new format
    pass
```

## Configuration System

The system uses a hierarchical configuration approach:

### Default Configuration
```python
DEFAULT_CONFIG = {
    'image_processing': {...},
    'models': {...},
    'vectorization': {...},
    '3d_reconstruction': {...},
    'output': {...}
}
```

### Custom Configuration
Users can override any configuration value by providing a custom config file or by passing parameters to the AI Agent.

### Configuration Validation
All configuration values are validated against schemas to ensure compatibility and prevent runtime errors.

## Error Handling

### Error Types

1. **Input Errors**: Invalid files, unsupported formats
2. **Processing Errors**: AI model failures, memory issues
3. **Output Errors**: Export failures, file system issues
4. **System Errors**: Resource exhaustion, hardware issues

### Error Recovery

The system implements graceful error recovery:
- Fallback methods when AI models fail
- Retry mechanisms for transient errors
- Partial results when possible
- Comprehensive error logging

### Logging

Structured logging is used throughout the system:
- Different log levels (DEBUG, INFO, WARNING, ERROR)
- Contextual information in log messages
- Performance metrics and timing
- Error stack traces for debugging

## Testing

### Unit Tests

Each module has corresponding unit tests:
```bash
python -m pytest tests/unit/
```

### Integration Tests

End-to-end pipeline tests:
```bash
python -m pytest tests/integration/
```

### Performance Tests

Benchmark tests for performance monitoring:
```bash
python -m pytest tests/performance/
```

### Test Data

Test data is stored in `tests/data/`:
- Sample floor plan images
- Expected output files
- Test configurations

## Performance Optimization

### Memory Management

- Lazy loading of large models
- Memory cleanup after processing
- Streaming for large files
- Memory usage monitoring

### GPU Acceleration

- Automatic GPU detection
- Fallback to CPU when GPU unavailable
- Memory management for GPU operations
- Batch processing optimization

### Caching

- Model loading cache
- Intermediate result caching
- Configuration caching
- File system caching

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . /app
WORKDIR /app

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

### Production Considerations

1. **Resource Management**: Monitor CPU, memory, and GPU usage
2. **Scaling**: Use load balancers for multiple instances
3. **Security**: Implement authentication and rate limiting
4. **Monitoring**: Set up logging and metrics collection
5. **Backup**: Regular backup of models and configurations

## API Development

### REST API Design

The API follows RESTful principles:
- Resource-based URLs
- HTTP methods for operations
- JSON for data exchange
- Proper HTTP status codes

### API Versioning

API versioning is handled through URL paths:
- `/api/v1/` for current version
- Backward compatibility maintained
- Deprecation notices for old versions

### Documentation

API documentation is generated using:
- OpenAPI/Swagger specification
- Interactive documentation
- Code examples
- Error response documentation

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Set up development environment
4. Make changes
5. Add tests
6. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings
- Add unit tests
- Update documentation

### Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add changelog entry
4. Request code review
5. Address feedback
6. Merge after approval

## Monitoring and Debugging

### Performance Monitoring

- Processing time tracking
- Memory usage monitoring
- GPU utilization tracking
- Error rate monitoring

### Debugging Tools

- Verbose logging mode
- Debug visualization
- Performance profiling
- Memory profiling

### Health Checks

- System health endpoints
- Model availability checks
- Resource availability checks
- Dependency health checks

## Security Considerations

### Input Validation

- File type validation
- File size limits
- Malicious file detection
- Input sanitization

### Resource Protection

- Rate limiting
- Resource quotas
- Memory limits
- CPU limits

### Data Privacy

- No data persistence
- Secure file handling
- Temporary file cleanup
- Privacy compliance

## Future Enhancements

### Planned Features

1. **Real-time Processing**: WebSocket support for real-time updates
2. **Cloud Integration**: AWS/Azure deployment options
3. **Advanced AI**: More sophisticated models and techniques
4. **Mobile Support**: Mobile-optimized interface
5. **Collaboration**: Multi-user editing and sharing

### Research Areas

1. **Better Segmentation**: Improved room and object detection
2. **Style Transfer**: Apply different architectural styles
3. **Material Recognition**: Automatic material detection
4. **Lighting Simulation**: Advanced lighting and rendering
5. **VR/AR Support**: Virtual and augmented reality output

## Troubleshooting

### Common Issues

1. **Model Loading Failures**: Check file paths and permissions
2. **Memory Errors**: Reduce image size or increase system memory
3. **GPU Issues**: Verify CUDA installation and compatibility
4. **Export Failures**: Check output directory permissions
5. **Performance Issues**: Monitor resource usage and optimize

### Debug Commands

```bash
# Check system status
python main.py --status

# Enable verbose logging
python main.py input.jpg -o output/ --verbose

# Test with sample data
python examples/simple_floor_plan.py

# Run tests
python -m pytest tests/
```

### Getting Help

- Check the documentation
- Review example code
- Search existing issues
- Create new issue with details
- Join community discussions