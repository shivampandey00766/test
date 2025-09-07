# Contributing to Architectural AI Agent

Thank you for your interest in contributing to the Architectural AI Agent! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- 🐛 **Bug Reports**: Help us identify and fix issues
- ✨ **Feature Requests**: Suggest new functionality
- 🔧 **Code Contributions**: Implement features or fix bugs
- 📚 **Documentation**: Improve or expand documentation
- 🧪 **Testing**: Add test cases or improve test coverage
- 🎨 **UI/UX**: Enhance visualizations and user experience

### Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/architectural-ai-agent.git
   cd architectural-ai-agent
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📋 Development Guidelines

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **isort**: Import sorting

```bash
# Format code
black src/ tests/ examples/

# Check linting
flake8 src/ tests/ examples/

# Sort imports
isort src/ tests/ examples/
```

### Code Standards

- **Python Version**: 3.8+
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Include type hints for function parameters and returns
- **Error Handling**: Use appropriate exception handling
- **Logging**: Use the logging module instead of print statements

#### Example Code Style

```python
"""
Module docstring describing the purpose.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExampleClass:
    """
    Class docstring describing the class purpose.
    
    Args:
        param1: Description of parameter
        param2: Description of parameter
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None):
        self.param1 = param1
        self.param2 = param2 or 0
    
    def example_method(self, input_data: List[Dict]) -> Tuple[bool, str]:
        """
        Method docstring describing what it does.
        
        Args:
            input_data: Description of input parameter
            
        Returns:
            Tuple containing success flag and message
            
        Raises:
            ValueError: When input_data is invalid
        """
        try:
            # Implementation here
            result = self._process_data(input_data)
            return True, "Success"
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise ValueError(f"Invalid input data: {e}")
    
    def _process_data(self, data: List[Dict]) -> Dict:
        """Private method for internal processing."""
        # Implementation
        return {}
```

### Testing

We use pytest for testing. Please include tests for new features and bug fixes.

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_agent.py
```

#### Test Structure

```python
import pytest
import numpy as np
from src.agent import ArchitecturalAIAgent


class TestArchitecturalAIAgent:
    """Test class for ArchitecturalAIAgent."""
    
    @pytest.fixture
    def agent(self):
        """Fixture to create agent instance."""
        return ArchitecturalAIAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert hasattr(agent, 'config')
    
    def test_process_floor_plan_invalid_path(self, agent):
        """Test error handling for invalid image path."""
        with pytest.raises(ValueError):
            agent.process_floor_plan("nonexistent.png")
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Detailed steps to reproduce the bug
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, dependencies
6. **Sample Data**: If possible, include sample floor plan images

### Bug Report Template

```markdown
## Bug Description
Brief description of the bug.

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- Python version: 3.x
- OS: Windows/macOS/Linux
- GPU: Yes/No
- Dependencies: (paste pip freeze output)

## Additional Context
Any other relevant information.
```

## ✨ Feature Requests

When requesting features, please include:

1. **Use Case**: Why is this feature needed?
2. **Description**: Detailed description of the feature
3. **Implementation Ideas**: Any thoughts on implementation
4. **Examples**: Mock-ups or examples if applicable

## 🔧 Code Contributions

### Pull Request Process

1. **Create an Issue**: For significant changes, create an issue first to discuss
2. **Fork and Branch**: Fork the repo and create a feature branch
3. **Implement**: Write code following our guidelines
4. **Test**: Add tests and ensure all tests pass
5. **Document**: Update documentation if needed
6. **Submit PR**: Create a pull request with clear description

### Pull Request Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Documentation
- [ ] Documentation updated
- [ ] Code comments added
- [ ] Examples updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] No merge conflicts
```

## 📚 Documentation

### Documentation Types

- **API Documentation**: Docstrings in code
- **User Guide**: README and tutorials
- **Developer Guide**: This contributing guide
- **Examples**: Working code examples

### Documentation Standards

- Use clear, concise language
- Include code examples
- Keep documentation up-to-date with code changes
- Use proper markdown formatting

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions/methods
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test speed and memory usage

### Test Data

- Use synthetic test data when possible
- Include edge cases and error conditions
- Keep test data small and focused
- Document test data requirements

## 🔄 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version bumped
- [ ] Changelog updated
- [ ] Performance benchmarks run
- [ ] Examples tested

## 🏗️ Architecture Guidelines

### Module Organization

```
src/
├── agent.py              # Main orchestrator
├── preprocessing/         # Image preprocessing
├── detection/            # Feature detection
├── segmentation/         # Semantic segmentation
├── reconstruction/       # 3D reconstruction
└── utils/               # Utilities
```

### Design Principles

1. **Modularity**: Keep components loosely coupled
2. **Extensibility**: Design for easy extension
3. **Performance**: Optimize critical paths
4. **Maintainability**: Write clean, readable code
5. **Testability**: Design for easy testing

### Adding New Components

When adding new components:

1. Follow existing patterns
2. Include comprehensive tests
3. Add configuration options
4. Update documentation
5. Provide usage examples

## 🎯 Priority Areas

We're particularly interested in contributions in these areas:

### High Priority
- 🎯 **Model Accuracy**: Improve detection and segmentation accuracy
- 🚀 **Performance**: Optimize processing speed and memory usage
- 🧪 **Testing**: Increase test coverage
- 📊 **Benchmarking**: Add performance benchmarks

### Medium Priority
- 🎨 **Visualization**: New visualization features
- 🔧 **Configuration**: More flexible configuration options
- 📱 **Mobile Support**: Optimize for mobile/edge devices
- 🌐 **Web Interface**: Web-based user interface

### Low Priority
- 🎵 **Audio**: Audio feedback features
- 🔌 **Integrations**: Third-party integrations
- 📦 **Packaging**: Distribution improvements

## 💬 Communication

### Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code contributions
- **Email**: contact@architectural-ai-agent.com

### Communication Guidelines

- Be respectful and constructive
- Search existing issues before creating new ones
- Use clear, descriptive titles
- Provide context and examples
- Follow up on your contributions

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:

- README.md contributors section
- Release notes
- Documentation credits
- Optional: LinkedIn recommendations

## ❓ Questions?

If you have questions about contributing:

1. Check existing documentation
2. Search GitHub issues and discussions
3. Create a new discussion
4. Contact maintainers directly

Thank you for contributing to Architectural AI Agent! 🏗️✨