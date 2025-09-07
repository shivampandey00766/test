#!/usr/bin/env python3
"""
Setup script for the AI Architectural Converter.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-architectural-converter",
    version="1.0.0",
    author="AI Architectural Converter Team",
    author_email="contact@ai-architectural-converter.com",
    description="An AI-powered tool for converting 2D floor plans to 3D architectural models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai-architectural-converter/ai-architectural-converter",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-architectural-converter=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml", "*.yml"],
    },
    keywords=[
        "ai", "artificial-intelligence", "computer-vision", "3d-modeling",
        "architecture", "floor-plan", "deep-learning", "pytorch", "opencv",
        "blender", "3d-reconstruction", "semantic-segmentation"
    ],
    project_urls={
        "Bug Reports": "https://github.com/ai-architectural-converter/ai-architectural-converter/issues",
        "Source": "https://github.com/ai-architectural-converter/ai-architectural-converter",
        "Documentation": "https://ai-architectural-converter.readthedocs.io/",
    },
)