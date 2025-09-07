#!/usr/bin/env python3
"""
Advanced Usage Example for Architectural AI Agent

This script demonstrates advanced features including custom configuration,
model training, and detailed analysis.
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agent import ArchitecturalAIAgent
from utils import Config


def main():
    """Main function demonstrating advanced usage."""
    
    print("Architectural AI Agent - Advanced Usage Example")
    print("=" * 50)
    
    # Create custom configuration
    custom_config = {
        'model': {
            'segmentation': {
                'architecture': 'deeplabv3+',
                'encoder': 'resnet101',
                'num_classes': 15
            },
            'object_detection': {
                'confidence_threshold': 0.6
            }
        },
        'preprocessing': {
            'target_size': [1024, 1024],
            'noise_reduction': True,
            'perspective_correction': True,
            'line_enhancement': True
        },
        'reconstruction': {
            'output_format': 'gltf',  # Use GLTF format for better material support
            'standard_heights': {
                'wall_height': 2.7,  # Higher ceilings
                'door_height': 2.1,
                'window_height': 1.4
            }
        },
        'device': 'auto',
        'logging': {
            'level': 'DEBUG'
        }
    }
    
    # Initialize agent with custom configuration
    print("Initializing AI agent with custom configuration...")
    agent = ArchitecturalAIAgent(config=custom_config)
    
    # Save configuration for future use
    config_path = "examples/custom_config.yaml"
    agent.save_config(config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Example 1: Process a single floor plan with detailed analysis
    print("\n" + "=" * 50)
    print("Example 1: Detailed Single Image Processing")
    print("=" * 50)
    
    image_path = "sample_floor_plan.png"
    output_dir = "output/advanced_example"
    
    if Path(image_path).exists():
        results = process_with_detailed_analysis(agent, image_path, output_dir)
        
        if results.get('success', True):
            print("Processing completed successfully!")
            analyze_results(results)
        else:
            print(f"Processing failed: {results.get('error')}")
    else:
        print(f"Sample image not found: {image_path}")
        print("Skipping detailed analysis example.")
    
    # Example 2: Component-by-component processing
    print("\n" + "=" * 50)
    print("Example 2: Component-by-Component Processing")
    print("=" * 50)
    
    demonstrate_component_usage(agent)
    
    # Example 3: Custom visualization
    print("\n" + "=" * 50)
    print("Example 3: Custom Visualization")
    print("=" * 50)
    
    demonstrate_custom_visualization(agent)


def process_with_detailed_analysis(agent, image_path, output_dir):
    """Process image with detailed analysis and custom settings."""
    
    print(f"Processing: {image_path}")
    print("Enabling all analysis features...")
    
    # Process with all features enabled
    results = agent.process_floor_plan(
        image_path=image_path,
        output_dir=output_dir,
        visualize=True,
        save_intermediate=True
    )
    
    return results


def analyze_results(results):
    """Perform detailed analysis of processing results."""
    
    print("\nDetailed Results Analysis:")
    print("-" * 30)
    
    # Analyze processing stages
    stages = results.get('processing_stages', {})
    
    print("Processing Stage Analysis:")
    for stage_name, stage_data in stages.items():
        print(f"\n{stage_name.replace('_', ' ').title()}:")
        
        if stage_name == 'feature_detection':
            total_features = (stage_data.get('num_walls', 0) + 
                            stage_data.get('num_doors', 0) + 
                            stage_data.get('num_windows', 0) + 
                            stage_data.get('num_fixtures', 0))
            print(f"  Total architectural features detected: {total_features}")
            
            if stage_data.get('num_rooms', 0) > 0:
                print(f"  Average features per room: {total_features / stage_data['num_rooms']:.1f}")
        
        elif stage_name == 'segmentation':
            room_analysis = stage_data.get('room_analysis', {})
            if room_analysis:
                print(f"  Room types detected: {len(room_analysis)}")
                total_area = sum(info.get('total_area', 0) for info in room_analysis.values())
                print(f"  Total segmented area: {total_area:,} pixels²")
        
        elif stage_name == 'reconstruction':
            if stage_data.get('total_vertices', 0) > 0:
                complexity = stage_data['total_vertices'] / stage_data.get('num_objects', 1)
                print(f"  Average vertices per object: {complexity:.1f}")
    
    # Analyze performance
    timing = results.get('timing', {})
    if timing:
        print("\nPerformance Analysis:")
        total_time = timing.get('total', 0)
        
        # Find bottlenecks
        stage_times = [(stage, time) for stage, time in timing.items() if stage != 'total']
        stage_times.sort(key=lambda x: x[1], reverse=True)
        
        print("Processing time breakdown:")
        for stage, time in stage_times:
            percentage = (time / total_time) * 100 if total_time > 0 else 0
            print(f"  {stage.replace('_', ' ').title()}: {time:.2f}s ({percentage:.1f}%)")
    
    # Quality assessment
    assess_quality(results)


def assess_quality(results):
    """Assess the quality of the processing results."""
    
    print("\nQuality Assessment:")
    print("-" * 20)
    
    quality_score = 0
    max_score = 0
    
    # Check feature detection quality
    features_data = results.get('features_data', {})
    if features_data:
        walls = len(features_data.get('walls', []))
        doors = len(features_data.get('doors', []))
        windows = len(features_data.get('windows', []))
        
        # Basic quality checks
        if walls > 0:
            quality_score += 20
            print("✓ Walls detected")
        else:
            print("✗ No walls detected")
        max_score += 20
        
        if doors > 0:
            quality_score += 15
            print("✓ Doors detected")
        else:
            print("✗ No doors detected")
        max_score += 15
        
        if windows > 0:
            quality_score += 15
            print("✓ Windows detected")
        else:
            print("✗ No windows detected")
        max_score += 15
    
    # Check segmentation quality
    segmentation_results = results.get('segmentation_results', {})
    if segmentation_results:
        room_analysis = segmentation_results.get('room_analysis', {})
        if len(room_analysis) > 0:
            quality_score += 25
            print("✓ Room segmentation successful")
        else:
            print("✗ No rooms segmented")
        max_score += 25
    
    # Check 3D model quality
    model_3d = results.get('model_3d', {})
    if model_3d and model_3d.get('total_vertices', 0) > 0:
        quality_score += 25
        print("✓ 3D model generated")
        
        # Check model complexity
        vertices = model_3d.get('total_vertices', 0)
        if vertices > 1000:
            print("✓ Detailed 3D model (high vertex count)")
        else:
            print("⚠ Simple 3D model (low vertex count)")
    else:
        print("✗ No 3D model generated")
    max_score += 25
    
    # Calculate overall quality score
    if max_score > 0:
        quality_percentage = (quality_score / max_score) * 100
        print(f"\nOverall Quality Score: {quality_score}/{max_score} ({quality_percentage:.1f}%)")
        
        if quality_percentage >= 80:
            print("🟢 Excellent quality")
        elif quality_percentage >= 60:
            print("🟡 Good quality")
        elif quality_percentage >= 40:
            print("🟠 Fair quality")
        else:
            print("🔴 Poor quality - consider checking input image or configuration")


def demonstrate_component_usage(agent):
    """Demonstrate individual component usage."""
    
    print("Demonstrating individual component usage...")
    
    # Access individual components
    preprocessor = agent.image_preprocessor
    feature_detector = agent.feature_detector
    segmentation_model = agent.segmentation_model
    
    print("\nAvailable components:")
    print(f"- Image Preprocessor: {type(preprocessor).__name__}")
    print(f"- Feature Detector: {type(feature_detector).__name__}")
    print(f"- Segmentation Model: {type(segmentation_model).__name__}")
    print(f"- Depth Estimator: {type(agent.depth_estimator).__name__}")
    print(f"- 3D Reconstructor: {type(agent.model_reconstructor).__name__}")
    
    # Example of using individual components
    print("\nExample: Using preprocessor individually")
    
    sample_image_path = "sample_floor_plan.png"
    if Path(sample_image_path).exists():
        try:
            # Just preprocessing
            processed_image = preprocessor.preprocess_image(sample_image_path)
            print(f"Processed image shape: {processed_image.shape}")
            print(f"Scale factor: {preprocessor.get_scale_factor():.4f}")
        except Exception as e:
            print(f"Component usage example failed: {e}")
    else:
        print("Sample image not available for component demonstration")


def demonstrate_custom_visualization(agent):
    """Demonstrate custom visualization capabilities."""
    
    print("Custom visualization capabilities:")
    
    visualizer = agent.visualizer
    
    # Show available visualization methods
    viz_methods = [method for method in dir(visualizer) 
                  if method.startswith('visualize_') and callable(getattr(visualizer, method))]
    
    print(f"Available visualization methods:")
    for method in viz_methods:
        method_name = method.replace('visualize_', '').replace('_', ' ').title()
        print(f"- {method_name}")
    
    # Example of creating custom visualization
    print("\nCustom visualization example:")
    print("You can create custom visualizations by:")
    print("1. Accessing the visualizer: agent.visualizer")
    print("2. Using matplotlib/plotly directly with the results data")
    print("3. Creating custom analysis reports")
    
    # Show color palette
    print(f"\nDefault color palette: {visualizer.color_palette[:5]}...")


if __name__ == "__main__":
    main()