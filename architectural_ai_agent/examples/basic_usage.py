#!/usr/bin/env python3
"""
Basic Usage Example for Architectural AI Agent

This script demonstrates how to use the AI agent to convert a 2D floor plan
into a 3D architectural model.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agent import ArchitecturalAIAgent


def main():
    """Main function demonstrating basic usage."""
    
    print("Architectural AI Agent - Basic Usage Example")
    print("=" * 50)
    
    # Initialize the AI agent
    print("Initializing AI agent...")
    agent = ArchitecturalAIAgent()
    
    # Example configuration (optional)
    config_updates = {
        'preprocessing': {
            'target_size': [1024, 1024],
            'noise_reduction': True
        },
        'model': {
            'segmentation': {
                'architecture': 'unet',
                'num_classes': 15
            }
        },
        'reconstruction': {
            'output_format': 'obj'
        }
    }
    agent.update_config(config_updates)
    
    # Process a floor plan image
    # Note: You'll need to provide an actual floor plan image
    image_path = "sample_floor_plan.png"  # Replace with actual image path
    output_dir = "output/basic_example"
    
    print(f"Processing floor plan: {image_path}")
    
    try:
        # Check if sample image exists
        if not Path(image_path).exists():
            print(f"Sample image not found: {image_path}")
            print("Please provide a floor plan image to process.")
            print("\nTo use this example:")
            print("1. Place a floor plan image in the examples directory")
            print("2. Update the 'image_path' variable in this script")
            print("3. Run the script again")
            return
        
        # Process the floor plan
        results = agent.process_floor_plan(
            image_path=image_path,
            output_dir=output_dir,
            visualize=True,
            save_intermediate=True
        )
        
        if results.get('success', True):  # Default to True if not specified
            print("\nProcessing completed successfully!")
            print(f"Results saved to: {results['output_dir']}")
            
            # Print summary statistics
            print("\nProcessing Summary:")
            print("-" * 20)
            
            stages = results.get('processing_stages', {})
            for stage_name, stage_data in stages.items():
                print(f"{stage_name.replace('_', ' ').title()}:")
                for key, value in stage_data.items():
                    print(f"  {key}: {value}")
            
            # Print timing information
            timing = results.get('timing', {})
            if timing:
                print("\nTiming Information:")
                print("-" * 20)
                for stage, time_taken in timing.items():
                    print(f"{stage.replace('_', ' ').title()}: {time_taken:.2f}s")
            
            # Print 3D model information
            model_3d = results.get('model_3d', {})
            if model_3d:
                print("\n3D Model Information:")
                print("-" * 20)
                print(f"Total vertices: {model_3d.get('total_vertices', 'N/A'):,}")
                print(f"Total faces: {model_3d.get('total_faces', 'N/A'):,}")
                
                if results.get('export_success'):
                    print(f"3D model exported to: {results.get('model_output_path')}")
            
            # Print summary report
            if 'summary_report' in results:
                print("\n" + "=" * 50)
                print(results['summary_report'])
        
        else:
            print(f"Processing failed: {results.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Installed all required dependencies")
        print("2. Provided a valid floor plan image")
        print("3. Sufficient system resources")


if __name__ == "__main__":
    main()