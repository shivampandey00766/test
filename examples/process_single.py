#!/usr/bin/env python3
"""
Example script for processing a single floor plan.

This script demonstrates how to use the AI Agent to process
a single floor plan image and generate a 3D model.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.agent import ArchitecturalAgent


def main():
    """Process a single floor plan."""
    # Configuration
    input_image = "sample_floor_plan.jpg"  # Replace with your image path
    output_dir = "output"
    
    # Room types for classification (optional)
    room_types = [
        "living_room",
        "bedroom", 
        "kitchen",
        "bathroom",
        "dining_room"
    ]
    
    # Output formats
    output_formats = ["obj", "gltf", "svg"]
    
    # Custom configuration
    config = {
        'reconstruction': {
            'wall_height': 2.4,
            'ceiling_height': 2.7,
            'reconstruction_method': 'blender'
        },
        'export': {
            'formats': output_formats,
            'generate_visualizations': True
        }
    }
    
    print("AI Agent for 2D to 3D Architectural Conversion")
    print("=" * 50)
    
    # Check if input file exists
    if not os.path.exists(input_image):
        print(f"Error: Input file not found: {input_image}")
        print("Please place a floor plan image in the examples directory and update the script.")
        return
    
    try:
        # Initialize agent
        print("Initializing AI Agent...")
        agent = ArchitecturalAgent(config=config, output_dir=output_dir)
        
        # Start processing session
        session_id = agent.start_session()
        print(f"Started session: {session_id}")
        
        # Process floor plan
        print(f"Processing floor plan: {input_image}")
        result = agent.process_floor_plan(
            image_path=input_image,
            output_formats=output_formats,
            room_types=room_types,
            custom_config=config
        )
        
        # Display results
        if result['status'] == 'success':
            print("\n✅ Processing completed successfully!")
            print(f"Session ID: {result['session_id']}")
            print(f"Processing time: {result['processing_time']}")
            print("\nGenerated files:")
            for file_type, file_path in result['output_files'].items():
                print(f"  {file_type}: {file_path}")
            
            print(f"\nAll files saved to: {output_dir}/{session_id}/")
            
        else:
            print(f"\n❌ Processing failed: {result.get('error', 'Unknown error')}")
            return 1
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())