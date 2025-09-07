#!/usr/bin/env python3
"""
Batch Processing Example for Architectural AI Agent

This script demonstrates how to process multiple floor plan images in batch.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agent import ArchitecturalAIAgent


def main():
    """Main function demonstrating batch processing."""
    
    print("Architectural AI Agent - Batch Processing Example")
    print("=" * 50)
    
    # Initialize the AI agent
    print("Initializing AI agent...")
    agent = ArchitecturalAIAgent()
    
    # Configuration for batch processing
    config_updates = {
        'preprocessing': {
            'target_size': [1024, 1024]
        },
        'reconstruction': {
            'output_format': 'obj'
        },
        'logging': {
            'level': 'INFO'
        }
    }
    agent.update_config(config_updates)
    
    # Batch processing settings
    input_directory = "sample_floor_plans"  # Directory containing floor plan images
    output_directory = "output/batch_results"
    file_pattern = "*.png"  # Process all PNG files
    
    print(f"Processing floor plans from: {input_directory}")
    print(f"Output directory: {output_directory}")
    print(f"File pattern: {file_pattern}")
    
    try:
        # Check if input directory exists
        if not Path(input_directory).exists():
            print(f"Input directory not found: {input_directory}")
            print("\nTo use this example:")
            print("1. Create a directory called 'sample_floor_plans'")
            print("2. Place floor plan images (PNG format) in that directory")
            print("3. Run this script again")
            
            # Create sample directory structure
            Path(input_directory).mkdir(exist_ok=True)
            print(f"\nCreated sample directory: {input_directory}")
            print("Please add floor plan images to this directory.")
            return
        
        # Process all images in batch
        batch_results = agent.batch_process(
            input_directory=input_directory,
            output_directory=output_directory,
            file_pattern=file_pattern,
            visualize=True,
            save_intermediate=False  # Save space for batch processing
        )
        
        # Print batch summary
        print(f"\nBatch processing completed!")
        print(f"Total images processed: {len(batch_results)}")
        
        successful_results = [r for r in batch_results if r.get('success', False)]
        failed_results = [r for r in batch_results if not r.get('success', False)]
        
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        
        if successful_results:
            print("\nSuccessful Processing Results:")
            print("-" * 30)
            
            total_vertices = 0
            total_faces = 0
            total_time = 0
            
            for result in successful_results:
                image_name = Path(result['input_path']).name
                timing = result.get('timing', {})
                model_3d = result.get('model_3d', {})
                
                print(f"Image: {image_name}")
                if 'total' in timing:
                    print(f"  Processing time: {timing['total']:.2f}s")
                    total_time += timing['total']
                
                if model_3d:
                    vertices = model_3d.get('total_vertices', 0)
                    faces = model_3d.get('total_faces', 0)
                    print(f"  3D model: {vertices:,} vertices, {faces:,} faces")
                    total_vertices += vertices
                    total_faces += faces
                
                print()
            
            # Print aggregate statistics
            print("Aggregate Statistics:")
            print("-" * 20)
            print(f"Total vertices: {total_vertices:,}")
            print(f"Total faces: {total_faces:,}")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Average time per image: {total_time/len(successful_results):.2f}s")
        
        if failed_results:
            print("\nFailed Processing Results:")
            print("-" * 30)
            for result in failed_results:
                image_name = Path(result['input_path']).name
                error = result.get('error', 'Unknown error')
                print(f"Image: {image_name}")
                print(f"  Error: {error}")
                print()
        
        print(f"\nDetailed results saved to: {output_directory}/batch_summary.json")
    
    except Exception as e:
        print(f"Error during batch processing: {e}")


if __name__ == "__main__":
    main()