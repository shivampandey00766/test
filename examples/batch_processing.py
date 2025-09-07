#!/usr/bin/env python3
"""
Example script for batch processing multiple floor plans.

This script demonstrates how to process multiple floor plan images
in batch mode using the AI Agent.
"""

import sys
import os
import glob
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.agent import ArchitecturalAgent


def process_single_image(agent, image_path, output_dir, config):
    """Process a single image."""
    try:
        print(f"Processing: {os.path.basename(image_path)}")
        
        # Start processing session
        session_id = agent.start_session()
        
        # Process floor plan
        result = agent.process_floor_plan(
            image_path=image_path,
            output_formats=["obj", "gltf", "svg"],
            custom_config=config
        )
        
        if result['status'] == 'success':
            print(f"✅ Completed: {os.path.basename(image_path)} (Session: {session_id})")
            return {
                'image': image_path,
                'session_id': session_id,
                'status': 'success',
                'output_files': result['output_files']
            }
        else:
            print(f"❌ Failed: {os.path.basename(image_path)} - {result.get('error', 'Unknown error')}")
            return {
                'image': image_path,
                'session_id': session_id,
                'status': 'failed',
                'error': result.get('error', 'Unknown error')
            }
    
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(image_path)}: {e}")
        return {
            'image': image_path,
            'status': 'error',
            'error': str(e)
        }


def main():
    """Process multiple floor plans in batch."""
    # Configuration
    input_dir = "sample_images"  # Directory containing floor plan images
    output_dir = "batch_output"
    max_workers = 2  # Number of parallel workers
    
    # Supported image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # Custom configuration
    config = {
        'reconstruction': {
            'wall_height': 2.4,
            'ceiling_height': 2.7,
            'reconstruction_method': 'blender'
        },
        'export': {
            'formats': ['obj', 'gltf', 'svg'],
            'generate_visualizations': True
        }
    }
    
    print("AI Agent - Batch Processing")
    print("=" * 40)
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(input_dir, ext)
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(input_dir, ext.upper())
        image_files.extend(glob.glob(pattern))
    
    if not image_files:
        print(f"Error: No image files found in {input_dir}")
        print(f"Supported formats: {', '.join(image_extensions)}")
        return 1
    
    print(f"Found {len(image_files)} image files:")
    for img in image_files:
        print(f"  - {os.path.basename(img)}")
    
    print(f"\nProcessing with {max_workers} parallel workers...")
    print(f"Output directory: {output_dir}")
    
    # Initialize agent
    agent = ArchitecturalAgent(config=config, output_dir=output_dir)
    
    # Process images
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_image = {
            executor.submit(process_single_image, agent, img, output_dir, config): img
            for img in image_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_image):
            result = future.result()
            results.append(result)
    
    # Calculate statistics
    end_time = time.time()
    total_time = end_time - start_time
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"\n" + "=" * 40)
    print("BATCH PROCESSING COMPLETE")
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/len(image_files):.2f} seconds")
    
    # Display detailed results
    print(f"\nDetailed Results:")
    print("-" * 40)
    
    for result in results:
        status_icon = "✅" if result['status'] == 'success' else "❌"
        print(f"{status_icon} {os.path.basename(result['image'])}")
        
        if result['status'] == 'success':
            print(f"   Session: {result['session_id']}")
            print(f"   Output files: {len(result['output_files'])}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())