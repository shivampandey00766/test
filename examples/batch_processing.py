#!/usr/bin/env python3
"""
Example script for batch processing multiple floor plans.
Demonstrates how to process multiple images in a directory.
"""

import sys
import os
import logging
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agent import AIAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_single_image(ai_agent, image_path, output_dir, max_workers=1):
    """Process a single image."""
    try:
        image_name = Path(image_path).stem
        logger.info(f"Processing: {image_name}")
        
        result = ai_agent.convert_2d_to_3d(
            image_path=str(image_path),
            output_dir=str(output_dir / image_name),
            output_name=image_name
        )
        
        return {
            'image': str(image_path),
            'success': result['success'],
            'error': result.get('error') if not result['success'] else None,
            'outputs': result.get('exported_models', {}),
            'processing_time': result.get('metadata', {}).get('total_processing_time', 0)
        }
    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return {
            'image': str(image_path),
            'success': False,
            'error': str(e),
            'outputs': {},
            'processing_time': 0
        }


def main():
    """Main batch processing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process floor plans')
    parser.add_argument('input_dir', help='Directory containing floor plan images')
    parser.add_argument('-o', '--output', default='batch_output', help='Output directory')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--formats', nargs='+', default=['obj'], help='Output formats')
    parser.add_argument('--pattern', default='*.jpg', help='File pattern to match')
    
    args = parser.parse_args()
    
    try:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output)
        
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return 1
        
        # Find all matching images
        image_files = list(input_dir.glob(args.pattern))
        if not image_files:
            logger.error(f"No images found matching pattern: {args.pattern}")
            return 1
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize AI Agent
        config = {
            'output': {
                'formats': args.formats
            }
        }
        
        logger.info("Initializing AI Agent...")
        ai_agent = AIAgent(config)
        
        # Process images
        start_time = time.time()
        results = []
        
        if args.workers == 1:
            # Sequential processing
            for image_path in image_files:
                result = process_single_image(ai_agent, image_path, output_dir)
                results.append(result)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                # Submit all tasks
                future_to_image = {
                    executor.submit(process_single_image, ai_agent, image_path, output_dir): image_path
                    for image_path in image_files
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_image):
                    result = future.result()
                    results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total images: {len(image_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per image: {total_time/len(image_files):.2f} seconds")
        
        if successful > 0:
            print(f"\nSuccessful conversions:")
            for result in results:
                if result['success']:
                    print(f"  ✓ {Path(result['image']).name}")
                    for format_name, path in result['outputs'].items():
                        print(f"    {format_name.upper()}: {path}")
        
        if failed > 0:
            print(f"\nFailed conversions:")
            for result in results:
                if not result['success']:
                    print(f"  ✗ {Path(result['image']).name}: {result['error']}")
        
        print("="*60)
        
        # Cleanup
        ai_agent.cleanup()
        
        return 0 if failed == 0 else 1
    
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())