#!/usr/bin/env python3
"""
Command-line interface for the 2D to 3D Architectural Conversion AI Agent.
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import json
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent import AIAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Convert 2D floor plans to 3D models using AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.jpg -o output/
  python main.py input.png -o output/ -n my_model --formats obj gltf
  python main.py input.jpg -o output/ --render --resolution 1920 1080
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input',
        help='Path to the input 2D floor plan image'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory for the 3D model'
    )
    
    # Optional arguments
    parser.add_argument(
        '-n', '--name',
        default='converted_model',
        help='Name for the output files (default: converted_model)'
    )
    
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['obj', 'gltf', 'fbx'],
        default=['obj'],
        help='Output formats (default: obj)'
    )
    
    parser.add_argument(
        '--render',
        action='store_true',
        help='Generate a rendered image of the 3D model'
    )
    
    parser.add_argument(
        '--resolution',
        nargs=2,
        type=int,
        default=[1920, 1080],
        metavar=('WIDTH', 'HEIGHT'),
        help='Render resolution (default: 1920 1080)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output except errors'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show AI Agent status and exit'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    try:
        # Load configuration if provided
        config = None
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Initialize AI Agent
        logger.info("Initializing AI Agent...")
        ai_agent = AIAgent(config)
        
        # Show status if requested
        if args.status:
            status = ai_agent.get_status()
            print(json.dumps(status, indent=2))
            return 0
        
        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {args.input}")
            return 1
        
        if not input_path.is_file():
            logger.error(f"Input path is not a file: {args.input}")
            return 1
        
        # Validate output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update configuration for output formats
        if config is None:
            config = {}
        if 'output' not in config:
            config['output'] = {}
        config['output']['formats'] = args.formats
        
        # Reinitialize agent with updated config
        ai_agent = AIAgent(config)
        
        # Start conversion
        logger.info(f"Starting conversion: {args.input} -> {args.output}")
        start_time = time.time()
        
        result = ai_agent.convert_2d_to_3d(
            image_path=str(input_path),
            output_dir=str(output_dir),
            output_name=args.name
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result['success']:
            logger.info("Conversion completed successfully!")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            
            # Print results
            print("\n" + "="*60)
            print("CONVERSION RESULTS")
            print("="*60)
            print(f"Input file: {result['input_image']}")
            print(f"Output directory: {result['output_directory']}")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Models used: {', '.join(result['metadata']['models_used'])}")
            
            if result['exported_models']:
                print("\nExported models:")
                for format_name, path in result['exported_models'].items():
                    print(f"  {format_name.upper()}: {path}")
            
            if result['visualization']:
                print(f"\nVisualization: {result['visualization']}")
            
            # Generate render if requested
            if args.render:
                logger.info("Generating 3D render...")
                render_path = ai_agent.render_3d_model(
                    output_dir=str(output_dir),
                    output_name=args.name,
                    resolution=tuple(args.resolution)
                )
                if render_path:
                    print(f"3D render: {render_path}")
            
            print("="*60)
            return 0
        
        else:
            logger.error("Conversion failed!")
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        try:
            if 'ai_agent' in locals():
                ai_agent.cleanup()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


if __name__ == '__main__':
    sys.exit(main())