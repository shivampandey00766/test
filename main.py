#!/usr/bin/env python3
"""
Main entry point for the AI Agent for 2D to 3D Architectural Conversion.

This script provides a command-line interface for processing floor plans
and generating 3D architectural models.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.agent import ArchitecturalAgent


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='AI Agent for 2D to 3D Architectural Conversion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single floor plan
  python main.py --input floor_plan.jpg --output model.obj

  # Process with custom configuration
  python main.py --input floor_plan.jpg --output model.obj --config config.json

  # Process with specific room types
  python main.py --input floor_plan.jpg --output model.obj --room-types living_room bedroom kitchen

  # Process multiple formats
  python main.py --input floor_plan.jpg --output model --formats obj gltf svg

  # Start web server
  python main.py --web --port 5000
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', type=str, help='Input floor plan image file')
    parser.add_argument('--output', '-o', type=str, help='Output file or directory')
    parser.add_argument('--formats', '-f', nargs='+', default=['obj'], 
                       choices=['obj', 'gltf', 'svg', 'blend', 'ply', 'stl'],
                       help='Output formats')
    
    # Configuration arguments
    parser.add_argument('--config', '-c', type=str, help='Configuration file (JSON)')
    parser.add_argument('--room-types', nargs='+', help='Room types for classification')
    parser.add_argument('--wall-height', type=float, default=2.4, help='Wall height in meters')
    parser.add_argument('--ceiling-height', type=float, default=2.7, help='Ceiling height in meters')
    
    # Processing arguments
    parser.add_argument('--reconstruction-method', choices=['blender', 'open3d', 'mesh'], 
                       default='blender', help='3D reconstruction method')
    parser.add_argument('--use-cnn', action='store_true', default=True, 
                       help='Use CNN for segmentation')
    parser.add_argument('--optimize-geometry', action='store_true', default=True,
                       help='Optimize vector geometry')
    
    # Web interface arguments
    parser.add_argument('--web', action='store_true', help='Start web interface')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web server host')
    
    # Other arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--version', action='version', version='AI Agent 1.0.0')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Start web interface
    if args.web:
        logger.info("Starting web interface...")
        from app import app
        app.run(debug=args.verbose, host=args.host, port=args.port)
        return
    
    # Validate arguments for file processing
    if not args.input:
        parser.error("--input is required for file processing")
    
    if not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input}")
    
    if not args.output:
        parser.error("--output is required for file processing")
    
    try:
        # Load configuration
        config = load_config(args)
        
        # Create output directory if needed
        output_path = Path(args.output)
        if len(args.formats) > 1 or output_path.suffix == '':
            output_path.mkdir(parents=True, exist_ok=True)
            output_dir = str(output_path)
        else:
            output_dir = str(output_path.parent)
        
        # Initialize agent
        logger.info("Initializing AI Agent...")
        agent = ArchitecturalAgent(config=config, output_dir=output_dir)
        
        # Start processing session
        session_id = agent.start_session()
        logger.info(f"Started session: {session_id}")
        
        # Process floor plan
        logger.info(f"Processing floor plan: {args.input}")
        result = agent.process_floor_plan(
            image_path=args.input,
            output_formats=args.formats,
            room_types=args.room_types,
            custom_config=config
        )
        
        # Handle results
        if result['status'] == 'success':
            logger.info("Processing completed successfully!")
            logger.info(f"Session ID: {result['session_id']}")
            logger.info(f"Processing time: {result['processing_time']}")
            logger.info("Output files:")
            for file_type, file_path in result['output_files'].items():
                logger.info(f"  {file_type}: {file_path}")
            
            # Copy files to specified output location if single format
            if len(args.formats) == 1 and output_path.suffix != '':
                import shutil
                source_file = result['output_files'].get(f'model_{args.formats[0]}')
                if source_file and os.path.exists(source_file):
                    shutil.copy2(source_file, args.output)
                    logger.info(f"Copied output to: {args.output}")
        
        else:
            logger.error(f"Processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def load_config(args) -> dict:
    """Load configuration from file and command line arguments."""
    config = {}
    
    # Load from file if specified
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        logging.getLogger(__name__).info(f"Loaded configuration from: {args.config}")
    
    # Override with command line arguments
    if args.wall_height:
        config.setdefault('reconstruction', {})['wall_height'] = args.wall_height
    
    if args.ceiling_height:
        config.setdefault('reconstruction', {})['ceiling_height'] = args.ceiling_height
    
    if args.reconstruction_method:
        config.setdefault('reconstruction', {})['reconstruction_method'] = args.reconstruction_method
    
    if args.use_cnn is not None:
        config.setdefault('segmentation', {})['use_cnn'] = args.use_cnn
    
    if args.optimize_geometry is not None:
        config.setdefault('vectorization', {})['optimize_geometry'] = args.optimize_geometry
    
    return config


if __name__ == '__main__':
    main()