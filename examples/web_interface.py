#!/usr/bin/env python3
"""
Example script for starting the web interface.

This script demonstrates how to start the web interface
for the AI Agent for 2D to 3D Architectural Conversion.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from app import app


def main():
    """Start the web interface."""
    print("AI Agent for 2D to 3D Architectural Conversion")
    print("Starting Web Interface...")
    print("=" * 50)
    
    # Configuration
    host = "0.0.0.0"
    port = 5000
    debug = True
    
    print(f"Web interface will be available at:")
    print(f"  http://localhost:{port}")
    print(f"  http://{host}:{port}")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start the Flask app
        app.run(host=host, port=port, debug=debug)
    
    except KeyboardInterrupt:
        print("\n\nWeb interface stopped by user")
        return 0
    
    except Exception as e:
        print(f"\nError starting web interface: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())