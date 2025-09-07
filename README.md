Plan2Scene: 2D Floor Plan to 3D Model (MVP)

Overview
This repository contains an MVP scaffold for an AI-assisted pipeline that converts a 2D floor plan image into a simple 3D model. It lays out a modular structure for preprocessing, vectorization, and 3D reconstruction, with room to integrate deep learning-based semantic segmentation.

Features (initial scaffold)
- Preprocessing module stub for image cleaning
- Vectorization module stub for detecting wall segments
- 3D reconstruction module stub for extruding walls into simple meshes
- IO helpers and configuration defaults

Setup
1) Create and activate a virtual environment
   - python3 -m venv .venv
   - source .venv/bin/activate
2) Install dependencies
   - pip install -r requirements.txt

Project Structure
- plan2scene/
  - __init__.py
  - config.py
  - preprocessing.py
  - vectorize.py
  - reconstruct3d.py
  - io.py
  - cli.py
  - __main__.py

Notes
- This is a starter scaffold. The semantic segmentation and robust vectorization are not yet implemented.
- A CLI entrypoint and end-to-end example will be added in subsequent iterations.

Usage
- Basic conversion (assumes DPI=300):
  - python -m plan2scene input_floorplan.png out/model.obj --output-json out/walls.json
- Options:
  - --dpi INT: pixels-per-inch to meters scale (default 300)
  - --wall-height FLOAT: wall extrusion height in meters (default 3.0)
  - --wall-thickness FLOAT: wall thickness in meters (default 0.2)

Web Preview (FastAPI)
- Start the server:
  - uvicorn plan2scene.web:app --host 0.0.0.0 --port 8000
- Open in your browser:
  - http://localhost:8000
# test