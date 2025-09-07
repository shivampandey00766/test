# 2D-to-3D Floor Plan AI Agent (Scaffold)

This repository provides a minimal, runnable scaffold for an AI agent pipeline that converts a 2D floor plan into a 3D model. It focuses on a synthetic demo that:

- Builds a simple 2D plan (rectangular room + partition with a door opening)
- Extrudes walls into 3D geometry
- Exports both OBJ (mesh) and JSON (structured plan)

It is intentionally dependency-light so you can run it anywhere with Python 3.

## Quick Start

Run the demo to generate outputs:

```bash
python3 -m floor3d --out-dir out --basename demo
```

Outputs:

- `out/demo.obj`: Triangular mesh of the extruded walls
- `out/demo.json`: Structured data of walls and openings

## Project Structure

```text
floor3d/
  __init__.py         # package metadata
  __main__.py         # CLI entry: python -m floor3d
  types.py            # dataclasses: Point2D, Opening, Wall, Mesh
  reconstruct.py      # wall segmentation by openings + extrusion to 3D mesh
  export_obj.py       # OBJ writer
  export_json.py      # JSON writer
requirements.txt      # empty for now (stdlib only)
```

## How It Works (High Level)

1. Synthetic plan construction: a small room perimeter plus an interior partition wall with a door-sized opening.
2. Openings create gaps along wall segments; remaining segments are extruded into rectangular prisms.
3. Mesh pieces are merged and written to OBJ; original plan is serialized to JSON.

This mirrors the final stages of a full AI pipeline (geometry reconstruction and export), while stubbing earlier AI stages (CV + DL segmentation).

## Next Steps (AI Pipeline Roadmap)

- Computer Vision Preprocessing: OpenCV for denoising, dewarping, vectorization
- Semantic Segmentation: Train CNN/Transformer (PyTorch/TensorFlow) for rooms/doors/windows
- Parametric Reconstruction: Convert semantic map + vectors into parametric `Wall` and `Opening` sets
- Rich 3D: Add window heights/sills, doors, staircases; furnish via asset library
- Export: glTF with materials; USD for DCC workflows; metadata-rich JSON/IFC mappings

## License

MIT
