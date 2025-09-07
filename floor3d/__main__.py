from __future__ import annotations

import argparse
import os
from typing import List

from .types import Point2D, Wall, Opening
from .reconstruct import walls_to_mesh
from .export_obj import write_obj
from .export_json import write_floorplan_json


def build_synthetic_plan() -> List[Wall]:
    """
    Build a simple rectangular room with one interior partition and a door opening.
    Units are meters; origin at lower-left corner.
    """
    width = 6.0
    depth = 4.0
    thickness = 0.20
    height = 3.0

    # Perimeter walls (clockwise)
    w0 = Wall(Point2D(0.0, 0.0), Point2D(width, 0.0), thickness, height)
    w1 = Wall(Point2D(width, 0.0), Point2D(width, depth), thickness, height)
    w2 = Wall(Point2D(width, depth), Point2D(0.0, depth), thickness, height)
    w3 = Wall(Point2D(0.0, depth), Point2D(0.0, 0.0), thickness, height)

    # Interior partition across the middle with a door gap
    w4 = Wall(Point2D(0.0, depth * 0.5), Point2D(width, depth * 0.5), thickness, height,
              openings=[Opening(position_t=0.25, width=0.9, kind="door")])

    return [w0, w1, w2, w3, w4]


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic 2D-to-3D floor plan demo")
    parser.add_argument("--out-dir", default="out", help="Output directory for OBJ and JSON")
    parser.add_argument("--basename", default="floorplan", help="Base name for output files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    walls = build_synthetic_plan()
    mesh = walls_to_mesh(walls)

    obj_path = os.path.join(args.out_dir, f"{args.basename}.obj")
    json_path = os.path.join(args.out_dir, f"{args.basename}.json")

    write_obj(obj_path, mesh)
    write_floorplan_json(json_path, walls, meta={"units": "meters", "note": "synthetic demo"})

    print(f"Wrote OBJ: {obj_path}")
    print(f"Wrote JSON: {json_path}")


if __name__ == "__main__":
    main()

