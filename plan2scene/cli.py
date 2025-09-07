from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .config import DEFAULTS
from .io import load_image, save_json
from .preprocessing import preprocess_pil_image
from .vectorize import vectorize_walls, to_structured_output
from .reconstruct3d import build_walls_mesh, export_mesh


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a 2D floor plan image into a simple 3D wall mesh",
    )
    parser.add_argument("input_image", help="Path to input floor plan image (png/jpg)")
    parser.add_argument("output_model", help="Path to output 3D model (.obj/.glb/.gltf/.stl)")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save structured wall data (JSON)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULTS.image_dpi,
        help=f"Assumed DPI for scale conversion (default: {DEFAULTS.image_dpi})",
    )
    parser.add_argument(
        "--wall-height",
        type=float,
        default=DEFAULTS.wall_height_meters,
        help=f"Wall height in meters (default: {DEFAULTS.wall_height_meters})",
    )
    parser.add_argument(
        "--wall-thickness",
        type=float,
        default=DEFAULTS.wall_thickness_meters,
        help=f"Wall thickness in meters (default: {DEFAULTS.wall_thickness_meters})",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"plan2scene {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_image_path = Path(args.input_image)
    output_model_path = Path(args.output_model)
    output_json_path = Path(args.output_json) if args.output_json else None

    pil_image = load_image(str(input_image_path))
    prep = preprocess_pil_image(pil_image)

    walls = vectorize_walls(prep.edges_image, dpi=int(args.dpi))
    structured = to_structured_output(
        walls,
        wall_thickness_m=float(args.wall_thickness),
        wall_height_m=float(args.wall_height),
    )

    result = build_walls_mesh(
        walls,
        wall_thickness_m=float(args.wall_thickness),
        wall_height_m=float(args.wall_height),
    )

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    export_mesh(result.wall_mesh, str(output_model_path))

    if output_json_path is not None:
        save_json(structured, str(output_json_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
