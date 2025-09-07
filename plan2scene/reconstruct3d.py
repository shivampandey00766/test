from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import shapely.ops as ops
from shapely.geometry import LineString, Polygon
import trimesh

from .vectorize import WallSegment


@dataclass
class ReconstructionResult:
    wall_mesh: trimesh.Trimesh


def wall_segments_to_polygons(
    wall_segments: Iterable[WallSegment],
    wall_thickness_m: float,
) -> List[Polygon]:
    half_thickness = wall_thickness_m / 2.0
    lines = [
        LineString([ws.start_xy_m, ws.end_xy_m]).buffer(half_thickness, cap_style=2, join_style=2)
        for ws in wall_segments
    ]
    if not lines:
        return []
    merged = ops.unary_union(lines)
    if isinstance(merged, Polygon):
        return [merged]
    return list(merged.geoms)


def extrude_walls_to_mesh(polygons: List[Polygon], wall_height_m: float) -> trimesh.Trimesh:
    meshes: List[trimesh.Trimesh] = []
    for poly in polygons:
        mesh = trimesh.creation.extrude_polygon(poly, height=wall_height_m)
        meshes.append(mesh)
    if not meshes:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(meshes)


def build_walls_mesh(wall_segments: Iterable[WallSegment], wall_thickness_m: float, wall_height_m: float) -> ReconstructionResult:
    polygons = wall_segments_to_polygons(wall_segments, wall_thickness_m)
    mesh = extrude_walls_to_mesh(polygons, wall_height_m)
    return ReconstructionResult(wall_mesh=mesh)


def export_mesh(mesh: trimesh.Trimesh, output_path: str) -> None:
    mesh.export(output_path)
