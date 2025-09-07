from __future__ import annotations

import math
from typing import List, Tuple

from .types import Point2D, Opening, Wall, Mesh


def _normalize(vec_x: float, vec_y: float) -> Tuple[float, float]:
    length = math.hypot(vec_x, vec_y)
    if length == 0:
        return 0.0, 0.0
    return vec_x / length, vec_y / length


def _compute_segments_with_gaps(start: Point2D, end: Point2D, openings: List[Opening]) -> List[Tuple[Point2D, Point2D]]:
    """
    Split the wall into sub-segments by carving out horizontal gaps for openings.
    Vertical extents are not modeled in this scaffold.
    """
    dx = end.x - start.x
    dy = end.y - start.y
    wall_length = math.hypot(dx, dy)
    if wall_length == 0:
        return []

    # Represent gaps as [t0, t1] along the wall parametric axis.
    gaps: List[Tuple[float, float]] = []
    for opening in openings:
        half = 0.5 * max(0.0, opening.width)
        center = max(0.0, min(1.0, opening.position_t)) * wall_length
        t0 = max(0.0, (center - half) / wall_length)
        t1 = min(1.0, (center + half) / wall_length)
        if t1 > t0:
            gaps.append((t0, t1))

    # Merge overlapping gaps
    gaps.sort(key=lambda g: g[0])
    merged: List[Tuple[float, float]] = []
    for gap in gaps:
        if not merged or gap[0] > merged[-1][1]:
            merged.append(list(gap))  # type: ignore[arg-type]
        else:
            merged[-1][1] = max(merged[-1][1], gap[1])  # type: ignore[index]

    # Complement intervals -> solid segments
    segments: List[Tuple[float, float]] = []
    last = 0.0
    for t0, t1 in merged:
        if t0 > last:
            segments.append((last, t0))
        last = max(last, t1)
    if last < 1.0:
        segments.append((last, 1.0))

    # Map param segments back to points
    points: List[Tuple[Point2D, Point2D]] = []
    for t0, t1 in segments:
        p0 = Point2D(start.x + dx * t0, start.y + dy * t0)
        p1 = Point2D(start.x + dx * t1, start.y + dy * t1)
        if p0.x != p1.x or p0.y != p1.y:
            points.append((p0, p1))
    return points


def _extrude_wall_segment(p0: Point2D, p1: Point2D, thickness: float, height: float, base_z: float = 0.0) -> Mesh:
    """
    Extrude a single wall segment [p0, p1] into a rectangular prism with given thickness and height.
    Returns a Mesh with triangles and 1-based indices.
    """
    dx, dy = p1.x - p0.x, p1.y - p0.y
    ux, uy = _normalize(dx, dy)
    # Outward normal in the XY plane (rotate by +90 deg)
    nx, ny = -uy, ux
    half_t = 0.5 * thickness

    # Four base corners (clockwise)
    a = (p0.x + nx * half_t, p0.y + ny * half_t, base_z)
    b = (p1.x + nx * half_t, p1.y + ny * half_t, base_z)
    c = (p1.x - nx * half_t, p1.y - ny * half_t, base_z)
    d = (p0.x - nx * half_t, p0.y - ny * half_t, base_z)

    # Top corners
    az = (a[0], a[1], base_z + height)
    bz = (b[0], b[1], base_z + height)
    cz = (c[0], c[1], base_z + height)
    dz = (d[0], d[1], base_z + height)

    vertices = [a, b, c, d, az, bz, cz, dz]

    # Triangulate faces (12 triangles)
    faces = [
        # bottom (a, b, c, d)
        (1, 2, 3), (1, 3, 4),
        # top (az, bz, cz, dz)
        (5, 7, 6), (5, 8, 7),
        # sides
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 8), (3, 8, 4),
        (4, 8, 5), (4, 5, 1),
    ]

    return Mesh(vertices=vertices, faces=faces)


def combine_meshes(meshes: List[Mesh]) -> Mesh:
    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    offset = 0
    for mesh in meshes:
        vertices.extend(mesh.vertices)
        faces.extend([(a + offset, b + offset, c + offset) for (a, b, c) in mesh.faces])
        offset += len(mesh.vertices)
    return Mesh(vertices=vertices, faces=faces)


def walls_to_mesh(walls: List[Wall]) -> Mesh:
    parts: List[Mesh] = []
    for wall in walls:
        segments = _compute_segments_with_gaps(wall.start, wall.end, wall.openings)
        for p0, p1 in segments:
            if (p0.x == p1.x) and (p0.y == p1.y):
                continue
            parts.append(_extrude_wall_segment(p0, p1, wall.thickness, wall.height))
    return combine_meshes(parts)

