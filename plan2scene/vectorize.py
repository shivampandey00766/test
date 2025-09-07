from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from shapely.geometry import LineString

from .config import DEFAULTS


@dataclass
class WallSegment:
    start_xy_m: Tuple[float, float]
    end_xy_m: Tuple[float, float]


def meters_per_pixel(dpi: int) -> float:
    return 0.0254 / float(dpi)


def detect_line_segments(edges_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    lines = cv2.HoughLinesP(
        edges_image,
        rho=1,
        theta=np.pi / 180,
        threshold=DEFAULTS.hough_threshold,
        minLineLength=DEFAULTS.min_line_length_pixels,
        maxLineGap=DEFAULTS.max_line_gap_pixels,
    )
    if lines is None:
        return []
    return [tuple(line[0]) for line in lines]


def simplify_colinear_segments(segments_px: List[Tuple[int, int, int, int]], tolerance_px: float = 2.0) -> List[Tuple[int, int, int, int]]:
    if not segments_px:
        return []
    shapely_lines = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in segments_px]
    merged: List[LineString] = []
    for line in shapely_lines:
        merged.append(line)
    simplified = [ln.simplify(tolerance_px, preserve_topology=False) for ln in merged]
    simplified_segments: List[Tuple[int, int, int, int]] = []
    for ln in simplified:
        x1, y1 = ln.coords[0]
        x2, y2 = ln.coords[-1]
        simplified_segments.append((int(x1), int(y1), int(x2), int(y2)))
    return simplified_segments


def convert_segments_to_meters(segments_px: List[Tuple[int, int, int, int]], dpi: int) -> List[WallSegment]:
    scale = meters_per_pixel(dpi)
    walls_m: List[WallSegment] = []
    for x1, y1, x2, y2 in segments_px:
        walls_m.append(
            WallSegment(
                start_xy_m=(x1 * scale, y1 * scale),
                end_xy_m=(x2 * scale, y2 * scale),
            )
        )
    return walls_m


def vectorize_walls(edges_image: np.ndarray, dpi: int = DEFAULTS.image_dpi) -> List[WallSegment]:
    segments_px = detect_line_segments(edges_image)
    simplified_px = simplify_colinear_segments(segments_px)
    walls_m = convert_segments_to_meters(simplified_px, dpi)
    return walls_m


def to_structured_output(walls: List[WallSegment], wall_thickness_m: float, wall_height_m: float) -> dict:
    return {
        "units": "meters",
        "wall_thickness": wall_thickness_m,
        "wall_height": wall_height_m,
        "walls": [
            {
                "start": [ws.start_xy_m[0], ws.start_xy_m[1]],
                "end": [ws.end_xy_m[0], ws.end_xy_m[1]],
            }
            for ws in walls
        ],
    }
