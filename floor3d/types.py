from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any


@dataclass
class Point2D:
    x: float
    y: float


@dataclass
class Opening:
    """
    An opening placed along a wall segment, parameterized by t in [0, 1].
    Only the horizontal span is modeled in this scaffold; vertical cutouts are not.
    """
    position_t: float  # 0..1 along the wall from start -> end
    width: float  # meters along wall direction on plan
    kind: str = "door"  # e.g., "door" or "window"


@dataclass
class Wall:
    start: Point2D
    end: Point2D
    thickness: float = 0.20  # meters
    height: float = 3.00  # meters
    openings: List[Opening] = field(default_factory=list)


@dataclass
class Mesh:
    """Simple triangular mesh container."""
    vertices: List[Tuple[float, float, float]]
    faces: List[Tuple[int, int, int]]  # 1-based indices for OBJ compatibility


def to_serializable_dict(obj: Any) -> Any:
    if isinstance(obj, (Point2D, Opening, Wall)):
        return asdict(obj)
    if isinstance(obj, list):
        return [to_serializable_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: to_serializable_dict(v) for k, v in obj.items()}
    return obj

