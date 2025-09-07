from __future__ import annotations

from typing import TextIO

from .types import Mesh


def mesh_to_obj_text(mesh: Mesh) -> str:
    lines = []
    for vx, vy, vz in mesh.vertices:
        lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}")
    for a, b, c in mesh.faces:
        lines.append(f"f {a} {b} {c}")
    return "\n".join(lines) + "\n"


def write_obj(path: str, mesh: Mesh) -> None:
    text = mesh_to_obj_text(mesh)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

