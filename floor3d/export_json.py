from __future__ import annotations

import json
from typing import Any, Dict, List

from .types import Wall, to_serializable_dict


def write_floorplan_json(path: str, walls: List[Wall], meta: Dict[str, Any] | None = None) -> None:
    data: Dict[str, Any] = {
        "walls": [to_serializable_dict(w) for w in walls],
    }
    if meta:
        data["meta"] = to_serializable_dict(meta)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

