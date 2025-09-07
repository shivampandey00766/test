from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image

from .config import DEFAULTS
from .preprocessing import preprocess_pil_image
from .vectorize import vectorize_walls, to_structured_output
from .reconstruct3d import build_walls_mesh, export_mesh
from .io import save_json


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / ".." / "out"

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(['html', 'xml'])
)

app = FastAPI(title="Plan2Scene Preview")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def render_template(name: str, **context) -> HTMLResponse:
    template = env.get_template(name)
    html = template.render(**context)
    return HTMLResponse(content=html)


@app.get("/", response_class=HTMLResponse)
def index():
    return render_template("index.html")


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    dpi: Optional[int] = Form(None),
    wall_height: Optional[float] = Form(None),
    wall_thickness: Optional[float] = Form(None),
):
    dpi_val = int(dpi) if dpi is not None else DEFAULTS.image_dpi
    wall_height_val = float(wall_height) if wall_height is not None else DEFAULTS.wall_height_meters
    wall_thickness_val = float(wall_thickness) if wall_thickness is not None else DEFAULTS.wall_thickness_meters

    contents = await file.read()
    pil = Image.open(io.BytesIO(contents)).convert("RGB")

    prep = preprocess_pil_image(pil)
    walls = vectorize_walls(prep.edges_image, dpi=dpi_val)
    structured = to_structured_output(walls, wall_thickness_val, wall_height_val)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "preview.glb"
    json_path = OUTPUT_DIR / "walls.json"

    mesh_result = build_walls_mesh(walls, wall_thickness_val, wall_height_val)
    export_mesh(mesh_result.wall_mesh, str(model_path))
    save_json(structured, str(json_path))

    return RedirectResponse(url=f"/result?model=preview.glb&json=walls.json", status_code=302)


@app.get("/result", response_class=HTMLResponse)
def result(model: str, json: str):
    return render_template("result.html", model=model, json=json)


@app.get("/download/{path:path}")
def download(path: str):
    safe_path = (OUTPUT_DIR / path).resolve()
    if not str(safe_path).startswith(str(OUTPUT_DIR.resolve())):
        return HTMLResponse(status_code=403, content="Forbidden")
    if not safe_path.exists():
        return HTMLResponse(status_code=404, content="Not Found")
    return FileResponse(str(safe_path))
