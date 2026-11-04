"""Pixel-Artifacts API and web server.

    .venv\\Scripts\\python -m uvicorn backend.app.main:app --port 8000      (from the repository root)

Serves the built frontend (frontend/dist) at / and the API under /api.
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.app import research
from backend.app.analysis import analyze
from backend.app.config import Settings
from backend.app.engine import Engine, ImageError, decode
from backend.app.evidence import Evidence
from backend.app.jobs import JobError, Jobs, is_image_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = await run_in_threadpool(Engine(settings).load)
    app.state.engine = engine
    app.state.evidence = Evidence(settings.model_dir, engine.threshold)
    app.state.jobs = Jobs(engine, settings)
    app.state.research = research.build()
    yield


app = FastAPI(title="Pixel-Artifacts", lifespan=lifespan)


@app.exception_handler(ImageError)
@app.exception_handler(JobError)
async def user_error(_: Request, exc: Exception):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


async def read_limited(f: UploadFile, limit: int) -> bytes:
    data = await f.read(limit + 1)
    if len(data) > limit:
        raise ImageError(f"{f.filename} is larger than {limit / 1e6:.0f} MB.")
    return data


@app.get("/api/health")
async def health(request: Request):
    return {"status": "ok", "model": request.app.state.engine.version}


@app.get("/api/model")
async def model(request: Request):
    return {**request.app.state.engine.describe(), "benchmark": request.app.state.evidence.benchmark}


@app.get("/api/research")
async def research_data(request: Request):
    return request.app.state.research


@app.post("/api/analyze")
async def analyze_images(request: Request, files: list[UploadFile] = File(...)):
    """1 to max_direct_images images, each with the full analysis and forensic views."""
    if len(files) > settings.max_direct_images:
        raise ImageError(f"Up to {settings.max_direct_images} images at once; use a folder or ZIP for more.")
    eng, ev = request.app.state.engine, request.app.state.evidence
    results = []
    for f in files:
        name = Path(f.filename or "image").name
        try:
            data = await read_limited(f, int(settings.max_image_mb * 1e6))
            img = decode(data, settings.max_image_pixels)
            results.append(await run_in_threadpool(analyze, eng, ev, data, name, img.format, img))
        except ImageError as e:
            results.append({"filename": name, "error": str(e)})
    return {"results": results}


@app.post("/api/jobs")
async def create_job(request: Request, files: list[UploadFile] = File(...), paths: str = Form("[]"),
                     name: str = Form("Batch")):
    """A folder (many files plus their relative paths) or one ZIP archive."""
    jobs: Jobs = request.app.state.jobs
    limit = int(settings.max_image_mb * 1e6)
    if len(files) == 1 and (files[0].filename or "").lower().endswith(".zip"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip", dir=settings.work_dir) as tmp:
            await run_in_threadpool(shutil.copyfileobj, files[0].file, tmp)
            tmp_path = Path(tmp.name)
        try:
            if tmp_path.stat().st_size > settings.max_batch_mb * 1e6:
                raise JobError(f"The ZIP is larger than {settings.max_batch_mb:.0f} MB.")
            jid = await run_in_threadpool(jobs.from_zip, tmp_path, name or files[0].filename)
        finally:
            tmp_path.unlink(missing_ok=True)
        return {"job_id": jid}
    rels = json.loads(paths) if paths else []
    entries = []
    for i, f in enumerate(files):
        rel = rels[i] if i < len(rels) and rels[i] else (f.filename or f"file{i}")
        if not is_image_name(rel):
            entries.append((rel, b""))
            continue
        entries.append((rel, await read_limited(f, limit)))
    return {"job_id": await run_in_threadpool(jobs.from_files, entries, name)}


def _job(request: Request, jid: str) -> Jobs:
    jobs: Jobs = request.app.state.jobs
    if jid not in jobs.jobs:
        raise HTTPException(404, "This batch was not found. Batches are kept for a few hours.")
    return jobs


@app.get("/api/jobs/{jid}")
async def job_status(request: Request, jid: str):
    return _job(request, jid).public(jid)


@app.get("/api/jobs/{jid}/report.csv")
async def job_csv(request: Request, jid: str):
    jobs = _job(request, jid)
    if jobs.jobs[jid]["status"] != "done":
        raise HTTPException(409, "The batch is still running.")
    return Response(jobs.csv_report(jid), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="pixel-artifacts-{jid}.csv"'})


@app.get("/api/jobs/{jid}/items/{index}/thumb")
async def job_thumb(request: Request, jid: str, index: int):
    path = _job(request, jid).thumb(jid, index)
    if not path.exists():
        raise HTTPException(404, "No preview")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/jobs/{jid}/items/{index}")
async def job_item(request: Request, jid: str, index: int):
    jobs = _job(request, jid)
    if not 0 <= index < len(jobs.jobs[jid]["items"]):
        raise HTTPException(404, "No such image in this batch")
    data, rel = jobs.file(jid, index)
    img = decode(data, settings.max_image_pixels)
    eng, ev = request.app.state.engine, request.app.state.evidence
    return await run_in_threadpool(analyze, eng, ev, data, Path(rel).name, img.format, img)


# ---------------------------------------------------------------- frontend (built SPA)
dist = settings.frontend_dist
if (dist / "assets").exists():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
async def spa(path: str):
    if path.startswith("api/"):
        raise HTTPException(404)
    f = (dist / path).resolve()
    if path and f.is_file() and dist.resolve() in f.parents:
        return FileResponse(f)
    index = dist / "index.html"
    if not index.exists():
        return JSONResponse({"detail": "Frontend not built. Run `npm run build` in frontend/."}, status_code=404)
    # index.html names the hashed bundles, so browsers must re-check it after every build
    return FileResponse(index, headers={"Cache-Control": "no-cache"})
