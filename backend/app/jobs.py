"""Batch jobs for a folder or a ZIP: store the images, score them on one worker thread, report.

Images are written under <work_dir>/<job>/ with generated names (never the uploaded paths), and
job folders older than job_ttl_hours are deleted when a new job starts.
"""
from __future__ import annotations

import csv
import hashlib
import io
import logging
import queue
import shutil
import threading
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np

from backend.app import metadata, visuals
from backend.app.analysis import LENSES
from backend.app.engine import Engine, ImageError, confidence_band, decode, verdict
from backend.app.evidence import binary_metrics

log = logging.getLogger("pixel_artifacts.jobs")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
REAL_NAMES = {"real", "0_real", "reals", "authentic", "genuine", "original", "camera"}
AI_NAMES = {"fake", "1_fake", "fakes", "ai", "generated", "synthetic", "ai_generated", "ai-generated"}
BATCH = 4                      # images per progress update (each image is a full scan)


class JobError(ValueError):
    """Bad batch upload; the message is safe to show to the user."""


def ground_truth(path: str) -> int | None:
    label = None
    for part in PurePosixPath(path.replace("\\", "/")).parts[:-1]:
        p = part.lower()
        if p in REAL_NAMES:
            label = 0
        elif p in AI_NAMES:
            label = 1
    return label


def is_image_name(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return p.suffix.lower() in IMAGE_EXTS and not any(s.startswith(".") or s == "__MACOSX" for s in p.parts)


class Jobs:
    def __init__(self, engine: Engine, settings):
        self.engine, self.s = engine, settings
        self.root = settings.work_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict] = {}
        self.q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._worker, daemon=True, name="batch-worker").start()

    # ------------------------------------------------------------------ intake
    def _new(self, source: str, name: str) -> tuple[str, Path]:
        self._cleanup()
        jid = uuid.uuid4().hex[:12]
        d = self.root / jid
        (d / "files").mkdir(parents=True)
        (d / "thumbs").mkdir()
        self.jobs[jid] = {"id": jid, "source": source, "name": name, "status": "queued", "total": 0,
                          "done": 0, "created": datetime.now().isoformat(timespec="seconds"),
                          "items": [], "skipped": [], "dir": d}
        return jid, d

    def _add(self, jid: str, rel: str, data: bytes) -> None:
        job = self.jobs[jid]
        i = len(job["items"])
        ext = PurePosixPath(rel).suffix.lower()
        stored = job["dir"] / "files" / f"{i:05d}{ext}"
        stored.write_bytes(data)
        job["items"].append({"index": i, "path": rel, "filename": PurePosixPath(rel).name,
                             "bytes": len(data), "ground_truth": ground_truth(rel), "file": stored.name})

    def from_files(self, files: list[tuple[str, bytes]], name: str) -> str:
        imgs = [(r, b) for r, b in files if is_image_name(r)]
        skipped = [r for r, _ in files if not is_image_name(r)]
        self._check(len(imgs), sum(len(b) for _, b in imgs))
        jid, _ = self._new("folder", name)
        for rel, data in imgs:
            self._add(jid, rel, data)
        return self._submit(jid, skipped)

    def from_zip(self, path: Path, name: str) -> str:
        try:
            zf = zipfile.ZipFile(path)
        except zipfile.BadZipFile:
            raise JobError("The file is not a valid ZIP archive.")
        with zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            imgs = [m for m in members if is_image_name(m.filename)]
            skipped = [m.filename for m in members if not is_image_name(m.filename)]
            self._check(len(imgs), sum(m.file_size for m in imgs))
            jid, _ = self._new("zip", name)
            limit = int(self.s.max_image_mb * 1e6)
            for m in imgs:
                if m.file_size > limit:
                    skipped.append(m.filename)
                    continue
                with zf.open(m) as f:
                    data = f.read(limit + 1)           # declared sizes can lie; never read past the limit
                if len(data) > limit:
                    skipped.append(m.filename)
                    continue
                self._add(jid, m.filename, data)
        if not self.jobs[jid]["items"]:
            raise JobError("The ZIP contains no supported images.")
        return self._submit(jid, skipped)

    def _check(self, n: int, total_bytes: int) -> None:
        if n == 0:
            raise JobError("No supported images found (JPG, PNG, WEBP, BMP or TIFF).")
        if n > self.s.max_batch_images:
            raise JobError(f"Too many images ({n}). The limit is {self.s.max_batch_images} per batch.")
        if total_bytes > self.s.max_batch_mb * 1e6:
            raise JobError(f"The images add up to more than {self.s.max_batch_mb:.0f} MB.")

    def _submit(self, jid: str, skipped: list[str]) -> str:
        job = self.jobs[jid]
        job["skipped"] = [{"path": s, "reason": "not a supported image"} for s in skipped[:500]]
        job["total"] = len(job["items"])
        self.q.put(jid)
        return jid

    def _cleanup(self) -> None:
        cutoff = time.time() - self.s.job_ttl_hours * 3600
        for d in self.root.iterdir():
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                self.jobs.pop(d.name, None)

    # ------------------------------------------------------------------ processing
    def _worker(self) -> None:
        while True:
            jid = self.q.get()
            job = self.jobs.get(jid)
            if job is None:
                continue
            job["status"], job["started"] = "running", time.time()
            try:
                for a in range(0, len(job["items"]), BATCH):
                    self._score(job, job["items"][a:a + BATCH])
                    job["done"] = min(a + BATCH, job["total"])
                job["summary"] = self.summary(job)
                job["status"] = "done"
            except Exception:
                log.exception("job %s failed", jid)
                job["status"] = "failed"
            job["seconds"] = time.time() - job["started"]

    def _score(self, job: dict, items: list[dict]) -> None:
        """Same whole-image verdict as a single check (Engine.scan, DECISION_LOG D-21)."""
        eng = self.engine
        for it in items:
            try:
                data = (job["dir"] / "files" / it["file"]).read_bytes()
                img = decode(data, self.s.max_image_pixels)
                meta = metadata.inspect(img, img.format)
                full, _, info = eng.prepare(img)
                visuals.thumbnail(full, 200).save(job["dir"] / "thumbs" / f"{it['index']:05d}.jpg", quality=82)
                sc = eng.scan(full)
            except ImageError as e:
                it["error"] = str(e)
                continue
            pi = sc["p_ai"]
            it.update({"sha256": hashlib.sha256(data).hexdigest(), "width": info["width"], "height": info["height"],
                       "format": img.format, "ai_markers": meta["ai_markers"], "camera_info": meta["camera_info"],
                       "verdict": verdict(pi, eng.threshold), "p_ai": pi, "band": confidence_band(pi),
                       "windows": len(sc["windows"]) or 1, "share_ai_windows": sc["share_ai"], "gates": sc["gates"]})
            if it["ground_truth"] is not None:
                it["correct"] = (it["verdict"] == "ai") == (it["ground_truth"] == 1)

    # ------------------------------------------------------------------ reporting
    def summary(self, job: dict) -> dict:
        items = job["items"]
        scored = [it for it in items if "p_ai" in it]
        p = np.array([it["p_ai"] for it in scored]) if scored else np.zeros(0)
        n_ai = int((p >= self.engine.threshold).sum())
        hist, _ = np.histogram(p, bins=10, range=(0, 1))
        groups: dict[str, list] = {}
        for it in scored:
            parts = PurePosixPath(it["path"].replace("\\", "/")).parts
            key = "/".join(parts[:-1][-2:]) or "(top level)"
            groups.setdefault(key, []).append(it["p_ai"])
        out = {
            "images": len(items), "scored": len(scored), "errors": len(items) - len(scored),
            "skipped": len(job["skipped"]), "ai": n_ai, "real": len(scored) - n_ai,
            "ai_share": n_ai / len(scored) if scored else 0.0,
            "mean_p_ai": float(p.mean()) if scored else None,
            "bands": {b: sum(1 for it in scored if it["band"] == b) for b in ("high", "moderate", "low")},
            "histogram": hist.tolist(),
            "with_ai_markers": sum(1 for it in scored if it.get("ai_markers")),
            "folders": sorted(({"folder": k, "images": len(v), "ai": int(sum(x >= self.engine.threshold for x in v)),
                                "mean_p_ai": float(np.mean(v))} for k, v in groups.items()),
                              key=lambda r: -r["images"])[:40],
            "mean_gates": {k: float(np.mean([it["gates"][k] for it in scored])) for k in LENSES} if scored else {},
        }
        labelled = [it for it in scored if it["ground_truth"] is not None]
        if labelled:
            y = np.array([it["ground_truth"] for it in labelled])
            out["labelled"] = {"images": len(labelled), "real": int((y == 0).sum()), "ai": int((y == 1).sum()),
                               **binary_metrics(y, np.array([it["p_ai"] for it in labelled]), self.engine.threshold)}
        return out

    def public(self, jid: str) -> dict:
        job = self.jobs.get(jid)
        if job is None:
            raise KeyError(jid)
        keys = ("index", "path", "filename", "bytes", "verdict", "p_ai", "band", "width", "height", "format",
                "ground_truth", "correct", "error", "ai_markers", "gates")
        out = {k: v for k, v in job.items() if k not in ("items", "dir")}
        out["items"] = [{k: it.get(k) for k in keys} for it in job["items"]] if job["status"] == "done" else []
        out["eta_seconds"] = None
        if job["status"] == "running" and job["done"]:
            rate = job["done"] / max(time.time() - job["started"], 1e-6)
            out["eta_seconds"] = (job["total"] - job["done"]) / rate
        return out

    def file(self, jid: str, index: int) -> tuple[bytes, str]:
        it = self.jobs[jid]["items"][index]
        return (self.jobs[jid]["dir"] / "files" / it["file"]).read_bytes(), it["path"]

    def thumb(self, jid: str, index: int) -> Path:
        return self.jobs[jid]["dir"] / "thumbs" / f"{index:05d}.jpg"

    def csv_report(self, jid: str) -> str:
        job = self.jobs[jid]
        s = job["summary"]
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Pixel-Artifacts batch report"])
        rows = [("source", job["source"]), ("name", job["name"]), ("created", job["created"]),
                ("model_version", self.engine.version), ("threshold", self.engine.threshold),
                ("protocol", "full-image scan: mean AI probability of 224 px windows covering every pixel"),
                ("images_found", s["images"]), ("images_scored", s["scored"]), ("unreadable", s["errors"]),
                ("non_image_files_skipped", s["skipped"]), ("verdict_ai", s["ai"]), ("verdict_real", s["real"]),
                ("ai_share", f"{s['ai_share']:.4f}"),
                ("mean_ai_probability", "" if s["mean_p_ai"] is None else f"{s['mean_p_ai']:.4f}"),
                ("confidence_high", s["bands"]["high"]), ("confidence_moderate", s["bands"]["moderate"]),
                ("confidence_low", s["bands"]["low"]), ("files_with_ai_tool_metadata", s["with_ai_markers"])]
        if "labelled" in s:
            L = s["labelled"]
            rows += [("labelled_images", L["images"]), ("labelled_accuracy", f"{L['accuracy']:.4f}"),
                     ("labelled_precision", f"{L['precision']:.4f}"), ("labelled_recall", f"{L['recall']:.4f}"),
                     ("labelled_f1", f"{L['f1']:.4f}"), ("labelled_auc", f"{L.get('auc', float('nan')):.4f}")]
        for r in rows:
            w.writerow(r)
        w.writerow([])
        cols = ["index", "path", "filename", "verdict", "ai_probability", "real_probability", "confidence",
                "confidence_band", "windows_scanned", "share_windows_ai", "width", "height", "format", "bytes", "sha256", "ai_tool_metadata",
                "camera_metadata", *(f"weight_{k}" for k in LENSES), "ground_truth", "correct", "error"]
        w.writerow(cols)
        for it in job["items"]:
            p = it.get("p_ai")
            gt = it.get("ground_truth")
            w.writerow([it["index"], it["path"], it["filename"], it.get("verdict", ""),
                        "" if p is None else f"{p:.6f}", "" if p is None else f"{1 - p:.6f}",
                        "" if p is None else f"{max(p, 1 - p):.6f}", it.get("band", ""), it.get("windows", ""),
                        "" if "share_ai_windows" not in it else f"{it['share_ai_windows']:.4f}",
                        it.get("width", ""), it.get("height", ""), it.get("format", ""), it["bytes"],
                        it.get("sha256", ""), "; ".join(it.get("ai_markers") or []),
                        "" if "camera_info" not in it else ("yes" if it["camera_info"] else "no"),
                        *("" if "gates" not in it else f"{it['gates'][k]:.4f}" for k in LENSES),
                        "" if gt is None else ("ai" if gt else "real"),
                        "" if "correct" not in it else ("yes" if it["correct"] else "no"), it.get("error", "")])
        return buf.getvalue()
