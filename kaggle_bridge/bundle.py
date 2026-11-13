"""Build the self-contained job archive that is uploaded to Kaggle.

Layout of ``job.zip``::

    manifest.json
    images/000000.jpg
    images/000001.jpg
    ...
    model/sam2.1_hiera_small.pt      (optional; SAM 2)
    model/yolov11.pt                 (YOLO predict)

Existing labels are embedded in ``manifest.json`` (small JSON) so a 500-frame
job does not create 500 extra files in the Kaggle dataset.  Only changed label
files are returned by the kernel.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .metadata import write_json_atomic
from .runner_lib import normalize_label_lines, read_label_lines
from .sam2 import SAM2_MODELS
from .spec import JobSpec

ProgressFn = Optional[Callable[[int, str], None]]
LogFn = Optional[Callable[[str], None]]


@dataclass
class BundleInfo:
    job_zip: Path
    size_bytes: int
    sha256: str
    image_count: int
    included_indices: List[int]
    model_included: bool
    manifest: Dict[str, Any]


def _call(progress: ProgressFn, percent: int, message: str) -> None:
    if progress is not None:
        try:
            progress(max(0, min(100, int(percent))), message)
        except Exception:
            pass


def _log(log: LogFn, message: str) -> None:
    if log is not None:
        try:
            log(message)
        except Exception:
            pass


def _imread(path: Path):
    import cv2
    import numpy as np

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _imwrite(path: Path, image, quality: int = 92) -> None:
    import cv2

    suffix = path.suffix.lower() or ".jpg"
    params = []
    if suffix in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buffer = cv2.imencode(suffix, image, params)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    buffer.tofile(str(path))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError:
        pass


def _checkpoint_copy(
    spec: JobSpec,
    bundle_dir: Path,
    include_checkpoint: bool,
    log: LogFn,
) -> Dict[str, Any]:
    info = SAM2_MODELS[spec.model_key]
    model_dir = bundle_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / info["file"]
    if include_checkpoint and spec.checkpoint_path:
        source = Path(spec.checkpoint_path)
        if not source.is_file():
            raise FileNotFoundError(f"SAM 2 checkpoint not found: {source}")
        size = source.stat().st_size
        if size < 100 * 1024 * 1024:
            raise RuntimeError(
                f"SAM 2 checkpoint looks incomplete ({size / 1024 / 1024:.1f} MB): {source}"
            )
        _log(log, f"Adding SAM 2 checkpoint: {source.name} ({size / 1024 / 1024:.0f} MB)")
        shutil.copyfile(source, dest)
        return {
            "kind": "sam2",
            "model_key": spec.model_key,
            "file": f"model/{info['file']}",
            "config": spec.sam2_config or info["config"],
            "checkpoint_url": info["url"],
            "checkpoint_included": True,
        }
    _log(
        log,
        "SAM 2 checkpoint will be downloaded inside the Kaggle kernel "
        f"({info['size']}).",
    )
    return {
        "kind": "sam2",
        "model_key": spec.model_key,
        "file": None,
        "config": spec.sam2_config or info["config"],
        "checkpoint_url": info["url"],
        "checkpoint_included": False,
    }


def _yolo_copy(spec: JobSpec, bundle_dir: Path, log: LogFn) -> Dict[str, Any]:
    source = Path(spec.model_path)
    if not source.is_file():
        raise FileNotFoundError(f"YOLO model not found: {source}")
    size = source.stat().st_size
    if size < 1024 * 1024:
        raise RuntimeError(f"YOLO model looks too small ({size} bytes): {source}")
    model_dir = bundle_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / source.name
    _log(log, f"Adding YOLO model: {source.name} ({size / 1024 / 1024:.0f} MB)")
    shutil.copyfile(source, dest)
    return {
        "kind": "yolo",
        "file": f"model/{source.name}",
        "model_name": source.name,
    }


def build_bundle(
    spec: JobSpec,
    job_dir: Path,
    include_sam2_checkpoint: bool = True,
    keep_bundle: bool = False,
    max_bundle_mb: float = 4096.0,
    progress: ProgressFn = None,
    log: LogFn = None,
    model_override: Optional[Dict[str, Any]] = None,
) -> BundleInfo:
    """Create ``<job_dir>/job.zip`` and ``<job_dir>/undo.json``."""
    spec.validate()
    job_dir = Path(job_dir)
    if job_dir.exists():
        # A retry may leave a partial bundle behind.
        _safe_rmtree(job_dir / "bundle")
        try:
            (job_dir / "job.zip").unlink()
        except OSError:
            pass
    job_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = job_dir / "bundle"
    images_out = bundle_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # Kaggle script kernels only upload ``code_file`` (runner.py); auxiliary
    # kernel-folder files are not sent.  Keep the shared label helpers inside
    # the job archive instead, and let runner.py import them after extraction.
    shutil.copyfile(
        Path(__file__).resolve().parent / "runner_lib.py",
        bundle_dir / "runner_lib.py",
    )

    included = spec.included_indices()
    if not included:
        raise RuntimeError("No frames selected for this job.")
    _call(progress, 2, "Validating frames and labels...")

    # Existing labels are embedded in the manifest and also snapshotted for undo.
    labels: Dict[str, List[List[str]]] = {}
    undo_lines: Dict[str, List[str]] = {}
    for index in included:
        name = spec.images[index]
        stem = Path(name).stem
        label_path = Path(spec.labels_dir) / (stem + ".txt")
        raw = read_label_lines(label_path)
        undo_lines[name] = raw
        labels[name] = normalize_label_lines(raw)

    write_json_atomic(
        job_dir / "undo.json",
        {
            "schema": 1,
            "created_at": _utc_now(),
            "labels_dir": str(Path(spec.labels_dir).resolve()),
            "lines": undo_lines,
        },
    )

    # Copy frames as a numeric JPEG sequence.  SAM 2's folder loader requires
    # numeric names; using the same sequence for YOLO keeps the protocol simple.
    entries: List[Dict[str, Any]] = []
    total = len(included)
    total_bytes = 0
    limit_bytes = max(32.0, float(max_bundle_mb)) * 1024 * 1024
    for position, index in enumerate(included):
        name = spec.images[index]
        source = Path(spec.images_dir) / name
        if not source.is_file():
            raise FileNotFoundError(f"Image not found: {source}")
        image = _imread(source)
        if image is None:
            raise RuntimeError(f"Could not read image: {source}")
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid image size: {source}")
        relative = f"images/{position:06d}.jpg"
        dest = bundle_dir / relative
        suffix = source.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            shutil.copyfile(source, dest)
        else:
            _imwrite(dest, image, quality=92)
        size = dest.stat().st_size
        total_bytes += size
        if total_bytes > limit_bytes:
            raise RuntimeError(
                f"Job bundle is larger than the configured limit "
                f"({max_bundle_mb:.0f} MB). Reduce the frame range or raise "
                "KAGGLE_MAX_BUNDLE_MB."
            )
        entries.append(
            {
                "name": name,
                "file": relative,
                "width": width,
                "height": height,
                "index": index + 1,
            }
        )
        if position % 10 == 0 or position == total - 1:
            _call(
                progress,
                3 + int(30 * (position + 1) / total),
                f"Packing frames {position + 1}/{total}...",
            )

    if model_override is not None:
        # The model/checkpoint lives in the persistent weights dataset; the
        # job bundle only carries images, labels and this small pointer.
        model = dict(model_override)
        _log(log, "Model weights are reused from the Kaggle weights dataset.")
    elif spec.job_type == "sam2_track":
        model = _checkpoint_copy(spec, bundle_dir, include_sam2_checkpoint, log)
    else:
        model = _yolo_copy(spec, bundle_dir, log)

    manifest: Dict[str, Any] = {
        "schema": 1,
        "job_id": job_dir.name,
        "type": spec.job_type,
        "created_at": _utc_now(),
        "classes": list(spec.classes),
        "images": entries,
        "labels": labels,
        "model": model,
        "params": {
            "start_index": spec.start_index if spec.job_type == "sam2_track" else 1,
            "end_index": spec.end_index if spec.job_type == "sam2_track" else len(entries),
            "source_start_index": spec.start_index,
            "source_end_index": spec.end_index,
            "direction": spec.direction,
            "min_confidence": float(spec.min_confidence),
            "boxes": [list(box) for box in spec.boxes],
            "selection_area": spec.selection_area,
        },
        "source": {"labelimg_plus_kaggle_bridge": "1.0.0"},
    }
    write_json_atomic(bundle_dir / "manifest.json", manifest)
    _call(progress, 35, "Compressing job bundle...")

    job_zip = job_dir / "job.zip"
    with zipfile.ZipFile(
        job_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as archive:
        files = sorted(
            (path for path in bundle_dir.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(bundle_dir).as_posix(),
        )
        for number, path in enumerate(files, start=1):
            arcname = path.relative_to(bundle_dir).as_posix()
            suffix = path.suffix.lower()
            compress = (
                zipfile.ZIP_STORED
                if suffix in (".jpg", ".jpeg", ".png", ".pt", ".zip")
                else zipfile.ZIP_DEFLATED
            )
            archive.write(path, arcname, compress_type=compress)
            if number % 10 == 0 or number == len(files):
                _call(
                    progress,
                    35 + int(12 * number / max(1, len(files))),
                    f"Compressing {number}/{len(files)}...",
                )

    size_bytes = job_zip.stat().st_size
    if size_bytes > limit_bytes:
        try:
            job_zip.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"Compressed job is {size_bytes / 1024 / 1024:.0f} MB, above the "
            f"{max_bundle_mb:.0f} MB limit."
        )
    sha = _sha256(job_zip)
    _log(
        log,
        f"Bundle ready: {job_zip} ({size_bytes / 1024 / 1024:.1f} MB, sha256 {sha[:12]}...)",
    )
    if not keep_bundle:
        _safe_rmtree(bundle_dir)
    _call(progress, 48, "Job bundle ready.")
    return BundleInfo(
        job_zip=job_zip,
        size_bytes=size_bytes,
        sha256=sha,
        image_count=len(entries),
        included_indices=included,
        model_included=bool(model.get("file")),
        manifest=manifest,
    )
