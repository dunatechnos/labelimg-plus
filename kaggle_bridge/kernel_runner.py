#!/usr/bin/env python3
"""Standalone Kaggle kernel runner for labelImg Plus.

This file and ``runner_lib.py`` are copied into a temporary Kaggle kernel
folder.  They do not import the local ``kaggle_bridge`` package.

Expected input:
    /kaggle/input/<dataset-slug>/job.zip

Output (collected from /kaggle/working):
    output/labels.json   {original_image_name: [[cls, xc, yc, w, h], ...]}
    output/result.json   status, counts and error message
    output/status.json   last progress marker
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# The local worker replaces this line with the actual per-click command when
# using the fast session dataset.  Legacy jobs leave it as None and carry a
# manifest inside job.zip instead.
EMBEDDED_COMMAND = None  # KAGGLE_BRIDGE_COMMAND

# ``runner_lib`` is shipped inside the session/job dataset and imported after
# extraction in main().  Kaggle script kernels do not upload sibling files
# from the kernel folder, so importing it at module import time would fail.
merge_tracked_labels = None
normalize_label_lines = None
parse_label_parts = None
remap_class_by_name = None

# Keep the extracted input outside /kaggle/working, otherwise Kaggle collects
# the frames/model again as kernel output.  Only output/ is persisted.
JOB_DIR = Path(os.environ.get("LABELIMG_JOB_DIR", "/tmp/labelimg_job"))
OUTPUT_DIR = Path(os.environ.get("LABELIMG_OUTPUT_DIR", "/kaggle/working/output"))
STATUS_FILE = OUTPUT_DIR / "status.json"
RESULT_FILE = OUTPUT_DIR / "result.json"
LABELS_FILE = OUTPUT_DIR / "labels.json"
START_TS = time.time()


def log(message: str) -> None:
    elapsed = time.time() - START_TS
    print(f"[{elapsed:8.1f}s] {message}", flush=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def write_status(stage: str, percent: int, message: str, **extra: Any) -> None:
    payload = {
        "stage": stage,
        "percent": max(0, min(100, int(percent))),
        "message": message,
        "elapsed_seconds": round(time.time() - START_TS, 1),
    }
    payload.update(extra)
    try:
        write_json(STATUS_FILE, payload)
    except OSError:
        pass


# ----------------------------------------------------------------------
# Input / dependency setup
# ----------------------------------------------------------------------
def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(str(source), str(destination))
    except OSError:
        shutil.copyfile(source, destination)


def find_session_input():
    """Find the extracted session dataset (session.json + images/)."""
    explicit = os.environ.get("LABELIMG_SESSION_DIR")
    if explicit and (Path(explicit) / "session.json").is_file():
        return Path(explicit)
    manifests = sorted(glob.glob("/kaggle/input/**/session.json", recursive=True))
    if manifests:
        return Path(manifests[0]).parent
    archives = sorted(glob.glob("/kaggle/input/**/session.zip", recursive=True))
    if archives:
        if JOB_DIR.exists():
            shutil.rmtree(JOB_DIR)
        safe_extract(Path(archives[0]), JOB_DIR)
        return JOB_DIR
    raise FileNotFoundError(
        "session.json/session.zip not found under /kaggle/input. "
        "Check that the session dataset is attached to the kernel."
    )


def build_manifest_from_session(session_root: Path, command: Dict[str, Any]) -> Dict[str, Any]:
    with open(session_root / "session.json", "r", encoding="utf-8") as handle:
        session = json.load(handle)
    if not isinstance(session, dict) or not isinstance(session.get("images"), list):
        raise RuntimeError("session.json is invalid")
    params = command.get("params") or {}
    indices = params.get("indices")
    all_images = session["images"]
    if indices is None:
        selected = list(all_images)
    else:
        selected = []
        for index in indices:
            try:
                selected.append(all_images[int(index)])
            except (IndexError, TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid image index in command: {index!r}") from exc

    if session_root.resolve() != JOB_DIR.resolve():
        if JOB_DIR.exists():
            shutil.rmtree(JOB_DIR)
        JOB_DIR.mkdir(parents=True, exist_ok=True)
        for entry in selected:
            relative = str(entry.get("file") or "")
            if not relative:
                raise RuntimeError("session image entry has no file")
            source = (session_root / relative).resolve()
            try:
                source.relative_to(session_root.resolve())
            except ValueError as exc:
                raise RuntimeError(f"session image escapes root: {relative!r}") from exc
            if not source.is_file():
                raise FileNotFoundError(f"session image missing: {source}")
            _link_or_copy(source, JOB_DIR / relative)
    # runner_lib.py lives in the session dataset.
    lib_source = session_root / "runner_lib.py"
    if lib_source.is_file() and not (JOB_DIR / "runner_lib.py").is_file():
        shutil.copyfile(lib_source, JOB_DIR / "runner_lib.py")

    manifest = {
        "schema": 1,
        "job_id": command.get("job_id") or "unknown",
        "type": command.get("type") or "yolo_predict",
        "classes": command.get("classes") or [],
        "images": selected,
        "labels": command.get("labels") or {},
        "model": command.get("model") or {},
        "params": params,
    }
    write_json(JOB_DIR / "manifest.json", manifest)
    return manifest


def find_job_input():
    """Return ``(path, kind)`` where kind is ``"zip"`` or ``"dir"``.

    Kaggle auto-extracts zip files uploaded inside a dataset, so a live run
    usually sees the extracted bundle root (manifest.json + images/ + model/)
    rather than job.zip.  Local tests still pass a real zip.
    """
    explicit = os.environ.get("LABELIMG_JOB_ZIP")
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path, "zip"
    explicit_dir = os.environ.get("LABELIMG_JOB_INPUT")
    if explicit_dir:
        path = Path(explicit_dir)
        if (path / "manifest.json").is_file():
            return path, "dir"

    manifests = sorted(glob.glob("/kaggle/input/**/manifest.json", recursive=True))
    if manifests:
        return Path(manifests[0]).parent, "dir"
    archives = sorted(glob.glob("/kaggle/input/**/job.zip", recursive=True))
    if archives:
        return Path(archives[0]), "zip"

    local_zip = Path("/kaggle/working/job.zip")
    if local_zip.is_file():
        return local_zip, "zip"
    local_dir = Path("/kaggle/working/job")
    if (local_dir / "manifest.json").is_file():
        return local_dir, "dir"
    raise FileNotFoundError(
        "Job bundle not found under /kaggle/input. "
        "Check that the dataset was attached to this kernel."
    )


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            name = member.filename.replace("\\", "/")
            if name.startswith("/") or ".." in name.split("/"):
                raise RuntimeError(f"Unsafe zip member: {member.filename!r}")
            target = (destination / name).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise RuntimeError(f"Zip member escapes destination: {member.filename!r}") from exc
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(target, "wb") as output:
                shutil.copyfileobj(source, output)


def pip_install(args: List[str], timeout: int = 1200) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-input"] + args
    log("Installing: " + " ".join(args))
    env = os.environ.copy()
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "pip install failed:\n"
            + (proc.stdout or "")[-3000:]
            + "\n"
            + (proc.stderr or "")[-3000:]
        )
    log("pip install finished.")


def ensure_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401

        log("ultralytics is already installed.")
        return
    except Exception:
        pass
    pip_install(["ultralytics"], timeout=1200)


def ensure_sam2() -> None:
    try:
        import sam2  # noqa: F401

        log("SAM 2 is already installed.")
        return
    except Exception:
        pass

    # The session dataset carries a prebuilt SAM 2 wheel.  This avoids the
    # slow git clone + build that used to cost ~4-5 minutes per kernel.
    wheels = sorted(glob.glob("/kaggle/input/**/sam_2*.whl", recursive=True))
    wheels += sorted(glob.glob("/kaggle/input/**/sam2*.whl", recursive=True))
    # Fallback when the session zip was extracted into JOB_DIR instead of
    # being auto-extracted by Kaggle.
    wheels += sorted(glob.glob(str(JOB_DIR / "**" / "sam_2*.whl"), recursive=True))
    wheels += sorted(glob.glob(str(JOB_DIR / "**" / "sam2*.whl"), recursive=True))
    if wheels:
        wheel = wheels[0]
        log(f"Installing SAM 2 from local wheel: {wheel}")
        try:
            pip_install(["--no-deps", wheel], timeout=600)
            pip_install(["hydra-core", "iopath"], timeout=900)
            import sam2  # noqa: F401
            log("SAM 2 installed from local wheel.")
            return
        except Exception as exc:
            log(f"Local SAM 2 wheel failed ({exc}); falling back to GitHub.")

    pip_install(["hydra-core", "iopath"], timeout=900)
    env = os.environ.copy()
    env["SAM2_BUILD_CUDA"] = "0"
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-input",
        "--no-deps",
        "git+https://github.com/facebookresearch/sam2.git",
    ]
    log("Installing SAM 2 from GitHub (CUDA extension disabled)...")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Could not install SAM 2. Internet must be enabled for this kernel.\n"
            + (proc.stdout or "")[-3000:]
            + "\n"
            + (proc.stderr or "")[-3000:]
        )
    try:
        import sam2  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"SAM 2 installed but cannot be imported: {exc}") from exc
    log("SAM 2 installed.")


def download_checkpoint(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 100 * 1024 * 1024:
        return destination
    log(f"Downloading SAM 2 checkpoint: {url}")
    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "labelimg-plus"})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if tmp.stat().st_size < 100 * 1024 * 1024:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("SAM 2 checkpoint download looks incomplete.")
    os.replace(tmp, destination)
    return destination


def load_manifest(job_dir: Path) -> Dict[str, Any]:
    manifest_path = job_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise RuntimeError("manifest.json is not an object")
    images = manifest.get("images")
    if not isinstance(images, list) or not images:
        raise RuntimeError("manifest.json has no images")
    names = set()
    for entry in images:
        name = entry.get("name")
        rel = entry.get("file")
        if not name or not rel:
            raise RuntimeError("manifest image entry is missing name/file")
        if "/" in name or "\\" in name:
            raise RuntimeError(f"Unsafe image name: {name!r}")
        # Do not resolve symlinks here: session images are symlinked from
        # /kaggle/input to /tmp to avoid copying multi-GB image sets.  Use the
        # link path for the containment check and let is_file() follow it.
        target = Path(os.path.abspath(str(job_dir / rel)))
        try:
            target.relative_to(Path(os.path.abspath(str(job_dir))))
        except ValueError as exc:
            raise RuntimeError(f"Image path escapes job dir: {rel!r}") from exc
        if not target.is_file():
            raise RuntimeError(f"Image file missing from bundle: {rel!r}")
        names.add(name)
    labels = manifest.get("labels") or {}
    if not isinstance(labels, dict):
        raise RuntimeError("manifest labels must be an object")
    for name in labels:
        if name not in names:
            raise RuntimeError(f"Labels reference an unknown image: {name!r}")
    return manifest


def imread(path: Path):
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def resolve_model_file(job_dir: Path, model_info: Dict[str, Any]) -> Path:
    """Resolve a bundle model or a file from the persistent weights dataset."""
    if model_info.get("source") == "weights":
        name = str(model_info.get("weights_file") or "").strip()
        if not name:
            raise RuntimeError("weights model entry has no weights_file")
        if "/" in name or "\\" in name or name in (".", ".."):
            raise RuntimeError(f"unsafe weights filename: {name!r}")
        explicit_dir = os.environ.get("LABELIMG_WEIGHTS_DIR")
        candidates = []
        if explicit_dir:
            candidates.extend(glob.glob(str(Path(explicit_dir) / name)))
        candidates.extend(
            glob.glob(str(Path("/kaggle/input") / "**" / name), recursive=True)
        )
        candidates = [path for path in candidates if Path(path).is_file()]
        if not candidates:
            attached = sorted(
                str(path) for path in Path("/kaggle/input").glob("*") if path.is_dir()
            )
            raise FileNotFoundError(
                f"Weights file {name!r} not found in the attached weights dataset. "
                f"Attached inputs: {attached}"
            )
        log(f"Using weights file: {candidates[0]}")
        return Path(candidates[0])
    relative = model_info.get("file")
    if not relative:
        raise RuntimeError("manifest model entry has neither file nor weights_file")
    path = (job_dir / relative).resolve()
    try:
        path.relative_to(job_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Model path escapes job dir: {relative!r}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Model file missing from bundle: {path}")
    return path


# ----------------------------------------------------------------------
# YOLO predict
# ----------------------------------------------------------------------
def run_yolo(manifest: Dict[str, Any], job_dir: Path) -> Dict[str, Any]:
    ensure_ultralytics()
    from ultralytics import YOLO
    import torch

    model_info = manifest.get("model") or {}
    model_path = resolve_model_file(job_dir, model_info)
    if not model_path.is_file():
        raise RuntimeError(f"YOLO model missing: {model_path}")
    device = 0 if torch.cuda.is_available() else "cpu"
    log(f"Loading YOLO model {model_path.name} (device={device})...")
    model = YOLO(str(model_path))
    names = getattr(model, "names", {}) or {}
    name_to_idx = {
        str(name).strip().lower(): index
        for index, name in enumerate(manifest.get("classes") or [])
    }
    classes = manifest.get("classes") or []
    area = (manifest.get("params") or {}).get("selection_area")
    images = manifest["images"]
    labels = manifest.get("labels") or {}
    updates: Dict[str, List[List[str]]] = {}
    skipped_classes: Dict[str, int] = {}
    total_boxes = 0

    for position, entry in enumerate(images):
        name = entry["name"]
        image = imread(job_dir / entry["file"])
        if image is None:
            log(f"WARNING: cannot read {name}; skipping")
            continue
        height, width = image.shape[:2]
        if width != int(entry.get("width") or width) or height != int(entry.get("height") or height):
            log(
                f"WARNING: size changed for {name}: "
                f"{width}x{height} vs manifest {entry.get('width')}x{entry.get('height')}"
            )

        original = normalize_label_lines(labels.get(name) or [])
        kept: List[List[str]] = []
        predictions: List[List[str]] = []

        if area:
            xmin_px = max(0, min(width - 1, int(float(area["xmin"]) * width)))
            ymin_px = max(0, min(height - 1, int(float(area["ymin"]) * height)))
            xmax_px = max(xmin_px + 1, min(width, int(float(area["xmax"]) * width)))
            ymax_px = max(ymin_px + 1, min(height, int(float(area["ymax"]) * height)))
            crop = image[ymin_px:ymax_px, xmin_px:xmax_px]
            if crop.size == 0:
                log(f"WARNING: empty crop for {name}; keeping labels unchanged")
                continue
            results = model.predict(crop, verbose=False, device=device)
            for parts in original:
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                bxmin, bxmax = xc - bw / 2.0, xc + bw / 2.0
                bymin, bymax = yc - bh / 2.0, yc + bh / 2.0
                if (
                    bxmax < float(area["xmin"])
                    or bxmin > float(area["xmax"])
                    or bymax < float(area["ymin"])
                    or bymin > float(area["ymax"])
                ):
                    kept.append(parts)
            for box in (results[0].boxes if results else []):
                model_cls = int(box.cls[0])
                mapped = remap_class_by_name(names, model_cls, name_to_idx)
                if mapped is None:
                    model_name = names.get(model_cls, str(model_cls)) if isinstance(names, dict) else str(model_cls)
                    skipped_classes[str(model_name)] = skipped_classes.get(str(model_name), 0) + 1
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 += xmin_px
                x2 += xmin_px
                y1 += ymin_px
                y2 += ymin_px
                parsed = parse_label_parts([
                    str(mapped),
                    (x1 + x2) / 2.0 / width,
                    (y1 + y2) / 2.0 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ])
                if parsed is not None:
                    predictions.append(parsed)
        else:
            results = model.predict(image, verbose=False, device=device)
            kept = list(original)
            for box in (results[0].boxes if results else []):
                model_cls = int(box.cls[0])
                mapped = remap_class_by_name(names, model_cls, name_to_idx)
                if mapped is None:
                    model_name = names.get(model_cls, str(model_cls)) if isinstance(names, dict) else str(model_cls)
                    skipped_classes[str(model_name)] = skipped_classes.get(str(model_name), 0) + 1
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                parsed = parse_label_parts([
                    str(mapped),
                    (x1 + x2) / 2.0 / width,
                    (y1 + y2) / 2.0 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ])
                if parsed is not None:
                    predictions.append(parsed)

        final = kept + predictions
        total_boxes += len(predictions)
        if final != original:
            updates[name] = final
        if position % 10 == 0 or position == len(images) - 1:
            percent = 55 + int(30 * (position + 1) / len(images))
            write_status("predict", percent, f"Predicted {position + 1}/{len(images)} frames")
            log(f"Predicted {position + 1}/{len(images)} frames, {len(updates)} changed")

    message = f"YOLO predicted {total_boxes} box(es) across {len(images)} frame(s)."
    if skipped_classes:
        skipped = sum(skipped_classes.values())
        detail = ", ".join(f"{key} x{value}" for key, value in sorted(skipped_classes.items()))
        message += f" Skipped {skipped} box(es) with classes not in classes.txt: {detail}."
    return {
        "labels": updates,
        "message": message,
        "stats": {
            "frames": len(images),
            "changed_frames": len(updates),
            "boxes": total_boxes,
            "skipped_classes": skipped_classes,
            "classes": classes,
            "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        },
    }


# ----------------------------------------------------------------------
# SAM 2 tracking
# ----------------------------------------------------------------------
def run_sam2(manifest: Dict[str, Any], job_dir: Path) -> Dict[str, Any]:
    ensure_sam2()
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    model_info = manifest.get("model") or {}
    checkpoint: Optional[Path] = None
    if model_info.get("source") == "weights" or model_info.get("file"):
        checkpoint = resolve_model_file(job_dir, model_info)
        if not checkpoint.is_file():
            raise RuntimeError(f"SAM 2 checkpoint missing: {checkpoint}")
    else:
        url = model_info.get("checkpoint_url")
        if not url:
            raise RuntimeError("No SAM 2 checkpoint and no checkpoint URL in manifest")
        filename = Path(url.split("?")[0]).name or "sam2_checkpoint.pt"
        checkpoint = download_checkpoint(url, job_dir / "model" / filename)

    config = model_info.get("config") or "configs/sam2.1/sam2.1_hiera_s.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading SAM 2 ({config}) on {device}...")
    write_status("sam2", 8, "Loading SAM 2 model")
    predictor = build_sam2_video_predictor(
        config,
        str(checkpoint),
        device=device,
        apply_postprocessing=True,
    )
    log("Initializing video state...")
    write_status("sam2", 12, "Initializing video state")
    state = predictor.init_state(
        str(job_dir / "images"),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
    )
    width = int(state["video_width"])
    height = int(state["video_height"])
    images = manifest["images"]
    labels = manifest.get("labels") or {}
    params = manifest.get("params") or {}
    start = max(0, min(int(params.get("start_index", 1)) - 1, len(images) - 1))
    direction = str(params.get("direction") or "entire")
    min_confidence = max(0.0, min(1.0, float(params.get("min_confidence", 0.5))))
    boxes = params.get("boxes") or []
    if not boxes:
        raise RuntimeError("SAM 2 job has no prompt boxes")

    obj_classes: Dict[int, int] = {}
    for index, parts in enumerate(boxes):
        parsed = parse_label_parts(parts)
        if parsed is None:
            continue
        cls = int(parsed[0])
        xc, yc, bw, bh = map(float, parsed[1:5])
        obj_id = len(obj_classes) + 1
        x1 = max(0.0, min(width - 1.0, (xc - bw / 2.0) * width))
        y1 = max(0.0, min(height - 1.0, (yc - bh / 2.0) * height))
        x2 = max(x1 + 1.0, min(float(width), (xc + bw / 2.0) * width))
        y2 = max(y1 + 1.0, min(float(height), (yc + bh / 2.0) * height))
        predictor.add_new_points_or_box(
            state,
            frame_idx=start,
            obj_id=obj_id,
            box=[x1, y1, x2, y2],
        )
        obj_classes[obj_id] = cls
    if not obj_classes:
        raise RuntimeError("No valid SAM 2 prompt boxes")

    total = len(images) if direction == "entire" else (
        len(images) - start if direction == "forward" else start + 1
    )
    total = max(1, total)
    seen = set()
    updates: Dict[str, List[List[str]]] = {}
    skipped_low_confidence = 0
    updated_frames = 0

    def process_frame(frame_idx, obj_ids, video_res_masks) -> None:
        nonlocal skipped_low_confidence, updated_frames
        frame_idx = int(frame_idx)
        if frame_idx in seen:
            return
        seen.add(frame_idx)
        tracked: List[List[str]] = []
        for mask_idx, obj_id in enumerate(obj_ids):
            cls = obj_classes.get(int(obj_id))
            if cls is None:
                continue
            score = None
            try:
                obj_idx = state["obj_id_to_idx"].get(int(obj_id))
                if obj_idx is not None:
                    obj_out = state["output_dict_per_obj"].get(obj_idx, {})
                    out = (
                        obj_out.get("cond_frame_outputs", {}).get(frame_idx)
                        or obj_out.get("non_cond_frame_outputs", {}).get(frame_idx)
                    )
                    if out is not None:
                        score_logits = out.get("object_score_logits")
                        if score_logits is not None:
                            score = float(score_logits.float().sigmoid().mean().item())
            except (KeyError, TypeError, ValueError, AttributeError):
                score = None
            if score is not None and score < min_confidence:
                skipped_low_confidence += 1
                continue
            mask = (video_res_masks[mask_idx, 0] > 0.0).cpu().numpy()
            label = _mask_to_label(mask, cls, width, height)
            if label is not None:
                tracked.append(label)

        if tracked and 0 <= frame_idx < len(images):
            name = images[frame_idx]["name"]
            original = labels.get(name) or []
            merged, changed = merge_tracked_labels(original, tracked)
            if changed:
                updates[name] = merged
                updated_frames += 1

        percent = 12 + int(78 * len(seen) / total)
        write_status(
            "sam2",
            percent,
            f"Tracked {len(seen)}/{total} frames, {len(obj_classes)} object(s)",
            skipped_low_confidence=skipped_low_confidence,
        )
        if len(seen) % 10 == 0 or len(seen) == total:
            log(
                f"Tracked {len(seen)}/{total} frames · "
                f"{len(obj_classes)} object(s) · skipped {skipped_low_confidence}"
            )

    if direction in ("forward", "entire"):
        for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
            state,
            start_frame_idx=start,
            max_frame_num_to_track=len(images) - 1 - start,
        ):
            process_frame(frame_idx, obj_ids, video_res_masks)

    if direction in ("backward", "entire"):
        if start > 0:
            for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
                state,
                start_frame_idx=start,
                max_frame_num_to_track=start,
                reverse=True,
            ):
                process_frame(frame_idx, obj_ids, video_res_masks)
        elif direction == "backward":
            for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
                state,
                start_frame_idx=0,
                max_frame_num_to_track=0,
            ):
                process_frame(frame_idx, obj_ids, video_res_masks)

    try:
        del state
    except Exception:
        pass
    try:
        del predictor
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    message = (
        f"SAM 2 tracked {len(obj_classes)} object(s) across {len(seen)} frame(s); "
        f"updated {updated_frames} label file(s); "
        f"skipped {skipped_low_confidence} low-confidence detection(s)."
    )
    return {
        "labels": updates,
        "message": message,
        "stats": {
            "frames": len(images),
            "tracked_frames": len(seen),
            "changed_frames": len(updates),
            "objects": len(obj_classes),
            "skipped_low_confidence": skipped_low_confidence,
            "device": device,
            "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        },
    }


def _mask_to_label(mask, cls: int, width: int, height: int) -> Optional[List[str]]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x1 = max(0, min(width - 1, int(xs.min())))
    x2 = max(0, min(width - 1, int(xs.max())))
    y1 = max(0, min(height - 1, int(ys.min())))
    y2 = max(0, min(height - 1, int(ys.max())))
    bw = (x2 - x1) / float(width)
    bh = (y2 - y1) / float(height)
    if bw <= 0 or bh <= 0:
        return None
    return parse_label_parts([
        str(cls),
        (x1 + x2) / 2.0 / width,
        (y1 + y2) / 2.0 / height,
        bw,
        bh,
    ])


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    global merge_tracked_labels, normalize_label_lines, parse_label_parts, remap_class_by_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_status("start", 1, "Preparing job")
    if EMBEDDED_COMMAND is not None:
        write_status("session", 2, "Preparing session images")
        session_root = find_session_input()
        log(f"Session input: {session_root}")
        build_manifest_from_session(session_root, EMBEDDED_COMMAND)
    else:
        job_input, kind = find_job_input()
        log(f"Job input ({kind}): {job_input}")
        if JOB_DIR.exists():
            shutil.rmtree(JOB_DIR)
        if kind == "zip":
            safe_extract(job_input, JOB_DIR)
        else:
            shutil.copytree(job_input, JOB_DIR)
    sys.path.insert(0, str(JOB_DIR))
    import runner_lib  # supplied inside the session/job dataset

    merge_tracked_labels = runner_lib.merge_tracked_labels
    normalize_label_lines = runner_lib.normalize_label_lines
    parse_label_parts = runner_lib.parse_label_parts
    remap_class_by_name = runner_lib.remap_class_by_name
    manifest = load_manifest(JOB_DIR)
    job_id = manifest.get("job_id", "unknown")
    job_type = manifest.get("type")
    log(f"Job {job_id} type={job_type} frames={len(manifest['images'])}")

    if job_type == "yolo_predict":
        write_status("predict", 5, "Preparing YOLO")
        outcome = run_yolo(manifest, JOB_DIR)
    elif job_type == "sam2_track":
        write_status("sam2", 5, "Preparing SAM 2")
        outcome = run_sam2(manifest, JOB_DIR)
    else:
        raise RuntimeError(f"Unknown job type: {job_type!r}")

    coordinates = outcome["labels"]
    write_json(LABELS_FILE, coordinates)
    # Explicit coordinates-only output.  No frames/masks/checkpoints are
    # written to /kaggle/working, so Kaggle returns just this small file.
    write_json(OUTPUT_DIR / "coordinates.json", coordinates)
    result = {
        "schema": 1,
        "job_id": job_id,
        "type": job_type,
        "status": "ok",
        "changed": len(coordinates),
        "message": outcome["message"],
        "coordinates_file": "coordinates.json",
        "stats": outcome.get("stats") or {},
        "elapsed_seconds": round(time.time() - START_TS, 1),
    }
    write_json(RESULT_FILE, result)
    write_status("done", 100, outcome["message"])
    log(outcome["message"])
    log("KAGGLE_JOB_DONE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        output = {
            "schema": 1,
            "job_id": "",
            "type": "",
            "status": "error",
            "changed": 0,
            "message": str(exc),
            "error": traceback.format_exc(),
            "elapsed_seconds": round(time.time() - START_TS, 1),
        }
        try:
            find_job_input()
        except Exception:
            pass
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            write_json(RESULT_FILE, output)
            write_json(LABELS_FILE, {})
            write_status("error", 100, str(exc))
        except Exception:
            traceback.print_exc()
        raise SystemExit(1)
