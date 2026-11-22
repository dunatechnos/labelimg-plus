"""Per-app-session image dataset + tiny command payload.

Design:
* When Kaggle mode is enabled (or the first remote job starts), the current
  ``images/`` folder is packed once into a private *session dataset*.
* Every Predict/SAM2 click then changes only a small command payload.  The
  command is embedded directly into ``runner.py`` (Kaggle script kernels upload
  one code file), so no per-click dataset is created.
* On LabelImg close, the session dataset is deleted from Kaggle.  The model
  weights dataset and the reusable fixed kernel stay.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .bundle import _imread, _imwrite
from .config import Config
from .kaggle_cli import KaggleCLI, KaggleCliError
from .metadata import dataset_metadata, validate_slug, write_json_atomic
from .runner_lib import normalize_label_lines, read_label_lines
from .spec import JobSpec

ProgressFn = Optional[Callable[[int, str], None]]
LogFn = Optional[Callable[[str], None]]
MARKER = "EMBEDDED_COMMAND = None  # KAGGLE_BRIDGE_COMMAND"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


@dataclass
class SessionInfo:
    session_id: str
    dataset_ref: str
    images_dir: str
    fingerprint: str
    image_count: int
    classes: List[str]
    created_at: str


class SessionError(RuntimeError):
    pass


def session_state_path(cfg: Config) -> Path:
    return Path(cfg.job_root) / "session.json"


def load_session_state(cfg: Config) -> Dict[str, Any]:
    try:
        data = json.loads(session_state_path(cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_session_state(cfg: Config, state: Dict[str, Any]) -> None:
    cfg.ensure_job_root()
    write_json_atomic(session_state_path(cfg), state)


def clear_session_state(cfg: Config) -> None:
    try:
        session_state_path(cfg).unlink()
    except OSError:
        pass


def image_fingerprint(images_dir: Path, images: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for name in images:
        path = Path(images_dir) / name
        try:
            stat = path.stat()
        except OSError as exc:
            raise SessionError(f"Image missing while preparing session: {path}") from exc
        digest.update(f"{name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def session_dataset_id(username: str, prefix: str, session_id: str) -> str:
    owner = validate_slug((username or "").strip().lower(), "Kaggle username")
    slug = f"{prefix}-{session_id}"[:50].rstrip("-")
    return f"{owner}/{validate_slug(slug, 'session dataset slug')}"


def session_title(session_id: str) -> str:
    return f"LabelImg Plus Session {session_id}"[:50]


def build_command_payload(
    spec: JobSpec,
    model: Optional[Dict[str, Any]],
    session: SessionInfo,
    job_id: str,
) -> Dict[str, Any]:
    """Build the tiny per-click command sent to Kaggle."""
    if spec.job_type == "yolo_predict":
        indices = list(range(spec.start_index - 1, spec.end_index))
    else:
        indices = list(range(len(spec.images)))
    labels: Dict[str, List[List[str]]] = {}
    for index in indices:
        if index < 0 or index >= len(spec.images):
            continue
        name = spec.images[index]
        label_path = Path(spec.labels_dir) / (Path(name).stem + ".txt")
        labels[name] = normalize_label_lines(read_label_lines(label_path))
    if spec.job_type == "yolo_predict":
        start_index = 1
        end_index = len(indices)
    else:
        start_index = spec.start_index
        end_index = spec.end_index
    return {
        "schema": 1,
        "job_id": job_id,
        "type": spec.job_type,
        "session_id": session.session_id,
        "classes": list(spec.classes),
        "labels": labels,
        "model": model,
        "params": {
            "indices": indices,
            "start_index": start_index,
            "end_index": end_index,
            "source_start_index": spec.start_index,
            "source_end_index": spec.end_index,
            "direction": spec.direction,
            "min_confidence": float(spec.min_confidence),
            "boxes": [list(box) for box in spec.boxes],
            "selection_area": spec.selection_area,
        },
    }


def embed_command(runner_source: str, command: Dict[str, Any]) -> str:
    if MARKER not in runner_source:
        raise SessionError(
            "kernel_runner.py is missing the KAGGLE_BRIDGE_COMMAND marker"
        )
    # repr() emits Python literals (None/True/False), not JSON's null/true.
    # The command contains only JSON-compatible primitives, so repr is safe.
    literal = "EMBEDDED_COMMAND = " + repr(command)
    return runner_source.replace(MARKER, literal, 1)


class RemoteSession:
    def __init__(self, cfg: Config, cli: KaggleCLI, log: LogFn = None):
        self.cfg = cfg
        self.cli = cli
        self.log = log

    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self.log:
            try:
                self.log(message)
            except Exception:
                pass

    def _progress(self, progress: ProgressFn, percent: int, message: str) -> None:
        if progress:
            try:
                progress(max(0, min(100, int(percent))), message)
            except Exception:
                pass

    def _session_dir(self, session_id: str) -> Path:
        return Path(self.cfg.job_root) / "sessions" / session_id

    def _dataset_id(self, username: str, session_id: str) -> str:
        return session_dataset_id(
            username, self.cfg.session_dataset_prefix, session_id
        )

    def _state_from_info(self, info: SessionInfo) -> Dict[str, Any]:
        return {
            "schema": 1,
            "session_id": info.session_id,
            "dataset_ref": info.dataset_ref,
            "images_dir": info.images_dir,
            "fingerprint": info.fingerprint,
            "image_count": info.image_count,
            "classes": info.classes,
            "created_at": info.created_at,
            "created_ts": time.time(),
            "last_used_ts": time.time(),
            "command_refs": [],
        }

    def _info_from_state(self, state: Dict[str, Any]) -> SessionInfo:
        return SessionInfo(
            session_id=str(state.get("session_id") or ""),
            dataset_ref=str(state.get("dataset_ref") or ""),
            images_dir=str(state.get("images_dir") or ""),
            fingerprint=str(state.get("fingerprint") or ""),
            image_count=int(state.get("image_count") or 0),
            classes=list(state.get("classes") or []),
            created_at=str(state.get("created_at") or ""),
        )

    # ------------------------------------------------------------------
    # Image packing / upload
    # ------------------------------------------------------------------
    def _build_zip(
        self,
        session_dir: Path,
        images_dir: Path,
        images: List[str],
        classes: List[str],
        session_id: str,
        progress: ProgressFn,
    ) -> Tuple[Path, List[Dict[str, Any]]]:
        bundle = session_dir / "bundle"
        shutil.rmtree(bundle, ignore_errors=True)
        images_out = bundle / "images"
        images_out.mkdir(parents=True, exist_ok=True)

        entries: List[Dict[str, Any]] = []
        total = len(images)
        total_bytes = 0
        for position, name in enumerate(images):
            source = Path(images_dir) / name
            if not source.is_file():
                raise SessionError(f"Image not found: {source}")
            image = _imread(source)
            if image is None:
                raise SessionError(f"Could not read image: {source}")
            height, width = image.shape[:2]
            relative = f"images/{position:06d}.jpg"
            dest = bundle / relative
            if source.suffix.lower() in (".jpg", ".jpeg"):
                shutil.copyfile(source, dest)
            else:
                _imwrite(dest, image, quality=90)
            total_bytes += dest.stat().st_size
            entries.append(
                {
                    "name": name,
                    "file": relative,
                    "width": width,
                    "height": height,
                }
            )
            if position % 10 == 0 or position == total - 1:
                self._progress(
                    progress,
                    2 + int(38 * (position + 1) / total),
                    f"Packing session images {position + 1}/{total}...",
                )

        write_json_atomic(
            bundle / "session.json",
            {
                "schema": 1,
                "session_id": session_id,
                "images": entries,
                "classes": list(classes),
                "created_at": _now(),
            },
        )
        # Shared label helper for the kernel.
        shutil.copyfile(
            Path(__file__).resolve().parent / "runner_lib.py",
            bundle / "runner_lib.py",
        )
        # SAM2 wheel(s): avoids a slow git/pip build inside every kernel.
        wheels_src = Path(self.cfg.weights_root) / "wheels"
        if wheels_src.is_dir():
            wheels_dst = bundle / "wheels"
            wheels_dst.mkdir(exist_ok=True)
            for wheel in sorted(wheels_src.glob("*.whl")):
                shutil.copyfile(wheel, wheels_dst / wheel.name)
                self._log(f"Bundled wheel: {wheel.name}")

        archive = session_dir / "session.zip"
        with zipfile.ZipFile(
            archive, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as zf:
            files = sorted(
                (path for path in bundle.rglob("*") if path.is_file()),
                key=lambda path: path.relative_to(bundle).as_posix(),
            )
            for number, path in enumerate(files, start=1):
                arcname = path.relative_to(bundle).as_posix()
                suffix = path.suffix.lower()
                compress = (
                    zipfile.ZIP_STORED
                    if suffix in (".jpg", ".jpeg", ".png", ".whl")
                    else zipfile.ZIP_DEFLATED
                )
                zf.write(path, arcname, compress_type=compress)
                if number % 20 == 0 or number == len(files):
                    self._progress(
                        progress,
                        40 + int(8 * number / max(1, len(files))),
                        f"Compressing session {number}/{len(files)}...",
                    )
        size_gb = archive.stat().st_size / (1024 ** 3)
        if size_gb > self.cfg.max_session_gb:
            raise SessionError(
                f"Session images are {size_gb:.1f} GB, above the configured "
                f"{self.cfg.max_session_gb:.1f} GB limit. Reduce the batch or "
                "raise KAGGLE_MAX_SESSION_GB."
            )
        self._log(
            f"Session archive: {archive} ({archive.stat().st_size / 1024 / 1024:.0f} MB)"
        )
        return archive, entries

    def _upload(
        self,
        session_id: str,
        dataset_id: str,
        archive: Path,
        progress: ProgressFn,
    ) -> None:
        session_dir = self._session_dir(session_id)
        dataset_dir = session_dir / "dataset"
        shutil.rmtree(dataset_dir, ignore_errors=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive, dataset_dir / "session.zip")
        write_json_atomic(
            dataset_dir / "dataset-metadata.json",
            dataset_metadata(dataset_id, session_title(session_id)),
        )
        self._progress(progress, 50, "Uploading session images to Kaggle (one-time)...")
        try:
            self.cli.dataset_create(dataset_dir)
        except KaggleCliError as exc:
            text = str(exc).lower()
            if "already in use" not in text and "already exists" not in text:
                raise
            self._log("Session dataset exists; creating a new version.")
            self.cli.dataset_version(dataset_dir, f"Session {session_id}")
        self.cli.wait_dataset_ready(dataset_id, timeout=3600, poll=5)
        self._progress(progress, 95, "Kaggle session dataset ready.")
        shutil.rmtree(dataset_dir, ignore_errors=True)

    def ensure_images(
        self,
        images_dir: str,
        images: List[str],
        classes: Optional[List[str]] = None,
        progress: ProgressFn = None,
        username: str = "",
    ) -> SessionInfo:
        if not images:
            raise SessionError("No images to upload for the Kaggle session.")
        source_dir = Path(images_dir)
        images = list(images)
        classes = list(classes or [])
        fingerprint = image_fingerprint(source_dir, images)
        state = load_session_state(self.cfg)

        if state and state.get("dataset_ref") and state.get("fingerprint") == fingerprint:
            if state.get("images_dir") == str(source_dir.resolve()):
                info = self._info_from_state(state)
                status = self.cli.dataset_status(info.dataset_ref)
                if status in ("ready", "ok", "complete"):
                    state["last_used_ts"] = time.time()
                    save_session_state(self.cfg, state)
                    self._log(f"Reusing Kaggle session dataset {info.dataset_ref}")
                    self._progress(progress, 100, "Kaggle session ready (reused).")
                    return info
                if status in ("processing", "pending"):
                    self._log("Session dataset is processing; waiting...")
                    try:
                        self.cli.wait_dataset_ready(info.dataset_ref, timeout=900, poll=5)
                        state["last_used_ts"] = time.time()
                        save_session_state(self.cfg, state)
                        self._progress(progress, 100, "Kaggle session ready.")
                        return info
                    except KaggleCliError as exc:
                        self._log(f"Could not reuse session dataset: {exc}")
                else:
                    self._log(f"Session dataset is not available ({status}); rebuilding.")

        # A previous session is obsolete (different images or missing remote).
        if state:
            self._delete_ref(state.get("dataset_ref"))
            for ref in state.get("command_refs") or []:
                self._delete_ref(ref)
            self._remove_local(str(state.get("session_id") or ""))
            clear_session_state(self.cfg)

        session_id = new_session_id()
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        dataset_id = self._dataset_id(username or self.cfg.username, session_id)
        self._progress(progress, 1, "Preparing session images (one-time)...")
        archive, entries = self._build_zip(
            session_dir,
            source_dir,
            images,
            classes,
            session_id,
            progress,
        )
        self._upload(session_id, dataset_id, archive, progress)

        info = SessionInfo(
            session_id=session_id,
            dataset_ref=dataset_id,
            images_dir=str(source_dir.resolve()),
            fingerprint=fingerprint,
            image_count=len(entries),
            classes=classes,
            created_at=_now(),
        )
        save_session_state(self.cfg, self._state_from_info(info))
        if not self.cfg.session_keep_local:
            shutil.rmtree(session_dir / "bundle", ignore_errors=True)
            try:
                archive.unlink()
            except OSError:
                pass
        self._progress(progress, 100, f"Kaggle session ready ({len(entries)} images).")
        return info

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _delete_ref(self, ref: Optional[str]) -> None:
        if not ref or "/" not in str(ref):
            return
        try:
            self.cli.dataset_delete(str(ref))
            self._log(f"Deleted Kaggle dataset {ref}")
        except Exception as exc:
            self._log(f"Could not delete {ref}: {exc}")

    def _remove_local(self, session_id: str) -> None:
        if not session_id:
            return
        shutil.rmtree(self._session_dir(session_id), ignore_errors=True)

    def cleanup(
        self,
        force: bool = True,
        stale_hours: Optional[float] = None,
    ) -> bool:
        state = load_session_state(self.cfg)
        if not state:
            return False
        if stale_hours is not None:
            age = time.time() - float(state.get("last_used_ts") or state.get("created_ts") or 0)
            if age < float(stale_hours) * 3600:
                self._log(
                    f"Session is recent ({age / 3600:.1f}h); leaving it alone."
                )
                return False
        self._delete_ref(state.get("dataset_ref"))
        for ref in state.get("command_refs") or []:
            self._delete_ref(ref)
        self._remove_local(str(state.get("session_id") or ""))
        clear_session_state(self.cfg)
        self._log("Kaggle session state cleared.")
        return True
