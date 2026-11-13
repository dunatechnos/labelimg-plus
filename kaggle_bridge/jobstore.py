"""Local job state, logs and the single-job lock."""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metadata import read_json, write_json_atomic
from .spec import JobSpec

ACTIVE_STATES = {
    "created",
    "bundling",
    "uploading_dataset",
    "waiting_dataset",
    "pushing_kernel",
    "queued",
    "running",
    "downloading",
    "importing",
}
TERMINAL_STATES = {"done", "failed", "cancelled"}

_JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{5,63}$")


class JobBusyError(RuntimeError):
    """Another Kaggle job is already running or waiting."""


class JobNotFoundError(FileNotFoundError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_job_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
            )
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        except Exception:
            return True
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def validate_job_id(self, job_id: str) -> str:
        if not _JOB_ID_RE.match(job_id or ""):
            raise ValueError(f"invalid job id: {job_id!r}")
        return job_id

    def job_dir(self, job_id: str) -> Path:
        return self.root / self.validate_job_id(job_id)

    def job_file(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def bundle_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "bundle"

    def bundle_zip(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.zip"

    def kernel_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "kernel"

    def output_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "output"

    def log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.log"

    def undo_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "undo.json"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def create(self, spec: JobSpec, job_id: Optional[str] = None) -> Dict[str, Any]:
        job_id = job_id or new_job_id()
        self.validate_job_id(job_id)
        job_dir = self.job_dir(job_id)
        if job_dir.exists():
            raise ValueError(f"job already exists: {job_id}")
        job_dir.mkdir(parents=True)
        now = _utc_now()
        record: Dict[str, Any] = {
            "schema": 1,
            "job_id": job_id,
            "created_at": now,
            "updated_at": now,
            "state": "created",
            "progress": 0,
            "message": "Job created",
            "job_type": spec.job_type,
            "spec": spec.to_dict(),
            "images_count": len(spec.images),
            "classes_count": len(spec.classes),
            "dataset_ref": None,
            "kernel_ref": None,
            "dataset_url": None,
            "kernel_url": None,
            "bundle_zip": None,
            "bundle_bytes": 0,
            "output_dir": None,
            "error": None,
            "stats": {},
        }
        self.save(record)
        return record

    def load(self, job_id: str) -> Dict[str, Any]:
        path = self.job_file(job_id)
        data = read_json(path)
        if not isinstance(data, dict):
            raise JobNotFoundError(f"job not found or unreadable: {job_id}")
        return data

    def save(self, record: Dict[str, Any]) -> None:
        record["updated_at"] = _utc_now()
        write_json_atomic(self.job_file(record["job_id"]), record)

    def update(self, job_id: str, **fields) -> Dict[str, Any]:
        record = self.load(job_id)
        record.update(fields)
        self.save(record)
        return record

    def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        jobs: List[Dict[str, Any]] = []
        try:
            entries = list(self.root.iterdir())
        except OSError:
            return jobs
        for entry in entries:
            if not entry.is_dir() or not _JOB_ID_RE.match(entry.name):
                continue
            data = read_json(entry / "job.json")
            if isinstance(data, dict) and data.get("job_id"):
                jobs.append(data)
        jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return jobs[: max(0, limit)]

    def find_active(self) -> Optional[Dict[str, Any]]:
        for job in self.list_jobs(limit=200):
            if job.get("state") in ACTIVE_STATES:
                return job
        return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def append_log(self, job_id: str, message: str) -> None:
        try:
            self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%H:%M:%S")
            with open(self.log_path(job_id), "a", encoding="utf-8") as handle:
                for line in str(message).splitlines() or [""]:
                    handle.write(f"[{stamp}] {line}\n")
        except OSError:
            pass

    def read_log_tail(self, job_id: str, lines: int = 80) -> str:
        try:
            content = self.log_path(job_id).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(content.splitlines()[-max(1, lines):])

    # ------------------------------------------------------------------
    # Single-job lock
    # ------------------------------------------------------------------
    @property
    def lock_path(self) -> Path:
        return self.root / ".active.lock"

    def _read_lock(self) -> Optional[Dict[str, Any]]:
        data = read_json(self.lock_path)
        return data if isinstance(data, dict) else None

    def acquire_lock(self, job_id: str) -> None:
        """Reserve the single remote-job slot.

        Kaggle jobs are serialized locally because they share dataset names,
        the user's GPU quota and the helper's action buttons.
        """
        self.validate_job_id(job_id)
        lock = self._read_lock()
        if lock:
            pid = int(lock.get("pid") or 0)
            created = float(lock.get("created") or 0)
            stale_after = 24 * 3600
            if _pid_alive(pid) and (time.time() - created) < stale_after:
                raise JobBusyError(
                    "Another Kaggle job is already active: "
                    f"{lock.get('job_id')} (pid {pid}). Wait for it to finish."
                )
            try:
                self.lock_path.unlink()
            except OSError:
                pass

        payload = {
            "job_id": job_id,
            "pid": os.getpid(),
            "created": time.time(),
            "created_at": _utc_now(),
        }
        try:
            fd = os.open(
                str(self.lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise JobBusyError("Another Kaggle job acquired the lock. Try again.") from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception:
            try:
                self.lock_path.unlink()
            except OSError:
                pass
            raise

    def release_lock(self, job_id: Optional[str] = None) -> None:
        lock = self._read_lock()
        if not lock:
            return
        if job_id and lock.get("job_id") != job_id:
            return
        try:
            self.lock_path.unlink()
        except OSError:
            pass
