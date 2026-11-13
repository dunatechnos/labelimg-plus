"""Kaggle dataset/kernel metadata generation."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict

ACCELERATOR_T4_X2 = "NvidiaTeslaT4"
DEFAULT_LICENSE = {"name": "CC0-1.0"}
DEFAULT_KERNEL_PREFIX = "labelimg-plus-run"

_OWNER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")


def slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return re.sub(r"-{2,}", "-", value).strip("-")


def validate_slug(slug: str, what: str = "slug") -> str:
    if not _SLUG_RE.match(slug or ""):
        raise ValueError(
            f"{what} must be lowercase letters/digits/hyphens and 2-63 chars: {slug!r}"
        )
    return slug


def job_slug(prefix: str, job_id: str) -> str:
    prefix = slugify(prefix) or "labelimg-plus-job"
    job_id = slugify(job_id)
    slug = f"{prefix}-{job_id}"[:50].rstrip("-")
    return validate_slug(slug, "dataset/kernel slug")


def job_title(job_id: str) -> str:
    # Kaggle requires 6..50 chars for dataset titles and >=5 for kernels.
    title = f"LabelImg Plus Job {job_id}"[:50].rstrip()
    if len(title) < 6:
        title = f"LabelImg Plus {job_id}"[:50]
    return title


def kernel_title(job_id: str) -> str:
    """Kernel title whose slug matches :func:`kernel_slug`."""
    title = f"LabelImg Plus Run {job_id}"[:50].rstrip()
    if len(title) < 5:
        title = f"LabelImg Run {job_id}"[:50]
    return title


def kernel_id_for(username: str, job_id: str, prefix: str = DEFAULT_KERNEL_PREFIX) -> str:
    owner = validate_slug((username or "").strip().lower(), "Kaggle username")
    return f"{owner}/{job_slug(prefix, job_id)}"


def dataset_id_for(username: str, prefix: str, job_id: str) -> str:
    owner = validate_slug((username or "").strip().lower(), "Kaggle username")
    return f"{owner}/{job_slug(prefix, job_id)}"


def dataset_metadata(dataset_id: str, title: str) -> Dict[str, Any]:
    owner, slug = dataset_id.split("/", 1)
    validate_slug(owner, "dataset owner")
    validate_slug(slug, "dataset slug")
    title = (title or "")[:50]
    if not (6 <= len(title) <= 50):
        raise ValueError(f"dataset title must be 6..50 chars, got {len(title)}")
    return {
        "title": title,
        "id": dataset_id,
        "licenses": [dict(DEFAULT_LICENSE)],
    }


def kernel_metadata(
    kernel_id: str,
    title: str,
    dataset_id: str,
    code_file: str = "runner.py",
    accelerator: str = ACCELERATOR_T4_X2,
    enable_internet: bool = True,
) -> Dict[str, Any]:
    owner, slug = kernel_id.split("/", 1)
    validate_slug(owner, "kernel owner")
    validate_slug(slug, "kernel slug")
    if len(title or "") < 5:
        raise ValueError("kernel title must be at least 5 characters")
    if "/" not in dataset_id:
        raise ValueError(f"invalid dataset id: {dataset_id!r}")
    return {
        "id": kernel_id,
        "title": title[:50],
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": bool(enable_internet),
        "machine_shape": accelerator,
        "dataset_sources": [dataset_id],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def write_json_atomic(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


def read_json(path: Path, default=None):
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default
