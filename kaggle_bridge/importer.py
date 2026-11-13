"""Merge Kaggle kernel output back into local YOLO labels."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .runner_lib import ApplyResult, apply_label_updates


class ImportError_(RuntimeError):
    """Raised when a kernel output is missing or unsafe."""


@dataclass
class ImportResult:
    changed: int
    boxes: int
    snapshot: Dict[str, List[str]]
    result: Dict[str, Any] = field(default_factory=dict)
    labels_path: Optional[Path] = None

    def summary(self) -> str:
        message = self.result.get("message") or ""
        pieces = [f"Merged {self.changed} label file(s), {self.boxes} box(es)."]
        if message:
            pieces.append(str(message))
        return " ".join(pieces)


def _find_file(root: Path, name: str) -> Optional[Path]:
    direct = root / name
    if direct.is_file():
        return direct
    matches = sorted(
        (path for path in root.rglob(name) if path.is_file()),
        key=lambda path: (len(path.parts), str(path)),
    )
    return matches[0] if matches else None


def _read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise ImportError_(f"Cannot read {path}: {exc}") from exc


def import_job_output(
    job_dir: Path,
    images: Iterable[str],
    classes: Iterable[str],
    labels_dir: Optional[Path] = None,
) -> ImportResult:
    """Read ``output/labels.json`` and apply it to the local label folder."""
    job_dir = Path(job_dir)
    output_dir = job_dir / "output"
    if not output_dir.is_dir():
        raise ImportError_(f"Job output folder not found: {output_dir}")

    result_path = _find_file(output_dir, "result.json")
    result: Dict[str, Any] = {}
    if result_path is not None:
        raw_result = _read_json(result_path)
        if isinstance(raw_result, dict):
            result = raw_result

    if result.get("status") == "error":
        message = result.get("error") or result.get("message") or "kernel failed"
        raise ImportError_(f"Kaggle kernel reported an error: {message}")

    if result.get("job_id") and result.get("job_id") != job_dir.name:
        raise ImportError_(
            f"Output belongs to job {result.get('job_id')!r}, expected {job_dir.name!r}"
        )

    labels_path = _find_file(output_dir, "labels.json")
    if labels_path is None:
        if result and result.get("changed", 0) == 0:
            return ImportResult(0, 0, {}, result, None)
        raise ImportError_(
            "labels.json was not found in the Kaggle output. "
            "Check the kernel log for an early failure."
        )

    updates = _read_json(labels_path)
    if not isinstance(updates, dict):
        raise ImportError_("labels.json must contain an object mapping image names to labels")

    if labels_dir is None:
        labels_dir = job_dir.parent.parent / "labels"
    image_set = set(images)
    class_list = list(classes)
    applied: ApplyResult = apply_label_updates(
        Path(labels_dir),
        updates,
        image_set,
        num_classes=len(class_list) if class_list else None,
    )

    return ImportResult(
        changed=applied.changed,
        boxes=applied.boxes,
        snapshot=applied.snapshot,
        result=result,
        labels_path=labels_path,
    )


def load_import_metadata(job_dir: Path) -> Dict[str, Any]:
    """Read the downloaded result (if any) without applying labels."""
    output_dir = Path(job_dir) / "output"
    result_path = _find_file(output_dir, "result.json")
    if result_path is None:
        return {}
    data = _read_json(result_path)
    return data if isinstance(data, dict) else {}
