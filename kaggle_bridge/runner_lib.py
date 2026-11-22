"""Shared, dependency-light label helpers.

This module is used by:

* the local importer (after a Kaggle job finishes), and
* the standalone runner executed inside the Kaggle kernel.

Keep it free of PyQt and of module-level heavy imports so it can be copied to
the kernel folder and imported there without the rest of labelImg Plus.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

LABEL_FIELDS = 5
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def parse_label_parts(parts) -> Optional[List[str]]:
    """Parse one YOLO line (string or sequence) into 5 normalized strings.

    Returns ``None`` for blank/invalid lines.  The class is emitted as an
    integer string and coordinates use six decimals.
    """
    if parts is None:
        return None
    if isinstance(parts, str):
        parts = parts.strip().split()
    try:
        values = list(parts)
    except TypeError:
        return None
    if len(values) < LABEL_FIELDS:
        return None
    try:
        cls = int(float(values[0]))
        coords = [float(values[i]) for i in range(1, LABEL_FIELDS)]
    except (TypeError, ValueError):
        return None
    if not all(abs(v) < 1e9 for v in coords):
        return None
    xc, yc, bw, bh = coords
    if bw <= 0.0 or bh <= 0.0:
        return None
    # Allow tiny float drift, then clamp.  Ultralytics can emit boxes a hair
    # outside the image; LabelImg expects normalized values in [0, 1].
    if not (-0.05 <= xc <= 1.05 and -0.05 <= yc <= 1.05):
        return None
    if not (-0.05 <= bw <= 1.05 and -0.05 <= bh <= 1.05):
        return None
    xc = min(1.0, max(0.0, xc))
    yc = min(1.0, max(0.0, yc))
    bw = min(1.0, max(1e-6, bw))
    bh = min(1.0, max(1e-6, bh))
    return [
        str(cls),
        f"{xc:.6f}",
        f"{yc:.6f}",
        f"{bw:.6f}",
        f"{bh:.6f}",
    ]


def normalize_label_lines(lines: Iterable) -> List[List[str]]:
    """Return valid normalized labels, preserving order, dropping bad lines."""
    out: List[List[str]] = []
    for line in lines:
        parsed = parse_label_parts(line)
        if parsed is not None:
            out.append(parsed)
    return out


def format_label_line(parts: Sequence) -> str:
    return " ".join(str(p) for p in parts)


def read_label_lines(path: Path) -> List[str]:
    """Read raw label lines (newlines preserved), matching helper behaviour."""
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            return handle.readlines()
    except OSError:
        return []


def write_label_lines_atomic(path: Path, labels: Iterable[Sequence]) -> None:
    """Atomically write YOLO labels; safe if the process is interrupted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for parts in labels:
                handle.write(format_label_line(parts) + "\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise


def mask_to_yolo_label(mask, cls, width: int, height: int) -> Optional[List[str]]:
    """Convert a binary SAM 2 mask to ``[cls, xc, yc, w, h]``.

    ``numpy`` is imported lazily because the local app may not need masks in
    every code path.
    """
    import numpy as np  # local import

    if width <= 0 or height <= 0:
        return None
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
    xc = (x1 + x2) / 2.0 / float(width)
    yc = (y1 + y2) / 2.0 / float(height)
    return [str(int(cls)), f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"]


def yolo_iou(a: Sequence, b: Sequence) -> float:
    """IoU for two YOLO lines (normalized ``xc yc w h``)."""
    try:
        ax1 = float(a[1]) - float(a[3]) / 2.0
        ax2 = float(a[1]) + float(a[3]) / 2.0
        ay1 = float(a[2]) - float(a[4]) / 2.0
        ay2 = float(a[2]) + float(a[4]) / 2.0
        bx1 = float(b[1]) - float(b[3]) / 2.0
        bx2 = float(b[1]) + float(b[3]) / 2.0
        by1 = float(b[2]) - float(b[4]) / 2.0
        by2 = float(b[2]) + float(b[4]) / 2.0
    except (IndexError, TypeError, ValueError):
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def merge_tracked_labels(
    original_raw_lines: Iterable,
    tracked_labels: Iterable,
    iou_thresh: float = 0.5,
) -> Tuple[List[List[str]], bool]:
    """Merge SAM 2 boxes into existing labels.

    Same policy as the local helper:

    * other classes are preserved,
    * a same-class existing box with IoU > ``iou_thresh`` is replaced,
    * non-overlapping same-class boxes are preserved,
    * tracked boxes are appended.
    """
    original = normalize_label_lines(original_raw_lines)
    tracked = normalize_label_lines(tracked_labels)
    tracked_meta = [
        (int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        for parts in tracked
    ]

    final: List[List[str]] = []
    for parts in original:
        cls = int(parts[0])
        remove = False
        for tcls, txc, tyc, tw, th in tracked_meta:
            if cls != tcls:
                continue
            candidate = [str(tcls), f"{txc:.6f}", f"{tyc:.6f}", f"{tw:.6f}", f"{th:.6f}"]
            if yolo_iou(parts, candidate) > iou_thresh:
                remove = True
                break
        if not remove:
            final.append(list(parts))
    final.extend([list(parts) for parts in tracked])
    return final, final != original


def safe_basename(name: str) -> str:
    """Validate a filename that came from a remote job."""
    if not isinstance(name, str):
        raise ValueError("label key is not a string")
    name = name.strip()
    if not name or name in (".", ".."):
        raise ValueError("empty label key")
    if "\x00" in name or "/" in name or "\\" in name:
        raise ValueError(f"unsafe label key: {name!r}")
    if Path(name).name != name:
        raise ValueError(f"unsafe label key: {name!r}")
    if re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"unsafe label key: {name!r}")
    return name


def remap_class_by_name(
    model_names,
    model_cls_idx: int,
    name_to_idx: Dict[str, int],
) -> Optional[int]:
    """Map a model's class index to our ``classes.txt`` by class name."""
    name = None
    if isinstance(model_names, dict):
        name = model_names.get(model_cls_idx)
        if name is None:
            name = model_names.get(str(model_cls_idx))
    elif isinstance(model_names, (list, tuple)):
        if 0 <= model_cls_idx < len(model_names):
            name = model_names[model_cls_idx]
    if name is None:
        return None
    return name_to_idx.get(str(name).strip().lower())


@dataclass
class ApplyResult:
    snapshot: Dict[str, List[str]]
    changed: int
    boxes: int
    changed_names: List[str] = field(default_factory=list)


def apply_label_updates(
    labels_dir: Path,
    updates: Dict[str, Iterable],
    images: Set[str],
    num_classes: Optional[int] = None,
) -> ApplyResult:
    """Apply remote label updates to the local label folder.

    ``updates`` maps an original image filename (e.g. ``frame_0001.jpg``) to a
    list of YOLO labels.  Only files that actually change are written and
    included in the undo snapshot.
    """
    labels_dir = Path(labels_dir).resolve()
    labels_dir.mkdir(parents=True, exist_ok=True)
    snapshot: Dict[str, List[str]] = {}
    changed = 0
    boxes = 0
    changed_names: List[str] = []

    if not isinstance(updates, dict):
        raise ValueError("labels.json must contain an object mapping image names to labels")

    for raw_name, raw_labels in updates.items():
        name = safe_basename(raw_name)
        if name not in images:
            raise ValueError(f"remote result references an image that is not in this batch: {name!r}")
        if not isinstance(raw_labels, (list, tuple)):
            raise ValueError(f"labels for {name!r} must be a list")
        parsed = normalize_label_lines(raw_labels)
        if num_classes is not None:
            for parts in parsed:
                cls = int(parts[0])
                if cls < 0 or cls >= num_classes:
                    raise ValueError(
                        f"class index {cls} for {name!r} is outside classes.txt "
                        f"(0..{num_classes - 1})"
                    )
        label_path = (labels_dir / (Path(name).stem + ".txt")).resolve()
        try:
            label_path.relative_to(labels_dir)
        except ValueError as exc:
            raise ValueError(f"label path escapes labels dir: {name!r}") from exc

        original = read_label_lines(label_path)
        original_norm = normalize_label_lines(original)
        if parsed == original_norm:
            continue
        snapshot[str(label_path)] = original
        write_label_lines_atomic(label_path, parsed)
        changed += 1
        boxes += len(parsed)
        changed_names.append(name)

    return ApplyResult(
        snapshot=snapshot,
        changed=changed,
        boxes=boxes,
        changed_names=changed_names,
    )
