"""Batch label operations for the main LabelImg window.

Everything here works on YOLO ``.txt`` files and returns the original file
contents for the action undo stack.  The functions never touch the Qt widgets
directly, which keeps them easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

from kaggle_bridge.runner_lib import (
    normalize_label_lines,
    read_label_lines,
    write_label_lines_atomic,
)

from .context import DatasetContext, area_contains, box_matches_any

ProgressFn = Optional[Callable[[int, int, str], bool]]
Box = List[str]


@dataclass
class OperationResult:
    files_changed: int = 0
    boxes_changed: int = 0
    changed_names: List[str] = field(default_factory=list)
    snapshot: Dict[str, List[str]] = field(default_factory=dict)
    cancelled: bool = False

    @property
    def message(self) -> str:
        if self.cancelled:
            return (
                f"Cancelled. {self.boxes_changed} box(es) changed in "
                f"{self.files_changed} file(s)."
            )
        return (
            f"{self.boxes_changed} box(es) changed in "
            f"{self.files_changed} file(s)."
        )


def _progress(progress: ProgressFn, done: int, total: int, name: str) -> bool:
    if progress is None:
        return True
    try:
        return bool(progress(done, total, name))
    except Exception:
        return True


def _read(context: DatasetContext, name: str):
    path = context.label_path(name)
    raw = read_label_lines(path)
    return path, raw, normalize_label_lines(raw)


def _matches(
    parts: Sequence[str],
    classes: Optional[Set[int]],
    area: Optional[Dict[str, float]],
    selected_boxes: Optional[Iterable[Sequence[str]]],
) -> bool:
    if classes is not None:
        try:
            if int(float(parts[0])) not in classes:
                return False
        except (TypeError, ValueError, IndexError):
            return False
    if area and not area_contains(area, parts):
        return False
    if selected_boxes and not box_matches_any(parts, selected_boxes):
        return False
    return True


def _finish(
    context: DatasetContext,
    result: OperationResult,
    path: Path,
    name: str,
    raw: List[str],
    labels: List[Box],
) -> None:
    original_text = "".join(raw)
    new_text = "".join(" ".join(parts) + "\n" for parts in labels)
    if new_text == original_text:
        return
    result.snapshot[str(path.resolve())] = list(raw)
    write_label_lines_atomic(path, labels)
    result.files_changed += 1
    result.changed_names.append(name)


def delete_boxes(
    context: DatasetContext,
    names: Sequence[str],
    classes: Optional[Set[int]] = None,
    area: Optional[Dict[str, float]] = None,
    selected_boxes: Optional[Iterable[Sequence[str]]] = None,
    progress: ProgressFn = None,
) -> OperationResult:
    """Delete boxes matching the supplied filters.

    With no class/area/selected-box filter every box in the target files is
    deleted.  The dialog makes that explicit before calling this function.
    """

    result = OperationResult()
    boxes = list(selected_boxes or [])
    total = len(names)
    for done, name in enumerate(names, start=1):
        path, raw, labels = _read(context, name)
        kept: List[Box] = []
        removed = 0
        for parts in labels:
            if _matches(parts, classes, area, boxes):
                removed += 1
            else:
                kept.append(parts)
        if removed:
            _finish(context, result, path, name, raw, kept)
            result.boxes_changed += removed
        if not _progress(progress, done, total, name):
            result.cancelled = True
            break
    return result


def change_class(
    context: DatasetContext,
    names: Sequence[str],
    new_class: int,
    old_classes: Optional[Set[int]] = None,
    area: Optional[Dict[str, float]] = None,
    selected_boxes: Optional[Iterable[Sequence[str]]] = None,
    progress: ProgressFn = None,
) -> OperationResult:
    """Change the class index of matching boxes while keeping the geometry."""

    result = OperationResult()
    boxes = list(selected_boxes or [])
    total = len(names)
    for done, name in enumerate(names, start=1):
        path, raw, labels = _read(context, name)
        new_labels: List[Box] = []
        changed = 0
        for parts in labels:
            if _matches(parts, old_classes, area, boxes):
                try:
                    current = int(float(parts[0]))
                except (TypeError, ValueError):
                    current = None
                if current == int(new_class):
                    new_labels.append(parts)
                else:
                    new_labels.append([str(int(new_class))] + list(parts[1:5]))
                    changed += 1
            else:
                new_labels.append(parts)
        if changed:
            _finish(context, result, path, name, raw, new_labels)
            result.boxes_changed += changed
        if not _progress(progress, done, total, name):
            result.cancelled = True
            break
    return result


def propagate_boxes(
    context: DatasetContext,
    target_names: Sequence[str],
    boxes: Sequence[Sequence[str]],
    progress: ProgressFn = None,
) -> OperationResult:
    """Copy the current selected boxes into the target label files."""

    result = OperationResult()
    if not boxes:
        return result
    planned = [list(parts) for parts in boxes]
    total = len(target_names)
    for done, name in enumerate(target_names, start=1):
        path, raw, labels = _read(context, name)
        existing = {tuple(parts) for parts in labels}
        added = 0
        for parts in planned:
            key = tuple(parts)
            if key not in existing:
                labels.append(list(parts))
                existing.add(key)
                added += 1
        if added:
            _finish(context, result, path, name, raw, labels)
            result.boxes_changed += added
        if not _progress(progress, done, total, name):
            result.cancelled = True
            break
    return result
