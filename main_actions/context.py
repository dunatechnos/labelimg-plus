from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from libs.constants import FORMAT_YOLO
from libs.labelFile import LabelFileFormat
from annotation_helper.annotation_helper_current import (
    list_images,
    read_classes,
    set_paths,
)
from kaggle_bridge.runner_lib import (
    normalize_label_lines,
    read_label_lines,
    write_label_lines_atomic,
    yolo_iou,
)


@dataclass
class DatasetContext:
    window: object
    images_dir: Path
    labels_dir: Path
    models_dir: Path
    image_paths: List[str]
    image_names: List[str]
    classes: List[str]
    current_index: int  # 0-based

    @property
    def current_name(self) -> str:
        if 0 <= self.current_index < len(self.image_names):
            return self.image_names[self.current_index]
        return ""

    @property
    def total(self) -> int:
        return len(self.image_names)

    def label_path(self, name: str) -> Path:
        return self.labels_dir / (Path(name).stem + ".txt")

    def names_in_range(self, start_1based: int, end_1based: int) -> List[str]:
        start = max(1, int(start_1based))
        end = min(self.total, int(end_1based))
        if end < start:
            return []
        return list(self.image_names[start - 1:end])

    def class_index(self, name: str) -> int:
        try:
            return self.classes.index(name)
        except ValueError:
            # A label typed in the main window may not be in the classes list
            # until the current frame is saved.  Fall back to label_hist so
            # selected-box actions keep working in that moment.
            try:
                return list(getattr(self.window, "label_hist", []) or []).index(name)
            except ValueError:
                return -1

    def selected_shapes(self) -> List[object]:
        canvas = self.window.canvas
        shapes = list(getattr(canvas, "selected_shapes", []) or [])
        if not shapes and getattr(canvas, "selected_shape", None) is not None:
            shapes = [canvas.selected_shape]
        return shapes

    def shape_to_box(self, shape) -> Optional[List[str]]:
        cls = self.class_index(shape.label)
        if cls < 0:
            return None
        canvas = self.window.canvas
        w = int(canvas.pixmap.width())
        h = int(canvas.pixmap.height())
        if w <= 0 or h <= 0 or not shape.points:
            return None
        xs = [p.x() for p in shape.points]
        ys = [p.y() for p in shape.points]
        x1 = max(0.0, min(float(w), min(xs)))
        x2 = max(0.0, min(float(w), max(xs)))
        y1 = max(0.0, min(float(h), min(ys)))
        y2 = max(0.0, min(float(h), max(ys)))
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        if bw <= 0 or bh <= 0:
            return None
        return [
            str(cls),
            f"{((x1 + x2) / 2.0 / w):.6f}",
            f"{((y1 + y2) / 2.0 / h):.6f}",
            f"{bw:.6f}",
            f"{bh:.6f}",
        ]

    def selected_boxes(self) -> List[List[str]]:
        boxes = []
        for shape in self.selected_shapes():
            box = self.shape_to_box(shape)
            if box is not None:
                boxes.append(box)
        return boxes

    @property
    def area(self) -> Optional[Dict[str, float]]:
        value = getattr(self.window.canvas, "area_value", None)
        return dict(value) if value else None

    def ensure_ready(self) -> None:
        """Switch the main app to YOLO and point helper globals at this batch."""
        if self.window.label_file_format != LabelFileFormat.YOLO:
            self.window.set_format(FORMAT_YOLO)
        self.window.default_save_dir = str(self.labels_dir)
        set_paths(str(self.images_dir), str(self.labels_dir), str(self.models_dir))
        classes_file = self.labels_dir / "classes.txt"
        if not classes_file.exists() and self.classes:
            classes_file.write_text(
                "\n".join(self.classes) + "\n", encoding="utf-8"
            )
        # The main window's YOLO writer uses label_hist for class indices.
        # Keep it in sync with classes.txt or saving the current frame would
        # renumber classes and corrupt batch edits.
        if self.classes:
            history = getattr(self.window, "label_hist", None)
            if history is not None and list(history) != list(self.classes):
                history[:] = list(self.classes)
            combo = getattr(self.window, "default_label_combo_box", None)
            if combo is not None and hasattr(combo, "cb"):
                box = combo.cb
                current = box.currentText()
                box.blockSignals(True)
                try:
                    box.clear()
                    box.addItems(list(self.classes))
                    index = box.findText(current)
                    box.setCurrentIndex(index if index >= 0 else 0)
                finally:
                    box.blockSignals(False)

    def save_current(self) -> bool:
        if not self.current_name:
            return False
        self.ensure_ready()
        target = self.labels_dir / Path(self.current_name).stem
        ok = self.window.save_labels(str(target))
        if ok:
            self.window.set_clean()
        return bool(ok)

    def reload(self) -> None:
        self.window.default_save_dir = str(self.labels_dir)
        area = self.area
        if self.window.file_path:
            self.window.load_file(self.window.file_path)
            if area:
                self.window.canvas.set_area_value(area)

    def snapshot(self, names: Iterable[str]) -> Dict[str, List[str]]:
        snap = {}
        for name in names:
            path = self.label_path(name)
            snap[str(path.resolve())] = read_label_lines(path)
        return snap

    def restore(self, snapshot: Dict[str, List[str]]) -> None:
        for path, lines in snapshot.items():
            write_label_lines_atomic(Path(path), [line.strip().split() for line in lines if line.strip()])


def build_context(window) -> DatasetContext:
    images_dir, labels_dir, models_dir = window.get_helper_paths()
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    models_dir = Path(models_dir)

    image_paths = list(getattr(window, "m_img_list", []) or [])
    if not image_paths and getattr(window, "file_path", None):
        image_paths = [window.file_path]
    if image_paths:
        actual_dir = Path(image_paths[0]).resolve().parent
        if actual_dir != images_dir.resolve():
            # A single file or an arbitrary folder was opened.  Do not act on
            # a sibling "images" folder that happens to exist.
            images_dir = actual_dir
            if getattr(window, "default_save_dir", None):
                labels_dir = Path(window.default_save_dir).resolve()
            else:
                labels_dir = actual_dir
    if not image_paths:
        set_paths(str(images_dir), str(labels_dir), str(models_dir))
        image_paths = [str(images_dir / name) for name in list_images()]
    image_names = [os.path.basename(path) for path in image_paths]

    current_path = getattr(window, "file_path", None)
    current_index = 0
    if current_path:
        try:
            current_index = image_paths.index(current_path)
        except ValueError:
            base = os.path.basename(current_path)
            if base in image_names:
                current_index = image_names.index(base)
    if image_names:
        current_index = max(0, min(current_index, len(image_names) - 1))

    classes_file = labels_dir / "classes.txt"
    classes: List[str] = []
    if classes_file.exists():
        try:
            classes = [
                line.strip()
                for line in classes_file.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
        except OSError:
            classes = []
    if not classes:
        classes = list(getattr(window, "label_hist", []) or [])
    if not classes:
        set_paths(str(images_dir), str(labels_dir), str(models_dir))
        classes = read_classes()
    return DatasetContext(
        window=window,
        images_dir=images_dir,
        labels_dir=labels_dir,
        models_dir=models_dir,
        image_paths=image_paths,
        image_names=image_names,
        classes=classes,
        current_index=current_index,
    )


def area_contains(area: Optional[Dict[str, float]], parts: Sequence) -> bool:
    """True when a normalized YOLO box overlaps the selected area."""
    if not area:
        return True
    try:
        xc, yc, bw, bh = map(float, parts[1:5])
    except (IndexError, TypeError, ValueError):
        return False
    x1, x2 = xc - bw / 2.0, xc + bw / 2.0
    y1, y2 = yc - bh / 2.0, yc + bh / 2.0
    return not (
        x2 < area["xmin"]
        or x1 > area["xmax"]
        or y2 < area["ymin"]
        or y1 > area["ymax"]
    )


def box_matches_any(parts: Sequence, boxes: Iterable[Sequence], iou_thresh: float = 0.5) -> bool:
    try:
        cls = int(float(parts[0]))
    except (IndexError, TypeError, ValueError):
        return False
    for box in boxes:
        try:
            if int(float(box[0])) != cls:
                continue
        except (IndexError, TypeError, ValueError):
            continue
        if yolo_iou(parts, box) > iou_thresh:
            return True
    return False


def normalize_lines(lines: Iterable) -> List[List[str]]:
    return normalize_label_lines(lines)
