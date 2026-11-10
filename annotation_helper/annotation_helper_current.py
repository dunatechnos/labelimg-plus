"""
Modified annotation tool UI (PyQt5)
- Right sidebar removed
- Modern left-side flexible vertical panel with all controls
- Image area kept large
- Zoom buttons removed (mouse-wheel zoom and pan still work)
- All original functionality preserved
"""

import os
import re
import sys
import cv2
import time
import shutil
import tempfile
import traceback
import urllib.request
import urllib.error
import numpy as np
from functools import partial
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem,
    QSpinBox, QComboBox, QCheckBox, QListWidget, QListWidgetItem,
    QLineEdit, QFileDialog, QMessageBox, QShortcut, QDialog,
    QDialogButtonBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QProgressBar, QRadioButton, QGroupBox, QScrollArea, QFrame,
    QGraphicsDropShadowEffect
)

from PyQt5.QtGui import QImage, QPixmap, QPen, QColor, QKeySequence, QFont
from PyQt5.QtCore import Qt, QRectF, QPointF, QThread, pyqtSignal

# ultralytics YOLO (ensure installed)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # Prediction features will check and alert if unavailable
def set_paths(images_dir=None, labels_dir=None, models_dir=None):
    global IMAGES_DIR, LABELS_DIR, CLASSES_FILE, MODELS_DIR

    if images_dir is not None:
        IMAGES_DIR = images_dir
        os.makedirs(IMAGES_DIR, exist_ok=True)

    if labels_dir is not None:
        LABELS_DIR = labels_dir
        CLASSES_FILE = os.path.join(labels_dir, "classes.txt")
        os.makedirs(LABELS_DIR, exist_ok=True)

    if models_dir is not None:
        MODELS_DIR = models_dir
        os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------- Config ----------------
IMAGES_DIR = "images"
LABELS_DIR = "labels"
CLASSES_FILE = os.path.join(LABELS_DIR, "classes.txt")
MODELS_DIR = "models"
UNDO_MAX = 20  # max undo stack depth

# ---- SAM 2 video tracking ----
HELPER_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_CHECKPOINT = os.path.join(HELPER_DIR, "sam2_models", "sam2.1_hiera_small.pt")
SAM2_CONFIG_FILE = "configs/sam2.1/sam2.1_hiera_s.yaml"
# Official Meta checkpoint (~180 MB). Downloaded automatically on first use,
# so end users never need to copy any .pt file manually.
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_small.pt"
)
SAM2_CHECKPOINT_MIN_BYTES = 100 * 1024 * 1024  # sanity check for partial downloads


def sam2_checkpoint_available():
    """True if the SAM 2 checkpoint exists and looks complete."""
    try:
        return (
            os.path.exists(SAM2_CHECKPOINT)
            and os.path.getsize(SAM2_CHECKPOINT) >= SAM2_CHECKPOINT_MIN_BYTES
        )
    except OSError:
        return False


def download_sam2_checkpoint(progress_cb=None):
    """Download the SAM 2 checkpoint to SAM2_CHECKPOINT.

    progress_cb(downloaded_bytes, total_bytes) is called as chunks arrive.
    Writes to a .part file first, then moves atomically into place.
    Raises RuntimeError on failure. Returns the checkpoint path.
    """
    os.makedirs(os.path.dirname(SAM2_CHECKPOINT), exist_ok=True)
    tmp_path = SAM2_CHECKPOINT + ".part"
    req = urllib.request.Request(
        SAM2_CHECKPOINT_URL, headers={"User-Agent": "labelimg-plus"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = resp.getheader("Content-Length")
            total = int(total) if total else 0
            downloaded = 0
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb is not None:
                        try:
                            progress_cb(downloaded, total)
                        except Exception:
                            pass
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(
            "Could not download the SAM 2 model (~180 MB).\n"
            "Check your internet connection and retry.\n"
            f"URL: {SAM2_CHECKPOINT_URL}\nDetails: {e}"
        )
    if os.path.getsize(tmp_path) < SAM2_CHECKPOINT_MIN_BYTES:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(
            "SAM 2 model download looks incomplete "
            "(file too small). Please retry."
        )
    os.replace(tmp_path, SAM2_CHECKPOINT)
    return SAM2_CHECKPOINT

# Create folders if not exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------- Helpers ----------------
def natural_key(s):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r'(\d+)', s)]

def list_images():
    files = [f for f in os.listdir(IMAGES_DIR) if not f.startswith(".") and os.path.isfile(os.path.join(IMAGES_DIR, f))]
    files = sorted(files, key=natural_key)
    return files

def read_classes():
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, "r", encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    return []

def read_yolo_labels(image_name):
    """Return list of lists (strings) for each label line."""
    txt = os.path.splitext(image_name)[0] + ".txt"
    path = os.path.join(LABELS_DIR, txt)
    out = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    out.append(parts[:5])
    return out

def write_yolo_labels(image_name, lines):
    txt = os.path.splitext(image_name)[0] + ".txt"
    path = os.path.join(LABELS_DIR, txt)
    with open(path, "w", encoding="utf-8") as f:
        for l in lines:
            # ensure all strings
            f.write(" ".join(map(str, l)) + "\n")

def scan_models():
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt") and os.path.isfile(os.path.join(MODELS_DIR, f))]
    files = sorted(files, key=natural_key)
    return files


# ---------------- SAM 2 label helpers ----------------
def read_label_lines(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
    return []


def write_label_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for parts in lines:
            f.write(" ".join(map(str, parts)) + "\n")


def mask_to_yolo_label(mask, cls, width, height):
    """Convert a binary SAM 2 mask to a YOLO [cls xc yc w h] line."""
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
    return [str(cls), f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"]


def yolo_iou(a, b):
    """IoU for two YOLO label lines (normalized xc yc w h)."""
    ax1 = float(a[1]) - float(a[3]) / 2.0
    ax2 = float(a[1]) + float(a[3]) / 2.0
    ay1 = float(a[2]) - float(a[4]) / 2.0
    ay2 = float(a[2]) + float(a[4]) / 2.0
    bx1 = float(b[1]) - float(b[3]) / 2.0
    bx2 = float(b[1]) + float(b[3]) / 2.0
    by1 = float(b[2]) - float(b[4]) / 2.0
    by2 = float(b[2]) + float(b[4]) / 2.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def merge_tracked_labels(original_raw_lines, tracked_labels, iou_thresh=0.5):
    """
    Replace same-class existing boxes that overlap a tracked box by > iou_thresh.
    All other existing labels (other classes, non-overlapping same class) are kept.
    """
    original = []
    for raw in original_raw_lines:
        parts = raw.strip().split()
        if parts:
            original.append(parts)

    tracked_meta = []
    for parts in tracked_labels:
        try:
            tracked_meta.append((
                int(float(parts[0])),
                float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]),
            ))
        except (TypeError, ValueError):
            continue

    final = []
    for parts in original:
        if len(parts) < 5:
            final.append(parts)
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
        except (TypeError, ValueError):
            final.append(parts)
            continue

        remove = False
        for tcls, txc, tyc, tw, th in tracked_meta:
            if cls != tcls:
                continue
            if yolo_iou(parts, [str(tcls), txc, tyc, tw, th]) > iou_thresh:
                remove = True
                break
        if not remove:
            final.append(parts)

    final.extend([list(x) for x in tracked_labels])
    changed = final != original
    return final, changed


# ---------------- QGraphics Box Item ----------------
class BBoxItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF, parts):
        super().__init__(rect)
        self.parts = parts  # ['cls','xc','yc','w','h'] strings
        self.setPen(QPen(Qt.red, 2))
        # selectable
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        # don't move (not dragging)
        self.setFlag(QGraphicsRectItem.ItemIsMovable, False)

    def set_highlight(self, v: bool):
        self.setPen(QPen(Qt.green if v else Qt.red, 2))

# ---------------- Predict Worker (QThread) ----------------
class PredictThread(QThread):
    progress = pyqtSignal(int)          # percent
    finished = pyqtSignal(int, str)     # (count_updated, message)
    error = pyqtSignal(str)
    snapshot = pyqtSignal(dict)         # emit files snapshot for undo

    def __init__(self, images, sel_area, model_obj, start_idx, end_idx):
        super().__init__()
        self.images = images
        self.sel_area = sel_area  # normalized coords or None
        self.model = model_obj
        self.start_idx = start_idx
        self.end_idx = end_idx

        # Build a name -> index map from classes.txt (our current class list)
        self.our_classes = read_classes()
        self.name_to_idx = {n.strip().lower(): i for i, n in enumerate(self.our_classes)}

        # Model's own index -> name map (from the loaded YOLO model)
        raw_names = getattr(self.model, "names", {}) or {}
        # ultralytics sometimes gives a dict {0:'name',...}, sometimes a list ['name',...]
        if isinstance(raw_names, dict):
            self.model_names = raw_names
        elif isinstance(raw_names, (list, tuple)):
            self.model_names = {i: n for i, n in enumerate(raw_names)}
        else:
            self.model_names = {}

        print("DEBUG our_classes:", self.our_classes)
        print("DEBUG model_names:", self.model_names)

        # track any predicted class names that couldn't be matched to classes.txt
        self.unmatched_names = set()

    def remap_cls(self, model_cls_idx):
        """
        Convert a class index coming from the model's own prediction
        (model_cls_idx, based on the model's training order) into the
        index used in our classes.txt, by matching on the class NAME.
        Falls back to the original index if the name isn't found
        (and records it in self.unmatched_names for a warning).
        """
        model_name = self.model_names.get(model_cls_idx, None)
        if model_name is None:
            # model has no name info for this index, keep as-is
            return model_cls_idx
        mapped = self.name_to_idx.get(str(model_name).strip().lower())
        if mapped is None:
            self.unmatched_names.add(model_name)
            return model_cls_idx  # fallback: keep original index
        print(f"DEBUG remap: model_idx={model_cls_idx} name={model_name} -> our_idx={mapped}")
        return mapped

    def run(self):
        updated = 0
        files_snapshot = {}  # collect original file contents for undo
        try:
            total = max(1, (self.end_idx - self.start_idx + 1))
            cnt = 0
            for i in range(self.start_idx, self.end_idx + 1):
                if i < 1 or i > len(self.images):
                    cnt += 1
                    self.progress.emit(int(cnt / total * 100))
                    continue
                name = self.images[i - 1]
                path = os.path.join(IMAGES_DIR, name)
                img = cv2.imread(path)
                if img is None:
                    cnt += 1
                    self.progress.emit(int(cnt / total * 100))
                    continue
                h, w = img.shape[:2]
                final = []

                # read original label file (store for undo if changed later)
                txt = os.path.splitext(name)[0] + ".txt"
                label_file = os.path.join(LABELS_DIR, txt)
                original_lines = []
                if os.path.exists(label_file):
                    with open(label_file, "r", encoding="utf-8") as f:
                        original_lines = f.readlines()

                if self.sel_area:
                    # compute pixel crop coords
                    xmin_px = int(self.sel_area["xmin"] * w)
                    ymin_px = int(self.sel_area["ymin"] * h)
                    xmax_px = int(self.sel_area["xmax"] * w)
                    ymax_px = int(self.sel_area["ymax"] * h)

                    # clamp
                    xmin_px = max(0, min(w - 1, xmin_px))
                    ymin_px = max(0, min(h - 1, ymin_px))
                    xmax_px = max(xmin_px + 1, min(w, xmax_px))
                    ymax_px = max(ymin_px + 1, min(h, ymax_px))

                    crop = img[ymin_px:ymax_px, xmin_px:xmax_px]
                    results = self.model(crop)

                    # keep labels outside area from existing file
                    kept = []
                    for p in read_yolo_labels(name):
                        cls, xc, yc, bw, bh = map(float, p)
                        bxmin = xc - bw/2
                        bxmax = xc + bw/2
                        bymin = yc - bh/2
                        bymax = yc + bh/2
                        A = self.sel_area
                        if (bxmax < A["xmin"] or bxmin > A["xmax"] or bymax < A["ymin"] or bymin > A["ymax"]):
                            kept.append(p)
                    final = kept

                    # parse predictions (tensor -> list)
                    for b in results[0].boxes:
                        cls = self.remap_cls(int(b.cls[0]))
                        x1, y1, x2, y2 = b.xyxy[0].tolist()   # safe conversion
                        # shift to original image coords (pixel)
                        x1 = x1 + xmin_px
                        x2 = x2 + xmin_px
                        y1 = y1 + ymin_px
                        y2 = y2 + ymin_px
                        xc = (x1 + x2) / 2 / w
                        yc = (y1 + y2) / 2 / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        final.append([str(cls), f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"] )

                else:
                    results = self.model(img)
                    for b in results[0].boxes:
                        cls = self.remap_cls(int(b.cls[0]))
                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        xc = (x1 + x2) / 2 / w
                        yc = (y1 + y2) / 2 / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        final.append([str(cls), f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"])

                # If labels changed, save original lines for undo and write new labels
                new_lines = [ " ".join(x) + "\n" for x in final ]
                if new_lines != original_lines:
                    # store snapshot only once per file
                    files_snapshot[label_file] = original_lines
                    write_yolo_labels(name, final)
                    updated += 1

                cnt += 1
                self.progress.emit(int(cnt / total * 100))

            # Emit snapshot for undo (main thread)
            self.snapshot.emit(files_snapshot)
            msg = f"Prediction done for {updated} files"
            if self.unmatched_names:
                msg += f" | WARNING: {len(self.unmatched_names)} class name(s) from model not found in classes.txt: {', '.join(sorted(self.unmatched_names))} (used model's original index for these)"
            self.finished.emit(updated, msg)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"Predict thread error: {e}\n{tb}")

# ---------------- SAM 2 Model Download Worker (QThread) ----------------
class Sam2DownloadThread(QThread):
    progress = pyqtSignal(int)          # percent
    status = pyqtSignal(str)
    done = pyqtSignal(str)              # checkpoint path
    failed = pyqtSignal(str)

    def run(self):
        try:
            if sam2_checkpoint_available():
                self.progress.emit(100)
                self.done.emit(SAM2_CHECKPOINT)
                return

            def _cb(downloaded, total):
                if total > 0:
                    pct = int(downloaded * 100 / total)
                    self.progress.emit(max(0, min(99, pct)))
                    mb_done = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    self.status.emit(
                        f"Downloading SAM 2 model... {mb_done:.0f}/{mb_total:.0f} MB"
                    )
                else:
                    self.status.emit(
                        f"Downloading SAM 2 model... {downloaded / (1024 * 1024):.0f} MB"
                    )

            self.status.emit("Downloading SAM 2 model (~180 MB, first use only)...")
            path = download_sam2_checkpoint(progress_cb=_cb)
            self.progress.emit(100)
            self.done.emit(path)
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


# ---------------- SAM 2 Tracking Worker (QThread) ----------------
class Sam2TrackThread(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    snapshot = pyqtSignal(dict)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, images, start_index, direction, boxes, images_dir, labels_dir, checkpoint, min_confidence=0.5):
        super().__init__()
        self.images = list(images)
        self.start_index = int(start_index)
        self.direction = direction
        self.boxes = [list(parts) for parts in boxes]
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.checkpoint = checkpoint
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))

    def run(self):
        tmp_dir = None
        predictor = None
        state = None
        files_snapshot = {}
        snapshot_sent = False
        updated_files = 0
        skipped_detections = 0
        seen_frames = set()
        try:
            try:
                import torch
            except ImportError:
                raise RuntimeError(
                    "PyTorch is not installed.\n"
                    "Install it from https://pytorch.org/get-started/locally/ "
                    "then retry."
                )
            try:
                from sam2.build_sam import build_sam2_video_predictor
            except ImportError:
                raise RuntimeError(
                    "The 'sam2' package is not installed.\n"
                    "Install it with:\n"
                    "  pip install hydra-core iopath\n"
                    "  $env:SAM2_BUILD_CUDA=\"0\"\n"
                    "  pip install -e <sam2-repo> --no-deps  (or pip install sam2)\n"
                    "then retry."
                )

            if not self.images:
                raise RuntimeError("No frames found.")
            # First use: fetch the checkpoint automatically so the user
            # never has to copy any .pt file manually.
            if not sam2_checkpoint_available():
                self.status.emit("SAM 2 model not found. Downloading (~180 MB)...")
                self.progress.emit(1)

                def _dl_cb(downloaded, total):
                    if total > 0:
                        self.progress.emit(max(1, min(4, int(downloaded * 4 / total))))

                download_sam2_checkpoint(progress_cb=_dl_cb)
                self.status.emit("SAM 2 model downloaded.")
                self.progress.emit(4)

            n = len(self.images)
            start = max(0, min(int(self.start_index) - 1, n - 1))
            direction = self.direction

            if direction == "forward":
                total = n - start
            elif direction == "backward":
                total = start + 1
            else:
                total = n
            total = max(1, total)

            valid_boxes = []
            for parts in self.boxes:
                if len(parts) < 5:
                    continue
                try:
                    cls = str(int(float(parts[0])))
                    xc, yc, bw, bh = map(float, parts[1:5])
                except (TypeError, ValueError):
                    continue
                valid_boxes.append((cls, xc, yc, bw, bh))
            if not valid_boxes:
                raise RuntimeError("No valid selected YOLO boxes.")

            # SAM 2's folder loader only accepts integer frame names, so mirror the
            # folder into a temporary 000000.jpg sequence in the same order as the UI.
            self.status.emit("Preparing frames for SAM 2...")
            tmp_dir = tempfile.mkdtemp(prefix=".sam2_track_", dir=self.images_dir)
            for i, name in enumerate(self.images):
                src = os.path.join(self.images_dir, name)
                dst = os.path.join(tmp_dir, f"{i:06d}.jpg")
                ext = os.path.splitext(src)[1].lower()
                if ext in (".jpg", ".jpeg"):
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                else:
                    img = cv2.imread(src)
                    if img is None:
                        raise RuntimeError("Cannot read frame: " + name)
                    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if i % 15 == 0:
                    self.status.emit(f"Preparing frames {i + 1}/{n}...")
                    self.progress.emit(max(1, int(3 * (i + 1) / n)))

            self.status.emit("Loading SAM 2 model...")
            self.progress.emit(4)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            predictor = build_sam2_video_predictor(
                SAM2_CONFIG_FILE,
                self.checkpoint,
                device=device,
                apply_postprocessing=True,
            )

            self.status.emit("Loading video frames...")
            self.progress.emit(5)
            state = predictor.init_state(
                tmp_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False,
            )
            width = int(state["video_width"])
            height = int(state["video_height"])

            obj_classes = {}
            for idx, (cls, xc, yc, bw, bh) in enumerate(valid_boxes):
                obj_id = idx + 1
                x1 = (xc - bw / 2.0) * width
                y1 = (yc - bh / 2.0) * height
                x2 = (xc + bw / 2.0) * width
                y2 = (yc + bh / 2.0) * height
                x1 = max(0.0, min(width - 1.0, x1))
                y1 = max(0.0, min(height - 1.0, y1))
                x2 = max(x1 + 1.0, min(float(width), x2))
                y2 = max(y1 + 1.0, min(float(height), y2))
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=start,
                    obj_id=obj_id,
                    box=[x1, y1, x2, y2],
                )
                obj_classes[obj_id] = cls

            def process_frame(frame_idx, obj_ids, video_res_masks):
                nonlocal updated_files, skipped_detections
                frame_idx = int(frame_idx)
                if frame_idx in seen_frames:
                    return
                seen_frames.add(frame_idx)

                tracked_labels = []
                for mask_idx, obj_id in enumerate(obj_ids):
                    cls = obj_classes.get(int(obj_id))
                    if cls is None:
                        continue

                    # SAM 2's object-presence score for this frame/object.
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
                                    score = float(
                                        score_logits.float().sigmoid().mean().item()
                                    )
                    except (KeyError, TypeError, ValueError, AttributeError):
                        score = None

                    if score is not None and score < self.min_confidence:
                        skipped_detections += 1
                        continue

                    mask = (video_res_masks[mask_idx, 0] > 0.0).cpu().numpy()
                    label = mask_to_yolo_label(mask, cls, width, height)
                    if label is not None:
                        tracked_labels.append(label)

                if tracked_labels and 0 <= frame_idx < len(self.images):
                    name = self.images[frame_idx]
                    label_file = os.path.join(
                        self.labels_dir, os.path.splitext(name)[0] + ".txt"
                    )
                    original_lines = read_label_lines(label_file)
                    merged, changed = merge_tracked_labels(original_lines, tracked_labels)
                    if changed:
                        if label_file not in files_snapshot:
                            files_snapshot[label_file] = original_lines
                        write_label_lines(label_file, merged)
                        updated_files += 1

                pct = int(6 + 93 * len(seen_frames) / total)
                self.progress.emit(min(99, max(6, pct)))
                self.status.emit(
                    f"Tracking {len(seen_frames)}/{total} frames  ·  "
                    f"{len(obj_classes)} object(s)  ·  "
                    f"skipped {skipped_detections} low-confidence"
                )

            if direction in ("forward", "entire"):
                for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
                    state,
                    start_frame_idx=start,
                    max_frame_num_to_track=n - 1 - start,
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
                    # Reverse tracking yields nothing at frame 0; process the prompt frame once.
                    for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
                        state,
                        start_frame_idx=0,
                        max_frame_num_to_track=0,
                    ):
                        process_frame(frame_idx, obj_ids, video_res_masks)

            if files_snapshot:
                snapshot_sent = True
                self.snapshot.emit(files_snapshot)
            self.progress.emit(100)
            self.status.emit("SAM 2 tracking finished.")
            self.done.emit(
                f"SAM 2 tracked {len(obj_classes)} object(s) across "
                f"{len(seen_frames)} frame(s); updated {updated_files} label file(s); "
                f"skipped {skipped_detections} low-confidence detection(s)."
            )
        except Exception as e:
            if files_snapshot and not snapshot_sent:
                self.snapshot.emit(files_snapshot)
            self.failed.emit(f"{e}\n{traceback.format_exc()}")
        finally:
            try:
                del state
            except Exception:
                pass
            try:
                del predictor
            except Exception:
                pass
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------- Select Images Dialog (for propagate) ----------------
class ChooseImagesDialog(QDialog):
    def __init__(self, image_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select target images")
        self.resize(400, 500)
        self.selected = []

        layout = QVBoxLayout()
        self.table = QTableWidget(len(image_list), 2)
        self.table.setHorizontalHeaderLabels(["Select", "Image"])
        self.table.verticalHeader().setVisible(False)
        self.table.setColumnWidth(0, 60)
        for i, name in enumerate(image_list):
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Unchecked)
            self.table.setItem(i, 0, chk_item)
            self.table.setItem(i, 1, QTableWidgetItem(name))
        layout.addWidget(self.table)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_selected_indices(self):
        sel = []
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item and item.checkState() == Qt.Checked:
                sel.append(r + 1)  # 1-based index
        return sel

# ---------------- Main Application ----------------
class AnnotationApp(QWidget):
    # Emitted when user wants to return to the host LabelImg page
    back_requested = pyqtSignal()

    def get_current_action(self):
        try:
            if self.radio_delete.isChecked():
                return "delete"
            if self.radio_propagate.isChecked():
                return "propagate"
            if self.radio_change.isChecked():
                return "change"
        except:
            pass
        return "delete"


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HUMAN VISION")
        self.resize(1400, 900)

        # state
        self.images = list_images()
        self.classes = read_classes()
        self.models = scan_models()   # filenames only
        self.model_cache = {}        # path->YOLO model object
        self.active_model_path = None
        self.active_model = None
        self.custom_target_indices = None    # <-- ADD THIS HERE
        self.sam2_thread = None


        self.current_index = 1 if self.images else 0
        self.current_image = None
        self.current_pixitem = None
        self.bbox_items = []         # list of BBoxItem
        self.selection_area = None   # normalized dict or None
        self.rubber = None
        self.dragging = False
        self.start_pt = None

        # undo stack: deque of dict {file_path: original_text_lines}
        self.undo_stack = deque(maxlen=UNDO_MAX)

        # Target images list (for propagate) - default None => use range

        # UI build
        self.build_ui()
        self.on_action_changed("delete")
        self.update_sam2_model_label()

        # keyboard shortcuts
        self.setup_shortcuts()

        # load initial
        if self.images:
            self.load_image(self.current_index)

    def build_ui(self):
        # root layout
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # ===================== LEFT PANEL (modern card) =====================
        left_panel_width = 380
        left_frame = QFrame()
        left_frame.setObjectName("leftPanel")
        left_frame.setFixedWidth(left_panel_width)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(18, 18, 18, 18)
        left_layout.setSpacing(14)

        # soft drop-shadow for the card
        shadow = QGraphicsDropShadowEffect(left_frame)
        shadow.setBlurRadius(30)
        shadow.setXOffset(0)
        shadow.setYOffset(6)
        shadow.setColor(QColor(31, 41, 55, 40))
        left_frame.setGraphicsEffect(shadow)

        # --- Header / brand ---
        title = QLabel("Annotation Helper")
        title.setObjectName("titleLabel")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        left_layout.addWidget(title)

        # --- Navigation Group ---
        nav_group = QGroupBox("Navigation")
        nav_group.setObjectName("cardGroup")
        nav_v = QVBoxLayout(nav_group)
        nav_v.setSpacing(8)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(8)
        nav_row.addWidget(QLabel("Index:"))
        self.spin_index = QSpinBox(); self.spin_index.setMinimum(1)
        self.spin_index.setMaximum(max(1, len(self.images))); self.spin_index.setValue(self.current_index or 1)
        nav_row.addWidget(self.spin_index)
        btn_load = QPushButton("Load")
        btn_load.setObjectName("secondaryButton")
        btn_load.clicked.connect(self.on_load_clicked)
        nav_row.addWidget(btn_load)
        nav_v.addLayout(nav_row)

        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("End:"))
        self.spin_end = QSpinBox(); self.spin_end.setMinimum(1)
        self.spin_end.setMaximum(max(1, len(self.images))); self.spin_end.setValue(len(self.images) or 1)
        end_row.addWidget(self.spin_end)
        nav_v.addLayout(end_row)

        # Select area mode
        self.chk_area_mode = QCheckBox("Select Area Mode")
        nav_v.addWidget(self.chk_area_mode)

        left_layout.addWidget(nav_group)

        # --- Actions Group ---
        act_group = QGroupBox("Actions")
        act_group.setObjectName("cardGroup")
        action_row = QVBoxLayout(act_group)
        action_row.setSpacing(8)

        # --- RADIO BUTTON ACTION SECTION (segmented style) ---
        radio_wrap = QFrame()
        radio_wrap.setObjectName("segmentWrap")
        radio_h = QHBoxLayout(radio_wrap)
        radio_h.setContentsMargins(4, 4, 4, 4)
        radio_h.setSpacing(4)

        self.radio_delete = QRadioButton("Delete")
        self.radio_propagate = QRadioButton("Propagate")
        self.radio_change = QRadioButton("Change Class")
        for r in (self.radio_delete, self.radio_propagate, self.radio_change):
            r.setObjectName("segmentRadio")
            r.setFont(QFont("Segoe UI", 8))

        # Default selection (same as old dropdown)
        self.radio_delete.setChecked(True)

        # Connect to your existing function
        self.radio_delete.toggled.connect(lambda checked: checked and self.on_action_changed("delete"))
        self.radio_propagate.toggled.connect(lambda checked: checked and self.on_action_changed("propagate"))
        self.radio_change.toggled.connect(lambda checked: checked and self.on_action_changed("change"))

        radio_h.addWidget(self.radio_delete)
        radio_h.addWidget(self.radio_propagate)
        radio_h.addWidget(self.radio_change)

        action_row.addWidget(radio_wrap)

        # Propagation options (visible only when Propagate is selected)
        self.propagation_options = QFrame()
        prop_v = QVBoxLayout(self.propagation_options)
        prop_v.setContentsMargins(0, 2, 0, 2)
        prop_v.setSpacing(8)

        prop_mode_wrap = QFrame()
        prop_mode_wrap.setObjectName("segmentWrap")
        prop_mode_h = QHBoxLayout(prop_mode_wrap)
        prop_mode_h.setContentsMargins(4, 4, 4, 4)
        prop_mode_h.setSpacing(4)

        self.radio_prop_static = QRadioButton("Static Propagation")
        self.radio_prop_sam2 = QRadioButton("Dynamic SAM 2")
        for r in (self.radio_prop_static, self.radio_prop_sam2):
            r.setObjectName("segmentRadio")
            r.setFont(QFont("Segoe UI", 8))
        self.radio_prop_static.setChecked(True)
        self.radio_prop_static.toggled.connect(
            lambda checked: checked and self.on_propagation_mode_changed()
        )
        self.radio_prop_sam2.toggled.connect(
            lambda checked: checked and self.on_propagation_mode_changed()
        )
        prop_mode_h.addWidget(self.radio_prop_static)
        prop_mode_h.addWidget(self.radio_prop_sam2)
        prop_v.addWidget(prop_mode_wrap)

        self.sam2_options = QFrame()
        sam2_opt_v = QVBoxLayout(self.sam2_options)
        sam2_opt_v.setContentsMargins(0, 0, 0, 0)
        sam2_opt_v.setSpacing(6)

        self.sam2_selected = QLabel("Selected boxes: 0")
        self.sam2_selected.setObjectName("statusLabel")
        sam2_opt_v.addWidget(self.sam2_selected)

        sam2_dir_row = QHBoxLayout()
        sam2_dir_row.addWidget(QLabel("SAM 2 Direction:"))
        self.combo_sam2_direction = QComboBox()
        self.combo_sam2_direction.addItems(["Backward", "Forward", "Entire video"])
        self.combo_sam2_direction.setCurrentIndex(2)
        sam2_dir_row.addWidget(self.combo_sam2_direction)
        sam2_opt_v.addLayout(sam2_dir_row)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Min confidence:"))
        self.spin_sam2_conf = QSpinBox()
        self.spin_sam2_conf.setRange(0, 100)
        self.spin_sam2_conf.setValue(50)
        self.spin_sam2_conf.setSingleStep(5)
        self.spin_sam2_conf.setSuffix("%")
        self.spin_sam2_conf.setToolTip(
            "SAM 2 object-presence score threshold. Lower-confidence detections "
            "are not written to YOLO labels. 0% disables filtering."
        )
        conf_row.addWidget(self.spin_sam2_conf)
        sam2_opt_v.addLayout(conf_row)

        self.sam2_status = QLabel(
            "Select boxes on the current frame, then Apply Action."
        )
        self.sam2_status.setObjectName("statusLabel")
        self.sam2_status.setWordWrap(True)
        sam2_opt_v.addWidget(self.sam2_status)

        # SAM 2 model row: status + one-click download (~180 MB, first use only)
        sam2_model_row = QHBoxLayout()
        sam2_model_row.setSpacing(6)
        self.sam2_model_status = QLabel()
        self.sam2_model_status.setObjectName("statusLabel")
        self.sam2_model_status.setWordWrap(True)
        sam2_model_row.addWidget(self.sam2_model_status, 1)
        self.btn_sam2_download = QPushButton("Download model")
        self.btn_sam2_download.setObjectName("secondaryButton")
        self.btn_sam2_download.setToolTip(
            "Download the SAM 2 tracking model (~180 MB, one time). "
            "It is fetched automatically when tracking starts."
        )
        self.btn_sam2_download.clicked.connect(self.on_sam2_download_clicked)
        sam2_model_row.addWidget(self.btn_sam2_download)
        sam2_opt_v.addLayout(sam2_model_row)
        self.sam2_download_thread = None
        self._sam2_autostart_after_download = False

        self.sam2_options.setVisible(False)
        prop_v.addWidget(self.sam2_options)
        self.propagation_options.setVisible(False)
        action_row.addWidget(self.propagation_options)

        # Old class dropdown (for delete/change)
        self.old_class_row = QFrame()
        line_old = QHBoxLayout(self.old_class_row)
        line_old.setContentsMargins(0, 0, 0, 0)
        self.combo_old_class = QComboBox()
        self.refresh_full_class_list()
        line_old.addWidget(QLabel("Old Class:"))
        line_old.addWidget(self.combo_old_class)
        action_row.addWidget(self.old_class_row)

        # Area classes dropdown (conditional, change only)
        self.area_class_row = QFrame()
        line_area_class = QHBoxLayout(self.area_class_row)
        line_area_class.setContentsMargins(0, 0, 0, 0)
        self.combo_area_class = QComboBox()
        self.combo_area_class.setVisible(False)
        line_area_class.addWidget(QLabel("Area Class:"))
        line_area_class.addWidget(self.combo_area_class)
        self.area_class_row.setVisible(False)
        action_row.addWidget(self.area_class_row)

        # New class (for change target)
        self.new_class_row = QFrame()
        line_new = QHBoxLayout(self.new_class_row)
        line_new.setContentsMargins(0, 0, 0, 0)
        self.combo_new_class = QComboBox()
        self.refresh_full_class_list()
        line_new.addWidget(QLabel("New Class:"))
        line_new.addWidget(self.combo_new_class)
        self.new_class_row.setVisible(False)
        action_row.addWidget(self.new_class_row)

        # Apply button centered
        self.btn_apply = QPushButton("Apply Action")
        self.btn_apply.setObjectName("primaryButton")
        self.btn_apply.setCursor(Qt.PointingHandCursor)
        self.btn_apply.clicked.connect(self.on_apply_action)
        action_row.addWidget(self.btn_apply)

        left_layout.addWidget(act_group)

        # --- Model / Predict Panel ---
        model_group = QGroupBox("Model & Predict")
        model_group.setObjectName("cardGroup")
        model_v = QVBoxLayout(model_group)
        model_v.setSpacing(8)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.combo_models = QComboBox()
        self.refresh_models()
        model_row.addWidget(self.combo_models)
        model_v.addLayout(model_row)

        reload_row = QHBoxLayout()
        btn_reload_models = QPushButton("Reload Models")
        btn_reload_models.setObjectName("secondaryButton")
        btn_reload_models.clicked.connect(self.on_reload_models)
        reload_row.addWidget(btn_reload_models)
        model_v.addLayout(reload_row)

        pred_row = QHBoxLayout()
        self.btn_predict = QPushButton("Predict")
        self.btn_predict.setObjectName("primaryButton")
        self.btn_predict.setCursor(Qt.PointingHandCursor)
        self.btn_predict.clicked.connect(self.on_predict_clicked)
        pred_row.addWidget(self.btn_predict)
        model_v.addLayout(pred_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        model_v.addWidget(self.progress_bar)

        left_layout.addWidget(model_group)

        # --- Selected boxes list (kept small) ---
        sel_group = QGroupBox("Selected Boxes")
        sel_group.setObjectName("cardGroup")
        sel_v = QVBoxLayout(sel_group)

        self.list_selected = QListWidget()
        self.list_selected.setFixedHeight(130)
        sel_v.addWidget(self.list_selected)

        left_layout.addWidget(sel_group)

        # Image path and copy
        path_row = QHBoxLayout()
        path_row.setSpacing(6)
        self.line_img_path = QLineEdit()
        self.line_img_path.setReadOnly(True)
        self.line_img_path.setPlaceholderText("Image path...")
        path_row.addWidget(self.line_img_path)
        btn_copy = QPushButton("Copy")
        btn_copy.setObjectName("iconButton")
        btn_copy.setToolTip("Copy image path")
        btn_copy.setFixedWidth(56)
        btn_copy.clicked.connect(self.copy_image_path)
        path_row.addWidget(btn_copy)
        left_layout.addLayout(path_row)

        # Undo / Status
        left_layout.addStretch()
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setObjectName("dangerButton")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.perform_undo)
        bottom_row.addWidget(self.btn_undo)
        self.label_status = QLabel("Ready")
        self.label_status.setObjectName("statusLabel")
        self.label_status.setWordWrap(True)
        bottom_row.addWidget(self.label_status, 1)
        left_layout.addLayout(bottom_row)

        # Put left frame inside a scroll area for better UX on smaller screens
        left_scroll = QScrollArea()
        left_scroll.setObjectName("leftScroll")
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_frame)
        left_scroll.setFixedWidth(left_panel_width + 8)
        left_scroll.setFrameShape(QFrame.NoFrame)

        # Graphics View (image) - fill remaining space and stay large
        self.scene = QGraphicsScene()
        self.scene.selectionChanged.connect(self.on_scene_selection_changed)
        self.view = QGraphicsView(self.scene)
        self.view.setObjectName("imageView")
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setRenderHints(self.view.renderHints())
        self.view.viewport().installEventFilter(self)

        # style the view to show a subtle background
        self.view.setBackgroundBrush(QColor(233, 236, 242))

        # wrap the view in a card frame for a consistent modern look
        view_frame = QFrame()
        view_frame.setObjectName("viewCard")
        view_layout = QVBoxLayout(view_frame)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(self.view)
        view_shadow = QGraphicsDropShadowEffect(view_frame)
        view_shadow.setBlurRadius(30)
        view_shadow.setXOffset(0)
        view_shadow.setYOffset(6)
        view_shadow.setColor(QColor(31, 41, 55, 35))
        view_frame.setGraphicsEffect(view_shadow)

        # add widgets to root
        root.addWidget(left_scroll, 0)
        root.addWidget(view_frame, 1)

        # ===================== Modern Style Sheet =====================
        self.setStyleSheet("""
            QWidget {
                background: #eef1f7;
                font-family: "Segoe UI";
                color: #1f2937;
                font-size: 10.5pt;
            }
            QScrollArea#leftScroll, QScrollArea#leftScroll > QWidget > QWidget {
                background: transparent;
                border: none;
            }
            #leftPanel {
                background: #ffffff;
                border-radius: 16px;
                border: 1px solid #e6e9f0;
            }
            #logoBadge {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6366f1, stop:1 #8b5cf6);
                border-radius: 10px;
                font-size: 16pt;
            }
            #titleLabel { color: #111827; }
            #subtitleLabel { color: #9ca3af; }

            QGroupBox#cardGroup {
                background: #f9fafc;
                border: 1px solid #e9ecf3;
                border-radius: 12px;
                margin-top: 10px;
                padding: 12px 10px 10px 10px;
                font-weight: 600;
                color: #374151;
            }
            QGroupBox#cardGroup::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #4b5563;
            }

            QLabel { color: #374151; }

            #segmentWrap {
                background: #eef0f6;
                border-radius: 10px;
            }
            QRadioButton#segmentRadio {
                padding: 6px 4px;
                border-radius: 8px;
            }
            QRadioButton#segmentRadio::indicator {
                width: 0px;
                height: 0px;
            }
            QRadioButton#segmentRadio:checked {
                background: #ffffff;
                color: #4f46e5;
                font-weight: 600;
            }

            QPushButton {
                padding: 7px 12px;
                border-radius: 8px;
                background: #ffffff;
                border: 1px solid #d7dbe4;
                color: #374151;
            }
            QPushButton:hover { background: #f3f4f8; }
            QPushButton:pressed { padding-left: 8px; padding-top: 8px; }

            QPushButton#primaryButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6366f1, stop:1 #7c3aed);
                color: white;
                border: none;
                font-weight: 600;
                padding: 9px 12px;
            }
            QPushButton#primaryButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5457e5, stop:1 #6d28d9);
            }
            QPushButton#secondaryButton {
                background: #f3f4f8;
                border: 1px solid #dfe2ea;
                color: #4b5563;
            }
            QPushButton#dangerButton {
                background: #fff1f2;
                border: 1px solid #fecdd3;
                color: #e11d48;
                font-weight: 600;
            }
            QPushButton#dangerButton:hover { background: #ffe4e6; }
            QPushButton#dangerButton:disabled {
                background: #f3f4f6; color: #9ca3af; border: 1px solid #e5e7eb;
            }
            QPushButton#iconButton {
                background: #f3f4f8;
                border: 1px solid #dfe2ea;
            }

            QLineEdit, QComboBox, QSpinBox, QListWidget {
                border: 1px solid #dfe2ea;
                border-radius: 8px;
                padding: 5px 8px;
                background: white;
                selection-background-color: #c7d2fe;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
                border: 1px solid #818cf8;
            }
            QListWidget { padding: 4px; }
            QListWidget::item { border-radius: 6px; padding: 3px; }
            QListWidget::item:selected { background: #eef0ff; color: #4338ca; }

            QCheckBox { spacing: 8px; }

            QProgressBar {
                border: none;
                border-radius: 7px;
                background: #e7e9f0;
                height: 14px;
            }
            QProgressBar::chunk {
                border-radius: 7px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6366f1, stop:1 #a855f7);
            }

            #statusLabel { color: #6b7280; font-size: 9.5pt; }

            #viewCard {
                background: #ffffff;
                border-radius: 16px;
                border: 1px solid #e6e9f0;
            }
            QGraphicsView#imageView {
                border: none;
                border-radius: 16px;
                background: #e9ecf2;
            }

            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #cbd0dc;
                border-radius: 5px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover { background: #b3b9c9; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)

    # ---------------- UI helpers ----------------
    def refresh_full_class_list(self):
        classes = read_classes()
        self.classes = classes
        items = [f"{i}:{n}" for i, n in enumerate(classes)]
        # set to old/new combos if present
        if hasattr(self, "combo_old_class"):
            self.combo_old_class.clear(); self.combo_old_class.addItems(items)
        if hasattr(self, "combo_new_class"):
            self.combo_new_class.clear(); self.combo_new_class.addItems(items)

    def refresh_models(self):
        self.models = scan_models()
        self.combo_models.clear()
        if self.models:
            self.combo_models.addItems(self.models)
        else:
            self.combo_models.addItem("(no models found)")

    def refresh_paths(self):
        """Reload datasets after caller changes IMAGES/LABELS/MODELS paths."""
        self.images = list_images()
        self.classes = read_classes()
        self.models = scan_models()
        self.refresh_full_class_list()
        self.refresh_models()
        max_count = max(1, len(self.images))
        self.spin_index.setMaximum(max_count)
        self.spin_end.setMaximum(max_count)
        if self.images:
            self.current_index = min(max(1, self.current_index or 1), len(self.images))
            self.spin_index.setValue(self.current_index)
            self.spin_end.setValue(len(self.images))
            self.load_image(self.current_index)
        else:
            self.current_index = 0
            self.spin_index.setValue(1)
            self.spin_end.setValue(1)
            self.scene.clear()
            self.label_status.setText("No images found.")

    def on_reload_models(self):
        self.refresh_models()
        self.label_status.setText("Models reloaded.")

    def on_load_clicked(self):
        idx = int(self.spin_index.value())
        self.load_image(idx)

    # ---------------- Image Loading / Drawing ----------------
    def safe_disconnect_scene(self):
        try:
            self.scene.selectionChanged.disconnect()
        except:
            pass

    def safe_connect_scene(self):
        try:
            self.scene.selectionChanged.connect(self.on_scene_selection_changed)
        except:
            pass

    def load_image(self, index):
        self.safe_disconnect_scene()
        self.scene.clear()
        self.bbox_items = []
        self.selection_area = None
        self.remove_rubber()
        self.list_selected.clear()
        self.copy_current_image_path("")

        self.images = list_images()
        if not self.images:
            self.label_status.setText("No images in images/ folder.")
            self.safe_connect_scene()
            return

        index = max(1, min(index, len(self.images)))
        self.current_index = index
        self.spin_index.setValue(index)
        self.spin_end.setMaximum(len(self.images))
        self.spin_index.setMaximum(len(self.images))
        self.spin_end.setValue(max(self.spin_end.value(), self.spin_end.minimum()))

        name = self.images[index - 1]
        self.current_image = name
        img_path = os.path.join(IMAGES_DIR, name)
        self.copy_current_image_path(img_path)

        img = cv2.imread(img_path)
        if img is None:
            self.label_status.setText("Failed to read image.")
            self.safe_connect_scene()
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.scene.setSceneRect(0, 0, w, h)
        self.current_pixitem = self.scene.addPixmap(pix)

        # draw bboxes from label file
        labels = read_yolo_labels(name)
        for parts in labels:
            try:
                cls = int(float(parts[0])); xc = float(parts[1]); yc = float(parts[2]); bw = float(parts[3]); bh = float(parts[4])
            except:
                continue
            x = (xc - bw/2) * w
            y = (yc - bh/2) * h
            W = bw * w
            H = bh * h
            rect = QRectF(x, y, W, H)
            item = BBoxItem(rect, parts)
            self.scene.addItem(item)
            self.bbox_items.append(item)

        # connect
        self.safe_connect_scene()
        self.update_sam2_selection_label()
        self.label_status.setText(f"Loaded {name} ({index}/{len(self.images)})")
        # fit by default
        self.fit_view()

    # ---------------- scene selection handler ----------------
    def on_scene_selection_changed(self):
        # update visuals and side list
        try:
            for it in self.bbox_items:
                it.set_highlight(it.isSelected())
        except RuntimeError:
            pass
        self.update_selected_panel()
        self.update_sam2_selection_label()
        # update area classes if in change
        if self.get_current_action() == "change" and self.selection_area:
            self.update_area_classes()

    def update_selected_panel(self):
        self.list_selected.clear()
        for it in self.bbox_items:
            if it.isSelected():
                cls_i = int(float(it.parts[0]))
                cls_name = self.classes[cls_i] if cls_i < len(self.classes) else str(cls_i)
                coords = f"{it.parts[1]} {it.parts[2]} {it.parts[3]} {it.parts[4]}"
                display = f"{cls_i}:{cls_name} | {coords}"
                lw = QListWidgetItem(display)
                lw.setData(Qt.UserRole, it)  # store ref
                self.list_selected.addItem(lw)
        self.copy_current_image_path(os.path.join(IMAGES_DIR, self.current_image) if self.current_image else "")

    def update_sam2_selection_label(self):
        try:
            count = sum(1 for it in self.bbox_items if it.isSelected())
            self.sam2_selected.setText(f"Selected boxes: {count}")
        except RuntimeError:
            pass
        except AttributeError:
            pass

    # ---------------- side-panel interactions ----------------
    def copy_current_image_path(self, path):
        self.line_img_path.setText(path)

    def copy_image_path(self):
        txt = self.line_img_path.text()
        if txt:
            QApplication.clipboard().setText(txt)
            self.label_status.setText("Image path copied to clipboard.")
        else:
            self.label_status.setText("No path to copy.")

    # ---------------- area selection via mouse events ----------------
    def eventFilter(self, obj, event):
        # handle zoom wheel and area selection and pan controls
        if obj is self.view.viewport():
            # wheel for zoom
            if event.type() == event.Wheel:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.zoom_view(1.15, center=event.pos())
                else:
                    self.zoom_view(1/1.15, center=event.pos())
                return True

            # mouse press for area selection or pan
            if event.type() == event.MouseButtonPress:
                button = event.button()
                pos = event.pos()
                scene_pos = self.view.mapToScene(pos)
                # Pan: Shift pressed or middle-button or Ctrl+Left
                if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) or button == Qt.MiddleButton or (event.modifiers() & Qt.KeyboardModifier.ControlModifier and button == Qt.LeftButton):
                    self.view.setDragMode(QGraphicsView.ScrollHandDrag)
                    self.view.viewport().setCursor(Qt.ClosedHandCursor)
                    return False  # let QGraphicsView handle pan
                # Area selection if area mode checked and left button
                if self.chk_area_mode.isChecked() and button == Qt.LeftButton:
                    if not self.scene.sceneRect().contains(scene_pos):
                        return False
                    self.dragging = True
                    self.start_pt = scene_pos
                    self.remove_rubber()
                    self.rubber = QGraphicsRectItem(QRectF(scene_pos, scene_pos))
                    pen = QPen(QColor(25, 118, 210))
                    pen.setStyle(Qt.DashLine)
                    pen.setWidth(2)
                    self.rubber.setPen(pen)
                    brush = QColor(25, 118, 210, 40)
                    self.rubber.setBrush(brush)
                    self.scene.addItem(self.rubber)
                    return True
            if event.type() == event.MouseMove:
                if self.dragging and self.rubber:
                    pos = event.pos()
                    scene_pos = self.view.mapToScene(pos)
                    self.rubber.setRect(QRectF(self.start_pt, scene_pos).normalized())
                    return True
            if event.type() == event.MouseButtonRelease:
                button = event.button()
                # finish pan
                if self.view.dragMode() == QGraphicsView.ScrollHandDrag:
                    self.view.setDragMode(QGraphicsView.NoDrag)
                    self.view.viewport().setCursor(Qt.ArrowCursor)
                    return False
                if self.dragging and button == Qt.LeftButton:
                    self.dragging = False
                    # finalize selection area
                    rect = self.rubber.rect().intersected(self.scene.sceneRect())
                    if rect.isNull() or rect.isEmpty():
                        self.remove_rubber()
                        self.selection_area = None
                        self.update_area_classes()
                        return True
                    W = self.scene.sceneRect().width(); H = self.scene.sceneRect().height()
                    self.selection_area = {"xmin": rect.left() / W, "ymin": rect.top() / H, "xmax": rect.right() / W, "ymax": rect.bottom() / H}
                    self.update_area_classes()
                    return True
        return False

    def remove_rubber(self):
        if self.rubber:
            try:
                self.scene.removeItem(self.rubber)
            except:
                pass
        self.rubber = None
        self.selection_area = None
        # hide area class row
        self.area_class_row.setVisible(False)
        if hasattr(self, "combo_area_class"):
            self.combo_area_class.setVisible(False)

    def update_area_classes(self):
        # compute classes inside selection_area and populate combo_area_class (and show)
        show_row = self.get_current_action() == "change"
        if not self.selection_area:
            self.area_class_row.setVisible(False)
            if hasattr(self, "combo_area_class"):
                self.combo_area_class.clear()
                self.combo_area_class.setVisible(False)
            self.label_status.setText("No area selected")
            return
        found = []
        for it in self.bbox_items:
            cls_i = int(float(it.parts[0]))
            xc = float(it.parts[1]); yc = float(it.parts[2]); bw = float(it.parts[3]); bh = float(it.parts[4])
            xmin = xc - bw/2; xmax = xc + bw/2; ymin = yc - bh/2; ymax = yc + bh/2
            A = self.selection_area
            if not (xmax < A["xmin"] or xmin > A["xmax"] or ymax < A["ymin"] or ymin > A["ymax"]):
                if cls_i not in found:
                    found.append(cls_i)
        self.area_class_row.setVisible(show_row)
        if found:
            items = [f"{i}:{self.classes[i] if i < len(self.classes) else str(i)}" for i in found]
            self.combo_area_class.clear()
            self.combo_area_class.addItems(items)
            self.combo_area_class.setVisible(show_row)
            self.label_status.setText("Area selected. Classes: " + ", ".join([self.classes[i] for i in found if i < len(self.classes)]))
        else:
            self.combo_area_class.clear()
            self.combo_area_class.setVisible(show_row)
            self.combo_area_class.addItem("(no classes in area)")
            self.label_status.setText("Area selected but no known classes inside.")

    # ---------------- zoom/pan helpers ----------------
    def zoom_view(self, factor, center=None):
        # center: QPoint viewport coords
        if center is None:
            center = self.view.viewport().rect().center()
        old_pos = self.view.mapToScene(center)
        self.view.scale(factor, factor)
        new_pos = self.view.mapToScene(center)
        delta = new_pos - old_pos
        self.view.translate(delta.x(), delta.y())



    def reset_zoom(self):
        self.view.resetTransform()
        self.fit_view()

    def fit_view(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    # ---------------- action change behavior ----------------
    def on_action_changed(self, text):
        is_propagate = text == "propagate"
        self.propagation_options.setVisible(is_propagate)
        self.old_class_row.setVisible(text in ("delete", "change"))
        self.new_class_row.setVisible(text == "change")

        if text == "change" and self.selection_area:
            self.update_area_classes()
        else:
            self.area_class_row.setVisible(False)
            self.combo_area_class.setVisible(False)

        if is_propagate:
            self.on_propagation_mode_changed()
        else:
            self.sam2_options.setVisible(False)
            self.btn_apply.setText("Apply Action")
            if self.label_status.text().startswith(
                ("Propagate", "Static propagation", "Dynamic SAM 2")
            ):
                self.label_status.setText("")

    def on_propagation_mode_changed(self):
        if not self.radio_propagate.isChecked():
            self.sam2_options.setVisible(False)
            self.btn_apply.setText("Apply Action")
            return
        dynamic = self.radio_prop_sam2.isChecked()
        self.sam2_options.setVisible(dynamic)
        self.btn_apply.setText("Start SAM 2 Tracking" if dynamic else "Apply Action")
        if dynamic:
            self.label_status.setText(
                "Dynamic SAM 2: select boxes, choose direction, then Apply Action."
            )
        else:
            self.label_status.setText(
                "Static propagation: select boxes, then Apply Action."
            )

    def update_apply_button_text(self):
        if self.radio_propagate.isChecked() and self.radio_prop_sam2.isChecked():
            self.btn_apply.setText("Start SAM 2 Tracking")
        else:
            self.btn_apply.setText("Apply Action")

    # ---------------- Apply action (delete/propagate/change) ----------------
    def on_apply_action(self):
        if self.sam2_thread is not None and self.sam2_thread.isRunning():
            self.label_status.setText("SAM 2 tracking is already running.")
            return
        act = self.get_current_action()
        start_idx = self.current_index
        end_idx = int(self.spin_end.value())
        if act == "delete":
            self.action_delete(start_idx, end_idx)
        elif act == "propagate":
            if self.radio_prop_sam2.isChecked():
                self.start_sam2_tracking()
            else:
                self.action_propagate(start_idx, end_idx)
        elif act == "change":
            self.action_change(start_idx, end_idx)
        else:
            self.label_status.setText("Unknown action.")

    # --------------- Undo helpers ---------------
    def push_undo_snapshot(self, files_changed_dict):
        # files_changed_dict: {file_path: original_lines_list}
        if not files_changed_dict:
            return
        # store minimal data: mapping filename->original text (list of lines)
        self.undo_stack.append(files_changed_dict.copy())
        self.btn_undo.setEnabled(True)
        self.label_status.setText("Undo snapshot taken.")

    def perform_undo(self):
        if not self.undo_stack:
            self.label_status.setText("Nothing to undo.")
            self.btn_undo.setEnabled(False)
            return
        snapshot = self.undo_stack.pop()
        for fp, lines in snapshot.items():
            # write back
            with open(fp, "w", encoding="utf-8") as f:
                for ln in lines:
                    f.write(ln.rstrip("\n") + "\n")
        self.label_status.setText("Undo applied.")
        # refresh current
        self.load_image(self.current_index)
        if not self.undo_stack:
            self.btn_undo.setEnabled(False)

    # --------------- Delete action ---------------
    def action_delete(self, start_idx, end_idx):
        A = self.selection_area if self.chk_area_mode.isChecked() else None

        target_class = None
        if A and self.combo_area_class.isVisible() and self.combo_area_class.count() and not self.combo_area_class.currentText().startswith("(no"):
            txt = self.combo_area_class.currentText()
            try:
                target_class = txt.split(":")[0]
            except:
                target_class = None

        if target_class is None:
            for it in self.bbox_items:
                if it.isSelected():
                    target_class = str(int(float(it.parts[0])))
                    break

        if target_class is None:
            if self.combo_old_class.count():
                target_class = self.combo_old_class.currentText().split(":")[0]
            else:
                target_class = None

        files_snapshot = {}
        images = list_images()
        deleted = 0
        for i in range(start_idx, end_idx + 1):
            if i < 1 or i > len(images):
                continue
            name = images[i - 1]
            label_file = os.path.join(LABELS_DIR, os.path.splitext(name)[0] + ".txt")
            original_lines = []
            if os.path.exists(label_file):
                with open(label_file, "r", encoding="utf-8") as f:
                    original_lines = f.readlines()
            labels = [ln.strip().split() for ln in original_lines if ln.strip()]
            new_labels = []
            for parts in labels:
                if not parts: continue
                cls = parts[0]
                if target_class is None and A is None:
                    new_labels.append(parts)
                elif target_class is None and A is not None:
                    x,y,w,h = map(float, parts[1:5])
                    xmin = x - w/2; xmax = x + w/2; ymin = y - h/2; ymax = y + h/2
                    if (xmax < A["xmin"] or xmin > A["xmax"] or ymax < A["ymin"] or ymin > A["ymax"]):
                        new_labels.append(parts)
                    else:
                        deleted += 1
                else:
                    if cls != target_class:
                        new_labels.append(parts)
                    else:
                        if A:
                            x,y,w,h = map(float, parts[1:5])
                            xmin = x - w/2; xmax = x + w/2; ymin = y - h/2; ymax = y + h/2
                            if (xmax < A["xmin"] or xmin > A["xmax"] or ymax < A["ymin"] or ymin > A["ymax"]):
                                new_labels.append(parts)
                            else:
                                deleted += 1
                        else:
                            deleted += 1
            if [ " ".join(x) + "\n" for x in new_labels ] != original_lines:
                files_snapshot[label_file] = original_lines
                write_yolo_labels(name, new_labels)
        if files_snapshot:
            self.push_undo_snapshot(files_snapshot)
        self.label_status.setText(f"Deleted {deleted} boxes across files.")
        self.load_image(self.current_index)

    # --------------- Propagate action ---------------
    def action_propagate(self, start_idx, end_idx):
        selected_boxes = [it.parts for it in self.bbox_items if it.isSelected()]
        if not selected_boxes:
            self.label_status.setText("No boxes selected to propagate.")
            return
        if self.custom_target_indices:
            target_indices = [i for i in self.custom_target_indices if i >= 1 and i <= len(self.images)]
        else:
            target_indices = list(range(start_idx + 1, end_idx + 1))
        if not target_indices:
            self.label_status.setText("No target images selected.")
            return

        files_snapshot = {}
        updated = 0
        for idx in target_indices:
            name = self.images[idx - 1]
            label_file = os.path.join(LABELS_DIR, os.path.splitext(name)[0] + ".txt")
            original_lines = []
            if os.path.exists(label_file):
                with open(label_file, "r", encoding="utf-8") as f:
                    original_lines = f.readlines()
            exist = [ln.strip().split() for ln in original_lines if ln.strip()]
            changed = False
            for lbl in selected_boxes:
                if lbl not in exist:
                    exist.append(lbl)
                    changed = True
            if changed:
                files_snapshot[label_file] = original_lines
                write_yolo_labels(name, exist)
                updated += 1

        if files_snapshot:
            self.push_undo_snapshot(files_snapshot)
        self.label_status.setText(f"Propagated to {updated} files.")
        self.load_image(self.current_index)

    def choose_target_images(self):
        dlg = ChooseImagesDialog(self.images, parent=self)
        if dlg.exec_():
            sel = dlg.get_selected_indices()  # list of 1-based indices
            if sel:
                self.custom_target_indices = sel
                self.lbl_tgt_info.setText(f"Custom targets: {len(sel)} images")
            else:
                self.custom_target_indices = None
                self.lbl_tgt_info.setText("Using range by default")
        else:
            pass

    # --------------- Class Change action ---------------
    def action_change(self, start_idx, end_idx):
        if self.combo_new_class.count() == 0 or self.combo_old_class.count() == 0:
            self.label_status.setText("Class list not available.")
            return
        try:
            new_cls = int(self.combo_new_class.currentText().split(":")[0])
        except:
            self.label_status.setText("Select valid new class.")
            return

        old_cls = None
        if self.chk_area_mode.isChecked() and self.combo_area_class.isVisible() and not self.combo_area_class.currentText().startswith("(no"):
            try:
                old_cls = int(self.combo_area_class.currentText().split(":")[0])
            except:
                old_cls = None

        if old_cls is None:
            for it in self.bbox_items:
                if it.isSelected():
                    old_cls = int(float(it.parts[0]))
                    break

        if old_cls is None:
            try:
                old_cls = int(self.combo_old_class.currentText().split(":")[0])
            except:
                self.label_status.setText("Select the class to replace (old class).")
                return

        A = self.selection_area if self.chk_area_mode.isChecked() else None

        files_snapshot = {}
        changed_total = 0
        for i in range(start_idx, end_idx + 1):
            if i < 1 or i > len(self.images):
                continue
            name = self.images[i - 1]
            label_file = os.path.join(LABELS_DIR, os.path.splitext(name)[0] + ".txt")
            original_lines = []
            if os.path.exists(label_file):
                with open(label_file, "r", encoding="utf-8") as f:
                    original_lines = f.readlines()
            labels = [ln.strip().split() for ln in original_lines if ln.strip()]
            new_labels = []
            changed = False
            for p in labels:
                cls = int(float(p[0]))
                if cls != old_cls:
                    new_labels.append(p)
                else:
                    if A:
                        x,y,w,h = map(float, p[1:5])
                        xmin = x - w/2; xmax = x + w/2; ymin = y - h/2; ymax = y + h/2
                        if (xmax < A["xmin"] or xmin > A["xmax"] or ymax < A["ymin"] or ymin > A["ymax"]):
                            new_labels.append(p)
                        else:
                            new_labels.append([str(new_cls), str(x), str(y), str(w), str(h)])
                            changed_total += 1
                            changed = True
                    else:
                        new_labels.append([str(new_cls), p[1], p[2], p[3], p[4]])
                        changed_total += 1
                        changed = True
            if changed:
                files_snapshot[label_file] = original_lines
                write_yolo_labels(name, new_labels)
        if files_snapshot:
            self.push_undo_snapshot(files_snapshot)
        self.label_status.setText(f"Changed {changed_total} labels.")
        self.load_image(self.current_index)

    # --------------- SAM 2 tracking action ---------------
    def start_sam2_tracking(self):
        if self.sam2_thread is not None and self.sam2_thread.isRunning():
            self.label_status.setText("SAM 2 tracking is already running.")
            return
        predict_thread = getattr(self, "predict_thread", None)
        if predict_thread is not None and predict_thread.isRunning():
            self.label_status.setText("YOLO prediction is running. Wait for it to finish.")
            return

        selected = [list(it.parts) for it in self.bbox_items if it.isSelected()]
        if not selected:
            self.sam2_status.setText("Select one or more boxes on the current frame first.")
            self.label_status.setText("SAM 2: no boxes selected.")
            return
        # First use: model (~180 MB) is fetched automatically. Ask once,
        # then download in the background and auto-start tracking.
        if not sam2_checkpoint_available():
            answer = QMessageBox.question(
                self,
                "Download SAM 2 model?",
                "The SAM 2 tracking model (~180 MB) is not downloaded yet.\n"
                "Download it now from the official Meta release?\n"
                "(One-time download, then tracking starts automatically.)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                self.sam2_status.setText(
                    "SAM 2 model download skipped. Tracking needs the model."
                )
                return
            self._sam2_autostart_after_download = True
            self.start_sam2_model_download()
            return

        images = list_images()
        if not images:
            self.label_status.setText("No frames found.")
            return

        direction = ["backward", "forward", "entire"][
            self.combo_sam2_direction.currentIndex()
        ]
        min_confidence = self.spin_sam2_conf.value() / 100.0
        self.sam2_thread = Sam2TrackThread(
            images=images,
            start_index=self.current_index,
            direction=direction,
            boxes=selected,
            images_dir=IMAGES_DIR,
            labels_dir=LABELS_DIR,
            checkpoint=SAM2_CHECKPOINT,
            min_confidence=min_confidence,
        )
        self.sam2_thread.progress.connect(self.on_sam2_progress)
        self.sam2_thread.status.connect(self.on_sam2_status)
        self.sam2_thread.snapshot.connect(self.on_sam2_snapshot)
        self.sam2_thread.done.connect(self.on_sam2_done)
        self.sam2_thread.failed.connect(self.on_sam2_failed)

        self.set_action_controls_enabled(False)
        self.btn_apply.setText("Tracking...")
        self.progress_bar.setValue(0)
        self.sam2_status.setText(
            f"Starting: {len(selected)} object(s), {direction}, "
            f"confidence >= {self.spin_sam2_conf.value()}%..."
        )
        self.label_status.setText(
            f"SAM 2: tracking {len(selected)} selected box(es) ({direction})..."
        )
        self.sam2_thread.start()

    def on_sam2_progress(self, value):
        self.progress_bar.setValue(int(value))

    def on_sam2_status(self, message):
        self.sam2_status.setText(message)
        self.label_status.setText(message)

    def on_sam2_snapshot(self, files_snapshot):
        try:
            if files_snapshot:
                self.push_undo_snapshot(files_snapshot)
        except Exception as e:
            print("on_sam2_snapshot error:", e)

    def set_action_controls_enabled(self, enabled):
        for widget in (
            self.btn_apply,
            self.btn_predict,
            self.radio_delete,
            self.radio_propagate,
            self.radio_change,
            self.radio_prop_static,
            self.radio_prop_sam2,
            self.combo_sam2_direction,
            self.spin_sam2_conf,
            self.btn_sam2_download,
        ):
            widget.setEnabled(enabled)
        self.update_apply_button_text()

    def on_sam2_done(self, message):
        self.progress_bar.setValue(100)
        self.set_action_controls_enabled(True)
        self.load_image(self.current_index)
        self.sam2_status.setText(message)
        self.label_status.setText(message)
        self.update_sam2_selection_label()

    def on_sam2_failed(self, message):
        self.progress_bar.setValue(0)
        self.set_action_controls_enabled(True)
        self.load_image(self.current_index)
        self.sam2_status.setText("SAM 2 tracking failed. Partial label changes can be undone.")
        self.label_status.setText("SAM 2 tracking failed. See console for traceback.")
        print(message)

    # --------------- SAM 2 model download ---------------
    def update_sam2_model_label(self):
        """Show whether the SAM 2 model is ready or needs downloading."""
        try:
            if sam2_checkpoint_available():
                size_mb = os.path.getsize(SAM2_CHECKPOINT) / (1024 * 1024)
                self.sam2_model_status.setText(
                    f"SAM 2 model: ready ({size_mb:.0f} MB)"
                )
                self.btn_sam2_download.setVisible(False)
            else:
                self.sam2_model_status.setText(
                    "SAM 2 model: not downloaded (~180 MB, auto-downloads)"
                )
                self.btn_sam2_download.setVisible(True)
        except Exception:
            pass

    def on_sam2_download_clicked(self):
        self._sam2_autostart_after_download = False
        self.start_sam2_model_download()

    def start_sam2_model_download(self):
        if self.sam2_download_thread is not None and self.sam2_download_thread.isRunning():
            self.sam2_status.setText("SAM 2 model is already downloading...")
            return
        if sam2_checkpoint_available():
            self.update_sam2_model_label()
            self.sam2_status.setText("SAM 2 model is already downloaded.")
            if self._sam2_autostart_after_download:
                self._sam2_autostart_after_download = False
                self.start_sam2_tracking()
            return
        self.sam2_download_thread = Sam2DownloadThread()
        self.sam2_download_thread.progress.connect(self.on_sam2_download_progress)
        self.sam2_download_thread.status.connect(self.on_sam2_download_status)
        self.sam2_download_thread.done.connect(self.on_sam2_download_done)
        self.sam2_download_thread.failed.connect(self.on_sam2_download_failed)
        self.set_action_controls_enabled(False)
        self.progress_bar.setValue(0)
        self.sam2_download_thread.start()

    def on_sam2_download_progress(self, value):
        self.progress_bar.setValue(int(value))

    def on_sam2_download_status(self, message):
        self.sam2_status.setText(message)
        self.label_status.setText(message)

    def on_sam2_download_done(self, path):
        self.progress_bar.setValue(100)
        self.set_action_controls_enabled(True)
        self.update_sam2_model_label()
        self.sam2_status.setText("SAM 2 model ready.")
        self.label_status.setText("SAM 2 model downloaded.")
        # User pressed Apply Action: start tracking automatically now.
        if self._sam2_autostart_after_download:
            self._sam2_autostart_after_download = False
            self.start_sam2_tracking()

    def on_sam2_download_failed(self, message):
        self.progress_bar.setValue(0)
        self.set_action_controls_enabled(True)
        self._sam2_autostart_after_download = False
        self.update_sam2_model_label()
        self.sam2_status.setText(
            "SAM 2 model download failed. Check internet and retry."
        )
        self.label_status.setText("SAM 2 model download failed.")
        print(message)

    # --------------- Predict action ---------------
    def on_predict_clicked(self):
        if self.sam2_thread is not None and self.sam2_thread.isRunning():
            self.label_status.setText("SAM 2 tracking is running. Wait for it to finish.")
            return
        model_name = self.combo_models.currentText()
        if not model_name or model_name == "(no models found)":
            self.label_status.setText("No model selected.")
            return
        model_path = os.path.join(MODELS_DIR, model_name)
        self.active_model_path = model_path
        if model_path in self.model_cache:
            self.active_model = self.model_cache[model_path]
        else:
            if YOLO is None:
                self.label_status.setText("ultralytics not installed (YOLO unavailable).")
                return
            try:
                self.label_status.setText("Loading model...")
                QApplication.processEvents()
                mdl = YOLO(model_path)
                self.model_cache[model_path] = mdl
                self.active_model = mdl
                self.label_status.setText("Model loaded.")
            except Exception as e:
                self.label_status.setText(f"Model load failed: {e}")
                return
        sel_area = self.selection_area if self.chk_area_mode.isChecked() else None
        start_idx = self.current_index
        end_idx = int(self.spin_end.value())
        images = list_images()
        self.predict_thread = PredictThread(images, sel_area, self.active_model, start_idx, end_idx)
        self.predict_thread.progress.connect(self.on_predict_progress)
        self.predict_thread.finished.connect(self.on_predict_finished)
        self.predict_thread.error.connect(self.on_predict_error)
        self.predict_thread.snapshot.connect(self.on_predict_snapshot)  # <-- connect snapshot
        self.btn_predict.setEnabled(False)
        self.btn_apply.setEnabled(False)
        self.predict_thread.start()
        self.label_status.setText("Predict started...")

    def on_predict_snapshot(self, files_snapshot):
        # Called in main thread: store undo snapshot so Undo works for predictions
        try:
            if files_snapshot:
                self.push_undo_snapshot(files_snapshot)
                self.label_status.setText("Undo snapshot saved for predict changes.")
        except Exception as e:
            print("on_predict_snapshot error:", e)

    def on_predict_progress(self, p):
        self.progress_bar.setValue(p)

    def on_predict_finished(self, updated, message):
        self.btn_predict.setEnabled(True)
        self.btn_apply.setEnabled(True)
        self.update_apply_button_text()
        self.progress_bar.setValue(100)
        self.label_status.setText(message)
        self.load_image(self.current_index)

    def on_predict_error(self, msg):
        self.btn_predict.setEnabled(True)
        self.btn_apply.setEnabled(True)
        self.update_apply_button_text()
        self.label_status.setText("Predict error. See console.")
        print(msg)

    # ---------------- Shortcuts ----------------
    def setup_shortcuts(self):
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self.perform_undo)
        fit_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        fit_shortcut.activated.connect(self.fit_view)

# Alias to make embedding name clearer without breaking existing imports
AnnotationHelperPage = AnnotationApp

# ---------------- Run App ----------------
def main():
    app = QApplication(sys.argv)
    win = AnnotationHelperPage()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
