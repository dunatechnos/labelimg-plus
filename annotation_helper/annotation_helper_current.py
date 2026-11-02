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
import traceback
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

    def __init__(self, images, sel_area, model_obj, start_idx, end_idx,
                 merge_mode="replace_area"):
        super().__init__()
        self.images = images
        self.sel_area = sel_area  # normalized coords or None
        self.model = model_obj
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.merge_mode = merge_mode if merge_mode in ("add", "replace_area", "replace_all") else "replace_area"
        # merge_mode: "add" (append, keep all existing),
        #   "replace_area" (keep outside area, replace inside; full image = replace all),
        #   "replace_all" (drop all existing). Applies to YOLO + SAM3 models.

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

    def _append_box(self, final, seen, nl):
        """Append predicted box honoring merge_mode (dedup for 'add')."""
        if self.merge_mode == "add":
            key = " ".join(nl)
            if key in seen:
                return
            seen.add(key)
        final.append(nl)

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
                    # ("add" keeps ALL existing; "replace_all" drops everything)
                    kept = []
                    for p in read_yolo_labels(name):
                        if self.merge_mode == "replace_all":
                            continue
                        cls, xc, yc, bw, bh = map(float, p)
                        bxmin = xc - bw/2
                        bxmax = xc + bw/2
                        bymin = yc - bh/2
                        bymax = yc + bh/2
                        A = self.sel_area
                        if self.merge_mode == "add" or (bxmax < A["xmin"] or bxmin > A["xmax"] or bymax < A["ymin"] or bymin > A["ymax"]):
                            kept.append(p)
                    final = kept
                    seen = {" ".join(x) for x in final}

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
                        self._append_box(final, seen, [str(cls), f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"] )

                else:
                    if self.merge_mode == "add":
                        final = [p for p in read_yolo_labels(name)]
                    else:
                        final = []
                    seen = {" ".join(x) for x in final}
                    results = self.model(img)
                    for b in results[0].boxes:
                        cls = self.remap_cls(int(b.cls[0]))
                        x1, y1, x2, y2 = b.xyxy[0].tolist()
                        xc = (x1 + x2) / 2 / w
                        yc = (y1 + y2) / 2 / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        self._append_box(final, seen, [str(cls), f"{xc:.6f}", f"{yc:.6f}", f"{bw:.6f}", f"{bh:.6f}"])

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

SAM3_MODEL_ENTRY = "SAM3.1-16bit (text prompt)"
MERGE_MODES = {"Replace inside area": "replace_area",
               "Add to existing": "add",
               "Replace all": "replace_all"}


def _import_sam3():
    """Lazy import (torch/sam3 heavy). Works both as package and standalone script."""
    try:
        from annotation_helper.sam3_helper import Sam3PropagateThread, Sam3TextModel
    except ImportError:
        from sam3_helper import Sam3PropagateThread, Sam3TextModel
    return Sam3PropagateThread, Sam3TextModel

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
            if self.radio_sam3track.isChecked():
                return "sam3track"
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
        self.sam3_model_cache = {}   # (prompt, cls)->Sam3TextModel adapter
        self.active_model_path = None
        self.active_model = None
        self.custom_target_indices = None    # <-- ADD THIS HERE


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
        self.radio_sam3track = QRadioButton("SAM3 Track")
        for r in (self.radio_delete, self.radio_propagate, self.radio_change, self.radio_sam3track):
            r.setObjectName("segmentRadio")
            r.setFont(QFont("Segoe UI", 8))

        # Default selection (same as old dropdown)
        self.radio_delete.setChecked(True)

        # Connect to your existing function
        self.radio_delete.toggled.connect(lambda checked: checked and self.on_action_changed("delete"))
        self.radio_propagate.toggled.connect(lambda checked: checked and self.on_action_changed("propagate"))
        self.radio_change.toggled.connect(lambda checked: checked and self.on_action_changed("change"))
        self.radio_sam3track.toggled.connect(lambda checked: checked and self.on_action_changed("sam3track"))

        radio_h.addWidget(self.radio_delete)
        radio_h.addWidget(self.radio_propagate)
        radio_h.addWidget(self.radio_change)
        radio_h.addWidget(self.radio_sam3track)

        action_row.addWidget(radio_wrap)

        # Old class dropdown (for change)
        line_old = QHBoxLayout()
        self.combo_old_class = QComboBox()
        self.refresh_full_class_list()
        line_old.addWidget(QLabel("Old Class:"))
        line_old.addWidget(self.combo_old_class)
        action_row.addLayout(line_old)

        # Area classes dropdown (conditional)
        line_area_class = QHBoxLayout()
        self.combo_area_class = QComboBox()
        self.combo_area_class.setVisible(False)
        line_area_class.addWidget(QLabel("Area Class:"))
        line_area_class.addWidget(self.combo_area_class)
        action_row.addLayout(line_area_class)

        # New class (for change target)
        line_new = QHBoxLayout()
        self.combo_new_class = QComboBox()
        self.refresh_full_class_list()
        line_new.addWidget(QLabel("New Class:"))
        line_new.addWidget(self.combo_new_class)
        action_row.addLayout(line_new)

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
        self.combo_models.currentTextChanged.connect(self._update_sam3_prompt_ui)
        model_row.addWidget(self.combo_models)
        model_v.addLayout(model_row)

        # Merge mode: what happens to existing labels on Predict (YOLO + SAM3)
        merge_row = QHBoxLayout()
        merge_row.addWidget(QLabel("Merge:"))
        self.combo_merge = QComboBox()
        self.combo_merge.addItems(list(MERGE_MODES.keys()))
        self.combo_merge.setToolTip("Add to existing: keep all + append predictions\n"
                                    "Replace inside area: keep outside, replace inside (area mode)\n"
                                    "Replace all: drop all existing labels")
        merge_row.addWidget(self.combo_merge)
        model_v.addLayout(merge_row)

        # SAM3 text prompt (visible only when SAM3 model selected)
        sam3_row = QHBoxLayout()
        self.label_sam3_prompt = QLabel("Text:")
        self.line_sam3_prompt = QLineEdit()
        self.line_sam3_prompt.setPlaceholderText("e.g. person")
        self.line_sam3_prompt.setToolTip("SAM3 text prompt. Predicted boxes take the 'New Class' below.")
        sam3_row.addWidget(self.label_sam3_prompt)
        sam3_row.addWidget(self.line_sam3_prompt)
        model_v.addLayout(sam3_row)
        self._update_sam3_prompt_ui()

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
        self.progress_bar.setTextVisible(False)
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
        # SAM3 text-prompt detection (needs sam3 conda env + GPU)
        if self.combo_models.findText(SAM3_MODEL_ENTRY) < 0:
            self.combo_models.addItem(SAM3_MODEL_ENTRY)
        self._update_sam3_prompt_ui()

    def _update_sam3_prompt_ui(self, *a):
        is_sam3 = hasattr(self, "combo_models") and self.combo_models.currentText() == SAM3_MODEL_ENTRY
        for w in (getattr(self, "label_sam3_prompt", None), getattr(self, "line_sam3_prompt", None)):
            if w is not None:
                w.setVisible(bool(is_sam3))

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
        # hide area class combo
        if hasattr(self, "combo_area_class"):
            self.combo_area_class.setVisible(False)

    def update_area_classes(self):
        # compute classes inside selection_area and populate combo_area_class (and show)
        if not self.selection_area:
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
        if found:
            items = [f"{i}:{self.classes[i] if i < len(self.classes) else str(i)}" for i in found]
            self.combo_area_class.clear()
            self.combo_area_class.addItems(items)
            self.combo_area_class.setVisible(True)
            self.label_status.setText("Area selected. Classes: " + ", ".join([self.classes[i] for i in found if i < len(self.classes)]))
        else:
            self.combo_area_class.clear(); self.combo_area_class.setVisible(True)
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
        # Show/hide controls per action
        if text == "change":
            self.combo_old_class.setVisible(True)
            self.combo_new_class.setVisible(True)
        else:
            self.combo_old_class.setVisible(True)
            self.combo_new_class.setVisible(False)
            self.combo_area_class.setVisible(False)
        if text == "propagate":
            self.label_status.setText("Propagate mode: select multiple boxes (Ctrl/Shift-click).")
        elif text == "sam3track":
            self.label_status.setText("SAM3 Track: select exactly 1 box, set End index, Apply = track forward.")
        else:
            # clear message but not override useful status
            if self.label_status.text().startswith("Propagate mode"):
                self.label_status.setText("")

    # ---------------- Apply action (delete/propagate/change) ----------------
    def on_apply_action(self):
        act = self.get_current_action()
        start_idx = self.current_index
        end_idx = int(self.spin_end.value())
        if act == "delete":
            self.action_delete(start_idx, end_idx)
        elif act == "propagate":
            self.action_propagate(start_idx, end_idx)
        elif act == "change":
            self.action_change(start_idx, end_idx)
        elif act == "sam3track":
            self.action_sam3_track(start_idx, end_idx)
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

    # --------------- SAM3 single-object tracking ---------------
    def action_sam3_track(self, start_idx, end_idx):
        """Track the single selected box from current image forward to end_idx.

        Frames = consecutive images in the folder. Target frames keep their
        existing labels; same-class boxes overlapping the tracked box
        (IoU>0.3) are replaced by it. Undo supported via snapshot.
        """
        selected = [it.parts for it in self.bbox_items if it.isSelected()]
        if len(selected) != 1:
            self.label_status.setText("SAM3 Track: select exactly 1 box on the current image.")
            return
        if end_idx <= start_idx:
            self.label_status.setText("SAM3 Track: set End index greater than current image.")
            return
        try:
            Sam3PropagateThread, _ = _import_sam3()
        except Exception as e:
            self.label_status.setText(f"SAM3 unavailable (run in sam3 env with GPU): {e}")
            return
        images = list_images()
        cls_id = int(float(selected[0][0]))
        cls_name = self.classes[cls_id] if cls_id < len(self.classes) else None
        self.sam3_thread = Sam3PropagateThread(
            images, IMAGES_DIR, LABELS_DIR, start_idx, end_idx, selected[0],
            class_name=cls_name)
        self.sam3_thread.progress.connect(self.on_predict_progress)
        self.sam3_thread.finished.connect(self.on_predict_finished)
        self.sam3_thread.error.connect(self.on_predict_error)
        self.sam3_thread.snapshot.connect(self.on_predict_snapshot)
        self.btn_apply.setEnabled(False)
        self.sam3_thread.finished.connect(lambda *a: self.btn_apply.setEnabled(True))
        self.sam3_thread.error.connect(lambda *a: self.btn_apply.setEnabled(True))
        self.sam3_thread.start()
        self.label_status.setText("SAM3 tracking started (single object, forward)...")

    # --------------- Predict action ---------------
    def on_predict_clicked(self):
        model_name = self.combo_models.currentText()
        if not model_name or model_name == "(no models found)":
            self.label_status.setText("No model selected.")
            return
        merge_mode = MERGE_MODES.get(self.combo_merge.currentText(), "replace_area")
        sel_area = self.selection_area if self.chk_area_mode.isChecked() else None
        start_idx = self.current_index
        end_idx = int(self.spin_end.value())
        images = list_images()
        if model_name == SAM3_MODEL_ENTRY:
            self._on_predict_sam3(images, sel_area, start_idx, end_idx, merge_mode)
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
        self.predict_thread = PredictThread(images, sel_area, self.active_model,
                                            start_idx, end_idx, merge_mode=merge_mode)
        self.predict_thread.progress.connect(self.on_predict_progress)
        self.predict_thread.finished.connect(self.on_predict_finished)
        self.predict_thread.error.connect(self.on_predict_error)
        self.predict_thread.snapshot.connect(self.on_predict_snapshot)  # <-- connect snapshot
        self.btn_predict.setEnabled(False)
        self.predict_thread.start()
        self.label_status.setText("Predict started...")

    def _on_predict_sam3(self, images, sel_area, start_idx, end_idx, merge_mode):
        """SAM3 text-prompt detection; predicted boxes get the New-Class."""
        prompt = self.line_sam3_prompt.text().strip()
        if not prompt:
            self.label_status.setText("SAM3: enter a text prompt first (e.g. person).")
            return
        try:
            cls_idx = int(self.combo_new_class.currentText().split(":")[0])
            cls_name = self.classes[cls_idx] if cls_idx < len(self.classes) else str(cls_idx)
        except Exception:
            self.label_status.setText("SAM3: pick a valid New Class for predicted boxes.")
            return
        try:
            _, Sam3TextModel = _import_sam3()
        except Exception as e:
            self.label_status.setText(f"SAM3 unavailable (run in sam3 env with GPU): {e}")
            return
        key = (prompt, cls_idx)
        if key in self.sam3_model_cache:
            adapter = self.sam3_model_cache[key]
        else:
            try:
                self.label_status.setText("Loading SAM3 image model (16-bit)...")
                QApplication.processEvents()
                adapter = Sam3TextModel(prompt, cls_idx, cls_name)
                self.sam3_model_cache[key] = adapter
            except Exception as e:
                self.label_status.setText(f"SAM3 model load failed: {e}")
                print(traceback.format_exc())
                return
        self.predict_thread = PredictThread(images, sel_area, adapter,
                                            start_idx, end_idx, merge_mode=merge_mode)
        self.predict_thread.progress.connect(self.on_predict_progress)
        self.predict_thread.finished.connect(self.on_predict_finished)
        self.predict_thread.error.connect(self.on_predict_error)
        self.predict_thread.snapshot.connect(self.on_predict_snapshot)  # <-- connect snapshot
        self.btn_predict.setEnabled(False)
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
        self.progress_bar.setValue(100)
        self.label_status.setText(message)
        self.load_image(self.current_index)

    def on_predict_error(self, msg):
        self.btn_predict.setEnabled(True)
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
