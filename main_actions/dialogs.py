"""Compact dialogs for the main-window annotation actions.

Each action gets a small focused dialog instead of a separate helper page.
The dialogs only collect parameters; the controller performs the actual work
so the same operation can run locally or on Kaggle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from annotation_helper.annotation_helper_current import (
    KAGGLE_UI_AVAILABLE,
    SAM2_DEFAULT_MODEL,
    SAM2_MODELS,
    ChooseImagesDialog,
)

from .context import DatasetContext


# ----------------------------------------------------------------------
# Shared widgets
# ----------------------------------------------------------------------
class ScopePanel(QGroupBox):
    """Current frame / range / all frames selector."""

    def __init__(
        self,
        context: DatasetContext,
        default: str = "current",
        default_start: Optional[int] = None,
        default_end: Optional[int] = None,
        parent=None,
    ):
        super().__init__("Apply to", parent)
        self.context = context
        self.total = max(1, context.total)
        current = max(1, min(context.current_index + 1, self.total))

        self.radio_current = QRadioButton(
            "Current frame (%d/%d)" % (current, self.total)
        )
        self.radio_range = QRadioButton("Frame range")
        self.radio_all = QRadioButton("All frames (%d)" % self.total)
        self.group = QButtonGroup(self)
        for button in (self.radio_current, self.radio_range, self.radio_all):
            self.group.addButton(button)
            button.toggled.connect(self._update_enabled)

        self.spin_start = QSpinBox()
        self.spin_start.setRange(1, self.total)
        self.spin_start.setValue(default_start or current)
        self.spin_end = QSpinBox()
        self.spin_end.setRange(1, self.total)
        self.spin_end.setValue(default_end or current)
        self.spin_start.valueChanged.connect(self._fix_end)
        self.spin_end.valueChanged.connect(self._fix_start)

        range_row = QHBoxLayout()
        range_row.setContentsMargins(20, 0, 0, 0)
        range_row.addWidget(QLabel("From"))
        range_row.addWidget(self.spin_start)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self.spin_end)
        range_row.addStretch(1)

        if default == "range":
            self.radio_range.setChecked(True)
        elif default == "all":
            self.radio_all.setChecked(True)
        else:
            self.radio_current.setChecked(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.addWidget(self.radio_current)
        layout.addWidget(self.radio_range)
        layout.addLayout(range_row)
        layout.addWidget(self.radio_all)
        self._update_enabled()

    def _fix_end(self, value: int) -> None:
        if self.spin_end.value() < value:
            self.spin_end.setValue(value)

    def _fix_start(self, value: int) -> None:
        if self.spin_start.value() > value:
            self.spin_start.setValue(value)

    def _update_enabled(self) -> None:
        active = self.radio_range.isChecked()
        self.spin_start.setEnabled(active)
        self.spin_end.setEnabled(active)

    def mode(self) -> str:
        if self.radio_range.isChecked():
            return "range"
        if self.radio_all.isChecked():
            return "all"
        return "current"

    def target_names(self) -> List[str]:
        mode = self.mode()
        if mode == "all":
            return list(self.context.image_names)
        if mode == "range":
            return self.context.names_in_range(
                self.spin_start.value(), self.spin_end.value()
            )
        name = self.context.current_name
        return [name] if name else []

    def range_indices(self) -> Tuple[int, int]:
        """1-based start/end for predictor threads."""
        mode = self.mode()
        if mode == "all":
            return 1, max(1, self.context.total)
        if mode == "range":
            return self.spin_start.value(), self.spin_end.value()
        current = max(1, min(self.context.current_index + 1, self.total))
        return current, current


class BoxFilterPanel(QGroupBox):
    """Class / area / selected-box filters shared by delete and change."""

    def __init__(
        self,
        context: DatasetContext,
        class_title: str = "Box class",
        parent=None,
    ):
        super().__init__("Filter", parent)
        self.context = context

        self.class_combo = QComboBox()
        self.class_combo.addItem(class_title + ": all classes", None)
        for index, name in enumerate(context.classes):
            self.class_combo.addItem("%d: %s" % (index, name), index)

        self.selected_check = QCheckBox(
            "Only boxes matching the current selection (class + IoU)"
        )
        selected = context.selected_boxes()
        if selected:
            self.selected_check.setChecked(True)
        else:
            self.selected_check.setEnabled(False)
            self.selected_check.setToolTip(
                "Select one or more boxes on the canvas first (Ctrl+click)."
            )

        self.area_check = QCheckBox("Only boxes inside the canvas area")
        area = context.area
        if area:
            self.area_check.setChecked(True)
            area_text = "Area: %.1f%% %.1f%% -> %.1f%% %.1f%%" % (
                area["xmin"] * 100,
                area["ymin"] * 100,
                area["xmax"] * 100,
                area["ymax"] * 100,
            )
        else:
            self.area_check.setEnabled(False)
            area_text = "No canvas area selected (Ctrl+Alt+R to mark one)."
            self.area_check.setToolTip(area_text)
        self.area_label = QLabel(area_text)
        self.area_label.setWordWrap(True)

        form = QFormLayout()
        form.setContentsMargins(8, 6, 8, 6)
        form.setSpacing(5)
        form.addRow("Class", self.class_combo)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.addLayout(form)
        layout.addWidget(self.selected_check)
        layout.addWidget(self.area_check)
        layout.addWidget(self.area_label)

    def class_index(self) -> Optional[int]:
        value = self.class_combo.currentData()
        return int(value) if value is not None else None

    def classes(self) -> Optional[set]:
        index = self.class_index()
        return {index} if index is not None else None

    def area(self) -> Optional[Dict[str, float]]:
        if self.area_check.isEnabled() and self.area_check.isChecked():
            return self.context.area
        return None

    def selected_boxes(self) -> Optional[List[List[str]]]:
        if self.selected_check.isEnabled() and self.selected_check.isChecked():
            boxes = self.context.selected_boxes()
            return boxes or None
        return None

    def has_filter(self) -> bool:
        return (
            self.class_index() is not None
            or self.selected_boxes() is not None
            or self.area() is not None
        )


class CompactDialog(QDialog):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(430)
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(12, 10, 12, 10)
        self.root.setSpacing(8)

    def add_buttons(self, ok_text: str) -> QDialogButtonBox:
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.buttons.button(QDialogButtonBox.Ok).setText(ok_text)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.root.addWidget(self.buttons)
        return self.buttons

    def add_hint(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        self.root.addWidget(label)
        return label


def _count_label(context: DatasetContext) -> str:
    total = len(getattr(context.window.canvas, "shapes", []) or [])
    selected = len(context.selected_boxes())
    return "Current frame: %d box(es), %d selected." % (total, selected)


# ----------------------------------------------------------------------
# Delete / change / propagate
# ----------------------------------------------------------------------
class DeleteBoxesDialog(CompactDialog):
    def __init__(self, context: DatasetContext, parent=None):
        super(DeleteBoxesDialog, self).__init__("Delete boxes", parent)
        self.context = context
        self.add_hint(_count_label(context))

        self.scope = ScopePanel(context, default="current", parent=self)
        self.filters = BoxFilterPanel(context, class_title="Delete class", parent=self)
        self.root.addWidget(self.scope)
        self.root.addWidget(self.filters)

        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #b00020;")
        self.root.addWidget(self.warning)
        self.filters.class_combo.currentIndexChanged.connect(self._refresh_warning)
        self.filters.selected_check.toggled.connect(self._refresh_warning)
        self.filters.area_check.toggled.connect(self._refresh_warning)
        self.scope.group.buttonClicked.connect(self._refresh_warning)
        self._refresh_warning()

        self.add_buttons("Delete boxes")

    def _refresh_warning(self) -> None:
        count = len(self.scope.target_names())
        if not self.filters.has_filter():
            self.warning.setText(
                "No filter: every box on %d target file(s) will be deleted."
                % count
            )
        else:
            self.warning.setText(
                "Only matching boxes on %d target file(s) will be deleted."
                % count
            )


class ChangeClassDialog(CompactDialog):
    def __init__(self, context: DatasetContext, parent=None):
        super(ChangeClassDialog, self).__init__("Change class", parent)
        self.context = context
        self.add_hint(_count_label(context))

        self.scope = ScopePanel(context, default="current", parent=self)
        self.filters = BoxFilterPanel(context, class_title="Old class", parent=self)
        self._preselect_old_class()

        self.new_combo = QComboBox()
        for index, name in enumerate(context.classes):
            self.new_combo.addItem("%d: %s" % (index, name), index)
        self._preselect_new_class()

        form = QFormLayout()
        form.setContentsMargins(8, 6, 8, 6)
        form.addRow("New class", self.new_combo)
        new_box = QGroupBox("Replace with")
        new_box.setLayout(form)

        self.root.addWidget(self.scope)
        self.root.addWidget(self.filters)
        self.root.addWidget(new_box)
        self.add_buttons("Change class")

        if not context.classes:
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            self.add_hint("classes.txt is empty; add classes before changing them.")
        elif self.new_combo.count() == 0:
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)

    def _preselect_old_class(self) -> None:
        # When boxes are already selected, the selection itself is the filter;
        # leave the class on "all" so every selected box changes.
        if self.context.selected_shapes():
            return
        labels = [
            shape.label
            for shape in getattr(self.context.window.canvas, "shapes", [])
        ]
        chosen = None
        if labels:
            chosen = max(set(labels), key=labels.count)
        index = self.context.class_index(chosen) if chosen else -1
        if index >= 0:
            combo_index = self.filters.class_combo.findData(index)
            if combo_index >= 0:
                self.filters.class_combo.setCurrentIndex(combo_index)

    def _preselect_new_class(self) -> None:
        old = self.filters.class_index()
        if old is None:
            labels = [shape.label for shape in self.context.selected_shapes()]
            if labels:
                common = max(set(labels), key=labels.count)
                index = self.context.class_index(common)
                if index >= 0:
                    old = index
        for row in range(self.new_combo.count()):
            value = self.new_combo.itemData(row)
            if value != old:
                self.new_combo.setCurrentIndex(row)
                return

    def new_class(self) -> int:
        value = self.new_combo.currentData()
        return int(value) if value is not None else 0


class PropagateDialog(CompactDialog):
    def __init__(self, context: DatasetContext, parent=None):
        super(PropagateDialog, self).__init__("Propagate selected boxes", parent)
        self.context = context
        boxes = context.selected_boxes()
        self.boxes = boxes
        current = max(1, min(context.current_index + 1, context.total))
        hint = (
            "Source: %d selected box(es) on frame %d/%d."
            % (len(boxes), current, max(1, context.total))
        )
        self.add_hint(hint)
        if not boxes:
            warning = QLabel(
                "Select one or more boxes on the canvas first "
                "(click, or Ctrl+click for several)."
            )
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #b00020;")
            self.root.addWidget(warning)

        next_frame = current + 1
        if next_frame > context.total:
            next_frame = current
        self.scope = ScopePanel(
            context,
            default="range",
            default_start=next_frame,
            default_end=max(next_frame, context.total),
            parent=self,
        )
        self.root.addWidget(self.scope)

        self.custom_indices: Optional[List[int]] = None
        self.custom_label = QLabel("Using the frame range above.")
        choose_row = QHBoxLayout()
        choose_button = QPushButton("Choose frames...")
        choose_button.setObjectName("secondaryButton")
        choose_button.clicked.connect(self._choose_targets)
        clear_button = QPushButton("Use range")
        clear_button.setObjectName("secondaryButton")
        clear_button.clicked.connect(self._clear_custom)
        choose_row.addWidget(choose_button)
        choose_row.addWidget(clear_button)
        choose_row.addWidget(self.custom_label, 1)
        self.root.addLayout(choose_row)

        self.add_hint(
            "Boxes are added to the target label files; existing boxes are kept."
        )
        self.add_buttons("Propagate")
        if not boxes:
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)

    def _choose_targets(self) -> None:
        dialog = ChooseImagesDialog(self.context.image_names, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        indices = dialog.get_selected_indices()
        self.custom_indices = indices or None
        if indices:
            self.custom_label.setText(
                "Custom target: %d image(s)." % len(indices)
            )
        else:
            self.custom_label.setText("Using the frame range above.")

    def _clear_custom(self) -> None:
        self.custom_indices = None
        self.custom_label.setText("Using the frame range above.")

    def target_names(self) -> List[str]:
        if self.custom_indices:
            return [
                self.context.image_names[index - 1]
                for index in self.custom_indices
                if 1 <= index <= self.context.total
            ]
        return self.scope.target_names()


# ----------------------------------------------------------------------
# Predict / SAM 2
# ----------------------------------------------------------------------
class PredictDialog(CompactDialog):
    def __init__(self, context: DatasetContext, parent=None):
        super(PredictDialog, self).__init__("YOLO predict", parent)
        self.context = context
        self.add_hint(_count_label(context))

        self.model_combo = QComboBox()
        self._reload_models()
        browse = QPushButton("Browse...")
        browse.setObjectName("secondaryButton")
        browse.clicked.connect(self._browse_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(browse)
        model_box = QGroupBox("Model (.pt)")
        model_box.setLayout(model_row)

        self.scope = ScopePanel(
            context,
            default="range",
            default_start=max(1, context.current_index + 1),
            default_end=max(1, context.total),
            parent=self,
        )

        self.area_check = QCheckBox("Predict only inside the canvas area")
        area = context.area
        if area:
            self.area_check.setChecked(True)
            area_text = "Area: %.1f%% %.1f%% -> %.1f%% %.1f%%" % (
                area["xmin"] * 100,
                area["ymin"] * 100,
                area["xmax"] * 100,
                area["ymax"] * 100,
            )
        else:
            self.area_check.setEnabled(False)
            area_text = "No canvas area selected (Ctrl+Alt+R to mark one)."
        self.area_label = QLabel(area_text)
        self.area_label.setWordWrap(True)

        self.merge_check = QCheckBox("Keep existing boxes and add predictions")
        self.merge_check.setToolTip(
            "When off, labels on the target frames are replaced by predictions."
        )
        self.kaggle_check = QCheckBox("Run on Kaggle (T4 x2)")
        self.kaggle_check.setEnabled(KAGGLE_UI_AVAILABLE)

        self.root.addWidget(model_box)
        self.root.addWidget(self.scope)
        self.root.addWidget(self.area_check)
        self.root.addWidget(self.area_label)
        self.root.addWidget(self.merge_check)
        self.root.addWidget(self.kaggle_check)
        self.add_hint(
            "Without an area or merge, existing labels on the target frames "
            "are replaced by the model's predictions."
        )
        self.add_buttons("Run predict")

    def _reload_models(self) -> None:
        self.model_combo.clear()
        try:
            models = sorted(
                path
                for path in Path(self.context.models_dir).glob("*.pt")
                if path.is_file()
            )
        except OSError:
            models = []
        for path in models:
            self.model_combo.addItem(path.name, str(path))
        if not models:
            self.model_combo.addItem("(no .pt model in models folder)", None)

    def _browse_model(self) -> None:
        start = str(self.context.models_dir)
        filename, _ = QFileDialog.getOpenFileName(
            self, "Choose YOLO model", start, "PyTorch model (*.pt)"
        )
        if not filename:
            return
        index = self.model_combo.findData(filename)
        if index < 0:
            self.model_combo.addItem(Path(filename).name, filename)
            index = self.model_combo.count() - 1
        self.model_combo.setCurrentIndex(index)

    def model_path(self) -> str:
        value = self.model_combo.currentData()
        return str(value) if value else ""

    def start_end(self) -> Tuple[int, int]:
        return self.scope.range_indices()

    def area(self) -> Optional[Dict[str, float]]:
        if self.area_check.isEnabled() and self.area_check.isChecked():
            return self.context.area
        return None

    def merge_existing(self) -> bool:
        return self.merge_check.isChecked()

    def run_on_kaggle(self) -> bool:
        return self.kaggle_check.isChecked()


class Sam2Dialog(CompactDialog):
    def __init__(self, context: DatasetContext, parent=None):
        super(Sam2Dialog, self).__init__("SAM 2 tracking", parent)
        self.context = context
        boxes = context.selected_boxes()
        self.boxes = boxes
        self.add_hint(_count_label(context))
        if not boxes:
            warning = QLabel(
                "Select one or more boxes on the canvas first. SAM 2 needs "
                "them as the prompt for the first frame."
            )
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #b00020;")
            self.root.addWidget(warning)

        self.model_combo = QComboBox()
        for key, info in SAM2_MODELS.items():
            self.model_combo.addItem(
                "%s - %s" % (info["label"], info["size"]), key
            )
        self.model_combo.setCurrentIndex(
            max(0, self.model_combo.findData(SAM2_DEFAULT_MODEL))
        )

        self.direction_combo = QComboBox()
        self.direction_combo.addItem("Forward (current -> last frame)", "forward")
        self.direction_combo.addItem("Backward (current -> first frame)", "backward")
        self.direction_combo.addItem("Entire sequence", "entire")

        self.start_spin = QSpinBox()
        self.start_spin.setRange(1, max(1, context.total))
        self.start_spin.setValue(max(1, context.current_index + 1))

        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(5, 95)
        self.conf_spin.setValue(50)
        self.conf_spin.setSuffix(" %")

        self.kaggle_check = QCheckBox("Run on Kaggle (T4 x2)")
        self.kaggle_check.setEnabled(KAGGLE_UI_AVAILABLE)

        form = QFormLayout()
        form.setContentsMargins(8, 6, 8, 6)
        form.addRow("Model", self.model_combo)
        form.addRow("Direction", self.direction_combo)
        form.addRow("Start frame", self.start_spin)
        form.addRow("Min confidence", self.conf_spin)
        box = QGroupBox("Tracking")
        box.setLayout(form)

        self.root.addWidget(box)
        self.root.addWidget(self.kaggle_check)
        self.add_hint(
            "The model downloads automatically on first use. Existing boxes "
            "that overlap the tracked object are replaced."
        )
        self.add_buttons("Start tracking")
        if not boxes:
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)

    def model_key(self) -> str:
        value = self.model_combo.currentData()
        return str(value) if value else SAM2_DEFAULT_MODEL

    def direction(self) -> str:
        return str(self.direction_combo.currentData() or "forward")

    def start_index(self) -> int:
        return self.start_spin.value()

    def min_confidence(self) -> float:
        return self.conf_spin.value() / 100.0

    def run_on_kaggle(self) -> bool:
        return self.kaggle_check.isChecked()


# ----------------------------------------------------------------------
# Kaggle hub
# ----------------------------------------------------------------------
class KaggleHubDialog(CompactDialog):
    settings_requested = pyqtSignal()
    jobs_requested = pyqtSignal()
    session_requested = pyqtSignal()
    test_requested = pyqtSignal()
    gpu_requested = pyqtSignal()

    def __init__(self, status_lines: Sequence[str], parent=None):
        super(KaggleHubDialog, self).__init__("Kaggle controls", parent)
        self.status_label = QLabel("\n".join(status_lines))
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.root.addWidget(self.status_label)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.root.addWidget(line)

        grid = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()
        grid.addLayout(left)
        grid.addLayout(right)

        for text, signal, layout in (
            ("Kaggle settings...", self.settings_requested, left),
            ("Job history...", self.jobs_requested, left),
            ("Upload session images", self.session_requested, left),
            ("Test connection", self.test_requested, right),
            ("GPU check", self.gpu_requested, right),
        ):
            button = QPushButton(text)
            button.setObjectName("secondaryButton")
            button.clicked.connect(signal.emit)
            layout.addWidget(button)
        left.addStretch(1)
        right.addStretch(1)
        self.root.addLayout(grid)

        hint = QLabel(
            "Predict and SAM 2 dialogs have a 'Run on Kaggle' checkbox. "
            "Images are uploaded once per batch; later clicks send only the "
            "command and download the resulting coordinates."
        )
        hint.setWordWrap(True)
        self.root.addWidget(hint)
        self.add_buttons("Close")
        self.buttons.button(QDialogButtonBox.Cancel).hide()

    def set_status(self, lines: Sequence[str]) -> None:
        self.status_label.setText("\n".join(lines))


# ----------------------------------------------------------------------
# Palette / help
# ----------------------------------------------------------------------
class ActionPalette(CompactDialog):
    def __init__(self, items: Sequence[Tuple[str, str, str, str]], parent=None):
        super(ActionPalette, self).__init__("Action palette", parent)
        self.resize(520, 470)
        self._items = list(items)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Type to filter actions...")
        self.search.textChanged.connect(self._filter)
        self.search.setFocus()
        self.root.addWidget(self.search)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(lambda _item: self.accept())
        self.list_widget.itemActivated.connect(lambda _item: self.accept())
        self.root.addWidget(self.list_widget, 1)

        self._populate(self._items)
        buttons = self.add_buttons("Run")
        buttons.button(QDialogButtonBox.Ok).setDefault(True)

    def _populate(self, items: Sequence[Tuple[str, str, str, str]]) -> None:
        self.list_widget.clear()
        for action_id, title, shortcut, category in items:
            text = title if not shortcut else "%s    %s" % (title, shortcut)
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, action_id)
            item.setToolTip(category)
            self.list_widget.addItem(item)
        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)

    def _filter(self, text: str) -> None:
        needle = text.strip().lower()
        self._populate(
            [
                item
                for item in self._items
                if not needle
                or needle in item[1].lower()
                or needle in item[2].lower()
                or needle in item[3].lower()
            ]
        )

    def selected_action_id(self) -> str:
        item = self.list_widget.currentItem()
        return str(item.data(Qt.UserRole)) if item is not None else ""


class ShortcutHelpDialog(CompactDialog):
    def __init__(self, rows: Sequence[Tuple[str, str]], parent=None):
        super(ShortcutHelpDialog, self).__init__("Keyboard shortcuts", parent)
        self.resize(520, 520)
        hint = QLabel(
            "Canvas: click a box to select it, Ctrl+click to add/remove boxes. "
            "Selected boxes are used by Delete, Change, Propagate and SAM 2."
        )
        hint.setWordWrap(True)
        self.root.addWidget(hint)

        table = QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.horizontalHeader().setStretchLastSection(True)
        for row, (action, shortcut) in enumerate(rows):
            table.setItem(row, 0, QTableWidgetItem(action))
            table.setItem(row, 1, QTableWidgetItem(shortcut))
        table.resizeColumnsToContents()
        self.root.addWidget(table, 1)
        self.add_buttons("Close")
        self.buttons.button(QDialogButtonBox.Cancel).hide()
