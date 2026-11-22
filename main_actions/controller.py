"""Main-window controller for all Annotation Helper features.

The controller lives on the normal LabelImg window.  It owns the Actions menu,
the shortcuts, the compact dialogs and the background predictor/tracker
threads.  The old helper page is not used here.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt5.QtCore import QObject, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QMessageBox,
    QMenu,
    QProgressDialog,
)

from annotation_helper.annotation_helper_current import (
    KAGGLE_UI_AVAILABLE,
    KAGGLE_UI_ERROR,
    PredictThread,
    SAM2_MODELS,
    Sam2TrackThread,
    SelectClassesDialog,
    YOLO,
    sam2_checkpoint_available,
    sam2_checkpoint_path,
)

from . import ops
from .context import DatasetContext, build_context
from .dialogs import (
    ActionPalette,
    ChangeClassDialog,
    DeleteBoxesDialog,
    KaggleHubDialog,
    PredictDialog,
    PropagateDialog,
    Sam2Dialog,
    ShortcutHelpDialog,
)
from .undo import ActionUndo

if KAGGLE_UI_AVAILABLE:
    from annotation_helper.kaggle_ui import (
        KaggleGPUCheckThread,
        KaggleJobThread,
        KaggleJobsDialog,
        KaggleSessionThread,
        KaggleSettingsDialog,
        KaggleTestThread,
    )
    from kaggle_bridge.config import load_config
    from kaggle_bridge.spec import JobSpec, SpecError
else:  # pragma: no cover - depends on optional local setup
    KaggleGPUCheckThread = None
    KaggleJobThread = None
    KaggleJobsDialog = None
    KaggleSessionThread = None
    KaggleSettingsDialog = None
    KaggleTestThread = None
    load_config = None
    JobSpec = None
    SpecError = Exception


class MainActionController(QObject):
    """Installs the Actions menu and implements every main-window action."""

    def __init__(self, window):
        super(MainActionController, self).__init__(window)
        self.window = window
        self.undo = ActionUndo()
        self.actions: Dict[str, QAction] = {}
        self._handlers = {}
        self._palette_items: List[Tuple[str, str, str, str]] = []

        self._predict_thread = None
        self._sam2_thread = None
        self._kaggle_thread = None
        self._session_thread = None
        self._aux_threads = []
        self._last_changed_names: List[str] = []

        self._build_menu()
        try:
            self.window.canvas.areaSelected.connect(self._on_area_selected)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Menu and shortcuts
    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menu = QMenu("&Actions", self.window)
        try:
            self.window.menuBar().insertMenu(
                self.window.menus.help.menuAction(), menu
            )
        except Exception:
            self.window.menuBar().addMenu(menu)
        self.menu = menu

        specs = [
            ("select_all", "Select all boxes", "Ctrl+Alt+A", "Selection",
             self.select_all,
             "Select every box on the current frame."),
            ("select_by_class", "Select boxes by class...", "Ctrl+Shift+B",
             "Selection", self.select_by_class,
             "Select every box of one or more classes on the current frame."),
            ("clear_selection", "Clear box selection", "Ctrl+Shift+X",
             "Selection", self.clear_selection,
             "Clear the current canvas selection."),
            ("area_select", "Select area on canvas", "Ctrl+Alt+R",
             "Selection", self.select_area,
             "Drag a rectangle on the canvas for area-filtered actions."),
            ("clear_area", "Clear canvas area", "", "Selection",
             self.clear_area, "Remove the orange area rectangle."),
            (None, None, None, None, None, None),
            ("delete_boxes", "Delete boxes...", "Ctrl+Alt+D", "Edit",
             self.delete_boxes,
             "Delete all/class/selected/area boxes in a frame range."),
            ("change_class", "Change class...", "Ctrl+Shift+C", "Edit",
             self.change_class,
             "Replace a class on all/selected/area boxes."),
            ("propagate", "Propagate selected boxes...", "Ctrl+Alt+F", "Edit",
             self.propagate,
             "Copy the selected boxes into later/all frames."),
            (None, None, None, None, None, None),
            ("predict", "YOLO predict...", "Ctrl+Shift+Y", "AI",
             self.predict,
             "Run a YOLO .pt model locally or on Kaggle."),
            ("sam2", "SAM 2 track...", "Ctrl+Shift+T", "AI",
             self.sam2_track,
             "Track the selected boxes with SAM 2 locally or on Kaggle."),
            ("kaggle", "Kaggle controls...", "Ctrl+Shift+K", "AI",
             self.kaggle_hub,
             "Configure Kaggle, upload the session, inspect/restart jobs."),
            (None, None, None, None, None, None),
            ("undo", "Undo last label action", "Ctrl+Alt+Z", "History",
             self.undo_action,
             "Restore the label files from before the last batch action."),
            ("palette", "Action palette...", "Ctrl+Shift+H", "Help",
             self.open_palette,
             "Search every action and run it."),
            ("shortcuts", "Keyboard shortcuts...", "Ctrl+Shift+F1", "Help",
             self.show_shortcuts,
             "Show every main-window shortcut."),
        ]

        last_was_separator = True
        for action_id, title, shortcut, category, handler, tip in specs:
            if action_id is None:
                if not last_was_separator:
                    self.menu.addSeparator()
                    last_was_separator = True
                continue
            action = QAction(title, self.window)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
                action.setShortcutContext(Qt.WindowShortcut)
            action.setStatusTip(tip)
            action.triggered.connect(handler)
            self.menu.addAction(action)
            self.window.addAction(action)
            self.actions[action_id] = action
            self._handlers[action_id] = handler
            self._palette_items.append((action_id, title, shortcut, category))
            last_was_separator = False

        self.actions["undo"].setEnabled(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def status(self, message: str, timeout: int = 6000) -> None:
        try:
            self.window.status(message, timeout)
        except Exception:
            print(message)

    def _context(self, ensure: bool = False) -> Optional[DatasetContext]:
        try:
            context = build_context(self.window)
            if ensure:
                context.ensure_ready()
            return context
        except Exception:
            QMessageBox.warning(
                self.window,
                "Actions",
                "Could not read the current dataset:\n\n"
                + traceback.format_exc(limit=2),
            )
            return None

    def _require_image(self, context: Optional[DatasetContext]) -> bool:
        if context is None:
            return False
        if not context.image_names or not context.current_name:
            QMessageBox.information(
                self.window,
                "Actions",
                "Open an image folder first (Ctrl+U).",
            )
            return False
        return True

    def _stop_area_mode(self) -> None:
        try:
            if getattr(self.window.canvas, "area_mode", False):
                self.window.canvas.set_area_mode(False)
        except Exception:
            pass

    def _busy_thread(self):
        for attr in ("_predict_thread", "_sam2_thread", "_kaggle_thread", "_session_thread"):
            thread = getattr(self, attr, None)
            if thread is not None:
                try:
                    if thread.isRunning():
                        return thread
                except RuntimeError:
                    continue
        for thread in list(self._aux_threads):
            try:
                if thread.isRunning():
                    return thread
            except RuntimeError:
                continue
        return None

    def _push_undo(self, snapshot) -> None:
        if not snapshot:
            return
        try:
            self.undo.push(dict(snapshot))
        except Exception:
            return
        self.actions["undo"].setEnabled(True)

    def _progress_dialog(self, title: str, cancel: bool = True) -> QProgressDialog:
        dialog = QProgressDialog(
            "Preparing...", "Cancel" if cancel else "", 0, 100, self.window
        )
        if not cancel:
            dialog.setCancelButton(None)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        dialog.show()
        QApplication.processEvents()
        return dialog

    def _run_operation(self, title: str, context: DatasetContext, func) -> None:
        if getattr(self.window, "file_path", None):
            try:
                context.save_current()
            except Exception:
                pass

        dialog = self._progress_dialog(title, cancel=True)

        def progress(done: int, total: int, name: str) -> bool:
            maximum = max(1, total)
            if dialog.maximum() != maximum:
                dialog.setMaximum(maximum)
            dialog.setValue(min(done, maximum))
            dialog.setLabelText(
                "%s\n%s (%d/%d)" % (title, name, done, total)
            )
            QApplication.processEvents()
            return not dialog.wasCanceled()

        try:
            result = func(progress)
        except Exception:
            dialog.close()
            QMessageBox.warning(
                self.window,
                title,
                "The operation failed:\n\n" + traceback.format_exc(limit=3),
            )
            return
        dialog.close()

        if result.snapshot:
            self._push_undo(result.snapshot)
        if result.files_changed:
            try:
                context.reload()
            except Exception:
                pass
        self.status(result.message, 9000)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def select_all(self) -> None:
        canvas = self.window.canvas
        if not canvas.shapes:
            self.status("No boxes on this frame.")
            return
        canvas.select_all_shapes()
        self.status("Selected all %d box(es)." % len(canvas.selected_shapes))

    def select_by_class(self) -> None:
        context = self._context()
        if not self._require_image(context):
            return
        canvas = self.window.canvas
        shapes = list(canvas.shapes)
        if not shapes:
            self.status("No boxes on this frame.")
            return

        labels = [shape.label for shape in shapes]
        ordered: List[str] = []
        for name in context.classes:
            if name in labels and name not in ordered:
                ordered.append(name)
        for name in labels:
            if name not in ordered:
                ordered.append(name)
        counts = {index: labels.count(name) for index, name in enumerate(ordered)}

        dialog = SelectClassesDialog(ordered, counts, parent=self.window)
        if dialog.exec_() != QDialog.Accepted:
            return
        chosen = {ordered[index] for index in dialog.selected_classes()}
        if not chosen:
            canvas.clear_selection()
            self.status("No classes checked; selection cleared.")
            return
        canvas.select_shapes(lambda shape: shape.label in chosen)
        self.status(
            "Selected %d box(es) for: %s"
            % (len(canvas.selected_shapes), ", ".join(sorted(chosen)))
        )

    def clear_selection(self) -> None:
        self.window.canvas.clear_selection()
        self.status("Box selection cleared.")

    def select_area(self) -> None:
        canvas = self.window.canvas
        if canvas.pixmap is None or canvas.pixmap.isNull():
            self.status("Open an image first.")
            return
        self._stop_area_mode()
        canvas.set_area_mode(True)
        canvas.setFocus()
        self.status("Drag a rectangle on the image. Esc cancels area mode.")

    def clear_area(self) -> None:
        self.window.canvas.clear_area()
        self.status("Canvas area cleared.")

    def _on_area_selected(self, area: dict) -> None:
        if not area:
            return
        self.status(
            "Area set: x %.1f%%-%.1f%%, y %.1f%%-%.1f%%."
            % (
                area.get("xmin", 0) * 100,
                area.get("xmax", 0) * 100,
                area.get("ymin", 0) * 100,
                area.get("ymax", 0) * 100,
            ),
            8000,
        )

    # ------------------------------------------------------------------
    # Delete / change / propagate
    # ------------------------------------------------------------------
    def delete_boxes(self) -> None:
        context = self._context()
        if not self._require_image(context):
            return
        self._stop_area_mode()
        dialog = DeleteBoxesDialog(context, self.window)
        if dialog.exec_() != QDialog.Accepted:
            return
        names = dialog.scope.target_names()
        if not names:
            self.status("No target frames selected.")
            return
        if not dialog.filters.has_filter() and len(names) > 1:
            answer = QMessageBox.question(
                self.window,
                "Delete all boxes?",
                "No class/selection/area filter is active.\n\n"
                "Every box in %d label files will be deleted. Continue?" % len(names),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        context.ensure_ready()
        classes = dialog.filters.classes()
        area = dialog.filters.area()
        selected = dialog.filters.selected_boxes()
        self._run_operation(
            "Deleting boxes",
            context,
            lambda progress: ops.delete_boxes(
                context, names, classes, area, selected, progress
            ),
        )

    def change_class(self) -> None:
        context = self._context()
        if not self._require_image(context):
            return
        if not context.classes:
            QMessageBox.information(
                self.window,
                "Change class",
                "classes.txt is empty. Add class names first.",
            )
            return
        self._stop_area_mode()
        dialog = ChangeClassDialog(context, self.window)
        if dialog.exec_() != QDialog.Accepted:
            return
        names = dialog.scope.target_names()
        if not names:
            self.status("No target frames selected.")
            return
        if (
            dialog.filters.class_index() is None
            and dialog.filters.selected_boxes() is None
            and dialog.filters.area() is None
            and len(names) > 1
        ):
            answer = QMessageBox.question(
                self.window,
                "Change every class?",
                "No old-class/selection/area filter is active.\n\n"
                "Every box in %d label files will get the new class. Continue?"
                % len(names),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        context.ensure_ready()
        new_class = dialog.new_class()
        old_classes = dialog.filters.classes()
        area = dialog.filters.area()
        selected = dialog.filters.selected_boxes()
        self._run_operation(
            "Changing class",
            context,
            lambda progress: ops.change_class(
                context, names, new_class, old_classes, area, selected, progress
            ),
        )

    def propagate(self) -> None:
        context = self._context()
        if not self._require_image(context):
            return
        self._stop_area_mode()
        dialog = PropagateDialog(context, self.window)
        if dialog.exec_() != QDialog.Accepted:
            return
        boxes = dialog.boxes
        if not boxes:
            self.status("Select boxes on the current frame first.")
            return
        names = dialog.target_names()
        if not names:
            self.status("No target frames selected.")
            return
        context.ensure_ready()
        self._run_operation(
            "Propagating boxes",
            context,
            lambda progress: ops.propagate_boxes(context, names, boxes, progress),
        )

    # ------------------------------------------------------------------
    # YOLO predict
    # ------------------------------------------------------------------
    def predict(self) -> None:
        context = self._context(ensure=True)
        if not self._require_image(context):
            return
        self._stop_area_mode()
        dialog = PredictDialog(context, self.window)
        if dialog.exec_() != QDialog.Accepted:
            return
        model_path = dialog.model_path()
        if not model_path or not Path(model_path).is_file():
            QMessageBox.warning(
                self.window,
                "YOLO predict",
                "Choose an existing .pt model first.",
            )
            return
        start, end = dialog.start_end()
        area = dialog.area()
        if dialog.run_on_kaggle():
            self._run_kaggle_predict(context, model_path, start, end, area)
        else:
            self._run_local_predict(
                context,
                model_path,
                start,
                end,
                area,
                dialog.merge_existing(),
            )

    def _run_local_predict(
        self,
        context: DatasetContext,
        model_path: str,
        start: int,
        end: int,
        area,
        merge_existing: bool,
    ) -> None:
        busy = self._busy_thread()
        if busy is not None:
            self.status("Another prediction/tracking job is already running.")
            return
        if YOLO is None:
            QMessageBox.warning(
                self.window,
                "YOLO predict",
                "ultralytics is not installed or could not be imported.",
            )
            return
        self.status("Loading model %s..." % os.path.basename(model_path), 0)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            model = YOLO(model_path)
        except Exception:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(
                self.window,
                "YOLO predict",
                "Could not load the model:\n\n" + traceback.format_exc(limit=3),
            )
            return
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

        progress = self._progress_dialog("YOLO predict", cancel=False)
        progress.setLabelText("Predicting %d frame(s)..." % (end - start + 1))
        thread = PredictThread(
            context.image_names,
            area,
            model,
            start,
            end,
            merge_existing=merge_existing,
        )
        thread.progress.connect(lambda value: progress.setValue(int(value)))
        thread.snapshot.connect(self._push_undo)
        thread.finished.connect(
            lambda updated, message: self._on_predict_finished(
                context, progress, updated, message
            )
        )
        thread.error.connect(
            lambda message: self._on_predict_error(context, progress, message)
        )
        self._predict_thread = thread
        self.status("YOLO prediction running...", 0)
        thread.start()

    def _on_predict_finished(
        self,
        context: DatasetContext,
        progress: QProgressDialog,
        updated: int,
        message: str,
    ) -> None:
        progress.close()
        self._predict_thread = None
        try:
            context.reload()
        except Exception:
            pass
        self.status(message, 10000)
        if "WARNING" in message:
            QMessageBox.warning(self.window, "YOLO predict", message)

    def _on_predict_error(
        self,
        context: DatasetContext,
        progress: QProgressDialog,
        message: str,
    ) -> None:
        progress.close()
        self._predict_thread = None
        try:
            context.reload()
        except Exception:
            pass
        text = str(message)
        if len(text) > 2200:
            text = text[:2200] + "\n..."
        QMessageBox.warning(self.window, "YOLO predict failed", text)

    def _run_kaggle_predict(
        self,
        context: DatasetContext,
        model_path: str,
        start: int,
        end: int,
        area,
    ) -> None:
        if not KAGGLE_UI_AVAILABLE:
            QMessageBox.warning(
                self.window,
                "Kaggle predict",
                "The Kaggle bridge is unavailable:\n\n" + (KAGGLE_UI_ERROR or ""),
            )
            return
        spec = JobSpec(
            job_type="yolo_predict",
            images_dir=str(context.images_dir),
            labels_dir=str(context.labels_dir),
            models_dir=str(context.models_dir),
            images=list(context.image_names),
            classes=list(context.classes),
            start_index=start,
            end_index=end,
            model_path=model_path,
            selection_area=area,
        )
        self._start_kaggle_spec(spec, "YOLO predict on Kaggle")

    # ------------------------------------------------------------------
    # SAM 2
    # ------------------------------------------------------------------
    def sam2_track(self) -> None:
        context = self._context(ensure=True)
        if not self._require_image(context):
            return
        self._stop_area_mode()
        dialog = Sam2Dialog(context, self.window)
        if dialog.exec_() != QDialog.Accepted:
            return
        if not dialog.boxes:
            self.status("Select boxes on the current frame first.")
            return
        if dialog.run_on_kaggle():
            self._run_kaggle_sam2(context, dialog)
        else:
            self._run_local_sam2(context, dialog)

    def _run_local_sam2(self, context: DatasetContext, dialog: Sam2Dialog) -> None:
        busy = self._busy_thread()
        if busy is not None:
            self.status("Another prediction/tracking job is already running.")
            return
        model_key = dialog.model_key()
        checkpoint = sam2_checkpoint_path(model_key)
        progress = self._progress_dialog("SAM 2 tracking", cancel=False)
        progress.setLabelText("Preparing SAM 2...")
        thread = Sam2TrackThread(
            images=context.image_names,
            start_index=dialog.start_index(),
            direction=dialog.direction(),
            boxes=dialog.boxes,
            images_dir=str(context.images_dir),
            labels_dir=str(context.labels_dir),
            checkpoint=checkpoint,
            min_confidence=dialog.min_confidence(),
            model_key=model_key,
        )
        thread.progress.connect(lambda value: progress.setValue(int(value)))
        thread.status.connect(progress.setLabelText)
        thread.snapshot.connect(self._push_undo)
        thread.done.connect(
            lambda message: self._on_sam2_done(context, progress, message)
        )
        thread.failed.connect(
            lambda message: self._on_sam2_failed(context, progress, message)
        )
        self._sam2_thread = thread
        self.status("SAM 2 tracking running...", 0)
        thread.start()

    def _on_sam2_done(
        self,
        context: DatasetContext,
        progress: QProgressDialog,
        message: str,
    ) -> None:
        progress.close()
        self._sam2_thread = None
        try:
            context.reload()
        except Exception:
            pass
        self.status(message, 10000)
        QMessageBox.information(self.window, "SAM 2 tracking", message)

    def _on_sam2_failed(
        self,
        context: DatasetContext,
        progress: QProgressDialog,
        message: str,
    ) -> None:
        progress.close()
        self._sam2_thread = None
        try:
            context.reload()
        except Exception:
            pass
        text = str(message)
        if len(text) > 2200:
            text = text[:2200] + "\n..."
        QMessageBox.warning(self.window, "SAM 2 failed", text)

    def _run_kaggle_sam2(self, context: DatasetContext, dialog: Sam2Dialog) -> None:
        if not KAGGLE_UI_AVAILABLE:
            QMessageBox.warning(
                self.window,
                "Kaggle SAM 2",
                "The Kaggle bridge is unavailable:\n\n" + (KAGGLE_UI_ERROR or ""),
            )
            return
        model_key = dialog.model_key()
        checkpoint = (
            sam2_checkpoint_path(model_key)
            if sam2_checkpoint_available(model_key)
            else ""
        )
        spec = JobSpec(
            job_type="sam2_track",
            images_dir=str(context.images_dir),
            labels_dir=str(context.labels_dir),
            models_dir=str(context.models_dir),
            images=list(context.image_names),
            classes=list(context.classes),
            start_index=dialog.start_index(),
            end_index=max(dialog.start_index(), context.total),
            direction=dialog.direction(),
            boxes=[list(box) for box in dialog.boxes],
            min_confidence=dialog.min_confidence(),
            model_key=model_key,
            checkpoint_path=checkpoint,
            sam2_config=SAM2_MODELS[model_key]["config"],
        )
        self._start_kaggle_spec(spec, "SAM 2 track on Kaggle")

    # ------------------------------------------------------------------
    # Kaggle
    # ------------------------------------------------------------------
    def _kaggle_blockers(self):
        if not KAGGLE_UI_AVAILABLE:
            return None, ["Kaggle bridge unavailable: " + (KAGGLE_UI_ERROR or "")]
        try:
            config = load_config()
        except Exception as exc:
            return None, ["Cannot load Kaggle configuration: %s" % exc]
        blockers = []
        if config.find_cli() is None:
            blockers.append(
                "Kaggle CLI not installed. Use Kaggle settings to install it."
            )
        if config.credentials_source() is None:
            blockers.append(
                "Kaggle credentials missing. Put kaggle.json in %s or paste a "
                "token in Kaggle settings." % config.kaggle_home
            )
        return config, blockers

    def _kaggle_status_lines(self) -> List[str]:
        config, blockers = self._kaggle_blockers()
        lines = []
        if config is not None:
            cli = config.find_cli()
            lines.append("CLI: %s" % (cli or "NOT FOUND"))
            lines.append(
                "Credentials: %s" % (config.credentials_source() or "NOT FOUND")
            )
            lines.append(
                "Accelerators: %s" % ", ".join(config.accelerators)
            )
            lines.append(
                "Mode: %s, queue timeout %g min"
                % (
                    "GPU only" if config.require_gpu else "GPU/CPU allowed",
                    config.queue_timeout_minutes,
                )
            )
        if blockers:
            lines.append("")
            lines.append("Setup needed:")
            lines.extend("- " + item for item in blockers)
        else:
            lines.append("")
            lines.append("Ready. Use Predict/SAM 2 with 'Run on Kaggle' checked.")
        return lines

    def kaggle_hub(self) -> None:
        dialog = KaggleHubDialog(self._kaggle_status_lines(), self.window)

        def refresh_after(handler):
            handler()
            dialog.set_status(self._kaggle_status_lines())

        def close_then(handler):
            dialog.accept()
            handler()

        dialog.settings_requested.connect(
            lambda: refresh_after(self.open_kaggle_settings)
        )
        dialog.jobs_requested.connect(
            lambda: refresh_after(self.open_kaggle_jobs)
        )
        dialog.session_requested.connect(
            lambda: close_then(self.start_kaggle_session)
        )
        dialog.test_requested.connect(lambda: close_then(self.test_kaggle))
        dialog.gpu_requested.connect(lambda: close_then(self.check_kaggle_gpu))
        dialog.exec_()

    def open_kaggle_settings(self) -> None:
        if not KAGGLE_UI_AVAILABLE:
            QMessageBox.warning(
                self.window, "Kaggle", KAGGLE_UI_ERROR or "Kaggle unavailable."
            )
            return
        KaggleSettingsDialog(self.window).exec_()

    def open_kaggle_jobs(self) -> None:
        if not KAGGLE_UI_AVAILABLE:
            return
        dialog = KaggleJobsDialog(self.window)
        dialog.resume_requested.connect(self._resume_kaggle_job)
        dialog.exec_()

    def start_kaggle_session(self) -> None:
        if not KAGGLE_UI_AVAILABLE:
            return
        context = self._context(ensure=True)
        if not self._require_image(context):
            return
        config, blockers = self._kaggle_blockers()
        if blockers:
            self._offer_kaggle_setup(blockers)
            return
        busy = self._busy_thread()
        if busy is not None:
            self.status("Another Kaggle/prediction job is already running.")
            return
        progress = self._progress_dialog("Kaggle session upload", cancel=False)
        progress.setLabelText("Preparing session images...")
        thread = KaggleSessionThread(
            str(context.images_dir),
            list(context.image_names),
            list(context.classes),
            parent=self.window,
        )
        thread.progress.connect(
            lambda value, message: (
                progress.setValue(int(value)),
                progress.setLabelText(message),
                self.status(message, 0),
            )
        )
        thread.status.connect(lambda message: progress.setLabelText(message))
        thread.done.connect(
            lambda info: self._on_session_done(progress, info)
        )
        thread.failed.connect(
            lambda message: self._on_session_failed(progress, message)
        )
        self._session_thread = thread
        thread.start()

    def _on_session_done(self, progress: QProgressDialog, info) -> None:
        progress.close()
        self._session_thread = None
        count = getattr(info, "image_count", 0)
        QMessageBox.information(
            self.window,
            "Kaggle session",
            "Session ready: %s image(s) uploaded once.\n"
            "Predict/SAM 2 clicks will now send only a command." % count,
        )

    def _on_session_failed(self, progress: QProgressDialog, message: str) -> None:
        progress.close()
        self._session_thread = None
        text = str(message)
        if len(text) > 2200:
            text = text[:2200] + "\n..."
        QMessageBox.warning(self.window, "Kaggle session failed", text)

    def test_kaggle(self) -> None:
        if not KAGGLE_UI_AVAILABLE:
            return
        self.status("Testing Kaggle connection...", 0)
        thread = KaggleTestThread(parent=self.window)
        self._aux_threads.append(thread)
        thread.done.connect(
            lambda message: QMessageBox.information(
                self.window, "Kaggle connection", message
            )
        )
        thread.failed.connect(
            lambda message: QMessageBox.warning(
                self.window, "Kaggle connection", str(message)[:2200]
            )
        )
        thread.finished.connect(lambda: self._forget_aux(thread))
        thread.start()

    def check_kaggle_gpu(self) -> None:
        if not KAGGLE_UI_AVAILABLE:
            return
        config, blockers = self._kaggle_blockers()
        if config is None or blockers:
            self._offer_kaggle_setup(blockers or ["Kaggle configuration missing."])
            return
        progress = self._progress_dialog("Kaggle GPU check", cancel=False)
        progress.setLabelText("Checking GPU accelerators (max 45 s each)...")
        thread = KaggleGPUCheckThread(
            config.accelerators, wait_seconds=45, parent=self.window
        )
        self._aux_threads.append(thread)
        thread.log.connect(lambda message: progress.setLabelText(message))
        thread.done.connect(
            lambda message: self._finish_gpu_check(progress, message, None)
        )
        thread.failed.connect(
            lambda message: self._finish_gpu_check(progress, None, message)
        )
        thread.finished.connect(lambda: self._forget_aux(thread))
        thread.start()

    def _finish_gpu_check(
        self,
        progress: QProgressDialog,
        message: Optional[str],
        error: Optional[str],
    ) -> None:
        progress.close()
        if error:
            QMessageBox.warning(self.window, "Kaggle GPU check", str(error)[:2200])
        else:
            QMessageBox.information(
                self.window, "Kaggle GPU check", message or "No result."
            )

    def _forget_aux(self, thread) -> None:
        try:
            self._aux_threads.remove(thread)
        except ValueError:
            pass

    def _offer_kaggle_setup(self, blockers: Sequence[str]) -> None:
        answer = QMessageBox.question(
            self.window,
            "Kaggle setup needed",
            "Kaggle is not ready:\n\n- "
            + "\n- ".join(blockers)
            + "\n\nOpen Kaggle settings now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            self.open_kaggle_settings()

    def _start_kaggle_spec(self, spec, label: str) -> None:
        try:
            spec.validate()
        except SpecError as exc:
            QMessageBox.warning(self.window, "Kaggle job", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self.window, "Kaggle job", str(exc))
            return
        if not KAGGLE_UI_AVAILABLE:
            return
        config, blockers = self._kaggle_blockers()
        if blockers:
            self._offer_kaggle_setup(blockers)
            return
        busy = self._busy_thread()
        if busy is not None:
            self.status("Another prediction/tracking job is already running.")
            return
        thread = KaggleJobThread(spec=spec, parent=self.window)
        self._start_kaggle_thread(thread, label)

    def _resume_kaggle_job(self, job_id: str) -> None:
        if not KAGGLE_UI_AVAILABLE:
            return
        config, blockers = self._kaggle_blockers()
        if blockers:
            self._offer_kaggle_setup(blockers)
            return
        busy = self._busy_thread()
        if busy is not None:
            self.status("Another prediction/tracking job is already running.")
            return
        thread = KaggleJobThread(job_id=job_id, resume=True, parent=self.window)
        self._start_kaggle_thread(thread, "Resuming %s" % job_id)

    def _start_kaggle_thread(self, thread, label: str) -> None:
        context = self._context(ensure=True)
        if context is None:
            return
        if getattr(self.window, "file_path", None):
            try:
                context.save_current()
            except Exception:
                pass
        progress = self._progress_dialog("%s" % label, cancel=True)
        progress.setCancelButtonText("Cancel job")
        progress.setLabelText("Preparing Kaggle job...")
        progress.canceled.connect(thread.cancel)

        self._last_changed_names = []
        thread.progress.connect(lambda value: progress.setValue(int(value)))
        thread.status.connect(progress.setLabelText)
        thread.snapshot.connect(self._push_undo)
        thread.changed_files.connect(self._on_kaggle_changed)
        thread.job_created.connect(
            lambda job_id: progress.setLabelText("Kaggle job: %s" % job_id)
        )
        thread.done.connect(
            lambda message: self._on_kaggle_done(context, progress, message)
        )
        thread.cancelled.connect(
            lambda message: self._on_kaggle_cancelled(progress, message)
        )
        thread.failed.connect(
            lambda message: self._on_kaggle_failed(context, progress, message)
        )
        thread.finished.connect(lambda: self._on_kaggle_thread_finished(thread))
        self._kaggle_thread = thread
        self.status("%s: preparing..." % label, 0)
        thread.start()

    def _on_kaggle_changed(self, names) -> None:
        self._last_changed_names = [str(name) for name in (names or [])]

    def _on_kaggle_done(
        self,
        context: DatasetContext,
        progress: QProgressDialog,
        message: str,
    ) -> None:
        progress.close()
        try:
            context.reload()
        except Exception:
            pass
        self._show_first_changed(context)
        self.status(message, 10000)
        QMessageBox.information(self.window, "Kaggle job finished", message)

    def _on_kaggle_cancelled(
        self, progress: QProgressDialog, message: str
    ) -> None:
        progress.close()
        self.status(message or "Kaggle job cancelled.", 8000)
        QMessageBox.information(
            self.window,
            "Kaggle job cancelled",
            "Kaggle job cancelled. Local labels were not changed.",
        )

    def _on_kaggle_failed(
        self,
        context: DatasetContext,
        progress: QProgressDialog,
        message: str,
    ) -> None:
        progress.close()
        try:
            context.reload()
        except Exception:
            pass
        text = str(message)
        if len(text) > 2400:
            text = text[:2400] + "\n..."
        QMessageBox.warning(self.window, "Kaggle job failed", text)

    def _on_kaggle_thread_finished(self, thread) -> None:
        if self._kaggle_thread is thread:
            self._kaggle_thread = None

    def _show_first_changed(self, context: DatasetContext) -> None:
        changed = set(self._last_changed_names)
        if not changed or context.current_name in changed:
            return
        for index, name in enumerate(context.image_names):
            if name in changed and index < len(context.image_paths):
                try:
                    self.window.load_file(context.image_paths[index])
                except Exception:
                    pass
                return

    # ------------------------------------------------------------------
    # Undo / palette / help
    # ------------------------------------------------------------------
    def undo_action(self) -> None:
        if not self.undo.can_undo():
            self.status("Nothing to undo.")
            return
        context = self._context()
        if context is None:
            return
        try:
            ok = self.undo.undo(context)
        except Exception:
            QMessageBox.warning(
                self.window,
                "Undo",
                "Could not restore the label files:\n\n"
                + traceback.format_exc(limit=3),
            )
            return
        self.actions["undo"].setEnabled(self.undo.can_undo())
        if ok:
            self.status("Restored the label files from before the last action.")

    def open_palette(self) -> None:
        self._stop_area_mode()
        dialog = ActionPalette(self._palette_items, self.window)
        if dialog.exec_() != QDialog.Accepted:
            return
        action_id = dialog.selected_action_id()
        if action_id and action_id in self.actions:
            self.actions[action_id].trigger()

    def show_shortcuts(self) -> None:
        rows = [
            (title, shortcut)
            for _action_id, title, shortcut, _category in self._palette_items
            if shortcut
        ]
        rows.extend(
            [
                ("Toggle box selection", "Ctrl+click box"),
                ("Cancel area mode", "Esc"),
                ("Finish area selection", "Release mouse"),
            ]
        )
        ShortcutHelpDialog(rows, self.window).exec_()
