"""Qt layer for the optional Kaggle T4 x2 offload.

Kept separate from ``annotation_helper_current.py`` so the core bridge stays
UI-free and testable.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PyQt5.QtCore import QThread, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from kaggle_bridge.config import (
    Config,
    ConfigError,
    load_config,
    parse_env_file,
    upsert_env_file,
)
from kaggle_bridge.doctor import doctor_lines
from kaggle_bridge.jobstore import JobBusyError, JobNotFoundError, JobStore
from kaggle_bridge.kaggle_cli import KaggleCLI, KaggleCliError
from kaggle_bridge.setup import create_venv
from kaggle_bridge.spec import JobSpec, SpecError
from kaggle_bridge.worker import KaggleJobError, KaggleJobRunner


# ----------------------------------------------------------------------
# Background workers
# ----------------------------------------------------------------------
class KaggleJobThread(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    snapshot = pyqtSignal(dict)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)
    job_created = pyqtSignal(str)

    def __init__(self, spec: Optional[JobSpec] = None, job_id: Optional[str] = None,
                 resume: bool = False, parent=None):
        super().__init__(parent)
        if spec is None and not job_id:
            raise ValueError("spec or job_id is required")
        self.spec = spec
        self.job_id = job_id
        self.resume = resume
        self._log_lines: List[str] = []

    def _log(self, message: str) -> None:
        self._log_lines.append(str(message))
        if len(self._log_lines) > 400:
            self._log_lines = self._log_lines[-400:]

    def run(self):
        try:
            cfg = load_config()
            store = JobStore(cfg.job_root)
            runner = KaggleJobRunner(
                cfg,
                store,
                progress=lambda percent, message: (
                    self.progress.emit(int(percent)),
                    self.status.emit(str(message)),
                ),
                status=lambda message: self.status.emit(str(message)),
                log=self._log,
                snapshot=lambda data: self.snapshot.emit(dict(data)),
            )
            if self.job_id:
                result = runner.run(job_id=self.job_id, resume=True)
            else:
                result = runner.run(spec=self.spec)
            self.job_created.emit(result.job_id)
            self.done.emit(result.message)
        except (KaggleJobError, JobBusyError, ConfigError, SpecError, JobNotFoundError) as exc:
            details = str(exc)
            job_id = getattr(exc, "job_id", "") or self.job_id or ""
            log_path = getattr(exc, "log_path", None)
            if job_id:
                details += f"\n\nJob: {job_id}"
            if log_path:
                details += f"\nLog: {log_path}"
            self.failed.emit(details)
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class KaggleSetupThread(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, force: bool = False, parent=None):
        super().__init__(parent)
        self.force = force

    def run(self):
        try:
            cfg = load_config()
            create_venv(cfg, force=self.force, log=self.log.emit)
            self.done.emit("Kaggle CLI is ready.")
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class KaggleTestThread(QThread):
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def run(self):
        try:
            cfg = load_config()
            cli = KaggleCLI(cfg)
            version = cli.version()
            username = cli.check_auth()
            self.done.emit(f"Connected as {username} ({version}).")
        except Exception as exc:
            self.failed.emit(str(exc))


# ----------------------------------------------------------------------
# Settings / setup dialog
# ----------------------------------------------------------------------
class KaggleSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kaggle T4 x2 setup")
        self.resize(640, 620)
        self.cfg: Config = load_config()
        self.setup_thread: Optional[KaggleSetupThread] = None
        self.test_thread: Optional[KaggleTestThread] = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        intro = QLabel(
            "<b>Heavy work runs on Kaggle, not on this laptop.</b><br>"
            "The job uploads frames, current labels and the selected model, runs "
            "YOLO/SAM 2 on a T4 x2 kernel, then merges the result back here.<br><br>"
            "Create a token at "
            '<a href="https://www.kaggle.com/settings/api">'
            "kaggle.com/settings/api</a> and save it as "
            "<code>%USERPROFILE%\\.kaggle\\kaggle.json</code>. "
            "The T4 x2 option needs a phone-verified Kaggle account."
        )
        intro.setWordWrap(True)
        intro.setTextInteractionFlags(Qt.TextBrowserInteraction)
        intro.setOpenExternalLinks(True)
        layout.addWidget(intro)

        source = QLabel(
            "Current credentials: "
            + (self.cfg.credentials_source() or "NOT FOUND")
        )
        source.setWordWrap(True)
        layout.addWidget(source)

        form_box = QGroupBox("Optional: store credentials / job settings in .env")
        form = QFormLayout(form_box)
        self.edit_username = QLineEdit(self.cfg.username)
        self.edit_username.setPlaceholderText("from kaggle.json if blank")
        form.addRow("Kaggle username:", self.edit_username)

        self.edit_key = QLineEdit()
        self.edit_key.setEchoMode(QLineEdit.Password)
        self.edit_key.setPlaceholderText("legacy API key (optional)")
        form.addRow("Legacy key:", self.edit_key)

        self.edit_token = QLineEdit()
        self.edit_token.setEchoMode(QLineEdit.Password)
        self.edit_token.setPlaceholderText("new access token (optional)")
        form.addRow("API token:", self.edit_token)

        self.edit_prefix = QLineEdit(self.cfg.dataset_prefix)
        form.addRow("Dataset prefix:", self.edit_prefix)

        self.spin_poll = QSpinBox()
        self.spin_poll.setRange(5, 3600)
        self.spin_poll.setValue(int(self.cfg.poll_seconds))
        self.spin_poll.setSuffix(" sec")
        form.addRow("Poll interval:", self.spin_poll)

        self.spin_timeout = QDoubleSpinBox()
        self.spin_timeout.setRange(0.25, 24.0)
        self.spin_timeout.setSingleStep(0.5)
        self.spin_timeout.setValue(float(self.cfg.timeout_hours))
        self.spin_timeout.setSuffix(" hours")
        form.addRow("Job timeout:", self.spin_timeout)

        self.chk_force = QCheckBox("Recreate .kaggle-venv if it is broken")
        form.addRow("", self.chk_force)
        layout.addWidget(form_box)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(130)
        self.log_view.setFont(QFont("Consolas", 8))
        layout.addWidget(self.log_view)

        buttons = QHBoxLayout()
        self.btn_venv = QPushButton("Install / repair Kaggle CLI")
        self.btn_venv.setObjectName("secondaryButton")
        self.btn_venv.clicked.connect(self.start_venv)
        buttons.addWidget(self.btn_venv)

        self.btn_test = QPushButton("Test connection")
        self.btn_test.setObjectName("secondaryButton")
        self.btn_test.clicked.connect(self.start_test)
        buttons.addWidget(self.btn_test)
        buttons.addStretch(1)

        self.btn_save = QPushButton("Save")
        self.btn_save.setObjectName("primaryButton")
        self.btn_save.clicked.connect(self.save_and_close)
        buttons.addWidget(self.btn_save)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        buttons.addWidget(self.btn_cancel)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------
    def _busy(self, busy: bool) -> None:
        for widget in (self.btn_venv, self.btn_test, self.btn_save, self.btn_cancel,
                       self.chk_force):
            widget.setEnabled(not busy)

    def _append(self, text: str) -> None:
        self.log_view.appendPlainText(str(text))

    def start_venv(self) -> None:
        if self.setup_thread is not None and self.setup_thread.isRunning():
            return
        self._append("Preparing .kaggle-venv (Python 3.11+)...")
        self.setup_thread = KaggleSetupThread(force=self.chk_force.isChecked(), parent=self)
        self.setup_thread.log.connect(self._append)
        self.setup_thread.done.connect(self.on_setup_done)
        self.setup_thread.failed.connect(self.on_setup_failed)
        self.setup_thread.finished.connect(self._setup_finished)
        self._busy(True)
        self.setup_thread.start()

    def _setup_finished(self) -> None:
        self._busy(False)

    def on_setup_done(self, message: str) -> None:
        self._append(message)
        self.cfg = load_config()
        QMessageBox.information(self, "Kaggle CLI", message)

    def on_setup_failed(self, message: str) -> None:
        self._append(message)
        QMessageBox.warning(self, "Kaggle CLI setup failed", message)

    def start_test(self) -> None:
        if self.test_thread is not None and self.test_thread.isRunning():
            return
        self._append("Testing Kaggle connection...")
        self.test_thread = KaggleTestThread(parent=self)
        self.test_thread.done.connect(self.on_test_done)
        self.test_thread.failed.connect(self.on_test_failed)
        self.test_thread.finished.connect(lambda: self._busy(False))
        self._busy(True)
        self.test_thread.start()

    def on_test_done(self, message: str) -> None:
        self._append(message)
        QMessageBox.information(self, "Kaggle connection", message)

    def on_test_failed(self, message: str) -> None:
        self._append(message)
        QMessageBox.warning(self, "Kaggle connection failed", message)

    def save_and_close(self) -> None:
        updates: Dict[str, str] = {}
        username = self.edit_username.text().strip()
        key = self.edit_key.text().strip()
        token = self.edit_token.text().strip()
        if username:
            updates["KAGGLE_USERNAME"] = username
        if key:
            if not username and not self.cfg.username:
                QMessageBox.warning(
                    self, "Kaggle key", "Enter the Kaggle username with a legacy key."
                )
                return
            updates["KAGGLE_KEY"] = key
        if token:
            updates["KAGGLE_API_TOKEN"] = token
        if key and token:
            QMessageBox.warning(
                self, "Kaggle credentials",
                "Fill either the legacy key or the new API token, not both."
            )
            return
        updates["KAGGLE_DATASET_PREFIX"] = self.edit_prefix.text().strip() or "labelimg-plus-job"
        updates["KAGGLE_POLL_SECONDS"] = str(self.spin_poll.value())
        updates["KAGGLE_TIMEOUT_HOURS"] = str(self.spin_timeout.value())
        try:
            upsert_env_file(self.cfg.env_path, updates)
        except OSError as exc:
            QMessageBox.critical(self, "Kaggle setup", f"Could not write .env: {exc}")
            return
        self.accept()

    def closeEvent(self, event):
        running = (
            (self.setup_thread is not None and self.setup_thread.isRunning())
            or (self.test_thread is not None and self.test_thread.isRunning())
        )
        if running:
            QMessageBox.information(
                self, "Kaggle setup", "A setup/test task is still running. Wait for it to finish."
            )
            event.ignore()
            return
        super().closeEvent(event)


# ----------------------------------------------------------------------
# Jobs dialog
# ----------------------------------------------------------------------
class KaggleJobsDialog(QDialog):
    resume_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kaggle jobs")
        self.resize(980, 560)
        self.cfg = load_config()
        self.store = JobStore(self.cfg.job_root)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(f"Local job history: {self.cfg.job_root}")
        )

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Job ID", "Type", "State", "Progress", "Created", "Message"]
        )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_selection)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(120)
        self.log_view.setFont(QFont("Consolas", 8))
        layout.addWidget(self.log_view)

        buttons = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setObjectName("secondaryButton")
        self.btn_refresh.clicked.connect(self.refresh)
        buttons.addWidget(self.btn_refresh)

        self.btn_open = QPushButton("Open job folder")
        self.btn_open.setObjectName("secondaryButton")
        self.btn_open.clicked.connect(self.open_folder)
        buttons.addWidget(self.btn_open)

        self.btn_resume = QPushButton("Resume selected")
        self.btn_resume.setObjectName("secondaryButton")
        self.btn_resume.clicked.connect(self.resume_selected)
        buttons.addWidget(self.btn_resume)

        buttons.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        self.jobs: List[dict] = []
        self.refresh()

    def refresh(self) -> None:
        try:
            self.jobs = self.store.list_jobs(limit=200)
        except OSError as exc:
            self.log_view.setPlainText(f"Cannot read jobs: {exc}")
            return
        self.table.setRowCount(len(self.jobs))
        for row, job in enumerate(self.jobs):
            values = [
                str(job.get("job_id", "")),
                str(job.get("job_type", "")),
                str(job.get("state", "")),
                f"{int(job.get('progress') or 0)}%",
                str(job.get("created_at", ""))[:19],
                str(job.get("message", "")),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.UserRole, job.get("job_id"))
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        if self.jobs:
            self.table.selectRow(0)

    def selected_job(self) -> Optional[dict]:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.jobs):
            return None
        return self.jobs[row]

    def on_selection(self) -> None:
        job = self.selected_job()
        if not job:
            self.log_view.clear()
            return
        job_id = job.get("job_id", "")
        self.log_view.setPlainText(self.store.read_log_tail(job_id, lines=60))

    def open_folder(self) -> None:
        job = self.selected_job()
        if not job:
            return
        path = self.store.job_dir(job["job_id"])
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def resume_selected(self) -> None:
        job = self.selected_job()
        if not job:
            return
        if job.get("state") == "done":
            QMessageBox.information(self, "Kaggle jobs", "This job is already done.")
            return
        self.resume_requested.emit(job["job_id"])
        self.accept()
