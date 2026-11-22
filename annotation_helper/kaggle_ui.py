"""Qt layer for the optional Kaggle T4 x2 offload.

Kept separate from ``annotation_helper_current.py`` so the core bridge stays
UI-free and testable.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import threading
import time
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
from kaggle_bridge.metadata import slugify
from kaggle_bridge.session import RemoteSession
from kaggle_bridge.setup import create_venv
from kaggle_bridge.spec import JobSpec, SpecError
from kaggle_bridge.weights import sync_weights
from kaggle_bridge.worker import JobCancelled, KaggleJobError, KaggleJobRunner


# ----------------------------------------------------------------------
# Background workers
# ----------------------------------------------------------------------
class KaggleJobThread(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    snapshot = pyqtSignal(dict)
    changed_files = pyqtSignal(list)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal(str)
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
        self._cancel_event = threading.Event()
        self._runner: Optional[KaggleJobRunner] = None

    def cancel(self) -> None:
        self._cancel_event.set()
        if self._runner is not None:
            self._runner.cancel()

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
                cancel_event=self._cancel_event,
            )
            self._runner = runner
            if self.job_id:
                result = runner.run(job_id=self.job_id, resume=True)
            else:
                result = runner.run(spec=self.spec)
            self.job_created.emit(result.job_id)
            self.changed_files.emit(list(result.changed_names or []))
            self.done.emit(result.message)
        except JobCancelled as exc:
            self.cancelled.emit(str(exc))
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


class KaggleSessionThread(QThread):
    progress = pyqtSignal(int, str)
    status = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, images_dir: str, images: List[str], classes: List[str], parent=None):
        super().__init__(parent)
        self.images_dir = images_dir
        self.images = list(images)
        self.classes = list(classes)

    def run(self):
        try:
            cfg = load_config()
            cli = KaggleCLI(cfg)
            username = cli.check_auth()
            session = RemoteSession(cfg, cli, log=self.status.emit)
            info = session.ensure_images(
                self.images_dir,
                self.images,
                self.classes,
                progress=lambda percent, message: (
                    self.progress.emit(int(percent), str(message)),
                    self.status.emit(str(message)),
                ),
                username=username,
            )
            self.done.emit(info)
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class KaggleGPUCheckThread(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    PROBE = (
        "import torch\n"
        "count = torch.cuda.device_count() if torch.cuda.is_available() else 0\n"
        "print('PROBE_GPU_COUNT', count)\n"
        "for i in range(count):\n"
        "    print('PROBE_GPU_NAME', i, torch.cuda.get_device_name(i))\n"
        "print('PROBE_DONE')\n"
    )

    def __init__(self, accelerators: List[str], wait_seconds: int = 45, parent=None):
        super().__init__(parent)
        self.accelerators = [a for a in accelerators if a and a.lower() != "cpu"]
        self.wait_seconds = max(15, int(wait_seconds))

    def _probe(self, cli: KaggleCLI, root: Path, accelerator: str) -> str:
        slug = slugify(accelerator)
        kernel_slug = f"labelimg-plus-gpu-check-{slug}"[:50].rstrip("-")
        kernel_id = f"{cli.cfg.username}/{kernel_slug}"
        title = f"LabelImg Plus GPU Check {accelerator}"[:50]
        folder = root / kernel_slug
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "probe.py").write_text(self.PROBE, encoding="utf-8")
        metadata = {
            "id": kernel_id,
            "title": title,
            "code_file": "probe.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "machine_shape": accelerator,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        try:
            push = cli.kernels_push(folder, accelerator=accelerator)
            match = re.search(r"Kernel version\s+(\d+)", push.output or "")
            version = int(match.group(1)) if match else None
            run_ref = f"{kernel_id}/{version}" if version else kernel_id
            status = None
            deadline = time.time() + self.wait_seconds
            while time.time() < deadline:
                status = cli.kernels_status(run_ref)
                if status.terminal:
                    break
                time.sleep(5)
            if status is not None and status.state == "COMPLETE":
                log_path = root / f"{slug}.log"
                text = cli.kernels_logs(run_ref, log_path)
                count = None
                names = []
                for line in text.splitlines():
                    if "PROBE_GPU_COUNT" in line:
                        match = re.search(r"PROBE_GPU_COUNT\s+(\d+)", line)
                        if match:
                            count = int(match.group(1))
                    if "PROBE_GPU_NAME" in line:
                        match = re.search(r"PROBE_GPU_NAME\s+\d+\s+(.+?)(?:\\n|\"|$)", line)
                        if match:
                            names.append(match.group(1).strip())
                if count is None:
                    return "COMPLETE (no GPU probe output)"
                if count <= 0:
                    return "CPU only (not usable)"
                return f"{count} GPU: {', '.join(names) or 'unknown'}"
            if status is not None and status.state == "QUEUED":
                return f"QUEUED after {self.wait_seconds}s (no allocation)"
            return status.state if status else "unknown"
        finally:
            try:
                cli.cancel_kernel(kernel_id)
            except Exception:
                pass
            shutil.rmtree(folder, ignore_errors=True)

    def run(self):
        try:
            cfg = load_config()
            cli = KaggleCLI(cfg, log=None)
            cli.check_auth()
            root = Path(cfg.job_root) / "gpu_check"
            shutil.rmtree(root, ignore_errors=True)
            root.mkdir(parents=True, exist_ok=True)
            lines = []
            for accelerator in self.accelerators:
                self.log.emit(f"Checking {accelerator} (max {self.wait_seconds}s)...")
                result = self._probe(cli, root, accelerator)
                self.log.emit(f"{accelerator}: {result}")
                lines.append(f"{accelerator}: {result}")
            self.done.emit("\n".join(lines))
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class KaggleWeightsThread(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def run(self):
        try:
            mapping = sync_weights(load_config(), log=self.log.emit)
            self.done.emit(
                f"Ready (uploaded once, now reused): {len(mapping)} weight file(s)."
            )
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


# ----------------------------------------------------------------------
# Settings / setup dialog
# ----------------------------------------------------------------------
class KaggleSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kaggle GPU setup")
        self.resize(700, 760)
        self.cfg: Config = load_config()
        self.setup_thread: Optional[KaggleSetupThread] = None
        self.test_thread: Optional[KaggleTestThread] = None
        self.weights_thread: Optional[KaggleWeightsThread] = None
        self.gpu_thread: Optional[KaggleGPUCheckThread] = None

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
            "The T4 x2 option needs a phone-verified Kaggle account.<br><br>"
            "<b>Models upload once:</b> use <i>Sync weights</i> (or the first "
            "real job). Later jobs send only images + labels and return "
            "coordinates, which are shown in LabelImg.<br><br>"
            "<b>GPU-only:</b> T4 x2 is tried first. If it stays queued longer "
            "than the fixed queue timeout, the next accelerator is tried. "
            "CPU is never used while <i>Require GPU</i> is checked."
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

        self.edit_accels = QLineEdit(",".join(self.cfg.accelerators))
        self.edit_accels.setToolTip(
            "Fallback order. T4 x2 first, then L4 x1/H100/A100 if available. "
            "CPU is not used while 'Require GPU' is checked."
        )
        form.addRow("Accelerators:", self.edit_accels)

        self.spin_queue = QDoubleSpinBox()
        self.spin_queue.setRange(0.5, 60.0)
        self.spin_queue.setSingleStep(0.5)
        self.spin_queue.setValue(float(self.cfg.queue_timeout_minutes))
        self.spin_queue.setSuffix(" min")
        self.spin_queue.setToolTip(
            "If an accelerator stays QUEUED for this long, cancel it and try "
            "the next one. 1 min is the fast default; use 5 for patient mode. "
            "If none starts, the job stops (CPU is not used)."
        )
        form.addRow("Queue timeout/GPU:", self.spin_queue)

        self.chk_require_gpu = QCheckBox("Require GPU (never import CPU results)")
        self.chk_require_gpu.setChecked(bool(self.cfg.require_gpu))
        form.addRow("", self.chk_require_gpu)

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

        self.btn_weights = QPushButton("Sync weights")
        self.btn_weights.setObjectName("secondaryButton")
        self.btn_weights.setToolTip(
            "Upload local YOLO models and SAM 2 checkpoints once; later jobs "
            "send only images and reuse these weights."
        )
        self.btn_weights.clicked.connect(self.start_weights)
        buttons.addWidget(self.btn_weights)

        self.btn_gpu_check = QPushButton("Check GPUs")
        self.btn_gpu_check.setObjectName("secondaryButton")
        self.btn_gpu_check.setToolTip(
            "Run tiny probe kernels to see which accelerators actually give a "
            "GPU. Nothing is changed locally."
        )
        self.btn_gpu_check.clicked.connect(self.start_gpu_check)
        buttons.addWidget(self.btn_gpu_check)
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
        for widget in (self.btn_venv, self.btn_test, self.btn_weights,
                       self.btn_gpu_check, self.btn_save, self.btn_cancel,
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

    def start_weights(self) -> None:
        if self.weights_thread is not None and self.weights_thread.isRunning():
            return
        self._append("Syncing local models/checkpoints to Kaggle (one-time)...")
        self.weights_thread = KaggleWeightsThread(parent=self)
        self.weights_thread.log.connect(self._append)
        self.weights_thread.done.connect(self.on_weights_done)
        self.weights_thread.failed.connect(self.on_weights_failed)
        self.weights_thread.finished.connect(lambda: self._busy(False))
        self._busy(True)
        self.weights_thread.start()

    def on_weights_done(self, message: str) -> None:
        self._append(message)
        QMessageBox.information(self, "Kaggle weights", message)

    def on_weights_failed(self, message: str) -> None:
        self._append(message)
        QMessageBox.warning(self, "Kaggle weights failed", message)

    def start_gpu_check(self) -> None:
        if self.gpu_thread is not None and self.gpu_thread.isRunning():
            return
        accelerators = [
            part.strip()
            for part in self.edit_accels.text().split(",")
            if part.strip() and part.strip().lower() != "cpu"
        ]
        if not accelerators:
            QMessageBox.warning(
                self, "GPU check", "Enter at least one GPU accelerator."
            )
            return
        self._append("Quick GPU check: up to 20s per accelerator...")
        self.gpu_thread = KaggleGPUCheckThread(
            accelerators, wait_seconds=20, parent=self
        )
        self.gpu_thread.log.connect(self._append)
        self.gpu_thread.done.connect(self.on_gpu_check_done)
        self.gpu_thread.failed.connect(self.on_gpu_check_failed)
        self.gpu_thread.finished.connect(lambda: self._busy(False))
        self._busy(True)
        self.gpu_thread.start()

    def on_gpu_check_done(self, message: str) -> None:
        self._append(message)
        QMessageBox.information(self, "GPU check result", message)

    def on_gpu_check_failed(self, message: str) -> None:
        self._append(message)
        QMessageBox.warning(self, "GPU check failed", message)

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
        require_gpu = self.chk_require_gpu.isChecked()
        accelerators = [
            part.strip()
            for part in self.edit_accels.text().split(",")
            if part.strip() and not (require_gpu and part.strip().lower() == "cpu")
        ]
        if not accelerators:
            QMessageBox.warning(
                self, "Kaggle setup", "Enter at least one accelerator."
            )
            return
        updates["KAGGLE_ACCELERATORS"] = ",".join(accelerators)
        updates["KAGGLE_QUEUE_TIMEOUT_MINUTES"] = str(self.spin_queue.value())
        updates["KAGGLE_REQUIRE_GPU"] = "true" if require_gpu else "false"
        updates["KAGGLE_CPU_FALLBACK"] = "false" if require_gpu else "true"
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
            or (self.weights_thread is not None and self.weights_thread.isRunning())
            or (self.gpu_thread is not None and self.gpu_thread.isRunning())
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
