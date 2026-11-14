"""End-to-end local orchestration for one Kaggle job.

Flow::

    bundle -> private dataset -> GPU kernel (T4 x2) -> output -> local labels

The worker is deliberately UI-free.  The Qt layer only supplies callbacks.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .bundle import BundleInfo, build_bundle
from .config import Config, ConfigError, load_config
from .importer import ImportResult, import_job_output, load_import_metadata
from .jobstore import JobBusyError, JobStore
from .kaggle_cli import KaggleCLI, KaggleCliError, KernelStatus
from .metadata import (
    dataset_id_for,
    dataset_metadata,
    job_title,
    kernel_id_for,
    kernel_metadata,
    kernel_title,
    title_from_slug,
    write_json_atomic,
)
from .session import (
    RemoteSession,
    build_command_payload,
    embed_command,
)
from .spec import JobSpec
from .weights import WeightsManager

ProgressFn = Optional[Callable[[int, str], None]]
StatusFn = Optional[Callable[[str], None]]
LogFn = Optional[Callable[[str], None]]
SnapshotFn = Optional[Callable[[Dict[str, List[str]]], None]]


class KaggleJobError(RuntimeError):
    def __init__(
        self,
        message: str,
        job_id: str = "",
        log_path: Optional[Path] = None,
    ):
        super().__init__(message)
        self.job_id = job_id
        self.log_path = log_path


class JobCancelled(RuntimeError):
    pass


class QueueTimeout(RuntimeError):
    """The GPU stayed queued longer than the configured timeout."""


@dataclass
class JobResult:
    job_id: str
    changed: int
    boxes: int
    message: str
    snapshot: Dict[str, List[str]] = field(default_factory=dict)
    output_dir: Optional[Path] = None
    result: Dict[str, Any] = field(default_factory=dict)
    changed_names: List[str] = field(default_factory=list)


class KaggleJobRunner:
    def __init__(
        self,
        cfg: Config,
        store: JobStore,
        progress: ProgressFn = None,
        status: StatusFn = None,
        log: LogFn = None,
        snapshot: SnapshotFn = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        cancel_event: Optional[threading.Event] = None,
    ):
        self.cfg = cfg
        self.store = store
        self.progress_cb = progress
        self.status_cb = status
        self.log_cb = log
        self.snapshot_cb = snapshot
        self.sleep = sleep_fn
        self.cancel_event = cancel_event or threading.Event()
        self._cli: Optional[KaggleCLI] = None

    def cancel(self) -> None:
        self.cancel_event.set()

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled("Cancelled by user.")

    # ------------------------------------------------------------------
    # Callbacks / state
    # ------------------------------------------------------------------
    def _progress(self, percent: int, message: str) -> None:
        if self.progress_cb:
            try:
                self.progress_cb(max(0, min(100, int(percent))), message)
            except Exception:
                pass
        self._log(message)

    def _status(self, message: str) -> None:
        if self.status_cb:
            try:
                self.status_cb(message)
            except Exception:
                pass

    def _log(self, message: str, job_id: str = "") -> None:
        if job_id:
            self.store.append_log(job_id, message)
        if self.log_cb:
            try:
                self.log_cb(message)
            except Exception:
                pass

    def _update(self, job_id: str, **fields) -> Dict[str, Any]:
        record = self.store.update(job_id, **fields)
        if "message" in fields:
            self._status(str(fields["message"]))
        return record

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        spec: Optional[JobSpec] = None,
        job_id: Optional[str] = None,
        resume: bool = False,
    ) -> JobResult:
        if spec is None and not job_id:
            raise ValueError("Provide a JobSpec or a job_id")
        if job_id:
            record = self.store.load(job_id)
            spec = JobSpec.from_dict(record.get("spec") or {})
            spec.validate()
        else:
            assert spec is not None
            spec.validate()
            record = self.store.create(spec)
            job_id = record["job_id"]

        assert spec is not None and job_id is not None
        self.store.acquire_lock(job_id)
        kernel_ref: Optional[str] = None
        try:
            self._check_cancel()
            self._log(f"Starting Kaggle job {job_id} ({spec.job_type})", job_id)
            cli = KaggleCLI(self.cfg, log=lambda msg: self._log(msg, job_id))
            self._cli = cli
            username = cli.check_auth()
            self.cfg.username = username
            self._log(f"Kaggle user: {username}", job_id)

            use_session = bool(self.cfg.use_session_dataset)
            if use_session:
                dataset_id = None
                if self.cfg.fixed_kernel:
                    kernel_id = f"{username}/{self.cfg.kernel_slug}"
                    fixed_title = title_from_slug(self.cfg.kernel_slug)
                else:
                    kernel_id = kernel_id_for(username, job_id)
                    fixed_title = kernel_title(job_id)
            else:
                dataset_id = dataset_id_for(
                    username, self.cfg.dataset_prefix, job_id
                )
                # Kaggle rejects a kernel and a dataset that share the same
                # slug (409 SaveKernel), so the kernel gets its own prefix.
                kernel_id = kernel_id_for(username, job_id)
                fixed_title = kernel_title(job_id)
            kernel_ref = kernel_id
            weights_dataset_id = None
            self._update(
                job_id,
                dataset_ref=dataset_id,
                kernel_ref=kernel_id,
                fixed_kernel_title=fixed_title,
                message=f"Kaggle user: {username}",
            )

            state = record.get("state", "created")
            if resume and kernel_ref:
                handled = self._try_resume(job_id, spec, cli, kernel_ref)
                if handled is not None:
                    return handled

            # ----------------------------------------------------------
            # 0. Reusable model weights (uploaded only once)
            # ----------------------------------------------------------
            model_override = None
            if self.cfg.use_weights_dataset:
                self._update(
                    job_id,
                    state="bundling",
                    progress=2,
                    message="Checking reusable model weights on Kaggle...",
                    error=None,
                )
                manager = WeightsManager(
                    self.cfg,
                    cli,
                    log=lambda msg: self._log(msg, job_id),
                )
                weights_dataset_id = manager.dataset_id(username)
                if spec.job_type == "yolo_predict":
                    remote_name = manager.ensure_file(Path(spec.model_path), username)
                    model_override = {
                        "kind": "yolo",
                        "source": "weights",
                        "weights_file": remote_name,
                        "weights_dataset": weights_dataset_id,
                    }
                else:
                    info = spec.model_info()
                    if (
                        self.cfg.include_sam2_checkpoint
                        and spec.checkpoint_path
                        and Path(spec.checkpoint_path).is_file()
                    ):
                        remote_name = manager.ensure_file(
                            Path(spec.checkpoint_path), username
                        )
                        model_override = {
                            "kind": "sam2",
                            "source": "weights",
                            "model_key": spec.model_key,
                            "weights_file": remote_name,
                            "weights_dataset": weights_dataset_id,
                            "config": info["config"],
                            "checkpoint_included": False,
                        }
                    else:
                        model_override = {
                            "kind": "sam2",
                            "source": "url",
                            "model_key": spec.model_key,
                            "file": None,
                            "config": info["config"],
                            "checkpoint_url": info["url"],
                            "checkpoint_included": False,
                        }
                self._update(
                    job_id,
                    weights_dataset=weights_dataset_id,
                    message=(
                        "Model weights ready on Kaggle; only images/labels "
                        "will be uploaded for this job."
                    ),
                )

            if use_session:
                return self._run_session_mode(
                    job_id=job_id,
                    spec=spec,
                    cli=cli,
                    username=username,
                    kernel_id=kernel_id,
                    fixed_title=fixed_title,
                    weights_dataset_id=weights_dataset_id,
                    model_override=model_override,
                )

            # ----------------------------------------------------------
            # 1. Bundle
            # ----------------------------------------------------------
            self._update(
                job_id,
                state="bundling",
                progress=2,
                message=(
                    "Packing frames and labels (model reused)..."
                    if model_override is not None
                    else "Packing frames, labels and model..."
                ),
                error=None,
            )
            bundle = build_bundle(
                spec,
                self.store.job_dir(job_id),
                include_sam2_checkpoint=self.cfg.include_sam2_checkpoint,
                keep_bundle=self.cfg.keep_bundle,
                max_bundle_mb=self.cfg.max_bundle_mb,
                progress=lambda pct, msg: self._progress(pct, msg),
                log=lambda msg: self._log(msg, job_id),
                model_override=model_override,
            )
            self._update(
                job_id,
                progress=48,
                bundle_zip=str(bundle.job_zip),
                bundle_bytes=bundle.size_bytes,
                images_count=bundle.image_count,
                message=(
                    f"Bundle ready ({bundle.size_bytes / 1024 / 1024:.1f} MB, "
                    f"{bundle.image_count} frames)"
                ),
            )

            # ----------------------------------------------------------
            # 2. Dataset
            # ----------------------------------------------------------
            self._update(
                job_id,
                state="uploading_dataset",
                progress=50,
                message=(
                    f"Uploading {bundle.size_bytes / 1024 / 1024:.1f} MB to Kaggle..."
                ),
            )
            dataset_dir = self._prepare_dataset_folder(job_id, bundle, dataset_id)
            record = self.store.load(job_id)
            dataset_created = bool(record.get("dataset_created"))
            try:
                if dataset_created:
                    self._log(
                        f"Dataset flag set; creating a new version: {dataset_id}",
                        job_id,
                    )
                    cli.dataset_version(dataset_dir, f"Job {job_id} retry")
                else:
                    try:
                        cli.dataset_create(dataset_dir)
                        self._update(job_id, dataset_created=True)
                    except KaggleCliError as exc:
                        message = str(exc).lower()
                        if "already in use" not in message and "already exists" not in message:
                            raise
                        # A previous attempt created the dataset but crashed
                        # before persisting the flag.  Update it instead.
                        self._log(
                            f"Dataset already exists; creating a version: {dataset_id}",
                            job_id,
                        )
                        cli.dataset_version(dataset_dir, f"Job {job_id} retry")
                        self._update(job_id, dataset_created=True)
                self._update(
                    job_id,
                    state="waiting_dataset",
                    progress=56,
                    dataset_ref=dataset_id,
                    dataset_url=f"https://www.kaggle.com/datasets/{dataset_id}",
                    message="Waiting for Kaggle to finish processing the dataset...",
                )
                cli.wait_dataset_ready(dataset_id, timeout=1800, poll=5)
            finally:
                if not self.cfg.keep_bundle:
                    shutil.rmtree(dataset_dir, ignore_errors=True)
            self._check_cancel()

            # ----------------------------------------------------------
            # 3-5. Kernel attempts with accelerator fallback
            # ----------------------------------------------------------
            last_error: Optional[Exception] = None
            for accelerator in self.cfg.accelerators:
                self._check_cancel()
                label = (
                    "CPU"
                    if str(accelerator).lower() == "cpu"
                    else str(accelerator)
                )
                self._update(
                    job_id,
                    state="pushing_kernel",
                    progress=60,
                    accelerator=accelerator,
                    message=f"Pushing Kaggle kernel on {label}...",
                )
                kernel_dir = self._prepare_kernel_folder(
                    job_id,
                    dataset_id,
                    kernel_id,
                    extra_dataset_sources=(
                        [weights_dataset_id] if weights_dataset_id else []
                    ),
                    accelerator=accelerator,
                )
                try:
                    push_result = cli.kernels_push(
                        kernel_dir, accelerator=accelerator
                    )
                except KaggleCliError as exc:
                    last_error = exc
                    self._log(
                        f"Accelerator {label} could not be used: {exc}", job_id
                    )
                    continue
                version = self._parse_kernel_version(push_result.output)
                kernel_run_ref = (
                    f"{kernel_id}/{version}" if version else kernel_id
                )
                self._update(
                    job_id,
                    state="queued",
                    progress=63,
                    kernel_ref=kernel_id,
                    kernel_run_ref=kernel_run_ref,
                    accelerator=accelerator,
                    kernel_url=f"https://www.kaggle.com/code/{kernel_id}",
                    message=(
                        f"Kernel version {version} queued on {label}."
                        if version
                        else f"Kernel queued on {label}."
                    ),
                )
                self._log(push_result.output, job_id)
                try:
                    status = self._wait_for_kernel(
                        job_id,
                        cli,
                        kernel_run_ref,
                        spec,
                        queue_timeout_minutes=self.cfg.queue_timeout_minutes,
                    )
                except QueueTimeout as exc:
                    last_error = exc
                    self._log(
                        f"{label} queue timeout; trying the next accelerator.",
                        job_id,
                    )
                    try:
                        cli.cancel_kernel(kernel_id)
                    except Exception as cancel_exc:
                        self._log(
                            f"Could not cancel queued kernel: {cancel_exc}",
                            job_id,
                        )
                    continue
                if not status.success:
                    logs = self._fetch_kernel_logs(
                        job_id, cli, kernel_run_ref
                    )
                    raise KaggleJobError(
                        "Kaggle kernel did not complete successfully: "
                        f"{status.state}. {status.failure or ''}\n{logs}".strip(),
                        job_id,
                        self.store.job_dir(job_id) / "logs" / "kernel.log",
                    )
                return self._download_and_import(
                    job_id, spec, cli, kernel_run_ref
                )

            raise self._no_accelerator_error(job_id, last_error)

        except JobCancelled as exc:
            try:
                record = self.store.load(job_id)
                ref = record.get("kernel_run_ref") or record.get("kernel_ref") or kernel_ref
                if ref:
                    cli = self._cli or KaggleCLI(self.cfg)
                    cli.cancel_kernel(ref)
                    self._log(f"Cancelled remote kernel {ref}", job_id)
            except Exception as cancel_exc:
                self._log(f"Could not cancel remote kernel: {cancel_exc}", job_id)
            self._update(
                job_id,
                state="cancelled",
                progress=0,
                message=str(exc),
                error=None,
            )
            raise
        except KaggleJobError as exc:
            self._mark_failed(job_id, str(exc))
            raise
        except JobBusyError:
            raise
        except (ConfigError, KaggleCliError, ValueError) as exc:
            self._mark_failed(job_id, str(exc))
            raise KaggleJobError(str(exc), job_id, self.store.log_path(job_id)) from exc
        except Exception as exc:
            tb = traceback.format_exc()
            self._log("ERROR:\n" + tb, job_id)
            self._mark_failed(job_id, f"{exc}\n{tb}")
            if kernel_ref:
                try:
                    cli = self._cli or KaggleCLI(self.cfg)
                    self._fetch_kernel_logs(job_id, cli, kernel_ref)
                except Exception:
                    pass
            raise KaggleJobError(
                f"{exc}\n\nJob folder: {self.store.job_dir(job_id)}",
                job_id,
                self.store.log_path(job_id),
            ) from exc
        finally:
            self.store.release_lock(job_id)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    def _try_resume(
        self,
        job_id: str,
        spec: JobSpec,
        cli: KaggleCLI,
        kernel_ref: str,
    ) -> Optional[JobResult]:
        record = self.store.load(job_id)
        state = record.get("state", "")
        if self.cfg.use_session_dataset and not record.get("kernel_run_ref"):
            # The fixed kernel's latest version may belong to a previous job;
            # never resume an unpushed session job from someone else's output.
            self._log(
                "Session job has no kernel version yet; preparing a fresh run.",
                job_id,
            )
            return None
        output_dir = self.store.output_dir(job_id)
        has_output = (output_dir / "result.json").is_file() or bool(
            list(output_dir.glob("**/result.json"))
        )
        if state == "done":
            stats = record.get("stats") or {}
            return JobResult(
                job_id=job_id,
                changed=int(stats.get("changed") or 0),
                boxes=int(stats.get("boxes") or 0),
                message=str(record.get("message") or "Job already done."),
                output_dir=output_dir if has_output else None,
                result=(stats.get("kernel_result") or {}),
                changed_names=list(stats.get("changed_names") or []),
            )
        if state in ("importing", "downloading") and has_output:
            self._log("Resuming at local import.", job_id)
            return self._import_only(job_id, spec, output_dir)

        run_ref = record.get("kernel_run_ref") or kernel_ref
        status = cli.kernels_status(run_ref)
        self._log(f"Resume status for {run_ref}: {status.state}", job_id)
        if status.state in ("QUEUED", "RUNNING"):
            self._update(
                job_id,
                state="running" if status.state == "RUNNING" else "queued",
                message=f"Kernel {status.state.lower()} on Kaggle.",
            )
            final = self._wait_for_kernel(
                job_id,
                cli,
                run_ref,
                spec,
                queue_timeout_minutes=self.cfg.queue_timeout_minutes,
            )
            if not final.success:
                logs = self._fetch_kernel_logs(job_id, cli, run_ref)
                raise KaggleJobError(
                    f"Kaggle kernel failed: {final.state}. {final.failure or ''}\n{logs}",
                    job_id,
                    self.store.job_dir(job_id) / "logs" / "kernel.log",
                )
            return self._download_and_import(job_id, spec, cli, run_ref)
        if status.state == "COMPLETE":
            return self._download_and_import(job_id, spec, cli, run_ref)
        if status.state in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"):
            logs = self._fetch_kernel_logs(job_id, cli, run_ref)
            raise KaggleJobError(
                f"Kaggle kernel ended with {status.state}. {status.failure or ''}\n{logs}",
                job_id,
                self.store.job_dir(job_id) / "logs" / "kernel.log",
            )
        # Unknown / missing kernel: fall through to a fresh bundle+push.
        self._log("Previous kernel is not resumable; preparing a fresh run.", job_id)
        return None

    # ------------------------------------------------------------------
    # Fast session mode: images once, command embedded per click
    # ------------------------------------------------------------------
    def _run_session_mode(
        self,
        job_id: str,
        spec: JobSpec,
        cli: KaggleCLI,
        username: str,
        kernel_id: str,
        fixed_title: str,
        weights_dataset_id: Optional[str],
        model_override: Optional[Dict[str, Any]],
    ) -> JobResult:
        self._check_cancel()
        self._update(
            job_id,
            state="bundling",
            progress=2,
            message="Preparing Kaggle session images (one-time per app session)...",
            error=None,
        )
        session = RemoteSession(
            self.cfg, cli, log=lambda msg: self._log(msg, job_id)
        )

        def session_progress(percent: int, message: str) -> None:
            self._progress(2 + int(48 * max(0, min(100, percent)) / 100), message)

        info = session.ensure_images(
            spec.images_dir,
            spec.images,
            spec.classes,
            progress=session_progress,
            username=username,
        )
        self._check_cancel()
        self._update(
            job_id,
            progress=50,
            session_dataset=info.dataset_ref,
            images_count=info.image_count,
            message=(
                f"Session ready: {info.image_count} image(s) on Kaggle. "
                "Only the command will be sent per click."
            ),
        )

        command = build_command_payload(spec, model_override, info, job_id)
        command_bytes = len(
            json.dumps(command, ensure_ascii=False).encode("utf-8")
        )
        self._update(
            job_id,
            progress=55,
            bundle_bytes=command_bytes,
            message="Command ready; pushing it to the fixed Kaggle kernel...",
        )

        last_error: Optional[Exception] = None
        for accelerator in self.cfg.accelerators:
            self._check_cancel()
            label = "CPU" if str(accelerator).lower() == "cpu" else str(accelerator)
            self._update(
                job_id,
                state="pushing_kernel",
                progress=60,
                accelerator=accelerator,
                message=f"Pushing fixed Kaggle kernel on {label}...",
            )
            kernel_dir = self._prepare_kernel_folder_session(
                job_id,
                info.dataset_ref,
                kernel_id,
                fixed_title,
                command,
                weights_dataset_id,
                accelerator=accelerator,
            )
            try:
                push_result = cli.kernels_push(
                    kernel_dir, accelerator=accelerator
                )
            except KaggleCliError as exc:
                last_error = exc
                self._log(
                    f"Accelerator {label} could not be used: {exc}", job_id
                )
                continue
            version = self._parse_kernel_version(push_result.output)
            kernel_run_ref = f"{kernel_id}/{version}" if version else kernel_id
            self._update(
                job_id,
                state="queued",
                progress=63,
                kernel_ref=kernel_id,
                kernel_run_ref=kernel_run_ref,
                accelerator=accelerator,
                kernel_url=f"https://www.kaggle.com/code/{kernel_id}",
                message=(
                    f"Kernel version {version} queued on {label}."
                    if version
                    else f"Kernel queued on {label}."
                ),
            )
            self._log(push_result.output, job_id)
            try:
                status = self._wait_for_kernel(
                    job_id,
                    cli,
                    kernel_run_ref,
                    spec,
                    queue_timeout_minutes=self.cfg.queue_timeout_minutes,
                )
            except QueueTimeout as exc:
                last_error = exc
                self._log(
                    f"{label} queue timeout; trying the next accelerator.",
                    job_id,
                )
                try:
                    cli.cancel_kernel(kernel_id)
                except Exception as cancel_exc:
                    self._log(f"Could not cancel queued kernel: {cancel_exc}", job_id)
                continue
            if not status.success:
                logs = self._fetch_kernel_logs(job_id, cli, kernel_run_ref)
                raise KaggleJobError(
                    "Kaggle kernel did not complete successfully: "
                    f"{status.state}. {status.failure or ''}\n{logs}".strip(),
                    job_id,
                    self.store.job_dir(job_id) / "logs" / "kernel.log",
                )
            return self._download_and_import(job_id, spec, cli, kernel_run_ref)

        raise self._no_accelerator_error(job_id, last_error)

    def _no_accelerator_error(
        self, job_id: str, last_error: Optional[Exception]
    ) -> KaggleJobError:
        if self.cfg.require_gpu:
            message = (
                "No usable GPU was allocated. "
                f"Tried: {', '.join(self.cfg.accelerators)} "
                f"({self.cfg.queue_timeout_minutes:.0f} min max each). "
                "CPU fallback is disabled (KAGGLE_REQUIRE_GPU=true), so the "
                "job stopped and local labels were not changed. Retry later "
                "or add an available GPU to KAGGLE_ACCELERATORS."
            )
        else:
            message = (
                "No accelerator could start the kernel. "
                f"Tried: {', '.join(self.cfg.accelerators)}. "
                "CPU fallback was allowed but did not start."
            )
        if last_error:
            message += f" Last error: {last_error}"
        return KaggleJobError(
            message, job_id, self.store.log_path(job_id)
        )

    def _prepare_kernel_folder_session(
        self,
        job_id: str,
        session_dataset: str,
        kernel_id: str,
        fixed_title: str,
        command: Dict[str, Any],
        weights_dataset_id: Optional[str],
        accelerator: str,
    ) -> Path:
        kernel_dir = self.store.kernel_dir(job_id)
        shutil.rmtree(kernel_dir, ignore_errors=True)
        kernel_dir.mkdir(parents=True, exist_ok=True)
        source = (Path(__file__).resolve().parent / "kernel_runner.py").read_text(
            encoding="utf-8"
        )
        embedded = embed_command(source, command)
        (kernel_dir / "runner.py").write_text(embedded, encoding="utf-8")
        extra = [weights_dataset_id] if weights_dataset_id else []
        write_json_atomic(
            kernel_dir / "kernel-metadata.json",
            kernel_metadata(
                kernel_id,
                fixed_title,
                session_dataset,
                code_file="runner.py",
                accelerator=accelerator,
                extra_dataset_sources=extra,
            ),
        )
        return kernel_dir

    @staticmethod
    def _parse_kernel_version(output: str) -> Optional[int]:
        text = output or ""
        match = re.search(r"Kernel version\s+(\d+)\s+successfully", text, re.IGNORECASE)
        if not match:
            match = re.search(r"Kernel version\s+(\d+)", text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------
    def _prepare_dataset_folder(
        self, job_id: str, bundle: BundleInfo, dataset_id: str
    ) -> Path:
        dataset_dir = self.store.job_dir(job_id) / "dataset"
        shutil.rmtree(dataset_dir, ignore_errors=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle.job_zip, dataset_dir / "job.zip")
        write_json_atomic(
            dataset_dir / "dataset-metadata.json",
            dataset_metadata(dataset_id, job_title(job_id)),
        )
        return dataset_dir

    def _prepare_kernel_folder(
        self,
        job_id: str,
        dataset_id: str,
        kernel_id: str,
        extra_dataset_sources: Optional[List[str]] = None,
        accelerator: Optional[str] = None,
    ) -> Path:
        kernel_dir = self.store.kernel_dir(job_id)
        shutil.rmtree(kernel_dir, ignore_errors=True)
        kernel_dir.mkdir(parents=True, exist_ok=True)
        package_dir = Path(__file__).resolve().parent
        shutil.copyfile(package_dir / "kernel_runner.py", kernel_dir / "runner.py")
        # runner_lib.py travels inside job.zip (Kaggle script kernels only
        # upload the single code_file).
        write_json_atomic(
            kernel_dir / "kernel-metadata.json",
            kernel_metadata(
                kernel_id,
                kernel_title(job_id),
                dataset_id,
                code_file="runner.py",
                accelerator=accelerator or self.cfg.accelerator,
                extra_dataset_sources=extra_dataset_sources,
            ),
        )
        return kernel_dir

    # ------------------------------------------------------------------
    # Polling / download
    # ------------------------------------------------------------------
    def _wait_for_kernel(
        self,
        job_id: str,
        cli: KaggleCLI,
        kernel_ref: str,
        spec: JobSpec,
        queue_timeout_minutes: Optional[float] = None,
    ) -> KernelStatus:
        started = time.time()
        deadline = started + max(0.25, self.cfg.timeout_hours) * 3600
        expected = max(300.0, min(3600.0, len(spec.images) * 2.0 + 300.0))
        last_state = ""
        queued_since: Optional[float] = None
        while time.time() < deadline:
            self._check_cancel()
            status = cli.kernels_status(kernel_ref)
            if status.state != last_state:
                last_state = status.state
                self._log(f"Kernel status: {status.state}", job_id)
            if status.state in ("QUEUED", "RUNNING"):
                elapsed = time.time() - started
                if status.state == "QUEUED":
                    if queued_since is None:
                        queued_since = time.time()
                    if (
                        queue_timeout_minutes is not None
                        and queue_timeout_minutes > 0
                        and time.time() - queued_since
                        > queue_timeout_minutes * 60
                    ):
                        raise QueueTimeout(
                            f"Kernel stayed QUEUED for more than "
                            f"{queue_timeout_minutes:.0f} minutes."
                        )
                    percent = 63
                    max_text = (
                        f", max {queue_timeout_minutes:.0f}m per accelerator"
                        if queue_timeout_minutes
                        else ""
                    )
                    message = (
                        "Kaggle queued — waiting for accelerator allocation "
                        f"({int(elapsed // 60)}m {int(elapsed % 60)}s elapsed"
                        f"{max_text}). Use Cancel Kaggle job to stop."
                    )
                else:
                    queued_since = None
                    percent = 65 + int(20 * min(1.0, elapsed / expected))
                    message = (
                        f"Kaggle running "
                        f"({int(elapsed // 60)}m {int(elapsed % 60)}s elapsed)"
                    )
                self._update(
                    job_id,
                    state="running" if status.state == "RUNNING" else "queued",
                    progress=percent,
                    message=message,
                )
                self.sleep(max(5, self.cfg.poll_seconds))
                continue
            if status.state == "COMPLETE":
                self._update(
                    job_id,
                    state="downloading",
                    progress=85,
                    message="Kernel complete. Downloading labels...",
                )
                return status
            if status.state in (
                "ERROR",
                "CANCEL_ACKNOWLEDGED",
                "CANCEL_REQUESTED",
                "NOT_FOUND",
            ):
                return status
            # Unknown state: keep polling, but do not spin.
            self._log(f"Unknown kernel state {status.state!r}; waiting...", job_id)
            self.sleep(max(5, self.cfg.poll_seconds))
        raise KaggleJobError(
            f"Kaggle kernel did not finish within {self.cfg.timeout_hours:.1f} hours. "
            f"Job folder: {self.store.job_dir(job_id)}",
            job_id,
            self.store.log_path(job_id),
        )

    def _download_and_import(
        self,
        job_id: str,
        spec: JobSpec,
        cli: KaggleCLI,
        kernel_ref: str,
    ) -> JobResult:
        output_dir = self.store.output_dir(job_id)
        existing_result = next(output_dir.glob("**/result.json"), None)
        if existing_result is not None:
            self._log("Using already downloaded Kaggle output.", job_id)
            return self._import_only(job_id, spec, output_dir)
        shutil.rmtree(output_dir, ignore_errors=True)
        self._update(
            job_id,
            state="downloading",
            progress=86,
            message="Downloading Kaggle output...",
        )
        cli.kernels_output(kernel_ref, output_dir)
        result_path = next(output_dir.glob("**/result.json"), None)
        if result_path is None:
            logs = self._fetch_kernel_logs(job_id, cli, kernel_ref)
            raise KaggleJobError(
                "Kaggle output has no result.json. The kernel likely failed early.\n"
                + logs,
                job_id,
                self.store.job_dir(job_id) / "logs" / "kernel.log",
            )
        return self._import_only(job_id, spec, output_dir)

    def _import_only(
        self, job_id: str, spec: JobSpec, output_dir: Path
    ) -> JobResult:
        self._check_cancel()
        if self.cfg.require_gpu:
            meta = load_import_metadata(self.store.job_dir(job_id))
            if meta.get("status") == "ok":
                stats = meta.get("stats") or {}
                gpu_count = int(stats.get("gpu_count") or 0)
                gpu_names = list(stats.get("gpu_names") or [])
                if gpu_count <= 0:
                    raise KaggleJobError(
                        "Kaggle ran the kernel on CPU, but GPU is required "
                        "(KAGGLE_REQUIRE_GPU=true). No labels were imported. "
                        "CPU fallback is disabled; retry when a GPU is available.",
                        job_id,
                        self.store.log_path(job_id),
                    )
                self._log(
                    f"Kernel GPU: {', '.join(gpu_names) or (str(gpu_count) + ' GPU(s)')}",
                    job_id,
                )
        self._update(
            job_id,
            state="importing",
            progress=93,
            message="Merging Kaggle labels into the local dataset...",
        )
        imported: ImportResult = import_job_output(
            self.store.job_dir(job_id),
            images=spec.images,
            classes=spec.classes,
            labels_dir=Path(spec.labels_dir),
        )
        if imported.snapshot and self.snapshot_cb:
            try:
                self.snapshot_cb(imported.snapshot)
            except Exception:
                pass
        stats = {
            "changed": imported.changed,
            "boxes": imported.boxes,
            "changed_names": imported.changed_names,
            "kernel_result": imported.result,
        }
        message = imported.summary()
        self._update(
            job_id,
            state="done",
            progress=100,
            message=message,
            stats=stats,
            output_dir=str(output_dir),
            error=None,
        )
        self._log(message, job_id)
        self._maybe_cleanup_remote(job_id)
        self._maybe_cleanup_local(job_id)
        return JobResult(
            job_id=job_id,
            changed=imported.changed,
            boxes=imported.boxes,
            message=message,
            snapshot=imported.snapshot,
            output_dir=output_dir,
            result=imported.result,
            changed_names=imported.changed_names,
        )

    def _fetch_kernel_logs(
        self, job_id: str, cli: KaggleCLI, kernel_ref: str
    ) -> str:
        log_path = self.store.job_dir(job_id) / "logs" / "kernel.log"
        try:
            text = cli.kernels_logs(kernel_ref, log_path)
        except Exception as exc:
            text = f"(could not fetch kernel logs: {exc})"
        lowered = text.lower()
        if "authentication required" in lowered:
            text = "Kaggle authentication failed while fetching logs.\n" + text
        tail = "\n".join(text.splitlines()[-80:])
        if tail:
            self._log("Kernel log tail:\n" + tail, job_id)
        return tail

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _maybe_cleanup_remote(self, job_id: str) -> None:
        record = self.store.load(job_id)
        dataset_ref = record.get("dataset_ref")
        kernel_ref = record.get("kernel_ref")
        cli = self._cli
        if cli is None:
            return
        if self.cfg.delete_dataset_after and dataset_ref:
            try:
                cli.dataset_delete(dataset_ref)
                self._log(f"Deleted remote dataset {dataset_ref}", job_id)
            except Exception as exc:
                self._log(f"Could not delete remote dataset {dataset_ref}: {exc}", job_id)
        if self.cfg.delete_kernel_after and kernel_ref:
            try:
                cli.kernels_delete(kernel_ref)
                self._log(f"Deleted remote kernel {kernel_ref}", job_id)
            except Exception as exc:
                self._log(f"Could not delete remote kernel {kernel_ref}: {exc}", job_id)

    def _maybe_cleanup_local(self, job_id: str) -> None:
        if self.cfg.keep_bundle:
            return
        try:
            self.store.bundle_zip(job_id).unlink()
        except OSError:
            pass
        shutil.rmtree(self.store.job_dir(job_id) / "dataset", ignore_errors=True)

    def _mark_failed(self, job_id: str, error: str) -> None:
        try:
            self.store.update(
                job_id,
                state="failed",
                error=error,
                message="Kaggle job failed. See the job log for details.",
            )
            self.store.append_log(job_id, "FAILED: " + error)
        except Exception:
            pass
