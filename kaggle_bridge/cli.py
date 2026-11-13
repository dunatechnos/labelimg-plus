"""Command line interface for the Kaggle bridge.

Examples::

    python -m kaggle_bridge doctor
    python -m kaggle_bridge setup --create-venv --write-env
    python -m kaggle_bridge jobs
    python -m kaggle_bridge status --job 20261113-121500-ab12cd
    python -m kaggle_bridge resume --job 20261113-121500-ab12cd
    python -m kaggle_bridge dry-run --spec job-spec.json
    python -m kaggle_bridge run --spec job-spec.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .bundle import build_bundle
from .config import Config, ConfigError, load_config
from .doctor import run_doctor
from .importer import import_job_output
from .jobstore import JobStore
from .spec import JobSpec, SpecError
from .weights import sync_weights
from .worker import KaggleJobError, KaggleJobRunner


def _print_progress(percent: int, message: str) -> None:
    print(f"[{percent:3d}%] {message}", flush=True)


def _print_status(message: str) -> None:
    print(message, flush=True)


def _print_log(message: str) -> None:
    print("      " + str(message).replace("\n", "\n      "), flush=True)


def _load_spec(path: Path) -> JobSpec:
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise SpecError(f"Cannot read spec file {path}: {exc}") from exc
    spec = JobSpec.from_json(text)
    spec.validate()
    return spec


def _job_summary(job: Dict[str, Any]) -> str:
    return (
        f"{job.get('job_id', '?'):<28} "
        f"{job.get('job_type', '?'):<13} "
        f"{job.get('state', '?'):<18} "
        f"{int(job.get('progress') or 0):3d}%  "
        f"{str(job.get('created_at', ''))[:19]}  "
        f"{job.get('message', '')}"
    )


def cmd_jobs(args) -> int:
    cfg = load_config()
    store = JobStore(cfg.job_root)
    jobs = store.list_jobs(limit=args.limit)
    if args.json:
        print(json.dumps(jobs, indent=2, ensure_ascii=False))
        return 0
    if not jobs:
        print(f"No jobs in {cfg.job_root}")
        return 0
    print(f"Jobs in {cfg.job_root}:")
    for job in jobs:
        print(_job_summary(job))
    return 0


def cmd_status(args) -> int:
    cfg = load_config()
    store = JobStore(cfg.job_root)
    record = store.load(args.job)
    if args.json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0
    print(f"Job:        {record.get('job_id')}")
    print(f"Type:       {record.get('job_type')}")
    print(f"State:      {record.get('state')}")
    print(f"Progress:   {record.get('progress')}%")
    print(f"Created:    {record.get('created_at')}")
    print(f"Updated:    {record.get('updated_at')}")
    print(f"Dataset:    {record.get('dataset_ref')}")
    print(f"Kernel:     {record.get('kernel_ref')}")
    print(f"Message:    {record.get('message')}")
    if record.get("error"):
        print("Error:")
        print(str(record["error"]))
    if args.log:
        print("")
        print("Log tail:")
        print(store.read_log_tail(args.job, lines=args.log_lines))
    return 0


def cmd_resume(args) -> int:
    cfg = load_config()
    store = JobStore(cfg.job_root)
    runner = KaggleJobRunner(
        cfg,
        store,
        progress=_print_progress,
        status=_print_status,
        log=_print_log,
    )
    try:
        result = runner.run(job_id=args.job, resume=True)
    except (KaggleJobError, ConfigError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("")
    print(f"Done: {result.message}")
    print(f"Changed label files: {result.changed}; boxes: {result.boxes}")
    return 0


def cmd_run(args) -> int:
    cfg = load_config()
    spec = _load_spec(Path(args.spec))
    store = JobStore(cfg.job_root)
    runner = KaggleJobRunner(
        cfg,
        store,
        progress=_print_progress,
        status=_print_status,
        log=_print_log,
    )
    try:
        result = runner.run(spec=spec)
    except (KaggleJobError, ConfigError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("")
    print(f"Done: {result.message}")
    print(f"Job: {result.job_id}")
    print(f"Changed label files: {result.changed}; boxes: {result.boxes}")
    return 0


def cmd_dry_run(args) -> int:
    cfg = load_config()
    spec = _load_spec(Path(args.spec))
    cfg.ensure_job_root()
    root = Path(tempfile.mkdtemp(prefix="labelimg-kaggle-dry-", dir=str(cfg.job_root)))
    job_id = "dryrun-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    job_dir = root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        info = build_bundle(
            spec,
            job_dir,
            include_sam2_checkpoint=cfg.include_sam2_checkpoint,
            keep_bundle=True,
            max_bundle_mb=cfg.max_bundle_mb,
            progress=_print_progress,
            log=_print_log,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("")
    print(f"Bundle: {info.job_zip}")
    print(f"Size:   {info.size_bytes / 1024 / 1024:.1f} MB")
    print(f"SHA256: {info.sha256}")
    print(f"Frames: {info.image_count}")
    print(f"Model included: {info.model_included}")
    if args.clean:
        import shutil

        shutil.rmtree(root, ignore_errors=True)
    else:
        print(f"Kept for inspection: {job_dir}")
    return 0


def cmd_import(args) -> int:
    cfg = load_config()
    store = JobStore(cfg.job_root)
    record = store.load(args.job)
    spec = JobSpec.from_dict(record["spec"])
    try:
        result = import_job_output(
            store.job_dir(args.job),
            images=spec.images,
            classes=spec.classes,
            labels_dir=Path(spec.labels_dir),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(result.summary())
    if result.snapshot:
        print(f"Undo snapshot: {len(result.snapshot)} file(s)")
    return 0


def cmd_cleanup(args) -> int:
    cfg = load_config()
    store = JobStore(cfg.job_root)
    job_dir = store.job_dir(args.job)
    if not job_dir.exists():
        print(f"Job folder does not exist: {job_dir}")
        return 0
    if not args.yes:
        print(f"Refusing to delete {job_dir} without --yes")
        return 2
    import shutil

    shutil.rmtree(job_dir)
    print(f"Deleted local job folder: {job_dir}")
    return 0


def cmd_doctor(args) -> int:
    return run_doctor(load_config(), check_auth=not args.no_auth)


def cmd_sync_weights(args) -> int:
    cfg = load_config()
    try:
        mapping = sync_weights(cfg, args.username, log=print)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("")
    print("Weights dataset is ready (uploaded once, now reused):")
    for local, remote in mapping.items():
        print(f"  {remote}  <-  {local}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kaggle_bridge",
        description="Run labelImg Plus heavy annotation jobs on Kaggle T4 x2",
    )
    parser.add_argument("--version", action="version", version=f"kaggle_bridge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="Check credentials, CLI and environment")
    p.add_argument("--no-auth", action="store_true", help="Skip network auth check")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser(
        "sync-weights",
        help="Upload all local YOLO/SAM2 weights once and reuse them",
    )
    p.add_argument("--username", default="", help="Kaggle username (auto-detected)")
    p.set_defaults(func=cmd_sync_weights)

    p = sub.add_parser("jobs", help="List local jobs")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_jobs)

    p = sub.add_parser("status", help="Show one job")
    p.add_argument("--job", required=True)
    p.add_argument("--json", action="store_true")
    p.add_argument("--log", action="store_true", help="Also print the log tail")
    p.add_argument("--log-lines", type=int, default=80)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("resume", help="Resume polling/download/import for a job")
    p.add_argument("--job", required=True)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("run", help="Run a job spec JSON end-to-end")
    p.add_argument("--spec", required=True)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("dry-run", help="Build a job bundle without contacting Kaggle")
    p.add_argument("--spec", required=True)
    p.add_argument("--clean", action="store_true", help="Remove the temporary bundle")
    p.set_defaults(func=cmd_dry_run)

    p = sub.add_parser("import", help="Import an already downloaded job output")
    p.add_argument("--job", required=True)
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("cleanup", help="Delete a local job folder")
    p.add_argument("--job", required=True)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_cleanup)

    return parser


def _safe_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def main(argv=None) -> int:
    _safe_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError,) as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
