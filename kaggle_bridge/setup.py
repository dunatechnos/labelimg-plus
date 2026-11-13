"""One-time local setup: dedicated Kaggle CLI venv and .env template.

Run with::

    python -m kaggle_bridge.setup --create-venv
    python -m kaggle_bridge.setup --write-env
    python -m kaggle_bridge.doctor

The bridge itself runs on Python 3.10; the Kaggle CLI 2.x (needed for the
``NvidiaTeslaT4`` accelerator) is installed into ``.kaggle-venv`` using any
Python 3.11+ interpreter found on the machine.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .config import (
    DEFAULT_ACCELERATOR,
    DEFAULT_DATASET_PREFIX,
    DEFAULT_POLL_SECONDS,
    DEFAULT_TIMEOUT_HOURS,
    Config,
    ConfigError,
    load_config,
    parse_env_file,
    upsert_env_file,
)

ENV_TEMPLATE = """# Kaggle cloud offload for labelImg Plus.
# This file is git-ignored. NEVER commit real credentials.

# ---------------------------------------------------------------------------
# Kaggle credentials (choose ONE method)
# ---------------------------------------------------------------------------
# Recommended: download a token from https://www.kaggle.com/settings/api
# and save it as %USERPROFILE%\\.kaggle\\kaggle.json (Windows) or
# ~/.kaggle/kaggle.json.  The CLI reads it automatically; nothing is needed
# in this file.
#
# Option B - legacy username + key:
#KAGGLE_USERNAME=
#KAGGLE_KEY=
#
# Option C - new API access token:
#KAGGLE_API_TOKEN=
#
# Optional custom config directory:
#KAGGLE_CONFIG_DIR=

# ---------------------------------------------------------------------------
# Local tools (usually auto-detected)
# ---------------------------------------------------------------------------
# Explicit Kaggle CLI 2.x executable:
#KAGGLE_CLI=
# Python 3.11+ interpreter used to build .kaggle-venv:
#KAGGLE_PYTHON=

# ---------------------------------------------------------------------------
# Job settings
# ---------------------------------------------------------------------------
KAGGLE_DATASET_PREFIX={dataset_prefix}
KAGGLE_JOB_ROOT=kaggle_jobs
KAGGLE_USE_WEIGHTS_DATASET=true
KAGGLE_WEIGHTS_DATASET=labelimg-plus-weights
KAGGLE_WEIGHTS_ROOT=kaggle_weights
KAGGLE_USE_SESSION_DATASET=true
KAGGLE_SESSION_DATASET_PREFIX=labelimg-plus-session
KAGGLE_FIXED_KERNEL=true
KAGGLE_KERNEL_SLUG=labelimg-plus-runner
KAGGLE_MAX_SESSION_GB=20
KAGGLE_SESSION_KEEP_LOCAL=false
KAGGLE_POLL_SECONDS={poll_seconds}
KAGGLE_TIMEOUT_HOURS={timeout_hours}
KAGGLE_ACCELERATOR={accelerator}
KAGGLE_DELETE_DATASET_AFTER=true
KAGGLE_DELETE_KERNEL_AFTER=false
KAGGLE_INCLUDE_SAM2_CHECKPOINT=true
KAGGLE_KEEP_BUNDLE=false
KAGGLE_MAX_BUNDLE_MB=4096
"""


def default_env_values() -> Dict[str, str]:
    return {
        "KAGGLE_DATASET_PREFIX": DEFAULT_DATASET_PREFIX,
        "KAGGLE_JOB_ROOT": "kaggle_jobs",
        "KAGGLE_USE_WEIGHTS_DATASET": "true",
        "KAGGLE_WEIGHTS_DATASET": "labelimg-plus-weights",
        "KAGGLE_WEIGHTS_ROOT": "kaggle_weights",
        "KAGGLE_USE_SESSION_DATASET": "true",
        "KAGGLE_SESSION_DATASET_PREFIX": "labelimg-plus-session",
        "KAGGLE_FIXED_KERNEL": "true",
        "KAGGLE_KERNEL_SLUG": "labelimg-plus-runner",
        "KAGGLE_MAX_SESSION_GB": "20",
        "KAGGLE_SESSION_KEEP_LOCAL": "false",
        "KAGGLE_POLL_SECONDS": str(DEFAULT_POLL_SECONDS),
        "KAGGLE_TIMEOUT_HOURS": str(DEFAULT_TIMEOUT_HOURS),
        "KAGGLE_ACCELERATOR": DEFAULT_ACCELERATOR,
        "KAGGLE_DELETE_DATASET_AFTER": "true",
        "KAGGLE_DELETE_KERNEL_AFTER": "false",
        "KAGGLE_INCLUDE_SAM2_CHECKPOINT": "true",
        "KAGGLE_KEEP_BUNDLE": "false",
        "KAGGLE_MAX_BUNDLE_MB": "4096",
    }


def write_env(cfg: Config, updates: Optional[Dict[str, str]] = None) -> Path:
    """Create ``.env`` with comments, or add missing defaults to it."""
    path = cfg.env_path
    updates = dict(updates or {})
    if not path.exists():
        text = ENV_TEMPLATE.format(
            dataset_prefix=DEFAULT_DATASET_PREFIX,
            poll_seconds=DEFAULT_POLL_SECONDS,
            timeout_hours=DEFAULT_TIMEOUT_HOURS,
            accelerator=DEFAULT_ACCELERATOR,
        )
        if updates:
            text += "\n# Values supplied during setup\n"
            for key, value in updates.items():
                text += f"{key}={value}\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    existing = parse_env_file(path)
    missing = {
        key: value
        for key, value in default_env_values().items()
        if key not in existing
    }
    missing.update({key: value for key, value in updates.items() if value != ""})
    if missing:
        upsert_env_file(path, missing)
    return path


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(
    cmd: List[str],
    timeout: int = 900,
    log: Callable[[str], None] = print,
) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    if output:
        for line in output.splitlines()[-30:]:
            log("  " + line)
    if proc.returncode != 0:
        raise ConfigError(
            f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}"
        )
    return proc


def build_sam2_wheel(
    cfg: Config,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> Path:
    """Build the SAM 2 wheel once (avoids a 4-5 minute git build per kernel)."""
    wheels = Path(cfg.weights_root) / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    existing = sorted(wheels.glob("sam_2*.whl")) + sorted(wheels.glob("sam2*.whl"))
    if existing and not force:
        log(f"SAM 2 wheel already exists: {existing[0]}")
        return existing[0]
    python = cfg.find_python_311()
    if not python:
        raise ConfigError(
            "No Python 3.11+ interpreter found to build the SAM 2 wheel. "
            "Set KAGGLE_PYTHON or install Python 3.11+."
        )
    log("Building SAM 2 wheel once (CUDA extension disabled)...")
    env = os.environ.copy()
    env["SAM2_BUILD_CUDA"] = "0"
    cmd = [
        python,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "git+https://github.com/facebookresearch/sam2.git",
        "-w",
        str(wheels),
    ]
    log("$ " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.stdout:
        for line in (proc.stdout or "").splitlines()[-15:]:
            log("  " + line)
    if proc.stderr:
        for line in (proc.stderr or "").splitlines()[-15:]:
            log("  " + line)
    if proc.returncode != 0:
        raise ConfigError("Failed to build the SAM 2 wheel; see output above.")
    found = sorted(wheels.glob("sam_2*.whl")) + sorted(wheels.glob("sam2*.whl"))
    if not found:
        raise ConfigError("SAM 2 wheel build finished but no wheel was found.")
    log(f"SAM 2 wheel ready: {found[0]}")
    return found[0]


def create_venv(
    cfg: Config,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> Path:
    venv_dir = cfg.venv_dir
    kaggle_exe = cfg.find_cli()

    if venv_dir.exists() and not force:
        if kaggle_exe and venv_dir in kaggle_exe.parents:
            try:
                version = subprocess.run(
                    [str(kaggle_exe), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if version.returncode == 0:
                    log(f"Kaggle CLI already installed: {version.stdout.strip()}")
                    return venv_dir
            except OSError:
                pass
        log(f"{venv_dir} exists but does not contain a working Kaggle CLI.")
        log("Re-run with --force to recreate it.")
        raise ConfigError(
            f"{venv_dir} exists but is not usable. Use --force to recreate."
        )

    if force and venv_dir.exists():
        log(f"Removing old venv: {venv_dir}")
        shutil.rmtree(venv_dir)

    python = cfg.find_python_311()
    if not python:
        raise ConfigError(
            "No Python 3.11+ interpreter found. Install Python 3.11/3.12/3.13 "
            "and optionally set KAGGLE_PYTHON in .env."
        )
    log(f"Creating {venv_dir} with {python}")
    _run([python, "-m", "venv", str(venv_dir)], timeout=300, log=log)

    venv_python = _venv_python(venv_dir)
    if not venv_python.exists():
        raise ConfigError(f"venv creation failed: {venv_python} not found")

    log("Installing Kaggle CLI 2.2+ (one-time, may take a minute)...")
    # Keep this list in sync with requirements-kaggle.txt.  ``six`` is an
    # undeclared dependency of kaggle 2.2.5's upload_file.py.
    _run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "-q",
            "--disable-pip-version-check",
            "kaggle>=2.2,<3",
            "six>=1.16",
        ],
        timeout=900,
        log=log,
    )

    cli_path = cfg.find_cli()
    if cli_path is None:
        # Config lookup may still be looking at the old path; check the venv.
        candidates = cfg.venv_kaggle_candidates()
        cli_path = next((path for path in candidates if path.exists()), None)
    if cli_path is None:
        raise ConfigError("Kaggle CLI installation finished but the executable was not found.")
    log(f"Kaggle CLI installed: {cli_path}")
    return venv_dir


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="Set up the Kaggle CLI venv and .env for labelImg Plus"
    )
    parser.add_argument(
        "--create-venv",
        action="store_true",
        help="Create/reuse .kaggle-venv with Kaggle CLI 2.2+",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate .kaggle-venv if it exists (explicit confirmation)",
    )
    parser.add_argument(
        "--write-env",
        action="store_true",
        help="Create .env from the template if missing",
    )
    parser.add_argument(
        "--build-sam2-wheel",
        action="store_true",
        help="Build the SAM 2 local wheel once (fast kernel installs)",
    )
    parser.add_argument(
        "--upload-weights",
        action="store_true",
        help="Upload all local YOLO models + SAM 2 checkpoints once",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run doctor after setup",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Skip the network auth check in doctor",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Set a .env value (repeatable; values are not echoed)",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    updates: Dict[str, str] = {}
    for assignment in args.set:
        if "=" not in assignment:
            parser.error(f"--set expects KEY=VALUE, got {assignment!r}")
        key, value = assignment.split("=", 1)
        key = key.strip()
        if not key:
            parser.error("--set key cannot be empty")
        updates[key] = value

    try:
        if args.write_env or updates:
            path = write_env(cfg, updates)
            print(f"Wrote {path}")
            cfg = load_config()

        if args.create_venv:
            create_venv(cfg, force=args.force)
            cfg = load_config()

        if args.build_sam2_wheel:
            build_sam2_wheel(cfg, force=args.force, log=print)

        if args.upload_weights:
            from .weights import sync_weights

            sync_weights(cfg, log=print)
            cfg = load_config()

        if args.check or not (
            args.create_venv
            or args.write_env
            or updates
            or args.build_sam2_wheel
            or args.upload_weights
        ):
            from .doctor import run_doctor

            return run_doctor(cfg, check_auth=not args.no_auth)
    except ConfigError as exc:
        print(f"SETUP ERROR: {exc}", file=sys.stderr)
        return 2

    print("")
    print("Setup complete. Next:")
    print("  1. Put kaggle.json in", cfg.kaggle_home)
    print("  2. Run: python -m kaggle_bridge.doctor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
