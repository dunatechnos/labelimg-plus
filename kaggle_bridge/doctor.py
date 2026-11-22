"""Environment diagnostics for the Kaggle bridge."""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .config import Config, load_config
from .kaggle_cli import KaggleCLI, KaggleCliError


def _fmt_gb(value: float) -> str:
    return f"{value / (1024 ** 3):.1f} GB"


def doctor_lines(
    cfg: Config,
    check_auth: bool = True,
) -> Tuple[List[str], List[str]]:
    """Return ``(lines, problems)`` without printing."""
    lines: List[str] = []
    problems: List[str] = []

    lines.append("Kaggle bridge doctor")
    lines.append("=" * 60)
    lines.append(f"Python:       {sys.version.split()[0]} ({sys.executable})")
    lines.append(f"Platform:     {platform.platform()}")
    lines.append(f"Repository:   {cfg.root}")
    lines.append(f".env:         {cfg.env_path} [{'found' if cfg.env_path.exists() else 'missing'}]")
    lines.append(f"Credentials:  {cfg.credentials_source() or 'NOT FOUND'}")
    lines.append(f"Config dir:   {cfg.kaggle_home}")
    lines.append(f"Job root:     {cfg.job_root}")
    lines.append(f"Dataset prefix: {cfg.dataset_prefix}")
    lines.append(f"Weights ds:   {cfg.weights_dataset}")
    lines.append(f"Reuse weights: {cfg.use_weights_dataset}")
    lines.append(f"Session ds:   {cfg.use_session_dataset} ({cfg.session_dataset_prefix})")
    lines.append(
        f"Kernel:       "
        f"{cfg.kernel_slug if cfg.fixed_kernel else '(new per click)'}"
    )
    lines.append(f"Accelerators: {', '.join(cfg.accelerators)}")
    lines.append(f"Queue timeout: {cfg.queue_timeout_minutes:.0f} min")
    lines.append("")

    if not cfg.env_path.exists():
        problems.append(
            "No .env found. Run: python -m kaggle_bridge.setup --write-env"
        )

    if cfg.credentials_source() is None:
        problems.append(
            "No Kaggle credentials. Create a token at "
            "https://www.kaggle.com/settings/api and save it as "
            f"{cfg.kaggle_json_path} (or set KAGGLE_API_TOKEN in .env)."
        )

    cli_path = cfg.find_cli()
    if cli_path is None:
        problems.append(
            "Kaggle CLI not found. Run: python -m kaggle_bridge.setup --create-venv"
        )
        lines.append("Kaggle CLI:   NOT FOUND")
    else:
        lines.append(f"Kaggle CLI:   {cli_path}")
        try:
            cli = KaggleCLI(cfg)
            version = cli.version()
            version_tuple = cli.version_tuple()
            lines.append(f"CLI version:  {version}")
            if version_tuple < (2, 2, 0):
                problems.append(
                    "Kaggle CLI 2.2+ is required for the T4 x2 accelerator. "
                    "Recreate the venv with: python -m kaggle_bridge.setup --create-venv --force"
                )
            if check_auth:
                try:
                    username = cli.check_auth()
                    lines.append(f"Authenticated: yes ({username})")
                    try:
                        from .weights import WeightsManager

                        weights_ref = WeightsManager(cfg, cli).dataset_id(username)
                        weights_files = cli.dataset_files(weights_ref)
                        lines.append(
                            f"Weights data: {weights_ref} "
                            f"({len(weights_files)} file(s))"
                        )
                    except Exception as exc:
                        lines.append(f"Weights data: could not check ({exc})")
                    try:
                        for item in cli.quota():
                            if str(item.get("resource", "")).upper() != "GPU":
                                continue
                            remaining = str(item.get("remaining", "?"))
                            total = str(item.get("total", "?"))
                            lines.append(
                                f"GPU quota:    {remaining} remaining of {total} "
                                f"(refresh {item.get('refreshAt', '?')})"
                            )
                            try:
                                hours = float(remaining.lower().replace("h", "").strip())
                            except ValueError:
                                hours = None
                            if hours is not None and hours <= 0:
                                problems.append(
                                    "No GPU quota left this week. Wait for the "
                                    "daily/weekly refresh."
                                )
                    except KaggleCliError:
                        pass
                except KaggleCliError as exc:
                    problems.append(str(exc))
                    lines.append("Authenticated: NO")
        except KaggleCliError as exc:
            problems.append(str(exc))
            lines.append(f"CLI check:    {exc}")

    py311 = cfg.find_python_311()
    lines.append(f"Python 3.11+: {py311 or 'NOT FOUND'}")
    if py311 is None and cli_path is None:
        problems.append(
            "No Python 3.11+ interpreter found. Install one, then run "
            "python -m kaggle_bridge.setup --create-venv"
        )

    try:
        cfg.job_root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(str(cfg.job_root))
        lines.append(
            f"Disk free:    {_fmt_gb(usage.free)} of {_fmt_gb(usage.total)} at {cfg.job_root}"
        )
        if usage.free < 5 * 1024 ** 3:
            problems.append("Less than 5 GB free disk space in the job root.")
    except OSError as exc:
        problems.append(f"Cannot create/check job root {cfg.job_root}: {exc}")

    if cfg.accelerator != "NvidiaTeslaT4":
        problems.append(
            "KAGGLE_ACCELERATOR should be NvidiaTeslaT4 for 2x T4 GPUs."
        )

    lines.append("")
    if problems:
        lines.append("Problems:")
        for problem in problems:
            lines.append("  - " + problem)
    else:
        lines.append("Everything looks ready.")
    lines.append("")
    lines.append("Next steps:")
    lines.append("  1. Open the main window and press Ctrl+Shift+K for Kaggle controls.")
    lines.append("  2. Use Ctrl+Shift+Y (Predict) or Ctrl+Shift+T (SAM 2).")
    lines.append("  3. Tick 'Run on Kaggle (T4 x2)' and start the job.")
    lines.append("     Results merge back automatically; Ctrl+Alt+Z undoes them.")
    return lines, problems


def run_doctor(
    cfg: Optional[Config] = None,
    check_auth: bool = True,
    print_fn: Callable[[str], None] = print,
) -> int:
    cfg = cfg or load_config()
    lines, problems = doctor_lines(cfg, check_auth=check_auth)
    for line in lines:
        print_fn(line)
    return 1 if problems else 0


def main(argv=None) -> int:
    import argparse

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Check Kaggle bridge setup")
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Skip the network authentication check",
    )
    args = parser.parse_args(argv)
    return run_doctor(check_auth=not args.no_auth)


if __name__ == "__main__":
    raise SystemExit(main())
