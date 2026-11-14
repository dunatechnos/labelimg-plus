"""Configuration and local environment discovery for the Kaggle bridge.

Design notes
------------
* The annotation helper may run on Python 3.10 (the ``lbl_env`` on the
  developer machine is 3.10).  Kaggle's current CLI requires Python 3.11+.
  Therefore the bridge never imports the ``kaggle`` Python package.  It calls
  the ``kaggle`` CLI as a subprocess and can use a dedicated ``.kaggle-venv``
  created from any Python 3.11+ interpreter.
* Secrets are read from the normal Kaggle locations (``~/.kaggle/kaggle.json``,
  ``~/.kaggle/access_token``) or from ``.env``.  ``.env`` is git-ignored.
* No secret is ever written into a job bundle or printed by the CLI.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_DATASET_PREFIX = "labelimg-plus-job"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
DEFAULT_POLL_SECONDS = 30
DEFAULT_TIMEOUT_HOURS = 6.0
DEFAULT_MAX_BUNDLE_MB = 4096.0
DEFAULT_WEIGHTS_DATASET = "labelimg-plus-weights"
DEFAULT_WEIGHTS_ROOT = "kaggle_weights"
DEFAULT_KERNEL_SLUG = "labelimg-plus-runner"
DEFAULT_SESSION_PREFIX = "labelimg-plus-session"
DEFAULT_MAX_SESSION_GB = 20.0
DEFAULT_QUEUE_TIMEOUT_MINUTES = 5.0
DEFAULT_ACCELERATORS = (
    "NvidiaTeslaT4",       # T4 x2 (Kaggle has no separate x1 T4 option)
    "NvidiaL4X1",          # currently resolves to T4 x2 on this account
    "NvidiaH100",          # currently resolves to T4 x2 on this account
    "NvidiaTeslaA100",     # may queue; tried before giving up
)
KAGGLE_VENV_DIR = ".kaggle-venv"
KAGGLE_JOB_DIR = "kaggle_jobs"


class ConfigError(RuntimeError):
    """Raised when the local Kaggle configuration cannot be used."""


def repo_root() -> Path:
    """Repository root (the directory containing ``annotation_helper``)."""
    return Path(__file__).resolve().parent.parent


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        quote = value[0]
        value = value[1:-1]
        if quote == '"':
            value = (
                value.replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
    return value


def parse_env_file(path: Path) -> Dict[str, str]:
    """Small, dependency-free ``.env`` parser.

    Supports comments, blank lines, ``export KEY=value`` and single/double
    quoted values.  It is deliberately not a full dotenv implementation.
    """
    result: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return result
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        # Inline comments are only removed for unquoted values.
        value = value.strip()
        if value and value[0] not in ("'", '"'):
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        result[key] = _strip_quotes(value)
    return result


def _format_env_value(value: str) -> str:
    if value == "":
        return ""
    if re.search(r"[\s#\"'\\]", value):
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )
        return f'"{escaped}"'
    return value


def upsert_env_file(path: Path, updates: Dict[str, str]) -> None:
    """Update keys in an existing ``.env``, preserving comments/order.

    New keys are appended.  Values are written as plain text; callers should
    never pass a secret unless the user explicitly asked to store it there.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines: List[str] = []
    if path.exists():
        try:
            existing_lines = path.read_text(encoding="utf-8-sig").splitlines()
        except OSError as exc:
            raise ConfigError(f"Cannot read {path}: {exc}") from exc

    pending = dict(updates)
    out: List[str] = []
    key_re = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
    for line in existing_lines:
        match = key_re.match(line)
        if match and match.group(1) in pending:
            key = match.group(1)
            out.append(f"{key}={_format_env_value(pending.pop(key))}")
        else:
            out.append(line)

    if pending:
        if out and out[-1].strip():
            out.append("")
        out.append("# Added by kaggle_bridge")
        for key, value in pending.items():
            out.append(f"{key}={_format_env_value(value)}")

    text = "\n".join(out).rstrip() + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _truthy(value: str, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on", "haan", "ha")


def _as_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _probe_python(executable: str) -> Optional[Tuple[int, int]]:
    """Return ``(major, minor)`` for an interpreter, or None."""
    try:
        proc = subprocess.run(
            [executable, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    match = re.search(r"(\d+)\.(\d+)", proc.stdout or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


@dataclass
class Config:
    """Resolved local configuration.

    Construct via :func:`load_config`; the object is mutable so the worker can
    fill in ``username`` after the Kaggle CLI authenticates.
    """

    root: Path
    env_path: Path
    env_values: Dict[str, str] = field(default_factory=dict)
    username: str = ""
    key: str = ""
    api_token: str = ""
    config_dir: str = ""
    cli_path: str = ""
    python_path: str = ""
    dataset_prefix: str = DEFAULT_DATASET_PREFIX
    job_root: Path = field(default_factory=lambda: repo_root() / KAGGLE_JOB_DIR)
    use_weights_dataset: bool = True
    weights_dataset: str = DEFAULT_WEIGHTS_DATASET
    weights_root: Path = field(default_factory=lambda: repo_root() / DEFAULT_WEIGHTS_ROOT)
    use_session_dataset: bool = True
    fixed_kernel: bool = True
    kernel_slug: str = DEFAULT_KERNEL_SLUG
    session_dataset_prefix: str = DEFAULT_SESSION_PREFIX
    max_session_gb: float = DEFAULT_MAX_SESSION_GB
    session_keep_local: bool = False
    accelerators: List[str] = field(
        default_factory=lambda: list(DEFAULT_ACCELERATORS)
    )
    queue_timeout_minutes: float = DEFAULT_QUEUE_TIMEOUT_MINUTES
    require_gpu: bool = True
    poll_seconds: int = DEFAULT_POLL_SECONDS
    timeout_hours: float = DEFAULT_TIMEOUT_HOURS
    delete_dataset_after: bool = True
    delete_kernel_after: bool = False
    include_sam2_checkpoint: bool = True
    keep_bundle: bool = False
    accelerator: str = DEFAULT_ACCELERATOR
    max_bundle_mb: float = DEFAULT_MAX_BUNDLE_MB

    # ------------------------------------------------------------------
    # Value lookup / secrets
    # ------------------------------------------------------------------
    def get(self, key: str, default: str = "") -> str:
        """OS environment wins over ``.env`` (standard dotenv precedence)."""
        if key in os.environ and os.environ[key] != "":
            return os.environ[key]
        return self.env_values.get(key, default)

    @property
    def kaggle_home(self) -> Path:
        if self.config_dir:
            return Path(os.path.expandvars(os.path.expanduser(self.config_dir)))
        return Path.home() / ".kaggle"

    @property
    def kaggle_json_path(self) -> Path:
        return self.kaggle_home / "kaggle.json"

    @property
    def access_token_path(self) -> Path:
        return self.kaggle_home / "access_token"

    def credentials_source(self) -> Optional[str]:
        if self.get("KAGGLE_API_TOKEN"):
            return "KAGGLE_API_TOKEN"
        if self.get("KAGGLE_USERNAME") and self.get("KAGGLE_KEY"):
            return "KAGGLE_USERNAME + KAGGLE_KEY"
        if self.kaggle_json_path.exists():
            return str(self.kaggle_json_path)
        if self.access_token_path.exists():
            return str(self.access_token_path)
        return None

    def auth_env(self) -> Dict[str, str]:
        """Environment for the Kaggle CLI, with ``.env`` values applied."""
        env = os.environ.copy()
        for key in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
            value = self.get(key)
            if value:
                env[key] = value
        if self.config_dir:
            env["KAGGLE_CONFIG_DIR"] = os.path.expandvars(
                os.path.expanduser(self.config_dir)
            )
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        return env

    def redacted_summary(self) -> List[str]:
        lines = [
            f"repo: {self.root}",
            f"env:  {self.env_path} ({'present' if self.env_path.exists() else 'missing'})",
            f"credentials: {self.credentials_source() or 'NOT FOUND'}",
            f"username: {self.username or '(resolved after auth)'}",
            f"dataset prefix: {self.dataset_prefix}",
            f"job root: {self.job_root}",
            f"weights dataset: {self.weights_dataset}",
            f"use weights dataset: {self.use_weights_dataset}",
            f"session dataset: {self.use_session_dataset}",
            f"fixed kernel: {self.kernel_slug if self.fixed_kernel else '(per job)'}",
            f"accelerators: {', '.join(self.accelerators)}",
            f"queue timeout: {self.queue_timeout_minutes} min",
            f"require GPU (no CPU): {self.require_gpu}",
        ]
        return lines

    # ------------------------------------------------------------------
    # External tools
    # ------------------------------------------------------------------
    @property
    def venv_dir(self) -> Path:
        return self.root / KAGGLE_VENV_DIR

    def venv_kaggle_candidates(self) -> List[Path]:
        if os.name == "nt":
            return [
                self.venv_dir / "Scripts" / "kaggle.exe",
                self.venv_dir / "Scripts" / "kaggle",
            ]
        return [self.venv_dir / "bin" / "kaggle"]

    def find_cli(self) -> Optional[Path]:
        """Locate a Kaggle CLI 2.x executable.

        Priority: explicit ``KAGGLE_CLI``, project ``.kaggle-venv``, PATH.
        """
        explicit = self.cli_path or self.get("KAGGLE_CLI")
        if explicit:
            path = Path(os.path.expandvars(os.path.expanduser(explicit)))
            if path.is_dir():
                for name in ("kaggle.exe", "kaggle"):
                    candidate = path / name
                    if candidate.exists():
                        return candidate
                for name in ("Scripts/kaggle.exe", "Scripts/kaggle", "bin/kaggle"):
                    candidate = path / name
                    if candidate.exists():
                        return candidate
            elif path.exists():
                return path

        for candidate in self.venv_kaggle_candidates():
            if candidate.exists():
                return candidate

        found = shutil.which("kaggle")
        if found:
            return Path(found)
        return None

    def find_python_311(self) -> Optional[str]:
        """Find a Python 3.11+ interpreter usable for the Kaggle CLI venv."""
        explicit = self.python_path or self.get("KAGGLE_PYTHON")
        candidates: List[str] = []
        if explicit:
            candidates.append(os.path.expandvars(os.path.expanduser(explicit)))
        candidates.append(sys.executable)

        if os.name == "nt":
            py_launcher = shutil.which("py")
            if py_launcher:
                for version in ("3.13", "3.12", "3.11"):
                    candidates.append(f"{py_launcher} -{version}")

        for name in (
            "python3.13",
            "python3.12",
            "python3.11",
            "python3",
            "python",
        ):
            found = shutil.which(name)
            if found:
                candidates.append(found)

        seen = set()
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            # First try the whole string as an executable path.  This handles
            # Windows paths containing spaces such as "C:\\Program Files\\...".
            version = _probe_python(candidate)
            if version is None:
                parts = candidate.split()
                if len(parts) > 1:
                    # ``py -3.13`` style.  Probe the launcher with its argument.
                    try:
                        proc = subprocess.run(
                            parts
                            + ["-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                            capture_output=True,
                            text=True,
                            timeout=20,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                        if proc.returncode == 0:
                            match = re.search(r"(\d+)\.(\d+)", proc.stdout or "")
                            if match:
                                version = (int(match.group(1)), int(match.group(2)))
                    except (OSError, subprocess.SubprocessError):
                        version = None
            if version and version >= (3, 11):
                return candidate
        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def basic_issues(self) -> List[str]:
        issues: List[str] = []
        cli = self.find_cli()
        if cli is None:
            issues.append(
                "Kaggle CLI not found. Run: python -m kaggle_bridge.setup --create-venv"
            )
        if self.credentials_source() is None:
            issues.append(
                "Kaggle credentials not found. Put kaggle.json in "
                f"{self.kaggle_home} or set KAGGLE_API_TOKEN in .env"
            )
        if self.accelerator != DEFAULT_ACCELERATOR:
            issues.append(
                f"Accelerator is {self.accelerator!r}; expected {DEFAULT_ACCELERATOR!r} for T4 x2"
            )
        return issues

    def ensure_job_root(self) -> Path:
        self.job_root.mkdir(parents=True, exist_ok=True)
        return self.job_root

    def ensure_weights_root(self) -> Path:
        self.weights_root.mkdir(parents=True, exist_ok=True)
        return self.weights_root


def _config_from_env(root: Path, env_path: Path) -> Config:
    values = parse_env_file(env_path)
    cfg = Config(root=root, env_path=env_path, env_values=values)
    cfg.username = cfg.get("KAGGLE_USERNAME").strip()
    cfg.key = cfg.get("KAGGLE_KEY").strip()
    cfg.api_token = cfg.get("KAGGLE_API_TOKEN").strip()
    cfg.config_dir = cfg.get("KAGGLE_CONFIG_DIR").strip()
    cfg.cli_path = cfg.get("KAGGLE_CLI").strip()
    cfg.python_path = cfg.get("KAGGLE_PYTHON").strip()
    cfg.dataset_prefix = (
        cfg.get("KAGGLE_DATASET_PREFIX", DEFAULT_DATASET_PREFIX).strip().lower()
        or DEFAULT_DATASET_PREFIX
    )
    job_root_value = cfg.get("KAGGLE_JOB_ROOT", KAGGLE_JOB_DIR).strip() or KAGGLE_JOB_DIR
    job_root = Path(os.path.expandvars(os.path.expanduser(job_root_value)))
    if not job_root.is_absolute():
        job_root = root / job_root
    cfg.job_root = job_root
    cfg.use_weights_dataset = _truthy(
        cfg.get("KAGGLE_USE_WEIGHTS_DATASET"), True
    )
    cfg.weights_dataset = (
        cfg.get("KAGGLE_WEIGHTS_DATASET", DEFAULT_WEIGHTS_DATASET).strip()
        or DEFAULT_WEIGHTS_DATASET
    )
    weights_root_value = (
        cfg.get("KAGGLE_WEIGHTS_ROOT", DEFAULT_WEIGHTS_ROOT).strip()
        or DEFAULT_WEIGHTS_ROOT
    )
    weights_root = Path(os.path.expandvars(os.path.expanduser(weights_root_value)))
    if not weights_root.is_absolute():
        weights_root = root / weights_root
    cfg.weights_root = weights_root
    cfg.use_session_dataset = _truthy(
        cfg.get("KAGGLE_USE_SESSION_DATASET"), True
    )
    cfg.fixed_kernel = _truthy(cfg.get("KAGGLE_FIXED_KERNEL"), True)
    cfg.kernel_slug = (
        cfg.get("KAGGLE_KERNEL_SLUG", DEFAULT_KERNEL_SLUG).strip().lower()
        or DEFAULT_KERNEL_SLUG
    )
    cfg.session_dataset_prefix = (
        cfg.get("KAGGLE_SESSION_DATASET_PREFIX", DEFAULT_SESSION_PREFIX).strip().lower()
        or DEFAULT_SESSION_PREFIX
    )
    cfg.max_session_gb = max(
        0.5, _as_float(cfg.get("KAGGLE_MAX_SESSION_GB"), DEFAULT_MAX_SESSION_GB)
    )
    cfg.session_keep_local = _truthy(
        cfg.get("KAGGLE_SESSION_KEEP_LOCAL"), False
    )
    raw_acc = (
        cfg.get("KAGGLE_ACCELERATORS", "").strip()
        or cfg.get("KAGGLE_ACCELERATOR", "").strip()
        or ",".join(DEFAULT_ACCELERATORS)
    )
    accelerators = [part.strip() for part in raw_acc.split(",") if part.strip()]
    cfg.require_gpu = _truthy(cfg.get("KAGGLE_REQUIRE_GPU"), True)
    if not cfg.require_gpu and _truthy(cfg.get("KAGGLE_CPU_FALLBACK"), False):
        accelerators.append("cpu")
    seen_acc = set()
    cfg.accelerators = []
    for acc in accelerators:
        key = acc.lower()
        if key == "cpu" and cfg.require_gpu:
            continue
        if key in seen_acc:
            continue
        seen_acc.add(key)
        cfg.accelerators.append("cpu" if key == "cpu" else acc)
    if not cfg.accelerators:
        cfg.accelerators = list(DEFAULT_ACCELERATORS)
    cfg.accelerator = cfg.accelerators[0]
    cfg.queue_timeout_minutes = max(
        0.0,
        _as_float(
            cfg.get("KAGGLE_QUEUE_TIMEOUT_MINUTES"),
            DEFAULT_QUEUE_TIMEOUT_MINUTES,
        ),
    )
    cfg.poll_seconds = max(5, _as_int(cfg.get("KAGGLE_POLL_SECONDS"), DEFAULT_POLL_SECONDS))
    cfg.timeout_hours = max(
        0.25, _as_float(cfg.get("KAGGLE_TIMEOUT_HOURS"), DEFAULT_TIMEOUT_HOURS)
    )
    cfg.delete_dataset_after = _truthy(
        cfg.get("KAGGLE_DELETE_DATASET_AFTER"), True
    )
    cfg.delete_kernel_after = _truthy(cfg.get("KAGGLE_DELETE_KERNEL_AFTER"), False)
    cfg.include_sam2_checkpoint = _truthy(
        cfg.get("KAGGLE_INCLUDE_SAM2_CHECKPOINT"), True
    )
    cfg.keep_bundle = _truthy(cfg.get("KAGGLE_KEEP_BUNDLE"), False)
    cfg.accelerator = (
        cfg.get("KAGGLE_ACCELERATOR", DEFAULT_ACCELERATOR).strip()
        or DEFAULT_ACCELERATOR
    )
    cfg.max_bundle_mb = max(
        32.0, _as_float(cfg.get("KAGGLE_MAX_BUNDLE_MB"), DEFAULT_MAX_BUNDLE_MB)
    )
    return cfg


def load_config(
    root: Optional[Path] = None,
    env_path: Optional[Path] = None,
) -> Config:
    """Load configuration from ``.env`` and the process environment."""
    root = Path(root or repo_root()).resolve()
    env_path = Path(env_path) if env_path else root / ".env"
    cfg = _config_from_env(root, env_path)
    return cfg


def require_ready(cfg: Config) -> None:
    """Raise a user-facing error if a remote job cannot be started."""
    issues = cfg.basic_issues()
    if issues:
        raise ConfigError("\n".join("- " + issue for issue in issues))
    if not cfg.username:
        # The worker resolves this through the CLI after authentication.
        pass
