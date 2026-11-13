"""One-time model/checkpoint upload and reuse.

Large weights (YOLO `.pt`, SAM 2 checkpoints) are uploaded to a private
Kaggle dataset once.  Every annotation job then sends only images + current
labels + a tiny manifest, and the kernel reads the model from the attached
weights dataset.  This is the "models once, images later" flow.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .config import Config
from .kaggle_cli import KaggleCLI, KaggleCliError
from .metadata import dataset_metadata, validate_slug, write_json_atomic

LogFn = Optional[Callable[[str], None]]


class WeightsError(RuntimeError):
    pass


def _sha8(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:8]


class WeightsManager:
    def __init__(self, cfg: Config, cli: KaggleCLI, log: LogFn = None):
        self.cfg = cfg
        self.cli = cli
        self.log = log
        self.state_path = Path(cfg.job_root) / "weights_state.json"

    def _load_state(self) -> Dict[str, Dict[str, str]]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_state(self, state: Dict[str, Dict[str, str]]) -> None:
        try:
            self.cfg.ensure_job_root()
            write_json_atomic(self.state_path, state)
        except OSError:
            pass

    @staticmethod
    def _fingerprint(path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def _log(self, message: str) -> None:
        if self.log:
            try:
                self.log(message)
            except Exception:
                pass

    def dataset_id(self, username: str) -> str:
        raw = (self.cfg.weights_dataset or "labelimg-plus-weights").strip()
        if "/" in raw:
            owner, slug = raw.split("/", 1)
            validate_slug(owner.strip().lower(), "weights dataset owner")
            validate_slug(slug.strip().lower(), "weights dataset slug")
            return f"{owner.strip().lower()}/{slug.strip().lower()}"
        owner = validate_slug((username or "").strip().lower(), "Kaggle username")
        return f"{owner}/{validate_slug(raw.lower(), 'weights dataset slug')}"

    def _remote_name(self, local_path: Path, staging: Path) -> str:
        name = local_path.name
        existing = staging / name
        if existing.exists() and existing.stat().st_size != local_path.stat().st_size:
            suffix = local_path.suffix
            name = f"{local_path.stem}-{_sha8(local_path)}{suffix}"
        return name

    def ensure_file(self, local_path: Path, username: str) -> str:
        source = Path(local_path)
        if not source.is_file():
            raise FileNotFoundError(f"Weight file not found: {source}")
        mapping = self.ensure_files([source], username)
        return mapping[str(source.resolve())]

    def ensure_files(self, local_paths: Iterable[Path], username: str) -> Dict[str, str]:
        """Ensure all files exist in the private weights dataset.

        Returns ``{absolute_local_path: remote_filename}``.
        """
        staging = self.cfg.ensure_weights_root()
        ref = self.dataset_id(username)
        mapping: Dict[str, str] = {}
        local_by_remote: Dict[str, Path] = {}
        state = self._load_state()
        changed = False

        for raw_path in local_paths:
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Weight file not found: {path}")
            if path.stat().st_size < 1024:
                raise WeightsError(f"Weight file is suspiciously small: {path}")
            remote_name = self._remote_name(path, staging)
            mapping[str(path)] = remote_name
            local_by_remote[remote_name] = path
            staging_path = staging / remote_name
            fingerprint = self._fingerprint(path)
            old_fingerprint = (state.get(remote_name) or {}).get("fingerprint")

            if not staging_path.exists():
                self._log(
                    f"Staging weight {path.name} "
                    f"({path.stat().st_size / 1024 / 1024:.0f} MB)..."
                )
                shutil.copyfile(path, staging_path)
            elif staging_path.stat().st_size != path.stat().st_size:
                self._log(f"Weight changed; restaging {path.name}...")
                shutil.copyfile(path, staging_path)
                changed = True

            if old_fingerprint and old_fingerprint != fingerprint:
                self._log(f"Weight changed; uploading a new version: {path.name}")
                shutil.copyfile(path, staging_path)
                changed = True

            state[remote_name] = {
                "fingerprint": fingerprint,
                "size": str(path.stat().st_size),
            }

        remote_files = {
            str(row.get("name") or "").replace("\\", "/"): str(row.get("size") or "")
            for row in self.cli.dataset_files(ref)
        }
        missing: List[str] = []
        for remote_name, path in local_by_remote.items():
            if remote_name not in remote_files:
                missing.append(remote_name)
                continue
            remote_size = remote_files.get(remote_name, "")
            if remote_size.isdigit() and int(remote_size) != path.stat().st_size:
                missing.append(remote_name)
        if changed:
            missing = sorted(set(missing) | set(local_by_remote))

        if not missing:
            self._save_state(state)
            self._log(
                "Weights already on Kaggle; reusing "
                + ", ".join(sorted(local_by_remote))
            )
            return mapping

        self._log(
            "Uploading model weights once to "
            f"{ref}: " + ", ".join(sorted(missing))
        )
        write_json_atomic(
            staging / "dataset-metadata.json",
            dataset_metadata(ref, "LabelImg Plus Weights"),
        )
        try:
            self.cli.dataset_create(staging)
        except KaggleCliError as exc:
            text = str(exc).lower()
            if "already in use" not in text and "already exists" not in text:
                raise
            self._log("Weights dataset already exists; adding a new version.")
            self.cli.dataset_version(
                staging,
                "Add weights: " + ", ".join(sorted(missing)),
            )
        # A 403 during creation is handled by wait_dataset_ready as
        # "processing"; a real failure is re-raised.
        self.cli.wait_dataset_ready(ref, timeout=1800, poll=5)
        self._save_state(state)
        self._log(f"Weights ready: {ref}")
        return mapping


def local_weight_files(root: Path) -> List[Path]:
    """Find YOLO models and SAM 2 checkpoints that belong in the weights dataset."""
    root = Path(root)
    files: List[Path] = []
    models_dir = root / "annotation_helper" / "models"
    if models_dir.is_dir():
        for path in sorted(models_dir.glob("*.pt")):
            low = path.name.lower()
            if low.startswith("sam3") or ".bf16." in low:
                continue
            if path.stat().st_size >= 1024 * 1024:
                files.append(path)
    sam2_dir = root / "annotation_helper" / "sam2_models"
    if sam2_dir.is_dir():
        for path in sorted(sam2_dir.glob("*.pt")):
            if path.stat().st_size >= 1024 * 1024:
                files.append(path)
    return files


def sync_weights(cfg: Config, username: str = "", log: LogFn = None) -> Dict[str, str]:
    """Upload/reuse every local YOLO model and SAM 2 checkpoint once."""
    files = local_weight_files(cfg.root)
    if not files:
        raise WeightsError(
            "No local YOLO models or SAM 2 checkpoints found in "
            f"{cfg.root / 'annotation_helper' / 'models'} and "
            f"{cfg.root / 'annotation_helper' / 'sam2_models'}."
        )
    cli = KaggleCLI(cfg, log=log)
    username = username or cli.check_auth()
    manager = WeightsManager(cfg, cli, log=log)
    mapping = manager.ensure_files(files, username)
    if log:
        log(f"Weights dataset: {manager.dataset_id(username)}")
    return mapping
