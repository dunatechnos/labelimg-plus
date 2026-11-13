"""Thin, well-behaved wrapper around the Kaggle CLI 2.x.

The annotation helper is allowed to run on Python 3.10; the Kaggle CLI 2.x
needs Python 3.11+.  Everything here therefore talks to the CLI as a
subprocess, never imports the ``kaggle`` Python package.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import Config

LogFn = Optional[Callable[[str], None]]


class KaggleCliError(RuntimeError):
    def __init__(
        self,
        message: str,
        returncode: int = -1,
        stdout: str = "",
        stderr: str = "",
    ):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass
class CliResult:
    args: List[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return (self.stdout or "") + (("\n" + self.stderr) if self.stderr else "")


@dataclass
class KernelStatus:
    raw: str
    state: str
    failure: str = ""

    @property
    def terminal(self) -> bool:
        return self.state in {
            "COMPLETE",
            "ERROR",
            "CANCEL_ACKNOWLEDGED",
            "CANCEL_REQUESTED",
            "NOT_FOUND",
        }

    @property
    def success(self) -> bool:
        return self.state == "COMPLETE"


def _norm_state(value: str) -> str:
    value = (value or "").strip().strip('"').strip("'")
    if "." in value:
        value = value.rsplit(".", 1)[-1]
    return value.upper().replace("-", "_").replace(" ", "_")


def _tail(text: str, limit: int = 4000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:1000] + "\n...\n" + text[-limit:]


class KaggleCLI:
    def __init__(self, cfg: Config, log: LogFn = None):
        self.cfg = cfg
        self.log = log
        self.exe = cfg.find_cli()
        if self.exe is None:
            raise KaggleCliError(
                "Kaggle CLI not found. Run:\n"
                "  python -m kaggle_bridge.setup --create-venv\n"
                "or put a Kaggle CLI 2.x executable on PATH and set KAGGLE_CLI."
            )

    # ------------------------------------------------------------------
    # Process execution
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self.log:
            try:
                self.log(message)
            except Exception:
                pass

    def _run(
        self,
        args: Sequence[str],
        timeout: float = 300,
        check: bool = True,
        cwd: Optional[Path] = None,
    ) -> CliResult:
        cmd = [str(self.exe)] + [str(arg) for arg in args]
        self._log("$ " + " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
                env=self.cfg.auth_env(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise KaggleCliError(
                f"Kaggle CLI command timed out after {timeout:.0f}s: {' '.join(cmd)}"
            ) from exc
        except OSError as exc:
            raise KaggleCliError(
                f"Could not start the Kaggle CLI ({self.exe}): {exc}"
            ) from exc

        result = CliResult(list(args), proc.returncode, proc.stdout or "", proc.stderr or "")
        output = result.output.strip()
        if output:
            self._log(_tail(output))
        if check and result.returncode != 0:
            combined = result.output.lower()
            if "authentication required" in combined or "401" in combined:
                raise KaggleCliError(
                    "Kaggle authentication failed. Put a fresh kaggle.json in "
                    f"{self.cfg.kaggle_home} or set KAGGLE_API_TOKEN in .env.",
                    result.returncode,
                    result.stdout,
                    result.stderr,
                )
            raise KaggleCliError(
                f"Kaggle CLI failed with exit code {result.returncode}: "
                f"{_tail(result.output, 2500)}",
                result.returncode,
                result.stdout,
                result.stderr,
            )
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def version(self) -> str:
        result = self._run(["--version"], timeout=60)
        return (result.stdout or result.stderr).strip()

    def version_tuple(self) -> Tuple[int, int, int]:
        match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", self.version())
        if not match:
            return (0, 0, 0)
        return (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3) or 0),
        )

    def check_auth(self) -> str:
        """Verify credentials and return the Kaggle username."""
        try:
            self._run(
                ["datasets", "list", "-m", "--page-size", "1", "--csv"],
                timeout=120,
            )
        except KaggleCliError as exc:
            combined = (exc.stdout + "\n" + exc.stderr).lower()
            if "authentication required" in combined or "401" in combined:
                raise KaggleCliError(
                    "Kaggle authentication failed.\n"
                    "1. Open https://www.kaggle.com/settings/api\n"
                    "2. Create a new token\n"
                    f"3. Save it as {self.cfg.kaggle_json_path} "
                    "(or set KAGGLE_API_TOKEN in .env)\n"
                    "4. Run: python -m kaggle_bridge.doctor",
                    exc.returncode,
                    exc.stdout,
                    exc.stderr,
                ) from exc
            raise
        return self.current_username()

    def current_username(self) -> str:
        if self.cfg.username:
            return self.cfg.username
        result = self._run(["config", "view"], timeout=60, check=False)
        text = result.output
        match = re.search(
            r"(?im)^\s*-\s*username\s*:\s*([^\s]+)\s*$", text
        )
        if match and match.group(1).lower() not in ("none", ""):
            self.cfg.username = match.group(1).strip()
            return self.cfg.username

        # Fallback: the first row of "my datasets" contains owner/slug.
        listing = self._run(
            ["datasets", "list", "-m", "--page-size", "1", "--csv"],
            timeout=120,
            check=False,
        )
        for line in (listing.stdout or "").splitlines():
            line = line.strip()
            if not line or line.lower().startswith("ref,"):
                continue
            ref = line.split(",")[0].strip()
            if "/" in ref:
                self.cfg.username = ref.split("/", 1)[0].strip()
                return self.cfg.username
        raise KaggleCliError(
            "Could not determine your Kaggle username. Set KAGGLE_USERNAME in .env."
        )

    def quota(self) -> List[Dict[str, str]]:
        """Return the weekly accelerator quota as a list of dicts."""
        result = self._run(["quota", "--format", "json"], timeout=90)
        try:
            data = json.loads(result.stdout or "[]")
        except ValueError:
            return []
        return data if isinstance(data, list) else []

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def dataset_exists(self, ref: str) -> bool:
        status = self.dataset_status(ref)
        return status in ("ready", "processing", "ok", "complete", "pending")

    def dataset_create(self, folder: Path) -> CliResult:
        result = self._run(
            ["datasets", "create", "-p", str(folder)], timeout=7200
        )
        text = result.output.lower()
        if "dataset creation error" in text:
            raise KaggleCliError(
                "Kaggle dataset creation failed:\n" + _tail(result.output, 2500),
                result.returncode,
                result.stdout,
                result.stderr,
            )
        self._log("Dataset creation request accepted.")
        return result

    def dataset_version(self, folder: Path, notes: str) -> CliResult:
        result = self._run(
            ["datasets", "version", "-p", str(folder), "-m", notes],
            timeout=7200,
        )
        text = result.output.lower()
        if "dataset version creation error" in text or "dataset creation error" in text:
            raise KaggleCliError(
                "Kaggle dataset version creation failed:\n"
                + _tail(result.output, 2500),
                result.returncode,
                result.stdout,
                result.stderr,
            )
        self._log("Dataset version request accepted.")
        return result

    def dataset_status(self, ref: str) -> str:
        result = self._run(["datasets", "status", ref], timeout=120, check=False)
        if result.returncode != 0:
            text = result.output.lower()
            # Kaggle returns 403 for a brand-new private dataset while it is
            # still being created/processed.  Treat it as a transient state
            # instead of failing the whole job (seen on a ~100 MB upload).
            if "403" in text or "forbidden" in text:
                self._log(
                    f"Dataset {ref} is not queryable yet (403); treating as processing."
                )
                return "processing"
            if "404" in text or "not found" in text:
                return "not_found"
            raise KaggleCliError(
                f"Could not read dataset status for {ref}:\n{_tail(result.output, 2000)}",
                result.returncode,
                result.stdout,
                result.stderr,
            )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return "unknown"
        value = lines[-1].lower()
        # The CLI may print extra decoration; take the last token if needed.
        for candidate in ("ready", "processing", "pending", "failed", "error", "ok"):
            if candidate in value:
                return candidate
        return value

    def dataset_files(self, ref: str) -> List[Dict[str, str]]:
        """Return dataset files as ``[{"name": ..., "size": ...}, ...]``."""
        result = self._run(
            ["datasets", "files", ref, "--csv"],
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            return []
        text = (result.stdout or "").strip()
        if not text:
            return []
        rows: List[Dict[str, str]] = []
        try:
            import csv
            import io

            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                name = (row.get("name") or row.get("fileName") or "").strip()
                if name:
                    rows.append(
                        {
                            "name": name,
                            "size": str(row.get("size") or row.get("totalBytes") or ""),
                        }
                    )
        except Exception:
            for line in text.splitlines()[1:]:
                parts = [part.strip() for part in line.split(",")]
                if parts and parts[0]:
                    rows.append(
                        {
                            "name": parts[0],
                            "size": parts[1] if len(parts) > 1 else "",
                        }
                    )
        return rows

    def wait_dataset_ready(
        self,
        ref: str,
        timeout: float = 1200,
        poll: float = 5,
    ) -> str:
        import time

        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            status = self.dataset_status(ref)
            if status != last:
                self._log(f"Dataset {ref} status: {status}")
                last = status
            if status in ("ready", "ok", "complete"):
                return status
            if status in ("error", "failed"):
                raise KaggleCliError(f"Kaggle dataset {ref} failed with status {status!r}")
            if status in ("not_found",):
                raise KaggleCliError(
                    f"Kaggle dataset {ref} was not found after creation."
                )
            time.sleep(max(2, poll))
        raise KaggleCliError(
            f"Kaggle dataset {ref} was not ready after {timeout:.0f}s "
            f"(last status: {last!r})."
        )

    def dataset_delete(self, ref: str) -> None:
        self._run(["datasets", "delete", ref, "-y"], timeout=180, check=True)

    # ------------------------------------------------------------------
    # Kernels
    # ------------------------------------------------------------------
    def kernels_push(self, folder: Path, accelerator: str = "NvidiaTeslaT4") -> CliResult:
        result = self._run(
            [
                "kernels",
                "push",
                "-p",
                str(folder),
                "--accelerator",
                accelerator,
            ],
            timeout=900,
        )
        text = result.output.lower()
        if "kernel push error" in text:
            raise KaggleCliError(
                "Kaggle kernel push failed:\n" + _tail(result.output, 2500),
                result.returncode,
                result.stdout,
                result.stderr,
            )
        for marker in (
            "not valid dataset sources",
            "not valid kernel sources",
            "not valid competition sources",
            "not valid model sources",
        ):
            if marker in text:
                raise KaggleCliError(
                    "Kaggle rejected one or more kernel sources:\n"
                    + _tail(result.output, 2500),
                    result.returncode,
                    result.stdout,
                    result.stderr,
                )
        if "pushed" not in text and "successfully" not in text:
            raise KaggleCliError(
                "Kaggle kernel push did not confirm success:\n"
                + _tail(result.output, 2500),
                result.returncode,
                result.stdout,
                result.stderr,
            )
        return result

    def kernels_status(self, ref: str) -> KernelStatus:
        result = self._run(["kernels", "status", ref], timeout=120, check=False)
        text = result.output
        match = re.search(r'has\s+status\s+"([^"]+)"', text, re.IGNORECASE)
        if not match:
            match = re.search(r"status\s*:\s*([^\s]+)", text, re.IGNORECASE)
        raw = match.group(1) if match else ""
        state = _norm_state(raw)
        failure = ""
        failure_match = re.search(
            r'Failure message:\s*"([^"]*)"', text, re.IGNORECASE | re.DOTALL
        )
        if failure_match:
            failure = failure_match.group(1).strip()
        if result.returncode != 0 and not state:
            lowered = text.lower()
            if "authentication required" in lowered or "401" in lowered:
                raise KaggleCliError(
                    "Kaggle authentication failed while checking kernel status.",
                    result.returncode,
                    result.stdout,
                    result.stderr,
                )
            if "404" in lowered or "not found" in lowered:
                state = "NOT_FOUND"
            else:
                state = "UNKNOWN"
        return KernelStatus(raw=text.strip(), state=state, failure=failure)

    def kernels_output(self, ref: str, dest: Path) -> CliResult:
        dest.mkdir(parents=True, exist_ok=True)
        result = self._run(
            ["kernels", "output", ref, "-p", str(dest), "-o"],
            timeout=3600,
        )
        return result

    def kernels_logs(self, ref: str, dest: Path) -> str:
        result = self._run(
            ["kernels", "logs", ref], timeout=300, check=False
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = result.output
        try:
            dest.write_text(text, encoding="utf-8")
        except OSError:
            pass
        return text

    def kernels_delete(self, ref: str) -> None:
        self._run(["kernels", "delete", ref, "-y"], timeout=180, check=True)
