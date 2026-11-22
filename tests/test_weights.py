from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from kaggle_bridge.config import Config
from kaggle_bridge.kaggle_cli import KaggleCliError
from kaggle_bridge.weights import WeightsManager, local_weight_files


class FakeCLI:
    def __init__(self, staging: Path):
        self.staging = staging
        self.create_calls = 0
        self.version_calls = 0
        self.uploaded: dict[str, str] = {}

    def _upload(self):
        self.uploaded = {
            path.name: str(path.stat().st_size)
            for path in self.staging.glob("*")
            if path.is_file() and path.name != "dataset-metadata.json"
        }

    def dataset_files(self, ref):
        return [
            {"name": name, "size": size}
            for name, size in sorted(self.uploaded.items())
        ]

    def dataset_create(self, folder):
        self.create_calls += 1
        if self.create_calls > 1:
            raise KaggleCliError("Dataset creation error: already in use")
        self._upload()
        return None

    def dataset_version(self, folder, notes):
        self.version_calls += 1
        self._upload()
        return None

    def wait_dataset_ready(self, ref, timeout=1800, poll=5):
        return "ready"


class WeightsTests(unittest.TestCase):
    def test_reuse_and_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = root / "jobs"
            staging = root / "weights"
            cfg = Config(
                root=root,
                env_path=root / ".env",
                job_root=jobs,
                weights_root=staging,
                weights_dataset="labelimg-plus-weights",
            )
            cfg.ensure_job_root()
            model = root / "model.pt"
            model.write_bytes(b"0" * (2 * 1024 * 1024))

            fake = FakeCLI(staging)
            manager = WeightsManager(cfg, fake)
            mapping = manager.ensure_file(model, "alice")
            self.assertEqual(mapping, "model.pt")
            self.assertEqual(fake.create_calls, 1)
            self.assertTrue((staging / "model.pt").is_file())

            # Second call reuses the remote/staged file.
            mapping = manager.ensure_file(model, "alice")
            self.assertEqual(mapping, "model.pt")
            self.assertEqual(fake.create_calls, 1)
            self.assertEqual(fake.version_calls, 0)

            # A changed model triggers a new dataset version.
            model.write_bytes(b"1" * (2 * 1024 * 1024))
            os.utime(model, (model.stat().st_atime + 10, model.stat().st_mtime + 10))
            manager.ensure_file(model, "alice")
            self.assertEqual(fake.version_calls, 1)

    def test_dataset_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Config(
                root=root,
                env_path=root / ".env",
                job_root=root / "jobs",
                weights_root=root / "weights",
                weights_dataset="labelimg-plus-weights",
            )
            manager = WeightsManager(cfg, FakeCLI(root / "weights"))
            self.assertEqual(
                manager.dataset_id("alice"),
                "alice/labelimg-plus-weights",
            )

    def test_local_weight_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "annotation_helper" / "models"
            sam2 = root / "annotation_helper" / "sam2_models"
            models.mkdir(parents=True)
            sam2.mkdir(parents=True)
            (models / "yolov8n.pt").write_bytes(b"0" * (2 * 1024 * 1024))
            (models / "sam3.pt").write_bytes(b"0" * (2 * 1024 * 1024))
            (sam2 / "sam2.1_hiera_small.pt").write_bytes(b"0" * (2 * 1024 * 1024))
            found = {path.name for path in local_weight_files(root)}
            self.assertIn("yolov8n.pt", found)
            self.assertIn("sam2.1_hiera_small.pt", found)
            self.assertNotIn("sam3.pt", found)


if __name__ == "__main__":
    unittest.main()
