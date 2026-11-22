from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from kaggle_bridge.kaggle_cli import _norm_state
from kaggle_bridge.kernel_runner import load_manifest, resolve_model_file, safe_extract


class KernelRunnerTests(unittest.TestCase):
    def test_norm_state(self):
        self.assertEqual(_norm_state("KernelWorkerStatus.COMPLETE"), "COMPLETE")
        self.assertEqual(_norm_state("running"), "RUNNING")

    def test_safe_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "ok.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("manifest.json", "{}")
                zf.writestr("images/000000.jpg", b"x")
            destination = root / "out"
            safe_extract(archive, destination)
            self.assertTrue((destination / "manifest.json").is_file())
            self.assertTrue((destination / "images" / "000000.jpg").is_file())

    def test_safe_extract_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("../evil.txt", "x")
            with self.assertRaises(RuntimeError):
                safe_extract(archive, root / "out")

    def test_load_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            (job / "images").mkdir()
            (job / "images" / "000000.jpg").write_bytes(b"x")
            manifest = {
                "images": [
                    {
                        "name": "frame.jpg",
                        "file": "images/000000.jpg",
                        "width": 10,
                        "height": 10,
                    }
                ],
                "labels": {"frame.jpg": []},
            }
            (job / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(load_manifest(job)["images"][0]["name"], "frame.jpg")

    def test_resolve_bundle_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            (job / "model").mkdir()
            (job / "model" / "x.pt").write_bytes(b"x")
            path = resolve_model_file(job, {"file": "model/x.pt"})
            self.assertEqual(path, (job / "model" / "x.pt").resolve())

    def test_resolve_weights_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights"
            weights.mkdir()
            (weights / "x.pt").write_bytes(b"x")
            old = os.environ.get("LABELIMG_WEIGHTS_DIR")
            os.environ["LABELIMG_WEIGHTS_DIR"] = str(weights)
            try:
                path = resolve_model_file(
                    root / "job",
                    {"source": "weights", "weights_file": "x.pt"},
                )
            finally:
                if old is None:
                    os.environ.pop("LABELIMG_WEIGHTS_DIR", None)
                else:
                    os.environ["LABELIMG_WEIGHTS_DIR"] = old
            self.assertEqual(path, (weights / "x.pt").resolve())

    def test_load_manifest_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            manifest = {
                "images": [
                    {
                        "name": "frame.jpg",
                        "file": "../outside.jpg",
                        "width": 10,
                        "height": 10,
                    }
                ],
                "labels": {},
            }
            (job / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_manifest(job)


if __name__ == "__main__":
    unittest.main()
