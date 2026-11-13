from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from kaggle_bridge.kaggle_cli import _norm_state
from kaggle_bridge.kernel_runner import load_manifest, safe_extract


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
