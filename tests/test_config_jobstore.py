from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from kaggle_bridge.config import (
    parse_env_file,
    upsert_env_file,
)
from kaggle_bridge.jobstore import JobBusyError, JobStore
from kaggle_bridge.spec import JobSpec


class ConfigTests(unittest.TestCase):
    def test_parse_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "# comment\n"
                "export A=1\n"
                'B="hello world"\n'
                "C='x#y'\n"
                "D=value # inline\n",
                encoding="utf-8",
            )
            values = parse_env_file(path)
            self.assertEqual(values["A"], "1")
            self.assertEqual(values["B"], "hello world")
            self.assertEqual(values["C"], "x#y")
            self.assertEqual(values["D"], "value")

    def test_upsert_preserves_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("# keep me\nA=1\nB=2\n", encoding="utf-8")
            upsert_env_file(path, {"B": "20", "C": "3"})
            text = path.read_text(encoding="utf-8")
            self.assertIn("# keep me", text)
            self.assertIn("A=1", text)
            self.assertIn("B=20", text)
            self.assertIn("C=3", text)


class JobStoreTests(unittest.TestCase):
    def _spec(self) -> JobSpec:
        return JobSpec(
            job_type="yolo_predict",
            images_dir="images",
            labels_dir="labels",
            models_dir="models",
            images=["a.jpg"],
            classes=["dog"],
            start_index=1,
            end_index=1,
            model_path="model.pt",
        )

    def test_create_update_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp))
            record = store.create(self._spec())
            job_id = record["job_id"]
            self.assertEqual(store.load(job_id)["state"], "created")
            store.update(job_id, state="bundling", progress=10)
            self.assertEqual(store.load(job_id)["progress"], 10)
            self.assertEqual(len(store.list_jobs()), 1)
            self.assertEqual(store.find_active()["job_id"], job_id)

    def test_lock_serializes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp))
            record = store.create(self._spec())
            job_id = record["job_id"]
            store.acquire_lock(job_id)
            with self.assertRaises(JobBusyError):
                store.acquire_lock(job_id)
            store.release_lock(job_id)
            store.acquire_lock(job_id)
            store.release_lock(job_id)

    def test_job_id_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp))
            with self.assertRaises(ValueError):
                store.job_dir("../escape")


if __name__ == "__main__":
    unittest.main()
