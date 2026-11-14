from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kaggle_bridge.config import Config
from kaggle_bridge.jobstore import JobStore
from kaggle_bridge.kaggle_cli import KernelStatus
from kaggle_bridge.spec import JobSpec
from kaggle_bridge.worker import KaggleJobError, KaggleJobRunner, QueueTimeout


class FakeQueuedCLI:
    def kernels_status(self, ref):
        return KernelStatus(raw="QUEUED", state="QUEUED")


class WorkerCancelTests(unittest.TestCase):
    def test_queue_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Config(
                root=root,
                env_path=root / ".env",
                job_root=root / "jobs",
            )
            store = JobStore(cfg.job_root)
            spec = JobSpec(
                job_type="yolo_predict",
                images_dir=str(root / "images"),
                labels_dir=str(root / "labels"),
                models_dir=str(root / "models"),
                images=["a.jpg"],
                classes=["dog"],
                model_path=str(root / "model.pt"),
            )
            record = store.create(spec)
            runner = KaggleJobRunner(cfg, store, sleep_fn=lambda seconds: None)
            with self.assertRaises(QueueTimeout):
                runner._wait_for_kernel(
                    record["job_id"],
                    FakeQueuedCLI(),
                    "alice/kernel/1",
                    spec,
                    queue_timeout_minutes=0.0001,
                )

    def test_require_gpu_rejects_cpu_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            cfg = Config(
                root=root,
                env_path=root / ".env",
                job_root=root / "jobs",
                require_gpu=True,
            )
            store = JobStore(cfg.job_root)
            spec = JobSpec(
                job_type="yolo_predict",
                images_dir=str(images),
                labels_dir=str(labels),
                models_dir=str(root / "models"),
                images=["a.jpg"],
                classes=["dog"],
                model_path=str(root / "model.pt"),
            )
            record = store.create(spec)
            job_id = record["job_id"]
            output = store.output_dir(job_id)
            output.mkdir(parents=True)
            (output / "result.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "stats": {"gpu_count": 0, "gpu_names": []},
                    }
                ),
                encoding="utf-8",
            )
            (output / "labels.json").write_text("{}", encoding="utf-8")
            runner = KaggleJobRunner(cfg, store)
            with self.assertRaises(KaggleJobError) as context:
                runner._import_only(job_id, spec, output)
            self.assertIn("CPU", str(context.exception))
            self.assertFalse((labels / "a.txt").exists())

    def test_cancel_sets_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Config(root=root, env_path=root / ".env", job_root=root / "jobs")
            store = JobStore(cfg.job_root)
            runner = KaggleJobRunner(cfg, store)
            self.assertFalse(runner.cancel_event.is_set())
            runner.cancel()
            self.assertTrue(runner.cancel_event.is_set())


if __name__ == "__main__":
    unittest.main()
