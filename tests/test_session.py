from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from kaggle_bridge import kernel_runner
from kaggle_bridge.config import Config
from kaggle_bridge.session import (
    RemoteSession,
    SessionInfo,
    build_command_payload,
    embed_command,
    image_fingerprint,
)
from kaggle_bridge.spec import JobSpec


def _make_images(root: Path, count: int = 3) -> tuple[Path, Path, list[str]]:
    import cv2
    import numpy as np

    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    names = []
    for index in range(1, count + 1):
        image = np.full((120, 160, 3), 30, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (100, 90), (20, 200, 20), -1)
        name = f"frame_{index:03d}.jpg"
        cv2.imwrite(str(images / name), image)
        (labels / f"frame_{index:03d}.txt").write_text(
            "0 0.300000 0.400000 0.300000 0.300000\n", encoding="utf-8"
        )
        names.append(name)
    return images, labels, names


class FakeSessionCLI:
    def __init__(self):
        self.created = 0
        self.deleted = []
        self.uploaded_files = []

    def dataset_status(self, ref):
        return "ready" if self.created else "not_found"

    def dataset_create(self, folder):
        self.created += 1
        self.uploaded_files = [p.name for p in Path(folder).glob("*")]

    def dataset_version(self, folder, notes):
        self.created += 1
        self.uploaded_files = [p.name for p in Path(folder).glob("*")]

    def wait_dataset_ready(self, ref, timeout=1800, poll=5):
        return "ready"

    def dataset_delete(self, ref):
        self.deleted.append(ref)


class SessionTests(unittest.TestCase):
    def test_ensure_upload_reuse_cleanup(self):
        try:
            import cv2  # noqa: F401
        except Exception:
            self.skipTest("opencv not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels, names = _make_images(root)
            wheels = root / "weights" / "wheels"
            wheels.mkdir(parents=True)
            (wheels / "sam_2-1.0-py3-none-any.whl").write_bytes(b"wheel")
            cfg = Config(
                root=root,
                env_path=root / ".env",
                job_root=root / "jobs",
                weights_root=root / "weights",
                session_keep_local=True,
                max_session_gb=1.0,
            )
            cfg.ensure_job_root()
            fake = FakeSessionCLI()
            session = RemoteSession(cfg, fake)
            info = session.ensure_images(
                str(images), names, ["dog"], username="alice"
            )
            self.assertEqual(info.image_count, 3)
            self.assertTrue(info.dataset_ref.startswith("alice/labelimg-plus-session-"))
            self.assertEqual(fake.created, 1)
            self.assertIn("session.zip", fake.uploaded_files)
            self.assertTrue((root / "jobs" / "session.json").is_file())

            archive = root / "jobs" / "sessions" / info.session_id / "session.zip"
            with zipfile.ZipFile(archive) as zf:
                names_in_zip = zf.namelist()
            self.assertIn("session.json", names_in_zip)
            self.assertIn("runner_lib.py", names_in_zip)
            self.assertIn("wheels/sam_2-1.0-py3-none-any.whl", names_in_zip)

            # Second call reuses the same dataset, no upload.
            info2 = session.ensure_images(
                str(images), names, ["dog"], username="alice"
            )
            self.assertEqual(info.dataset_ref, info2.dataset_ref)
            self.assertEqual(fake.created, 1)

            self.assertTrue(session.cleanup())
            self.assertEqual(fake.deleted, [info.dataset_ref])
            self.assertFalse((root / "jobs" / "session.json").exists())

    def test_fingerprint_changes(self):
        try:
            import cv2  # noqa: F401
        except Exception:
            self.skipTest("opencv not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, _labels, names = _make_images(root, count=2)
            first = image_fingerprint(images, names)
            (images / "extra.jpg").write_bytes(b"x")
            second = image_fingerprint(images, names + ["extra.jpg"])
            self.assertNotEqual(first, second)

    def test_command_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels, names = _make_images(root, count=3)
            spec = JobSpec(
                job_type="yolo_predict",
                images_dir=str(images),
                labels_dir=str(labels),
                models_dir=str(root / "models"),
                images=names,
                classes=["dog"],
                start_index=2,
                end_index=3,
                model_path=str(root / "model.pt"),
            )
            info = SessionInfo(
                session_id="s1",
                dataset_ref="alice/labelimg-plus-session-s1",
                images_dir=str(images),
                fingerprint="f",
                image_count=3,
                classes=["dog"],
                created_at="now",
            )
            payload = build_command_payload(
                spec, {"kind": "yolo", "source": "weights"}, info, "job1"
            )
            self.assertEqual(payload["params"]["indices"], [1, 2])
            self.assertEqual(set(payload["labels"]), {"frame_002.jpg", "frame_003.jpg"})

    def test_embed_command_and_manifest(self):
        try:
            import cv2  # noqa: F401
        except Exception:
            self.skipTest("opencv not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_root = root / "session"
            (session_root / "images").mkdir(parents=True)
            for index in range(2):
                shutil.copyfile(
                    Path(__file__).resolve().parent.parent / "README.md",
                    session_root / "images" / f"{index:06d}.jpg",
                )
            session_json = {
                "images": [
                    {"name": "a.jpg", "file": "images/000000.jpg", "width": 1, "height": 1},
                    {"name": "b.jpg", "file": "images/000001.jpg", "width": 1, "height": 1},
                ]
            }
            (session_root / "session.json").write_text(
                json.dumps(session_json), encoding="utf-8"
            )
            shutil.copyfile(
                Path(__file__).resolve().parent.parent / "kaggle_bridge" / "runner_lib.py",
                session_root / "runner_lib.py",
            )
            command = {
                "job_id": "job1",
                "type": "yolo_predict",
                "classes": ["dog"],
                "labels": {"b.jpg": []},
                "model": {"kind": "yolo", "source": "weights", "weights_file": "m.pt"},
                "params": {"indices": [1]},
            }
            old_job = kernel_runner.JOB_DIR
            old_out = kernel_runner.OUTPUT_DIR
            kernel_runner.JOB_DIR = root / "work"
            kernel_runner.OUTPUT_DIR = root / "out"
            try:
                manifest = kernel_runner.build_manifest_from_session(
                    session_root, command
                )
            finally:
                kernel_runner.JOB_DIR = old_job
                kernel_runner.OUTPUT_DIR = old_out
            self.assertEqual(len(manifest["images"]), 1)
            self.assertEqual(manifest["images"][0]["name"], "b.jpg")

            source = "EMBEDDED_COMMAND = None  # KAGGLE_BRIDGE_COMMAND\nprint('x')\n"
            embedded = embed_command(source, command)
            self.assertIn("'job_id': 'job1'", embedded)
            self.assertIn("'selection_area': None", embed_command(
                source,
                {"selection_area": None},
            ))
            self.assertNotIn("KAGGLE_BRIDGE_COMMAND", embedded)


if __name__ == "__main__":
    unittest.main()
