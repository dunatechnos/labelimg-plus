from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from kaggle_bridge.bundle import build_bundle
from kaggle_bridge.importer import import_job_output
from kaggle_bridge.spec import JobSpec


def _make_dataset(root: Path) -> tuple[Path, Path, Path, list[str]]:
    import cv2
    import numpy as np

    images = root / "images"
    labels = root / "labels"
    models = root / "models"
    for path in (images, labels, models):
        path.mkdir(parents=True, exist_ok=True)
    names = []
    for index in range(1, 4):
        image = np.full((240, 320, 3), 20, dtype=np.uint8)
        cv2.rectangle(image, (40, 50), (140, 160), (20, 200, 20), -1)
        name = f"frame_{index:03d}.jpg"
        cv2.imwrite(str(images / name), image)
        (labels / f"frame_{index:03d}.txt").write_text(
            "0 0.300000 0.450000 0.300000 0.300000\n", encoding="utf-8"
        )
        names.append(name)
    model = models / "dummy.pt"
    model.write_bytes(b"0" * (2 * 1024 * 1024))
    return images, labels, model, names


class BundleImportTests(unittest.TestCase):
    def test_yolo_bundle_and_import(self):
        try:
            import cv2  # noqa: F401
        except Exception:
            self.skipTest("opencv not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels, model, names = _make_dataset(root)
            spec = JobSpec(
                job_type="yolo_predict",
                images_dir=str(images),
                labels_dir=str(labels),
                models_dir=str(images.parent / "models"),
                images=names,
                classes=["dog"],
                start_index=1,
                end_index=2,
                model_path=str(model),
            )
            spec.validate()
            job_dir = root / "job"
            info = build_bundle(
                spec,
                job_dir,
                include_sam2_checkpoint=False,
                keep_bundle=True,
                progress=lambda percent, message: None,
            )
            self.assertTrue(info.job_zip.exists())
            with zipfile.ZipFile(info.job_zip) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["type"], "yolo_predict")
                self.assertEqual(len(manifest["images"]), 2)
                self.assertIn("labels", manifest)
                self.assertIn("runner_lib.py", archive.namelist())

            output = job_dir / "output"
            output.mkdir()
            (output / "result.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "job_id": job_dir.name,
                        "changed": 1,
                        "message": "ok",
                    }
                ),
                encoding="utf-8",
            )
            (output / "coordinates.json").write_text(
                json.dumps(
                    {
                        "frame_001.jpg": [
                            ["0", "0.3", "0.4", "0.2", "0.3"],
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = import_job_output(
                job_dir, names, ["dog"], labels_dir=labels
            )
            self.assertEqual(result.changed, 1)
            self.assertEqual(result.changed_names, ["frame_001.jpg"])
            self.assertEqual(result.labels_path.name, "coordinates.json")
            self.assertIn(str((labels / "frame_001.txt").resolve()), result.snapshot)

    def test_weights_override_keeps_model_out_of_bundle(self):
        try:
            import cv2  # noqa: F401
        except Exception:
            self.skipTest("opencv not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels, model, names = _make_dataset(root)
            spec = JobSpec(
                job_type="yolo_predict",
                images_dir=str(images),
                labels_dir=str(labels),
                models_dir=str(images.parent / "models"),
                images=names,
                classes=["dog"],
                start_index=1,
                end_index=1,
                model_path=str(model),
            )
            spec.validate()
            info = build_bundle(
                spec,
                root / "job_weights",
                include_sam2_checkpoint=False,
                keep_bundle=True,
                model_override={
                    "kind": "yolo",
                    "source": "weights",
                    "weights_file": "dummy.pt",
                    "weights_dataset": "alice/labelimg-plus-weights",
                },
            )
            self.assertFalse(info.model_included)
            with zipfile.ZipFile(info.job_zip) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertFalse(any(name.startswith("model/") for name in archive.namelist()))
            self.assertEqual(manifest["model"]["source"], "weights")
            self.assertEqual(manifest["model"]["weights_file"], "dummy.pt")

    def test_sam2_bundle_without_checkpoint(self):
        try:
            import cv2  # noqa: F401
        except Exception:
            self.skipTest("opencv not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels, _model, names = _make_dataset(root)
            spec = JobSpec(
                job_type="sam2_track",
                images_dir=str(images),
                labels_dir=str(labels),
                models_dir=str(images.parent / "models"),
                images=names,
                classes=["dog"],
                start_index=2,
                end_index=3,
                direction="entire",
                boxes=[["0", "0.3", "0.4", "0.2", "0.3"]],
                model_key="small",
                sam2_config="configs/sam2.1/sam2.1_hiera_s.yaml",
                checkpoint_path="",
            )
            spec.validate()
            info = build_bundle(
                spec,
                root / "job_sam2",
                include_sam2_checkpoint=False,
                keep_bundle=True,
                progress=lambda percent, message: None,
            )
            with zipfile.ZipFile(info.job_zip) as archive:
                manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["type"], "sam2_track")
            self.assertEqual(len(manifest["images"]), 3)
            self.assertFalse(manifest["model"]["checkpoint_included"])
            self.assertTrue(manifest["model"]["checkpoint_url"].startswith("https://"))


if __name__ == "__main__":
    unittest.main()
