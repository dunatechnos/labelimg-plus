from __future__ import annotations

import unittest

from kaggle_bridge.metadata import (
    ACCELERATOR_T4_X2,
    dataset_id_for,
    dataset_metadata,
    job_slug,
    job_title,
    kernel_id_for,
    kernel_metadata,
    kernel_title,
)


class MetadataTests(unittest.TestCase):
    def test_job_slug_and_title(self):
        slug = job_slug("labelimg-plus-job", "20261113-121500-ab12cd")
        self.assertEqual(slug, "labelimg-plus-job-20261113-121500-ab12cd")
        title = job_title("20261113-121500-ab12cd")
        self.assertTrue(6 <= len(title) <= 50)

    def test_dataset_id(self):
        self.assertEqual(
            dataset_id_for("alice", "labelimg-plus-job", "20261113-121500-ab12cd"),
            "alice/labelimg-plus-job-20261113-121500-ab12cd",
        )

    def test_kernel_id_is_different_from_dataset_id(self):
        job_id = "20261113-121500-ab12cd"
        dataset_id = dataset_id_for("alice", "labelimg-plus-job", job_id)
        kernel_id = kernel_id_for("alice", job_id)
        self.assertNotEqual(dataset_id, kernel_id)
        self.assertEqual(kernel_id, "alice/labelimg-plus-run-20261113-121500-ab12cd")
        self.assertEqual(
            kernel_title(job_id).lower().replace(" ", "-"),
            kernel_id.split("/")[1],
        )

    def test_dataset_metadata(self):
        meta = dataset_metadata(
            "alice/labelimg-plus-job-20261113-121500-ab12cd",
            "LabelImg Plus Job 20261113-121500-ab12cd",
        )
        self.assertEqual(meta["licenses"], [{"name": "CC0-1.0"}])
        self.assertIn("/", meta["id"])

    def test_kernel_t4_x2(self):
        meta = kernel_metadata(
            "alice/labelimg-plus-job-20261113-121500-ab12cd",
            "LabelImg Plus Job 20261113-121500-ab12cd",
            "alice/labelimg-plus-job-20261113-121500-ab12cd",
        )
        self.assertEqual(meta["machine_shape"], ACCELERATOR_T4_X2)
        self.assertTrue(meta["enable_gpu"])
        self.assertTrue(meta["enable_internet"])
        self.assertEqual(meta["kernel_type"], "script")
        self.assertEqual(meta["dataset_sources"], [meta["id"]])

        with_weights = kernel_metadata(
            "alice/labelimg-plus-run-job",
            "LabelImg Plus Run job",
            "alice/labelimg-plus-job",
            extra_dataset_sources=["alice/labelimg-plus-weights"],
        )
        self.assertEqual(
            with_weights["dataset_sources"],
            ["alice/labelimg-plus-job", "alice/labelimg-plus-weights"],
        )


if __name__ == "__main__":
    unittest.main()
