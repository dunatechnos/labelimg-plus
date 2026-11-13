from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kaggle_bridge.runner_lib import (
    apply_label_updates,
    merge_tracked_labels,
    normalize_label_lines,
    parse_label_parts,
    remap_class_by_name,
)


class RunnerLibTests(unittest.TestCase):
    def test_parse_and_normalize(self):
        self.assertEqual(
            parse_label_parts("0 0.5 0.5 0.2 0.2"),
            ["0", "0.500000", "0.500000", "0.200000", "0.200000"],
        )
        self.assertEqual(
            parse_label_parts(["1", "0.4", "0.4", "0.1", "0.1"]),
            ["1", "0.400000", "0.400000", "0.100000", "0.100000"],
        )
        self.assertIsNone(parse_label_parts("bad line"))
        self.assertIsNone(parse_label_parts("0 0.5 0.5 0 0.2"))
        self.assertIsNone(parse_label_parts("0 2.0 0.5 0.2 0.2"))

    def test_normalize_drops_bad_lines(self):
        result = normalize_label_lines(
            ["0 0.5 0.5 0.2 0.2", "", "nonsense", "1 0.1 0.1 0.1 0.1"]
        )
        self.assertEqual(len(result), 2)

    def test_merge_policy(self):
        original = [
            "0 0.5 0.5 0.2 0.2",
            "1 0.2 0.2 0.1 0.1",
            "0 0.9 0.9 0.05 0.05",
        ]
        tracked = ["0 0.5 0.5 0.25 0.25"]
        merged, changed = merge_tracked_labels(original, tracked)
        self.assertTrue(changed)
        self.assertEqual(len(merged), 3)
        self.assertIn(["0", "0.500000", "0.500000", "0.250000", "0.250000"], merged)
        self.assertIn(["1", "0.200000", "0.200000", "0.100000", "0.100000"], merged)
        self.assertIn(["0", "0.900000", "0.900000", "0.050000", "0.050000"], merged)

    def test_merge_no_change(self):
        original = ["0 0.5 0.5 0.2 0.2"]
        merged, changed = merge_tracked_labels(original, original)
        self.assertFalse(changed)

    def test_remap_by_name(self):
        self.assertEqual(
            remap_class_by_name({0: "Dog", 1: "person"}, 0, {"dog": 2, "person": 1}),
            2,
        )
        self.assertIsNone(remap_class_by_name({0: "unknown"}, 0, {"dog": 0}))

    def test_apply_updates_and_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp)
            (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            result = apply_label_updates(
                labels,
                {
                    "a.jpg": [["1", "0.4", "0.4", "0.1", "0.1"]],
                    "b.jpg": [],
                },
                {"a.jpg", "b.jpg"},
                num_classes=2,
            )
            self.assertEqual(result.changed, 1)
            self.assertIn(str((labels / "a.txt").resolve()), result.snapshot)
            self.assertEqual(
                (labels / "a.txt").read_text(encoding="utf-8").strip(),
                "1 0.400000 0.400000 0.100000 0.100000",
            )
            # An absent label file with an empty remote result is a no-op,
            # not a newly created empty file.
            self.assertFalse((labels / "b.txt").exists())

    def test_apply_rejects_traversal_and_bad_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp)
            with self.assertRaises(ValueError):
                apply_label_updates(labels, {"../evil.jpg": []}, {"evil.jpg"})
            with self.assertRaises(ValueError):
                apply_label_updates(
                    labels,
                    {"a.jpg": [["5", "0.5", "0.5", "0.2", "0.2"]]},
                    {"a.jpg"},
                    num_classes=2,
                )


if __name__ == "__main__":
    unittest.main()
