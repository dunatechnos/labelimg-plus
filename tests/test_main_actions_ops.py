from pathlib import Path
from types import SimpleNamespace

from main_actions.context import DatasetContext, area_contains, box_matches_any
from main_actions.ops import change_class, delete_boxes, propagate_boxes


def make_context(tmp_path: Path) -> DatasetContext:
    labels = tmp_path / "labels"
    images = tmp_path / "images"
    labels.mkdir()
    images.mkdir()
    (labels / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")
    (labels / "one.txt").write_text(
        "0 0.500000 0.500000 0.200000 0.200000\n"
        "1 0.700000 0.700000 0.200000 0.200000\n",
        encoding="utf-8",
    )
    (labels / "two.txt").write_text(
        "0 0.100000 0.100000 0.100000 0.100000\n",
        encoding="utf-8",
    )
    window = SimpleNamespace(label_hist=["cat", "dog"])
    return DatasetContext(
        window=window,
        images_dir=images,
        labels_dir=labels,
        models_dir=tmp_path / "models",
        image_paths=[str(images / "one.png"), str(images / "two.png")],
        image_names=["one.png", "two.png"],
        classes=["cat", "dog"],
        current_index=0,
    )


def test_delete_by_class_and_undo_snapshot(tmp_path: Path):
    context = make_context(tmp_path)
    result = delete_boxes(context, ["one.png"], classes={1})
    assert result.files_changed == 1
    assert result.boxes_changed == 1
    assert (context.labels_dir / "one.txt").read_text(encoding="utf-8") == (
        "0 0.500000 0.500000 0.200000 0.200000\n"
    )
    snapshot = result.snapshot
    assert len(snapshot) == 1
    original = list(snapshot.values())[0]
    assert len(original) == 2


def test_change_class_keeps_geometry(tmp_path: Path):
    context = make_context(tmp_path)
    result = change_class(context, ["one.png"], new_class=1, old_classes={0})
    assert result.boxes_changed == 1
    assert (context.labels_dir / "one.txt").read_text(encoding="utf-8") == (
        "1 0.500000 0.500000 0.200000 0.200000\n"
        "1 0.700000 0.700000 0.200000 0.200000\n"
    )


def test_propagate_adds_only_missing_boxes(tmp_path: Path):
    context = make_context(tmp_path)
    box = ["1", "0.700000", "0.700000", "0.200000", "0.200000"]
    result = propagate_boxes(context, ["one.png", "two.png"], [box])
    assert result.files_changed == 1
    assert result.boxes_changed == 1
    assert (context.labels_dir / "two.txt").read_text(encoding="utf-8") == (
        "0 0.100000 0.100000 0.100000 0.100000\n"
        "1 0.700000 0.700000 0.200000 0.200000\n"
    )


def test_area_and_selected_box_filters(tmp_path: Path):
    context = make_context(tmp_path)
    area = {"xmin": 0.0, "ymin": 0.0, "xmax": 0.55, "ymax": 0.55}
    assert area_contains(area, ["0", "0.2", "0.2", "0.1", "0.1"])
    # The dog box near (0.7, 0.7) is outside the area.
    assert not area_contains(area, ["0", "0.7", "0.7", "0.2", "0.2"])
    selected = [["0", "0.500000", "0.500000", "0.200000", "0.200000"]]
    assert box_matches_any(["0", "0.5", "0.5", "0.2", "0.2"], selected)
    assert not box_matches_any(["0", "0.1", "0.1", "0.2", "0.2"], selected)

    result = delete_boxes(context, ["one.png"], area=area)
    assert result.boxes_changed == 1
    assert (context.labels_dir / "one.txt").read_text(encoding="utf-8") == (
        "1 0.700000 0.700000 0.200000 0.200000\n"
    )
