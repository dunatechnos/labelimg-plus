"""Description of one heavy annotation job."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runner_lib import parse_label_parts
from .sam2 import SAM2_DEFAULT_MODEL, SAM2_MODELS

JOB_TYPES = ("sam2_track", "yolo_predict")
DIRECTIONS = ("backward", "forward", "entire")


class SpecError(ValueError):
    """Raised when a job cannot be prepared safely."""


@dataclass
class JobSpec:
    """Everything the bundle builder needs from the annotation helper."""

    job_type: str
    images_dir: str
    labels_dir: str
    models_dir: str
    images: List[str]
    classes: List[str]
    start_index: int = 1
    end_index: int = 1
    direction: str = "entire"
    boxes: List[List[str]] = field(default_factory=list)
    min_confidence: float = 0.5
    model_key: str = SAM2_DEFAULT_MODEL
    checkpoint_path: str = ""
    sam2_config: str = ""
    model_path: str = ""
    selection_area: Optional[Dict[str, float]] = None

    def validate(self) -> None:
        errors: List[str] = []
        if self.job_type not in JOB_TYPES:
            errors.append(f"job_type must be one of {JOB_TYPES}, got {self.job_type!r}")
        images_dir = Path(self.images_dir)
        labels_dir = Path(self.labels_dir)
        if not images_dir.is_dir():
            errors.append(f"images_dir does not exist: {images_dir}")
        if not labels_dir.is_dir():
            errors.append(f"labels_dir does not exist: {labels_dir}")

        if not self.images:
            errors.append("images list is empty")
        else:
            if self.start_index < 1 or self.start_index > len(self.images):
                errors.append(
                    f"start_index {self.start_index} is outside 1..{len(self.images)}"
                )
            if self.end_index < 1 or self.end_index > len(self.images):
                errors.append(
                    f"end_index {self.end_index} is outside 1..{len(self.images)}"
                )
            if self.end_index < self.start_index:
                errors.append("end_index must be >= start_index")
            stems = [Path(name).stem for name in self.images]
            if len(stems) != len(set(stems)):
                errors.append(
                    "two images share the same filename stem; YOLO labels would collide"
                )
            for name in self.images:
                if not isinstance(name, str) or "/" in name or "\\" in name:
                    errors.append(f"image name must be a plain filename: {name!r}")
                    break

        if not self.classes:
            errors.append("classes.txt is empty; cannot map model classes by name")

        if self.job_type == "sam2_track":
            if self.direction not in DIRECTIONS:
                errors.append(f"direction must be one of {DIRECTIONS}")
            if not self.boxes:
                errors.append("SAM 2 needs at least one selected bounding box")
            for index, parts in enumerate(self.boxes):
                parsed = parse_label_parts(parts)
                if parsed is None:
                    errors.append(f"selected box #{index + 1} is not a valid YOLO box")
            if self.model_key not in SAM2_MODELS:
                errors.append(f"unknown SAM 2 model: {self.model_key!r}")
            if self.checkpoint_path and not Path(self.checkpoint_path).exists():
                errors.append(f"SAM 2 checkpoint not found: {self.checkpoint_path}")
            if not self.checkpoint_path and self.model_key not in SAM2_MODELS:
                errors.append("a SAM 2 checkpoint path or a known model key is required")
            if self.min_confidence < 0.0 or self.min_confidence > 1.0:
                errors.append("min_confidence must be between 0 and 1")
        else:
            if not self.model_path or not Path(self.model_path).is_file():
                errors.append(f"YOLO model not found: {self.model_path}")

        if self.selection_area is not None:
            try:
                x1 = float(self.selection_area["xmin"])
                y1 = float(self.selection_area["ymin"])
                x2 = float(self.selection_area["xmax"])
                y2 = float(self.selection_area["ymax"])
            except (KeyError, TypeError, ValueError):
                errors.append("selection_area must contain xmin, ymin, xmax, ymax")
            else:
                if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                    errors.append(
                        "selection_area must be normalized 0..1 with xmin<xmax, ymin<ymax"
                    )

        if errors:
            raise SpecError("\n".join("- " + error for error in errors))

    def included_indices(self) -> List[int]:
        """0-based indices of local frames to send to Kaggle."""
        if self.job_type == "sam2_track":
            return list(range(len(self.images)))
        return list(range(self.start_index - 1, self.end_index))

    def model_info(self) -> Dict[str, Any]:
        if self.job_type == "sam2_track":
            info = SAM2_MODELS[self.model_key]
            return {
                "kind": "sam2",
                "model_key": self.model_key,
                "file": info["file"],
                "config": self.sam2_config or info["config"],
                "url": info["url"],
                "checkpoint_path": self.checkpoint_path,
            }
        return {
            "kind": "yolo",
            "file": Path(self.model_path).name,
            "model_path": self.model_path,
        }

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["schema"] = 1
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobSpec":
        if not isinstance(data, dict):
            raise SpecError("job spec must be a JSON object")
        known = {field_name for field_name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in data.items() if key in known})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "JobSpec":
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as exc:
            raise SpecError(f"invalid job spec JSON: {exc}") from exc
