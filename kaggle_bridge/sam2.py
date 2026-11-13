"""SAM 2 metadata shared by the local bundle builder and the UI."""

from __future__ import annotations

SAM2_MODELS = {
    "small": {
        "label": "Small (accurate, ~176 MB)",
        "file": "sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "size": "~176 MB",
    },
    "tiny": {
        "label": "Tiny (fast, ~149 MB)",
        "file": "sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "size": "~149 MB",
    },
}

SAM2_DEFAULT_MODEL = "small"
