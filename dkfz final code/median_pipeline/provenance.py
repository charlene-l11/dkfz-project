from __future__ import annotations

import hashlib
from pathlib import Path

HTC_FINGERPRINT_FILES = (
    "pyproject.toml",
    "htc/models/median_pixel/LightningMedianPixel.py",
    "htc/models/median_pixel/DatasetMedianPixel.py",
    "htc/models/median_pixel/configs/default.json",
    "htc/models/common/HTCLightning.py",
    "htc/models/common/HTCDataset.py",
    "htc/utils/Config.py",
    "htc/models/run_training.py",
)


def htc_source_fingerprint(root: Path) -> dict:
    root = root.resolve()
    files = {}
    for relative in HTC_FINGERPRINT_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required HTC source file is missing: {path}")
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    return {"root": str(root), "files": files}
