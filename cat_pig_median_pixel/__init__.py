from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HTC_PROJECT_ROOT = PROJECT_ROOT / "htc"
for import_root in (PROJECT_ROOT, HTC_PROJECT_ROOT):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from cat_pig_median_pixel.DatasetCatPigMedianPixel import DatasetCatPigMedianPixel
from cat_pig_median_pixel.LightningCatPigMedianPixel import LightningCatPigMedianPixel

__all__ = ["DatasetCatPigMedianPixel", "LightningCatPigMedianPixel"]
