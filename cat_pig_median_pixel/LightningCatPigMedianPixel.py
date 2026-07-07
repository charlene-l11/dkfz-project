from htc.models.common.HTCDataset import HTCDataset
from htc.models.median_pixel.LightningMedianPixel import LightningMedianPixel

from cat_pig_median_pixel.DatasetCatPigMedianPixel import DatasetCatPigMedianPixel


class LightningCatPigMedianPixel(LightningMedianPixel):
    @staticmethod
    def dataset(**kwargs) -> HTCDataset:
        return DatasetCatPigMedianPixel(**kwargs)
