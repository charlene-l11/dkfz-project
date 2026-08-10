import torch

from htc.models.common.HTCDataset import HTCDataset
from htc.models.median_pixel.LightningMedianPixel import LightningMedianPixel

from median_pipeline.dataset import DatasetCatPigMedianPixel


class LightningCatPigMedianPixel(LightningMedianPixel):
    @staticmethod
    def dataset(**kwargs) -> HTCDataset:
        return DatasetCatPigMedianPixel(**kwargs)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Collect predictions and log validation loss once per epoch."""
        if batch_idx == 0:
            assert all(len(values) == 0 for values in self.validation_results_epoch.values()), (
                "Validation results are not properly cleared"
            )

        logits = self(batch)
        validation_loss = self.ce_loss_weighted(logits, batch["labels"])
        self.log("validation/ce_loss", validation_loss, on_step=False, on_epoch=True)

        self.validation_results_epoch["labels"].append(batch["labels"])
        self.validation_results_epoch["predictions"].append(logits.argmax(dim=1))
        self.validation_results_epoch["image_names"].append(batch["image_name_annotations"])
