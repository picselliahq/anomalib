from skimage.transform import resize
from skimage.measure import approximate_polygon, find_contours
from typing import List, Tuple, Optional, Any

from anomalib.pre_processing.transforms import Denormalize
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning import Callback
from torchvision.transforms import ToPILImage

from picsellia import Experiment
from picsellia.types.enums import LogType

import numpy as np
from PIL import Image
import os


def show_image_and_mask(sample: dict[str, Any], index: int) -> Image:
    img = ToPILImage()(Denormalize()(sample["image"][index].clone()))
    msk = ToPILImage()(sample["mask"][index]).convert("RGB")

    return Image.fromarray(np.hstack((np.array(img), np.array(msk))))


def get_formatted_polygons_from_mask(mask: np.ndarray, img_shape: Tuple[int, int]) -> List:
    mask = resize(mask, img_shape)
    polygons = convert_mask_to_polygons(mask)
    formatted_polygons = format_polygons(polygons=polygons)
    return formatted_polygons


def convert_mask_to_polygons(mask: np.ndarray) -> List[np.ndarray]:
    polygons = []
    contours = find_contours(mask, 0)
    for contour in contours:
        approximated_contour = approximate_polygon(coords=contour, tolerance=0.2)
        shifted_contour = shift_x_and_y_coordinates(approximated_contour)
        polygons.append(shifted_contour)
    return polygons


def format_polygons(polygons: List[np.ndarray]) -> List[List[List[int]]]:
    formatted_polygons = []
    for polygon in polygons:
        formatted_polygon = list(polygon.ravel().astype(int))
        # format into a list of lists of coordinate pairs
        formatted_polygon = [[int(formatted_polygon[k]), int(formatted_polygon[k + 1])] for k in
                             range(0, len(formatted_polygon), 2)]
        formatted_polygon.append(formatted_polygon[0])

        formatted_polygons.append(formatted_polygon)

    return formatted_polygons


def shift_x_and_y_coordinates(polygon: np.ndarray) -> np.ndarray:
    shifted_contours = np.zeros_like(polygon)
    shifted_contours[:, 0] = polygon[:, 1]
    shifted_contours[:, 1] = polygon[:, 0]
    return shifted_contours


class SaveTrainingMetrics(Callback):

    def __init__(self, experiment: Experiment):
        super().__init__()
        self.experiment = experiment

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.experiment.log(name='training_pixel_F1score', type=LogType.LINE,
                            data=float(trainer.callback_metrics['pixel_F1Score']))
        self.experiment.log(name='training_pixel_AUROC', type=LogType.LINE,
                            data=float(trainer.callback_metrics['pixel_AUROC']))
        self.experiment.log(name='training_image_F1Score', type=LogType.LINE,
                            data=float(trainer.callback_metrics['image_F1Score']))
        self.experiment.log(name='training_image_AUROC', type=LogType.LINE,
                            data=float(trainer.callback_metrics['image_AUROC']))


class GetMaskPredictions(Callback):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            outputs: Optional[STEP_OUTPUT],
            batch_idx: int,
            dataloader_idx: int
    ):
        test_batch_size = batch["image"].shape[0]
        abnormal_dataset = self.experiment.get_dataset('abnormal')
        good_dataset = self.experiment.get_dataset('good')
        label = abnormal_dataset.get_or_create_label('anomaly')

        for i in range(test_batch_size):
            pred_score = batch["pred_scores"][i].cpu().numpy().item()
            gt_mask = batch["mask"][i].squeeze().int().cpu().numpy() if "mask" in batch else None
            pred_mask = batch["pred_masks"][i].squeeze().int().cpu().numpy() if "pred_masks" in batch else None

            if len(gt_mask) > 0:  # image is in abnormal dataset
                try:
                    asset = abnormal_dataset.find_asset(filename=os.path.basename(batch["image_path"][i]))
                except:
                    print('cannot find the asset')
            else:  # image with no gt_mask is in good dataset
                asset = good_dataset.find_asset(filename=os.path.basename(batch["image_path"][i]))

            try:
                gt_annotation = asset.get_annotation()
            except:
                gt_annotation = asset.create_annotation()

            gt_polygons = [(polygon, label) for polygon in
                           get_formatted_polygons_from_mask(gt_mask, (asset.width, asset.height))]

            if gt_polygons:
                gt_annotation.create_multiple_polygons(polygons=gt_polygons)  # attach annotation to asset

            pred_polygons = [(polygon, label, pred_score) for polygon in
                             get_formatted_polygons_from_mask(pred_mask, (asset.width, asset.height))]

            if pred_polygons:
                self.experiment.add_evaluation(asset=asset, polygons=pred_polygons)
