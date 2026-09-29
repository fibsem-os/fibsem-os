import os

import numpy as np
import torch
from nnunetv2.inference import predict_from_raw_data as nnunet_predict

from fibsem.segmentation import config as scfg
from fibsem.segmentation.utils import decode_segmap_v2


class SegmentationModelNNUnetV2:
    def __init__(self, checkpoint: str):
        """Initialize the NNUnet model for segmentation inference using nnunetv2.
        Args:
            checkpoint (str): Path to the model checkpoint file.
                Note: this should be the full path to the checkpoint file, not just the directory.
                The directory will be inferred from the checkpoint path, and needs to be in nnunet format...
        """

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint = checkpoint
        self._image_properties: dict = {
            "sitk_stuff": {
                "spacing": (1.0, 1.0),
                "origin": (0.0, 0.0),
                "direction": (1.0, 0.0, 0.0, 1.0),
            },
            "spacing": [999.0, 1.0, 1.0],
        }

        self.predictor = nnunet_predict.nnUNetPredictor(
            perform_everything_on_device=bool(self.device.type == "cuda"),
            device=self.device,
            allow_tqdm=False,
        )
        self.load_model()

    def load_model(self) -> None:
        """Load the model, and optionally load a checkpoint"""

        model_directory = os.path.dirname(self.checkpoint)
        checkpoint_name = os.path.basename(self.checkpoint)

        self.predictor.initialize_from_trained_model_folder(
            model_training_output_dir=model_directory,
            use_folds=(0,),
            checkpoint_name=checkpoint_name,
        )

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        """Pre-process the image for model inference"""

        # convert to 4D
        if img.ndim == 2:
            img = img[np.newaxis, np.newaxis, :, :]
        elif img.ndim == 3:
            img = img[np.newaxis, :, :, :]
        elif img.ndim == 4:
            img = img[:, :, :, :]
        else:
            raise ValueError(f"Invalid image shape: {img.shape}")

        if not isinstance(img.dtype, np.float32):
            img = img.astype(np.float32)

        return img

    def inference(self, img: np.ndarray, rgb: bool = True) -> np.ndarray:
        """Run model inference on the input image"""

        img = self.pre_process(img)  # requires (C, 1, H, W) or (1, 1, H, W)
        mask, scores = self.predictor.predict_single_npy_array(  #  type: ignore
            img, self._image_properties, None, None, True
        )
        if rgb:
            mask = decode_segmap_v2(mask[0].astype(np.uint8), scfg.get_colormap())
        else:
            mask = mask[0]
        return mask
