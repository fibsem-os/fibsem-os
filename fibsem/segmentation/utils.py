import logging
import os
from pathlib import Path
from typing import List

import numpy as np
from huggingface_hub import hf_hub_download, scan_cache_dir

from fibsem import config as cfg


# helper functions
def decode_segmap(image, nc=5):
    """
    Decode segmentation class mask into an RGB image mask
    ref: https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
    """

    # 0=background, 1=lamella, 2= needle
    label_colors = np.zeros((8, 3))
    label_colors[1, :] = (255, 0, 0)
    label_colors[2, :] = (0, 255, 0)
    label_colors[3, :] = (0, 255, 255)
    label_colors[4, :] = (255, 255, 0)
    label_colors[5, :] = (255, 0, 255)
    label_colors[6, :] = (0, 0, 255)
    label_colors[7, :] = (128, 0, 0)  # TODO: convert this to use CLASS_COLORS

    # pre-allocate r, g, b channels as zero
    r = np.zeros_like(image, dtype=np.uint8)
    g = np.zeros_like(image, dtype=np.uint8)
    b = np.zeros_like(image, dtype=np.uint8)

    # TODO: make this more efficient
    # apply the class label colours to each pixel
    for class_idx in range(1, nc + 1):
        idx = image == class_idx
        # class_idx = class_idx % len(label_colors)
        r[idx] = label_colors[class_idx, 0]
        g[idx] = label_colors[class_idx, 1]
        b[idx] = label_colors[class_idx, 2]

    # stack rgb channels to form an image
    rgb_mask = np.stack([r, g, b], axis=-1).squeeze()
    return rgb_mask


def decode_segmap_v2(image, colormap: List[tuple] = None) -> np.ndarray:
    """
    Decode segmentation class mask into an RGB image mask
    ref: https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
    """

    if colormap is None:
        from fibsem.segmentation.config import CLASS_COLORS_RGB

        colormap = CLASS_COLORS_RGB

    # convert class masks to rgb values
    rgb_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    unique_labels = np.unique(image)

    for class_idx in unique_labels:
        rgb_mask[image == class_idx] = colormap[class_idx]

    return rgb_mask


# All UNet encoders that work with Imagenet weights
unet_encoders = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x16d",
    "resnext101_32x32d",
    "resnext101_32x48d",
    "dpn68",
    "dpn98",
    "dpn131",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "senet154",
    "se_resnet50",
    "se_resnet101",
    "se_resnet152",
    "se_resnext50_32x4d",
    "se_resnext101_32x4d",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",
    "efficientnet-b5",
    "efficientnet-b6",
    "efficientnet-b7",
    "mobilenet_v2",
    "efficientnet-b0",
    "xception",
]


## Huggingface Utils


def download_checkpoint(checkpoint: str):
    if os.path.exists(checkpoint):
        checkpoint = checkpoint
    else:
        checkpoint = hf_hub_download(repo_id=cfg.HUGGINFACE_REPO, filename=checkpoint)
    return checkpoint


def _get_cached_huggingface_checkpoints() -> List[str]:
    """Get cached huggingface model checkpoints"""
    # TODO: de-duplicate checkpoints from the cache
    checkpoints = []

    # get cached repositories
    cache_dir = os.environ.get(
        "HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")
    )
    cached_repos = scan_cache_dir(cache_dir)

    latest_revision_date = None
    latest_revision_hash = None

    # find specify repo in cache
    selected_repo = None
    for repo_info in cached_repos.repos:
        if cfg.HUGGINFACE_REPO in repo_info.repo_id:
            selected_repo = repo_info

    if selected_repo is None:
        logging.warning(f"Cannot find repoistory {cfg.HUGGINFACE_REPO} in cache.")
        return checkpoints

    # get info about each revision
    for revision in selected_repo.revisions:
        # get datetime of last modified
        revision_date = revision.last_modified
        if latest_revision_date is None:
            latest_revision_date = revision_date
            latest_revision_hash = revision.commit_hash

        if revision_date < latest_revision_date:
            logging.debug(
                f"Skipping revision {revision.commit_hash} as it is older than the latest revision {latest_revision_hash}."
            )
            continue

        snapshot_path = Path(revision.snapshot_path)
        # Find all .pt files that don't have "archive" in their name
        for file_path in snapshot_path.glob("**/*.pt"):
            if "archive" not in str(file_path):
                # Get the basename of the file (e.g., "model.pt")
                checkpoints.append(str(file_path))

    return checkpoints
