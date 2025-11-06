import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from fibsem import config as cfg
from fibsem import utils
from fibsem.detection.detection import DetectedFeatures, Feature
from fibsem.structures import FibsemImage, FibsemImageMetadata, Point


def decode_segmap(image, nc=3):

    """ Decode segmentation class mask into an RGB image mask"""

    # 0=background, 1=lamella, 2= needle
    label_colors = np.array([(0, 0, 0),
                                (255, 0, 0),
                                (0, 255, 0)])

    # pre-allocate r, g, b channels as zero
    r = np.zeros_like(image, dtype=np.uint8)
    g = np.zeros_like(image, dtype=np.uint8)
    b = np.zeros_like(image, dtype=np.uint8)

    # apply the class label colours to each pixel
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    # stack rgb channels to form an image
    rgb_mask = np.stack([r, g, b], axis=2)
    return rgb_mask


def scale_pixel_coordinates(px: Point, from_image: FibsemImage, to_image: FibsemImage) -> Point:
    """Scale the pixel coordinate from one image to another"""

    invariant_pt = get_scale_invariant_coordinates(px, from_image.data.shape)

    scaled_px = scale_coordinate_to_image(invariant_pt, to_image.data.shape)

    return scaled_px


def get_scale_invariant_coordinates(point: Point, shape: tuple) -> Point:
    """Convert the point coordinates from image coordinates to scale invariant coordinates"""
    scaled_pt = Point(x=point.x / shape[1], y=point.y / shape[0])

    return scaled_pt


def scale_coordinate_to_image(point: Point, shape: tuple) -> Point:
    """Scale invariant coordinates to image shape"""
    scaled_pt = Point(x=int(point.x * shape[1]), y=int(point.y * shape[0]))

    return scaled_pt


# TODO: add experiment, method
# TODO: migrate this to fibsem.db
# filename should match the same filename that is used for feature detection logging -> can be associated

def save_ml_feature_data(det: DetectedFeatures, initial_features: Optional[List[Feature]] = None):
    """Save the feature data to disk"""
    
    # if initial features are not provided, use the current features 
    if initial_features is None:
        initial_features = det.features
        det.mask = det.mask.astype(np.uint8)
    
    try:
        if det.fibsem_image.metadata is None or det.fibsem_image.metadata.image_settings is None:
            raise AttributeError
        fname = det.fibsem_image.metadata.image_settings.filename
        beam_type = det.fibsem_image.metadata.image_settings.beam_type
    except AttributeError:
        logging.warning("No image metadata found for ml images, saving with timestamp instead")
        fname = f"ml-{utils.current_timestamp_v2()}"
        beam_type = "NULL"

    fd = [] # feature detections
    for f0, f1 in zip(det.features, initial_features):
        px_diff = f1.px - f0.px
        msgd = {"msg": "feature_detection",
                "fname": fname,                                             # filename
                "feature": f0.name,                                         # feature name
                "px": f0.px.to_dict(),                                      # pixel coordinates
                "dpx": px_diff.to_dict(),                                   # pixel difference
                "dm": px_diff._to_metres(det.pixelsize).to_dict(),          # metre difference
                "is_correct": not np.any(px_diff.to_list()),                # is the feature correct
                "beam_type": beam_type.name,                                # beam type         
                "pixelsize": det.pixelsize,                                 # pixelsize
                "checkpoint": det.checkpoint,                               # checkpoint
        }
        logging.debug(msgd)
        fd.append(deepcopy(msgd))                                           # to write to disk

    # save features data csv
    save_feature_data_to_csv(det, features=fd, filename=fname)

def save_feature_data_to_csv(det: DetectedFeatures, features: List[dict], filename: str):
    """Save the feature data to a csv file. Includes saving the image and mask to disk. 
    All data is saved at the cfg.DATA_ML_PATH location. This can be configured in the config.py file."""

    # save image
    image = det.fibsem_image
    filename = os.path.join(cfg.DATA_ML_PATH, f"{filename}")
    image.save(filename) # type: ignore 
    logging.debug(f"Saved detection image to {filename}")

    # save mask to disk
    os.makedirs(os.path.join(cfg.DATA_ML_PATH, "mask"), exist_ok=True)
    mask_fname = os.path.join(cfg.DATA_ML_PATH, "mask", os.path.basename(filename))
    mask_fname = Path(mask_fname).with_suffix(".tif")
    im = Image.fromarray(det.mask) 
    im.save(mask_fname)
    logging.debug(f"Saved detection mask to {mask_fname}")

    # convert to csv format
    csv_data = []
    for fd in features:

        dat = deepcopy(fd)
        dat["px.x"] = fd["px"]["x"]
        dat["px.y"] = fd["px"]["y"]
        dat["dpx.x"] = fd["dpx"]["x"]
        dat["dpx.y"] = fd["dpx"]["y"]
        dat["dm.x"] = fd["dm"]["x"]
        dat["dm.y"] = fd["dm"]["y"]
        
        del dat["px"]
        del dat["dpx"]
        del dat["dm"]

        csv_data.append(dat)
    logging.debug(f"Converted {len(csv_data)} features to csv format")
    
    df = pd.DataFrame(csv_data)
    
    # save the dataframe to a csv file, append if the file already exists
    DATAFRAME_PATH = os.path.join(cfg.DATA_ML_PATH, "data.csv")
    if os.path.exists(DATAFRAME_PATH):
        df_tmp = pd.read_csv(DATAFRAME_PATH)
        df = pd.concat([df_tmp, df], axis=0, ignore_index=True)
    df.to_csv(DATAFRAME_PATH, index=False)
    logging.debug(f"Saved feature data to {DATAFRAME_PATH}")