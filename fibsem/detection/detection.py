import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

from fibsem import config as cfg
from fibsem import conversions
from fibsem.constants import DATETIME_FILE_NANO
from fibsem.imaging import tiled
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
    Point,
)

try:
    from fibsem.segmentation.utils import decode_segmap_v2
except ImportError as e:
    logging.debug(f"Could not import segmentation util / config {e}")

if TYPE_CHECKING:
    from fibsem.segmentation.model import SegmentationModel


@dataclass
class Feature(ABC):
    px: Point
    feature_m: Point
    color = "white"
    name: str = None

    @abstractmethod
    def detect(
        self, img: np.ndarray, mask: np.ndarray = None, point: Point = None
    ) -> "Feature":
        pass

    def to_dict(self):
        return {
            "name": self.name,
            "px": self.px.to_dict(),
            "feature_m": self.feature_m.to_dict(),
        }


@dataclass
class ImageCentre(Feature):
    feature_m: Point = None
    px: Point = None
    color = "white"
    name: str = "ImageCentre"

    def detect(
        self, img: np.ndarray, mask: np.ndarray = None, point: Point = None
    ) -> "ImageCentre":
        self.px = Point(x=img.shape[1] // 2, y=img.shape[0] // 2)
        return self.px


@dataclass
class NeedleTip(Feature):
    feature_m: Point = None
    px: Point = None
    color = "green"
    name: str = "NeedleTip"
    class_id: int = 2
    class_name: str = "manipulator"

    def detect(
        self, img: np.ndarray, mask: np.ndarray = None, point: Point = None
    ) -> "NeedleTip":
        self.px = detect_needle_v5(mask, idx=self.class_id, edge="right")
        return self.px


@dataclass
class NeedleTipBottom(Feature):
    feature_m: Point = None
    px: Point = None
    color = "green"
    name: str = "NeedleTipBottom"
    class_id: int = 2
    class_name: str = "manipulator"

    def detect(
        self, img: np.ndarray, mask: np.ndarray = None, point: Point = None
    ) -> "NeedleTip":
        self.px = detect_needle_v5(mask, idx=self.class_id, edge="bottom")
        return self.px


@dataclass
class LamellaCentre(Feature):
    feature_m: Point = None
    px: Point = None
    color = "red"
    name: str = "LamellaCentre"

    def detect(
        self, img: np.ndarray, mask: np.ndarray = None, point: Point = None
    ) -> "LamellaCentre":
        self.px = detect_lamella(mask, self)
        return self.px


@dataclass
class LamellaLeftEdge(Feature):
    feature_m: Point = None
    px: Point = None
    color = "red"
    name: str = "LamellaLeftEdge"

    def detect(
        self, img: np.ndarray, mask: np.ndarray = None, point: Point = None
    ) -> "LamellaLeftEdge":
        self.px = detect_lamella(mask, self)
        return self.px


@dataclass
class LamellaRightEdge(Feature):
    feature_m: Point = None
    px: Point = None
    color = "red"
    name: str = "LamellaRightEdge"

    def detect(
        self, img: np.ndarray, mask: np.ndarray = None, point: Point = None
    ) -> "LamellaRightEdge":
        self.px = detect_lamella(mask, self)
        return self.px


@dataclass
class LamellaTopEdge(Feature):
    feature_m: Point = None
    px: Point = None
    color = "red"
    name: str = "LamellaTopEdge"

    def detect(
        self, img: np.ndarray, mask: np.ndarray = None, point: Point = None
    ) -> "LamellaTopEdge":
        self.px = detect_lamella(mask, self)
        return self.px


@dataclass
class LamellaBottomEdge(Feature):
    feature_m: Point = None
    px: Point = None
    color = "red"
    name: str = "LamellaBottomEdge"

    def detect(
        self, img: np.ndarray, mask: np.ndarray = None, point: Point = None
    ) -> "LamellaBottomEdge":
        self.px = detect_lamella(mask, self)
        return self.px


# Detection and Drawing Tools
def detect_centre_point(mask: np.ndarray, threshold: int = 500) -> Point:
    """Detect the centre (mean) point of the mask for a given color (label)

    args:
        mask: the detection mask (PIL.Image)
        idx: the index of the desired class in the mask (int)
        threshold: the minimum number of required pixels for a detection to count (int)

    return:

        centre_px: the pixel coordinates of the centre point of the feature (tuple)
    """
    centre_px = Point(x=0, y=0)
    # get mask px coordinates
    idx = np.where(mask)

    # only return a centre point if detection is above a threshold
    if len(idx[0]) > threshold:
        # get the centre point of each coordinate
        y_mid = int(np.mean(idx[0]))
        x_mid = int(np.mean(idx[1]))

        # centre coordinate as tuple
        centre_px = Point(x=x_mid, y=y_mid)
    return centre_px


def detect_median_edge(mask: np.ndarray, edge: str, threshold: int = 250) -> Point:

    # get mask px coordindates
    edge_mask = np.where(mask)
    edge_px = Point(0, 0)

    if len(edge_mask[0]) > threshold:
        try:
            # get the centre point of each coordinate
            py = int(np.mean(edge_mask[0]))
            px = int(np.mean(edge_mask[1]))

            # get the min and max x and y coordinates at these median points
            x_min = np.min(edge_mask[1][edge_mask[0] == py])
            x_max = np.max(edge_mask[1][edge_mask[0] == py])

            if edge == "left":
                edge_px = Point(x=x_min, y=py)
            if edge == "right":
                edge_px = Point(x=x_max, y=py)

            y_min = np.min(edge_mask[0][edge_mask[1] == px])
            y_max = np.max(edge_mask[0][edge_mask[1] == px])
            if edge == "top":
                edge_px = Point(x=px, y=y_min)
            if edge == "bottom":
                edge_px = Point(x=px, y=y_max)
        except Exception as e:
            logging.warning(f"Error detecting edge: {e}")
    return Point(x=int(edge_px.x), y=int(edge_px.y))


def detect_absolute_edge(
    mask,
    edge: str,
    _filter: str = "largest",
    _mode: str = "median",
    threshold: int = 150,
) -> Point:

    # _mode: median or strict
    #   median: take median of points with same extremity
    #   strict: first best selection

    # get individual objects
    labels = measure.label(mask)
    props = measure.regionprops(labels)

    if len(props) == 0:
        return Point(0, 0)

    obj = props[0]
    if _filter == "largest":
        # get the largest object
        for prop in props:
            if prop.area > obj.area:
                obj = prop

    # get the coords of the most right pixel
    px = obj.coords[0]
    for coord in obj.coords:
        if edge == "right":
            if coord[1] > px[1]:
                px = coord

        if edge == "left":
            if coord[1] < px[1]:
                px = coord

        if edge == "top":
            if coord[0] < px[0]:
                px = coord

        if edge == "bottom":
            if coord[0] > px[0]:
                px = coord

    # find coordinates that have same x or y value as the most right pixel
    same_y = [coord for coord in obj.coords if coord[1] == px[1]]
    same_x = [coord for coord in obj.coords if coord[0] == px[0]]

    # TODO: could add some fuzz to the selection too (e..g +/- 2px)

    if edge in ["left", "right"]:
        # take median of points with same y value
        px = np.median(same_y, axis=0)
    else:
        # take median of points with same x value
        px = np.median(same_x, axis=0)

    return Point(px[1], px[0])


def detect_lamella(
    mask: np.ndarray,
    feature: Feature,
    mask_radius: int = 512,
    idx: int = 1,
) -> Point:

    lamella_mask = mask == idx
    # lamella_mask = masks.apply_circular_mask(lamella_mask, radius=mask_radius)

    if isinstance(feature, LamellaCentre):
        px = detect_centre_point(lamella_mask)

    if isinstance(feature, LamellaLeftEdge):
        # px = detect_corner(lamella_mask, left=True)
        px = detect_median_edge(lamella_mask, edge="left")

    if isinstance(feature, LamellaRightEdge):
        # px = detect_corner(lamella_mask, left=False)
        px = detect_median_edge(lamella_mask, edge="right")

    if isinstance(feature, LamellaTopEdge):
        px = detect_median_edge(lamella_mask, edge="top")

    if isinstance(feature, LamellaBottomEdge):
        px = detect_median_edge(lamella_mask, edge="bottom")

    return px


def detect_needle_v5(mask: np.ndarray, idx: int = 2, edge: str = "right") -> Point:
    needle_mask = mask == idx
    return detect_absolute_edge(
        needle_mask, edge=edge, _filter="largest", _mode="median", threshold=150
    )


### v2


@dataclass
class DetectedFeatures:
    features: List[Feature]
    image: np.ndarray  # TODO: convert or add FIBSEMImage
    mask: np.ndarray  # class binary mask
    rgb: np.ndarray  # rgb mask
    pixelsize: float
    _distance: Point = None
    _offset: Point = field(default_factory=Point)
    fibsem_image: FibsemImage = None
    checkpoint: str = None

    @property
    def distance(self):
        assert len(self.features) >= 2, (
            "Need at least two features to calculate distance"
        )
        return (
            self.features[0].px.distance(self.features[1].px) * self.pixelsize
            + self._offset
        )

    @distance.setter
    def distance(self, value: Point) -> None:
        self._distance = value

    def get_feature(self, ftype: Union[str, Feature]) -> Feature:
        for feature in self.features:
            if feature.name == ftype or isinstance(feature, type(ftype)):
                return feature

    def to_dict(self):
        return {
            "features": [f.to_dict() for f in self.features],
            "pixelsize": self.pixelsize,
            "offset": self._offset.to_dict(),
        }


def detect_features_v2(
    img: np.ndarray,
    mask: np.ndarray,
    features: Sequence[Feature],
    filter: bool = True,
    point: Optional[Point] = None,
) -> List[Feature]:

    detection_features = []

    for feature in features:
        if isinstance(
            feature,
            (
                LamellaCentre,
                LamellaLeftEdge,
                LamellaRightEdge,
                LamellaTopEdge,
                LamellaBottomEdge,
            ),
        ):
            feature = detect_multi_features(img, mask, feature)
            if filter:
                feature = filter_best_feature(
                    mask, feature, method="closest", point=point
                )
        else:
            feature.detect(img=img, mask=mask)

        if isinstance(feature, list):
            detection_features.extend(feature)
        else:
            detection_features.append(feature)

    return detection_features


def detect_features(
    image: Union[np.ndarray, FibsemImage],
    model: "SegmentationModel",
    features: Sequence[Feature],
    pixelsize: Optional[float] = None,
    filter: bool = True,
    point: Optional[Point] = None,
) -> DetectedFeatures:

    if isinstance(image, FibsemImage):
        fibsem_image = deepcopy(image)
        image = image.data
    else:
        fibsem_image = None

    if pixelsize is None:
        try:
            pixelsize = image.metadata.pixel_size.x
        except Exception as e:  # default (wrong value)
            logging.debug(f"Error getting pixelsize: {e}, using default value of 25nm")
            pixelsize = 25e-9

    # model inference
    mask = model.inference(image, rgb=False)
    if mask.ndim == 3:
        rgb = model.postprocess(mask, model.num_classes)
        mask = mask[0]  # remove channel dim
    else:
        rgb = decode_segmap_v2(mask)

    # detect features
    features = detect_features_v2(
        img=image, mask=mask, features=features, filter=filter, point=point
    )

    det = DetectedFeatures(
        features=features,  # type: ignore
        image=image,
        mask=mask,
        rgb=rgb,
        pixelsize=pixelsize,
        fibsem_image=fibsem_image,
        checkpoint=model.checkpoint,
    )

    # distance in metres (from centre)
    for feature in det.features:
        feature.feature_m = conversions.image_to_microscope_image_coordinates(
            feature.px, det.image.data, det.pixelsize
        )

    return det


def take_image_and_detect_features(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    features: Sequence[Feature],
    point: Optional[Union[Point, FibsemStagePosition]] = None,
    checkpoint: str = cfg.DEFAULT_CHECKPOINT,
) -> DetectedFeatures:

    from fibsem import acquire, utils
    from fibsem.segmentation.model import load_model

    if image_settings.reduced_area is not None:
        logging.info(
            "Reduced area is not compatible with model detection, disabling..."
        )
        image_settings.reduced_area = None

    image_settings.filename = f"ml-{utils.current_timestamp(DATETIME_FILE_NANO)}"
    image_settings.save = True

    # take new image
    image = acquire.acquire_image(microscope, image_settings)

    # load model
    model = load_model(checkpoint=checkpoint)

    if isinstance(point, FibsemStagePosition):
        logging.debug(f"Reprojecting point {point} to image coordinates...")
        points = tiled.reproject_stage_positions_onto_image(
            image=image, positions=[point], bound=True
        )
        point = points[0] if len(points) == 1 else None
        logging.debug(f"Reprojected point: {point}")

    # detect features
    det = detect_features(
        deepcopy(image),
        model,
        features=features,
        pixelsize=image.metadata.pixel_size.x,
        point=point,
    )
    return det


def plot_detection(det: DetectedFeatures):
    """Plotting image with detected features

    Args:
        det (DetectedFeatures): detected features type
        inverse (bool, optional): Inverses the colour of the centre crosshair of the feature. Defaults to True.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    fig = plot_det(det, ax)

    return fig


def plot_det(
    det: DetectedFeatures, ax: plt.Axes, title: str = "Prediction", show: bool = True
):
    ax.imshow(det.image, cmap="gray")
    if det.rgb is not None:
        ax.imshow(det.rgb, alpha=0.3)
    ax.set_title(title)

    # get unique feature names
    names = []
    for f in det.features:
        if f.name not in names:
            names.append(f.name)

    for f in det.features:
        ax.plot(
            f.px.x,
            f.px.y,
            "o",
            color=f.color,
            markersize=5,
            markeredgecolor="w",
            label=f.name if names.count(f.name) == 1 else None,
        )

        # remove from names list
        if names.count(f.name) == 1:
            names.remove(f.name)

    # if len(det.features) < 5:
    if len(det.features) != 0:
        ax.legend(loc="best")
    ax.axis("off")

    # if len(det.features) == 2:
    #     # plot white line between features
    #     ax.plot([det.features[0].px.x, det.features[1].px.x],
    #             [det.features[0].px.y, det.features[1].px.y],
    #             color="w", linestyle="--")

    if show:
        plt.show()


_DETECTIONS_THAT_MOVE_MANIPULATOR = (
    NeedleTip,
    NeedleTipBottom,
    LamellaRightEdge,
    LamellaBottomEdge,
    LamellaTopEdge,
)
_DETECTIONS_THAT_MOVE_STAGE = (LamellaCentre,)


def move_based_on_detection(
    microscope: FibsemMicroscope,
    det: DetectedFeatures,
    beam_type: BeamType,
    move_x: bool = True,
    move_y: bool = True,
    _move_system: str = None,  # auto
):

    dx, dy = det.distance.x, det.distance.y

    # nulify movements in unsupported axes
    if not move_x:
        dx = 0
    if not move_y:
        dy = 0

    # f1 = move from, f2 = move to
    f1 = det.features[0]
    f2 = det.features[1]

    logging.debug(f"move_x: {move_x}, move_y: {move_y}")
    logging.debug(f"movement: x={dx:.2e}, y={dy:.2e}")
    logging.debug(f"features: {f1}, {f2}, beam_type: {beam_type}")

    if _move_system is None:
        if isinstance(f1, _DETECTIONS_THAT_MOVE_MANIPULATOR):
            _move_system = "manipulator"
        if isinstance(f1, _DETECTIONS_THAT_MOVE_STAGE):
            _move_system = "stage"

    if _move_system not in ["manipulator", "stage"]:
        raise ValueError(
            f"move_system must be one of ['manipulator', 'stage'], not {_move_system}"
        )

    # these movements move the needle...
    if _move_system == "manipulator":
        # account for scan_rotation
        if np.isclose(microscope.get_scan_rotation(beam_type), np.pi):
            dx *= -1.0
            dy *= -1.0

        # NOTE (pat): double check this on liftout?
        if beam_type == BeamType.ELECTRON:
            dy *= -1.0

        microscope.move_manipulator_corrected(
            dx=dx,
            dy=dy,
            beam_type=beam_type,
        )

    if _move_system == "stage":
        # need to reverse the direction to move correctly. investigate if this is to do with scan rotation?
        microscope.stable_move(
            dx=-dx,
            dy=dy,
            beam_type=beam_type,
        )

        # TODO: support other movements?
    return


def mask_contours(image):
    # Find contours
    contours = measure.find_contours(image, 0.5)

    # Create a mask with the same shape as the input image
    mask = np.zeros_like(image, dtype=np.uint8)

    # Mask the area inside each contour
    for i, contour in enumerate(contours):
        # Convert the contour coordinates to integers
        contour = np.round(contour).astype(int)

        # Create a polygonal mask for this contour
        rr, cc = np.meshgrid(
            np.arange(mask.shape[0]), np.arange(mask.shape[1]), indexing="ij"
        )
        inside_contour = np.zeros_like(mask, dtype=np.uint8)
        inside_contour[rr, cc] = 0
        inside_contour[contour[:, 0], contour[:, 1]] = i + 1

        # mask the area inside the contour
        ymin, ymax = np.min(contour[:, 0]), np.max(contour[:, 0])
        xmin, xmax = np.min(contour[:, 1]), np.max(contour[:, 1])

        inside_contour[ymin:ymax, xmin:xmax] = i + 1

        # Apply the mask for this contour
        mask = mask + inside_contour

    return mask


# TODO: need passthrough for the params
def detect_multi_features(
    image: np.ndarray, mask: np.ndarray, feature: Feature, class_idx: int = 1
):

    mask = mask == class_idx  # filter to class
    mask = mask_contours(mask)
    idxs = np.unique(mask)

    features = []
    for idx in idxs:
        if idx == 0:
            continue

        # create a new image
        feature_mask = np.zeros_like(mask)
        feature_mask[mask == idx] = class_idx

        # detect features
        feature.detect(image, feature_mask)
        features.append(deepcopy(feature))

    if features == []:
        logging.info(f"No features detected for {feature.name}")
        # set at centre of image
        feature.px = Point(x=image.shape[1] // 2, y=image.shape[0] // 2)
        features = [deepcopy(feature)]

    return features


def filter_best_feature(
    mask: np.ndarray,
    features: List[Feature],
    method: str = "closest",
    point: Optional[Point] = None,
):
    if method == "closest":
        # plot feature closest to point
        if point is None:
            point = Point(mask.shape[1] / 2, mask.shape[0] / 2)

        distances = []
        for feature in features:
            distances.append(np.linalg.norm(point - feature.px))
        idx = np.argmin(distances)

        return features[idx]

    else:
        raise ValueError(f"method {method} not recognised")
