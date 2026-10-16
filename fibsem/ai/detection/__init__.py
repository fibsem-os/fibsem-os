"""Detection: features found in a registered model's segmentation.

    from fibsem.ai import detection, models

    det, result = detection.detect(models.get("lamella"), image, ["LamellaCentre"])
    det.features[0].px, det.features[0].feature_m

The model segments the image through the manager, checked against its card as
any segmentation is; the features (`fibsem.detection.detection`'s, by name) are
then found in its mask. The workflow's detections and the agent's run this one
way, so a model is loaded once and every detection says which model made it.
"""

from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from fibsem.ai.models import Model, SegmentationResult
    from fibsem.detection.detection import DetectedFeatures, Feature
    from fibsem.structures import Point


def feature(name: str) -> "Feature":
    """A new feature of the class named *name* (``LamellaCentre``).
    ValueError for a name that is not one."""
    from fibsem.detection.detection import __FEATURES__

    for cls in __FEATURES__:
        if cls.__name__ == name:
            return cls()
    known = ", ".join(cls.__name__ for cls in __FEATURES__)
    raise ValueError(f"no feature '{name}'; the features: {known}")


def detect(
    model: "Model",
    image: Any,
    features: Sequence[Union[str, "Feature"]],
    point: Optional["Point"] = None,
    filter: bool = True,
    pixel_size: Optional[float] = None,
    beam: Optional[str] = None,
) -> Tuple["DetectedFeatures", "SegmentationResult"]:
    """The *features* (names or `Feature`s) found in *model*'s segmentation of
    *image*, and the segmentation itself. *image* is a `FibsemImage`, whose
    own pixel size and beam are used, or an array with *pixel_size* (metres)
    and *beam*. *point* (pixels) picks among several of one feature: the
    closest wins. The detection names its model as ``name`` or
    ``name@version``."""
    from copy import deepcopy

    from fibsem.detection.detection import features_from_mask

    found = [feature(f) if isinstance(f, str) else f for f in features]
    result = model.segment(image, pixel_size=pixel_size, beam=beam)
    if result.pixel_size is None:
        raise ValueError("detection needs the image's pixel size to place features")
    # A FibsemImage's pixels; an array's own .data is its buffer, not pixels.
    fibsem_image = hasattr(image, "metadata")
    data = image.data if fibsem_image else image
    version = model.card.version
    return (
        features_from_mask(
            data,
            result.mask,
            result.rgb(),
            found,
            pixelsize=result.pixel_size,
            filter=filter,
            point=point,
            fibsem_image=deepcopy(image) if fibsem_image else None,
            checkpoint=f"{model.name}@{version}" if version else model.name,
        ),
        result,
    )
