"""Stage geometry of a saved image, beam or fluorescence, from its own metadata:
the same projections the overview canvases use, read from the file."""

from typing import Any, Callable, Dict, Tuple


def image_point_to_stage_position(path: str, px=None, fraction=None):
    """The stage position under a point of the saved image at *path*, as the
    stored overview canvas resolves a click
    (`fibsem.geometry.projection.image_point_to_stage`). ``px`` is [x, y] in
    source pixels; ``fraction`` is [x, y] from 0 to 1. Raises ValueError."""
    from fibsem.fm.preview import is_fluorescence_image
    from fibsem.geometry.projection import image_point_to_stage

    if (px is None) == (fraction is None):
        raise ValueError("Give exactly one of px ([x, y] source pixels) or fraction.")
    point = px if px is not None else fraction
    if not (isinstance(point, (list, tuple)) and len(point) == 2):
        raise ValueError("The point is [x, y].")
    x, y = float(point[0]), float(point[1])

    if is_fluorescence_image(path):
        from fibsem.fm.structures import FluorescenceImage

        image = FluorescenceImage.load(path)
    else:
        from fibsem.structures import FibsemImage

        image = FibsemImage.load(path)
    height, width = image.data.shape[-2:]
    if fraction is not None:
        x, y = x * width, y * height
    if not (0 <= x <= width and 0 <= y <= height):
        raise ValueError(f"The point is outside the {width} x {height} image.")
    try:
        return image_point_to_stage(image, x, y)
    except ValueError:
        raise ValueError(
            "The overview has no stage geometry to convert from."
        ) from None


def stage_projector(path: str) -> Tuple[Dict[str, Any], Callable]:
    """Open the saved image at *path* and return its scale block plus a projector.

    The projector maps one stage position to a source-pixel Point through the
    canvas's own reprojection for that kind of image: beam overviews
    (FibsemImage) through ``reproject_stage_positions_onto_image2``,
    fluorescence overviews (OME-TIFF) through
    ``reproject_stage_positions_onto_fm_image``. Both raise, per position, when
    the image lacks the metadata a projection needs; the caller records the
    reason against the item rather than guessing. The scale block is what a
    client needs to place the points on its rendition: source size, pixel size
    (m), field width (m), the acquisition's stage position.
    """
    from fibsem.fm.preview import is_fluorescence_image

    if is_fluorescence_image(path):
        from fibsem.fm.reprojection import reproject_stage_positions_onto_fm_image
        from fibsem.fm.structures import FluorescenceImage

        image = FluorescenceImage.load(path)
        height, width = image.data.shape[-2:]
        metadata = image.metadata
        stage = getattr(metadata, "stage_position", None)
        block = {
            "width": int(width),
            "height": int(height),
            "pixel_size": metadata.pixel_size_x if metadata else None,
            "hfw": metadata.pixel_size_x * width if metadata else None,
            "beam_type": "FM",
            "stage_position": stage.to_dict() if stage is not None else None,
        }

        def project(position):
            (point,) = reproject_stage_positions_onto_fm_image(image, [position])
            return point

        return block, project

    from fibsem.imaging.tiling.reprojection import (
        reproject_stage_positions_onto_image2,
    )
    from fibsem.structures import FibsemImage

    image = FibsemImage.load(path)
    height, width = image.data.shape[:2]
    metadata = image.metadata
    state = getattr(metadata, "microscope_state", None) if metadata else None
    stage = state.stage_position if state is not None else None
    block = {
        "width": int(width),
        "height": int(height),
        "pixel_size": metadata.pixel_size.x if metadata else None,
        "hfw": metadata.image_settings.hfw if metadata else None,
        "beam_type": metadata.image_settings.beam_type.name if metadata else None,
        "stage_position": stage.to_dict() if stage is not None else None,
    }

    def project(position):
        (point,) = reproject_stage_positions_onto_image2(image, [position])
        return point

    return block, project
