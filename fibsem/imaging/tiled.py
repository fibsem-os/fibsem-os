from __future__ import annotations

import datetime
import logging
import os
import threading
from copy import deepcopy
from typing import List, Optional, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from fibsem import acquire, conversions
from fibsem.constants import DATETIME_FILE
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    AutoFocusMode,
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    OverviewAcquisitionSettings,
    Point,
)
if TYPE_CHECKING:
    import psygnal


POSITION_COLOURS = ["lime", "blue", "cyan", "magenta", "hotpink", "yellow", "orange", "red"]

##### TILED ACQUISITION
def tiled_image_acquisition(
    microscope: FibsemMicroscope,
    settings: OverviewAcquisitionSettings,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    """Tiled image acquisition.
    Args:
        microscope: The microscope connection.
        settings: Overview acquisition settings.
        stop_event: Optional threading.Event to cancel acquisition.
    Returns:
        A dictionary containing the acquisition details for stitching."""

    image_settings = settings.image_settings
    n_rows, n_cols = settings.nrows, settings.ncols
    cryo = image_settings.autogamma  # capture before clearing below
    use_focus_stack = settings.use_focus_stack
    overlap = settings.overlap  # fractional overlap, e.g. 0.1 = 10%
    af_mode = settings.autofocus_settings.mode

    # derive tile FOVs — non-square tiles have different x/y physical extents
    image_width, image_height = image_settings.resolution
    tile_fov_x = image_settings.hfw
    tile_fov_y = tile_fov_x * (image_height / image_width)

    # step between tile centres (reduced by overlap)
    dx = tile_fov_x * (1 - overlap)
    dy = -tile_fov_y * (1 - overlap)  # invert y-axis

    # fixed image settings
    image_settings.autogamma = False
    total_fov = (n_cols - 1) * dx + tile_fov_x  # total physical width covered

    # start in the middle of the grid
    start_state = microscope.get_microscope_state()

    # we save all intermediates into a folder with the same name as the filename, then save the stitched image into the parent folder
    prev_path = image_settings.path
    prev_label = image_settings.filename
    image_settings.path = os.path.join(prev_path, prev_label) # type: ignore
    os.makedirs(image_settings.path, exist_ok=True) # type: ignore

    # TOP LEFT CORNER START — offset from centre to first tile
    image_settings.filename = prev_label
    image_settings.autocontrast = False # required for cryo
    image_settings.save = True
    start_move_x = (n_cols - 1) * dx / 2
    start_move_y = (n_rows - 1) * abs(dy) / 2
    dxg, dyg = start_move_x, start_move_y
    dyg *= -1

    microscope.stable_move(dx=-dxg, dy=-dyg, beam_type=image_settings.beam_type)
    start_position = microscope.get_stage_position()
    images = []

    # stitched image canvas — accounts for overlap and non-square tiles
    eff_w = max(1, int(round(image_width * (1 - overlap))))
    eff_h = max(1, int(round(image_height * (1 - overlap))))
    full_w = eff_w * (n_cols - 1) + image_width
    full_h = eff_h * (n_rows - 1) + image_height
    arr = np.zeros(shape=(full_h, full_w), dtype=np.uint8)
    n_tiles_acquired = 0
    total_tiles = n_rows*n_cols
    try:
        if af_mode is AutoFocusMode.ONCE:
            microscope.auto_focus(beam_type=image_settings.beam_type, reduced_area=image_settings.reduced_area)

        for i in range(n_rows):

            microscope.safe_absolute_stage_movement(start_position)

            img_row: list[FibsemImage] = []
            microscope.stable_move(dx=0, dy=i*dy, beam_type=image_settings.beam_type)

            if af_mode is AutoFocusMode.EVERY_ROW:
                microscope.auto_focus(beam_type=image_settings.beam_type, reduced_area=image_settings.reduced_area)

            for j in range(n_cols):
                image_settings.filename = f"tile_{i}_{j}"
                microscope.stable_move(dx=dx*(j!=0),  dy=0, beam_type=image_settings.beam_type) # dont move on the first tile?

                if af_mode is AutoFocusMode.EVERY_TILE:
                    microscope.auto_focus(beam_type=image_settings.beam_type, reduced_area=image_settings.reduced_area)

                if stop_event and stop_event.is_set():
                    raise Exception("User Stopped Acquisition")

                logging.info(f"Acquiring Tile {i}, {j}")
                if use_focus_stack:
                    image = acquire.acquire_focus_stacked_image(
                        microscope=microscope,
                        image_settings=image_settings,
                        n_steps=3,
                    )
                else:
                    image = acquire.acquire_image(microscope, image_settings)

                # stitch tile into canvas (overlapping regions are overwritten by later tiles)
                arr[i*eff_h:i*eff_h+image_height, j*eff_w:j*eff_w+image_width] = image.filtered_data

                n_tiles_acquired += 1
                microscope.tiled_acquisition_signal.emit(
                    {
                        "msg": "Tile Collected",
                        "i": i,
                        "j": j,
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                        "image": arr,
                        "counter": n_tiles_acquired,
                        "total": total_tiles,
                    }
                )

                img_row.append(image)
            images.append(img_row)
    except Exception as e:
        logging.error(f"Tiled acquisition failed: {e}")
        raise
    finally:
        logging.info(f"Tiled acquisition complete, restoring initial position: {start_state.stage_position.pretty}")
        microscope.set_microscope_state(start_state)
    image_settings.path = prev_path

    ddict = {
        "total_fov": total_fov,
        "tile_size": tile_fov_x,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "image_settings": image_settings,
        "dx": dx,
        "dy": dy,
        "cryo": cryo,
        "start_state": start_state,
        "prev-filename": prev_label,
        "start_move_x": start_move_x,
        "start_move_y": start_move_y,
        "dxg": dxg,
        "dyg": dyg,
        "images": images,
        "stitched_image": arr,
    }

    return ddict

def stitch_images(images: list[list[FibsemImage]], 
                  ddict: dict, signal: Optional['psygnal.SignalInstance'] = None) -> FibsemImage:
    """Stitch an array (2D) of images together. Assumes images are ordered in a grid with no overlap.
    Args:
        images: The images.
        ddict: The dictionary containing the acquisition details for stitching.
        signal: Optional signal for emitting progress updates.
    Returns:
        The stitched image."""
    if signal is not None:
        total = ddict["n_rows"] * ddict["n_cols"]
        signal.emit({"msg": "Stitching Tiles", "counter": total, "total": total})
    arr = ddict["stitched_image"]

    # convert to fibsem image
    image = FibsemImage(data=arr, metadata=images[0][0].metadata)
    if image.metadata is None:
        raise ValueError("Image metadata is not set. Cannot update metadata for stitched image.")
    image.metadata.microscope_state = deepcopy(ddict["start_state"])
    image.metadata.image_settings = ddict["image_settings"]
    image.metadata.image_settings.hfw = deepcopy(float(ddict["total_fov"]))
    image.metadata.image_settings.resolution = deepcopy((arr.shape[0], arr.shape[1]))

    filename = os.path.join(image.metadata.image_settings.path, f'{ddict["prev-filename"]}') # type: ignore
    image.save(filename)
    # for cryo need to histogram equalise
    if ddict.get("cryo", False):
        from fibsem.imaging.autogamma import auto_gamma
        image = auto_gamma(image, method="autogamma")
        filename = os.path.join(image.metadata.image_settings.path, f'{ddict["prev-filename"]}-autogamma') # type: ignore
        image.save(filename)

    # for garbage collection
    del ddict["images"]
    import time
    time.sleep(5)

    if signal is not None:
        signal.emit({"msg": "Done", "counter": total, "total": total, "finished": True})

    return image

def tiled_image_acquisition_and_stitch(
    microscope: FibsemMicroscope,
    settings: OverviewAcquisitionSettings,
    stop_event: Optional[threading.Event] = None,
) -> FibsemImage:
    """Acquire a tiled image and stitch it together.
    Args:
        microscope: The microscope connection.
        settings: Overview acquisition settings (image_settings, nrows, ncols, overlap).
        stop_event: Optional threading.Event to cancel acquisition.
    Returns:
        The stitched image."""

    # add datetime to filename for uniqueness
    filename = settings.image_settings.filename
    timestamp = datetime.datetime.now().strftime(DATETIME_FILE)
    settings.image_settings.filename = f"{filename}-{timestamp}"

    ddict = tiled_image_acquisition(microscope=microscope, settings=settings, stop_event=stop_event)
    image = stitch_images(images=ddict["images"], ddict=ddict, signal=microscope.tiled_acquisition_signal)

    return image

##### REPROJECTION
# TODO: move these to fibsem.imaging.reprojection?
def calculate_reprojected_stage_position(image: FibsemImage, pos: FibsemStagePosition) -> Point:
    """Calculate the reprojected stage position on an image.
    Args:
        image: The image.
        pos: The stage position.
    Returns:
        The reprojected stage position on the image."""

    # difference between current position and image position
    delta = pos - image.metadata.stage_position

    # projection of the positions onto the image
    dx = delta.x
    dy = np.sqrt(delta.y**2 + delta.z**2) # TODO: correct for perspective here
    dy = dy if (delta.y<0) else -dy

    pt_delta = Point(dx, dy)
    px_delta = pt_delta._to_pixels(image.metadata.pixel_size.x)

    beam_type = image.metadata.image_settings.beam_type
    if beam_type is BeamType.ELECTRON:
        scan_rotation = image.metadata.microscope_state.electron_beam.scan_rotation
    if beam_type is BeamType.ION:
        scan_rotation = image.metadata.microscope_state.ion_beam.scan_rotation
    
    if np.isclose(scan_rotation, np.pi):
        px_delta.x *= -1.0
        px_delta.y *= -1.0

    # account for compustage tilt, when mounted upside down
    if np.isclose(image.metadata.stage_position.t, np.radians(-180), atol=np.radians(5)):
        px_delta.y *= -1.0

    image_centre = Point(x=image.data.shape[1]/2, y=image.data.shape[0]/2)
    point = image_centre + px_delta

    # NB: there is a small reprojection error that grows with distance from centre
    # print(f"ERROR: dy: {dy}, delta_y: {delta.y}, delta_z: {delta.z}")

    return point

def reproject_stage_positions_onto_image(
        image:FibsemImage, 
        positions: List[FibsemStagePosition], 
        bound: bool=False) -> List[Point]:
    """Reproject stage positions onto an image. Assumes image is flat to beam.
    Args:
        image: The image.
        positions: The positions.
        bound: Whether to only return points inside the image.
    Returns:
        The reprojected stage positions on the image plane."""
    from fibsem.ui.napari.utilities import is_inside_image_bounds

    # reprojection of positions onto image coordinates
    points = []
    for pos in positions:


        # hotfix (pat): demo returns None positions #240
        if image.metadata.microscope_state.stage_position.x is None:
            image.metadata.microscope_state.stage_position.x = 0
        if image.metadata.microscope_state.stage_position.y is None:
            image.metadata.microscope_state.stage_position.y = 0
        if image.metadata.microscope_state.stage_position.z is None:
            image.metadata.microscope_state.stage_position.z = 0
        if image.metadata.microscope_state.stage_position.r is None:
            image.metadata.microscope_state.stage_position.r = 0
        if image.metadata.microscope_state.stage_position.t is None:
            image.metadata.microscope_state.stage_position.t = 0      
                
        # automate logic for transforming positions
        # assume only two valid positions are when stage is flat to either beam...  
        # r needs to be 180 degrees different
        # currently only one way: Flat to Ion -> Flat to Electron
        dr = abs(np.rad2deg(image.metadata.microscope_state.stage_position.r - pos.r))
        if np.isclose(dr, 180, atol=2):     
            pos = _transform_position(pos)

        pt = calculate_reprojected_stage_position(image, pos)
        pt.name = pos.name
        
        if bound and not is_inside_image_bounds([pt.y, pt.x], image.data.shape):
            continue
        
        points.append(pt)
    
    return points

def calculate_reprojected_stage_position2(image: FibsemImage, pos: FibsemStagePosition) -> Point:
    """Calculate the reprojected stage position on an image.
    Args:
        image: The image.
        pos: The stage position.
    Returns:
        The reprojected stage position on the image."""

    if image.metadata is None or image.metadata.microscope_state is None:
        raise ValueError("Image metadata or microscope state is not set. Cannot reproject stage position.")

    if image.metadata.microscope_state.stage_position is None:
        raise ValueError("Image metadata does not contain a valid stage position. Cannot reproject stage position.")


    beam_type = image.metadata.image_settings.beam_type
    base_stage_position = image.metadata.microscope_state.stage_position 
    pixel_size = image.metadata.pixel_size.x

    scan_rotation = None
    if beam_type is BeamType.ELECTRON:
        if image.metadata.microscope_state.electron_beam is None:
            raise ValueError("Image metadata does not contain a valid electron beam state. Cannot reproject stage position.")
        scan_rotation = image.metadata.microscope_state.electron_beam.scan_rotation
    if beam_type is BeamType.ION:
        if image.metadata.microscope_state.ion_beam is None:
            raise ValueError("Image metadata does not contain a valid ion beam state. Cannot reproject stage position.")
        scan_rotation = image.metadata.microscope_state.ion_beam.scan_rotation

    if scan_rotation is None:
        raise ValueError("Image metadata does not contain a valid scan rotation. Cannot reproject stage position.")

    # difference between current position and image position
    delta = pos - base_stage_position

    # projection of the positions onto the image
    dx = delta.x
    if dx is None:
        raise ValueError("Stage position x coordinate is None. Cannot reproject stage position.")
    # dy = microscope._inverse_y_corrected_stage_movement(dy=delta.y, dz=delta.z, beam_type=beam_type) # type: ignore
    dy = _inverse_y_corrected_stage_movement(image, dy=delta.y, dz=delta.z, beam_type=beam_type) # type: ignore

    pt_delta = Point(dx, -dy)
    px_delta = pt_delta._to_pixels(pixel_size)

    if np.isclose(scan_rotation, np.pi):
        px_delta.x *= -1.0
        px_delta.y *= -1.0

    image_centre = Point(x=image.data.shape[1]/2, y=image.data.shape[0]/2)
    point = image_centre + px_delta

    return point

def reproject_stage_positions_onto_image2(
        image:FibsemImage, 
        positions: List[FibsemStagePosition], 
        bound: bool=False) -> List[Point]:
    """Reproject stage positions onto an image. Assumes image is flat to beam.
    Args:
        image: The image.
        positions: The positions.
        bound: Whether to only return points inside the image.
    Returns:
        The reprojected stage positions on the image plane."""
    from fibsem.ui.napari.utilities import is_inside_image_bounds

    # reprojection of positions onto image coordinates
    points = []
    for pos in positions:

        # compucentric rotation correction
        if image.metadata is None or image.metadata.microscope_state is None:
            raise ValueError("Image metadata or microscope state is not set. Cannot reproject stage position.")
        if image.metadata.microscope_state.stage_position is None:
            raise ValueError("Image metadata does not contain a valid stage position. Cannot reproject stage position.")
        if image.metadata.microscope_state.stage_position is None:
            raise ValueError("Image metadata does not contain a valid stage position. Cannot reproject stage position.")
        if image.metadata.microscope_state.stage_position.r is None:
            raise ValueError("Image metadata does not contain a valid stage position r coordinate. Cannot reproject stage position.")
        if pos.r is None:
            raise ValueError("Stage position r coordinate is None. Cannot reproject stage position.")
        # automate logic for transforming positions
        dr = abs(np.rad2deg(image.metadata.microscope_state.stage_position.r - pos.r))
        if np.isclose(dr, 180, atol=2):
            pos = _transform_position(pos)

        pt = calculate_reprojected_stage_position2(image, pos)
        pt.name = pos.name

        if bound and not is_inside_image_bounds((pt.y, pt.x), image.data.shape):
            continue

        points.append(pt)

    return points


def plot_stage_positions_on_image(
        image: FibsemImage,
        positions: List[FibsemStagePosition],
        show: bool = False,
        bound: bool = True,
        color: Optional[str] = None,
        show_scalebar: bool = False,
        show_names: bool = True,
        figsize: Optional[Tuple[int, int]] = (15, 15)) -> Figure:
    """Plot stage positions reprojected on an image as matplotlib figure. Assumes image is flat to beam.
    Args:
        image: The image.
        positions: The positions.
        show: Whether to show the plot.
        bound: Whether to only plot points inside the image.
        color: The color of the points. (None -> default colour cycle)
    Returns:
        The matplotlib figure."""
    from fibsem.ui.napari.utilities import is_inside_image_bounds
    if image.metadata is None or image.metadata.microscope_state is None:
        raise ValueError("Image metadata or microscope state is not set. Cannot reproject stage positions.")

    # reproject stage positions onto image 
    points = reproject_stage_positions_onto_image2(image=image, positions=positions)

    # construct matplotlib figure
    fig = plt.figure(figsize = figsize)
    plt.imshow(image.data, cmap="gray")

    for i, pt in enumerate(points):

        # if points outside image, don't plot
        if bound and not is_inside_image_bounds((pt.y, pt.x), (image.data.shape[0], image.data.shape[1])):
            continue

        if color is None:
            c = POSITION_COLOURS[i%len(POSITION_COLOURS)]
        else:
            c = color
        plt.plot(pt.x, pt.y, ms=20, c=c, marker="+", markeredgewidth=2, label=f"{pt.name}")

        if show_names:
            # draw position name next to point
            plt.text(pt.x, pt.y-50, pt.name, fontsize=14, color=c, alpha=0.75)

    if show_scalebar:
        try:
            # add scalebar
            from matplotlib_scalebar.scalebar import ScaleBar
            scalebar = ScaleBar(
                dx=image.metadata.pixel_size.x,
                color="black",
                box_color="white",
                box_alpha=0.5,
                location="lower right",
            )
            plt.gca().add_artist(scalebar)
        except Exception as e:
            logging.debug(f"Could not add scalebar: {e}")

    plt.axis("off")
    if show:
        plt.show()

    return fig

def plot_minimap(
        image: FibsemImage,
        positions: List[FibsemStagePosition],
        current_position: Optional[FibsemStagePosition] = None,
        grid_positions: Optional[List[FibsemStagePosition]] = None,
        show: bool = False,
        bound: bool = True,
        color: str = "cyan",
        show_scalebar: bool = False,
        show_names: bool = True,
        show_grid_radius: bool = False,
        fontsize: int = 12,
        markersize: int = 20,
        figsize: Optional[Tuple[int, int]] = (15, 15),
        ax: Optional[plt.Axes] = None) -> Figure:
    """Plot stage positions reprojected on an image as matplotlib figure. Assumes image is flat to beam.
    Args:
        image: The image.
        positions: The positions.
        current_position: Optional current position to highlight
        grid_positions: Optional grid positions to show
        show: Whether to show the plot.
        bound: Whether to only plot points inside the image.
        color: The color of the points.
        show_scalebar: Whether to show a scalebar
        show_names: Whether to show position names as labels
        fontsize: Font size for position name labels (default: 14)
        figsize: Figure size in inches (default: (15, 15))
    Returns:
        The matplotlib figure."""
    from fibsem.ui.napari.utilities import is_inside_image_bounds
    if image.metadata is None or image.metadata.microscope_state is None:
        raise ValueError("Image metadata or microscope state is not set. Cannot reproject stage positions.")

    all_positions = list(positions)
    if current_position is not None:
        all_positions.append(current_position)
    if grid_positions is not None:
        all_positions.extend(grid_positions)

    # construct matplotlib figure/axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.imshow(image.data, cmap="gray")

    # reproject stage positions onto image 
    points = reproject_stage_positions_onto_image2(image=image, positions=all_positions)

    marker_entries: List[dict] = []
    for i, pt in enumerate(points):

        # if points outside image, don't plot
        if bound and not is_inside_image_bounds((pt.y, pt.x), (image.data.shape[0], image.data.shape[1])):
            continue
        
        if pt.name is None:
            pt.name = f"Position {i:02d}"

        c = color
        if "Grid" in pt.name:
            c = "red"
        elif "Current Position" in pt.name:
            c = "yellow"

        marker_entries.append(
            {
                "point": (pt.x, pt.y),
                "color": c,
                "label": pt.name,
            }
        )

        # show grid radius
        if c == "red" and show_grid_radius:
            r_pixels = 1000e-6 / image.metadata.pixel_size.x 
            ax.add_artist(plt.Circle((pt.x, pt.y), radius=r_pixels, color=c, fill=False, linewidth=5)
            )

    if marker_entries:
        scatter_array = np.array([entry["point"] for entry in marker_entries])
        scatter_colors = [entry["color"] for entry in marker_entries]
        ax.scatter(
            scatter_array[:, 0],
            scatter_array[:, 1],
            c=scatter_colors,
            marker="+",
            s=markersize ** 2,
            linewidths=2,
        )

        if show_names:
            for entry in marker_entries:
                x, y = entry["point"]
                ax.text(
                    x + 10,
                    y - 10,
                    entry["label"],
                    fontsize=fontsize,
                    color=entry["color"],
                    alpha=0.75,
                    clip_on=True,
                )

    if show_scalebar:
        try:
            # add scalebar
            from matplotlib_scalebar.scalebar import ScaleBar
            ax.add_artist(
                ScaleBar(
                    dx=image.metadata.pixel_size.x,
                    color="black",
                    box_color="white",
                    box_alpha=0.5,
                    location="lower right",
                )
            )
        except Exception as e:
            logging.debug(f"Could not add scalebar: {e}")

    ax.axis("off")
    if show:
        plt.show()

    return fig

def convert_image_coord_to_stage_position(
    microscope: FibsemMicroscope, image: FibsemImage, coord: Tuple[float, float]
) -> FibsemStagePosition:
    """Convert a coordinate in the image to a stage position. Assume image is flat to beam.
    Args:
        microscope: The microscope connection.
        image: The image
        coord: The coordinate in the image (y,x).
    Returns:
        The stage position.
    """
    # convert image to microscope image coordinates
    point = conversions.image_to_microscope_image_coordinates(
        coord=Point(x=coord[1], y=coord[0]),
        image=image.data,
        pixelsize=image.metadata.pixel_size.x,
    )
    # project as stage position
    stage_position = microscope.project_stable_move(
        dx=point.x,
        dy=point.y,
        beam_type=image.metadata.image_settings.beam_type,
        base_position=image.metadata.microscope_state.stage_position,
    )

    return stage_position

def convert_image_coordinates_to_stage_positions(
    microscope: FibsemMicroscope, image: FibsemImage, coords: List[Tuple[float, float]]
) -> List[FibsemStagePosition]:
    """Convert a list of coordinates in the image to a list of stage positions. Assume image is flat to beam.
    Args:
        microscope: The microscope connection.
        image: The image
        coords: The coordinates in the image (y,x).
    Returns:
        The stage positions."""

    stage_positions = []
    for i, coord in enumerate(coords):
        stage_position = convert_image_coord_to_stage_position(
            microscope=microscope, image=image, coord=coord
        )
        stage_position.name = f"Position {i:02d}"
        stage_positions.append(stage_position)
    return stage_positions

##### THERMO ONLY

X_OFFSET = -0.0005127403888932854 
Y_OFFSET = 0.0007937916666666666

def _to_specimen_coordinate_system(pos: FibsemStagePosition):
    """Converts a position in the raw coordinate system to the specimen coordinate system"""

    specimen_offset = FibsemStagePosition(x=X_OFFSET, y=Y_OFFSET, z=0.0, r=0, t=0, coordinate_system="RAW")
    specimen_position = pos - specimen_offset

    return specimen_position

def _to_raw_coordinate_system(pos: FibsemStagePosition):
    """Converts a position in the raw coordinate system to the specimen coordinate system"""

    specimen_offset = FibsemStagePosition(x=X_OFFSET, y=Y_OFFSET, z=0.0, r=0, t=0, coordinate_system="RAW")
    raw_position = pos + specimen_offset

    return raw_position


def _transform_position(pos: FibsemStagePosition) -> FibsemStagePosition:
    """This function takes in a position flat to a beam, and outputs the position if stage was rotated / tilted flat to the other beam).
    Args:
        pos: The position flat to the beam.
    Returns:
        The position flat to the other beam."""

    specimen_position = _to_specimen_coordinate_system(pos)
    # print("raw      pos: ", pos)
    # print("specimen pos: ", specimen_position)

    # # inverse xy (rotate 180 degrees)
    specimen_position.x = -specimen_position.x
    specimen_position.y = -specimen_position.y

    # movement offset (calibration for compucentric rotation error)
    specimen_position.x += 50e-6
    specimen_position.y += 25e-6

    # print("rotated pos: ", specimen_position)

    # _to_raw_coordinates
    transformed_position = _to_raw_coordinate_system(specimen_position)
    transformed_position.name = pos.name

    # print("trans   pos: ", transformed_position)
    logging.info(f"Initial position {pos} was transformed to {transformed_position}")

    return transformed_position

def _inverse_y_corrected_stage_movement(
    image: FibsemImage,
    dy: float,
    dz: float,
    beam_type: BeamType = BeamType.ELECTRON,
) -> float:
        """
        Calculate the expected_y input from dy, dz stage movements and beam_type.
        This is the inverse of _y_corrected_stage_movement.

        Args:
            dy (float): actual y stage movement
            dz (float): actual z stage movement  
            beam_type (BeamType, optional): beam_type used. Defaults to BeamType.ELECTRON.

        Returns:
            float: expected_y input that would produce the given dy, dz movements
        """
        if image.metadata is None or image.metadata.system is None:
            raise ValueError("Image metadata or system metadata is not set. Cannot calculate inverse y corrected stage movement.")

        # all angles in radians
        sem_column_tilt = np.deg2rad(image.metadata.system.electron.column_tilt)
        fib_column_tilt = np.deg2rad(image.metadata.system.ion.column_tilt)

        stage_pretilt = np.deg2rad(image.metadata.system.stage.shuttle_pre_tilt)

        stage_rotation_flat_to_eb = np.deg2rad(image.metadata.system.stage.rotation_reference) % (2 * np.pi)
        stage_rotation_flat_to_ion = np.deg2rad(image.metadata.system.stage.rotation_180) % (2 * np.pi)

        # current stage position
        current_stage_position = image.metadata.stage_position
        stage_rotation = current_stage_position.r % (2 * np.pi) if current_stage_position.r is not None else 0.0
        stage_tilt = current_stage_position.t if current_stage_position.t is not None else 0.0

        # Handle compustage case
        compustage_sign = 1.0
        stage_is_compustage = "Arctis" in image.metadata.system.info.model or image.metadata.system.sim.get("is_compustage", False)
        if stage_is_compustage: # TODO: add compustage to metadata
            if stage_tilt <= 0 and stage_tilt > np.radians(-90):
                compustage_sign = -1.0
            stage_tilt += np.pi

        PRETILT_SIGN = 1.0
        # pretilt angle depends on rotation
        from fibsem import movement
        if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_eb, atol=5):
            PRETILT_SIGN = 1.0
        if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_ion, atol=5):
            PRETILT_SIGN = -1.0

        corrected_pretilt_angle = PRETILT_SIGN * (stage_pretilt + sem_column_tilt)

        # perspective tilt adjustment
        if beam_type == BeamType.ELECTRON:
            perspective_tilt_adjustment = -corrected_pretilt_angle
        elif beam_type == BeamType.ION:
            perspective_tilt_adjustment = (-corrected_pretilt_angle - fib_column_tilt)

        # Reverse the calculations from the forward function:
        # Forward: y_move = y_sample_move * cos(corrected_pretilt_angle)
        # Forward: z_move = -y_sample_move * sin(corrected_pretilt_angle)
        # Therefore: y_sample_move can be calculated from either dy or dz

        # Calculate y_sample_move from dy and dz (should be consistent)
        cos_pretilt = np.cos(corrected_pretilt_angle)
        sin_pretilt = np.sin(corrected_pretilt_angle)
        
        if abs(cos_pretilt) > abs(sin_pretilt):
            # Use dy calculation when cos component is larger
            y_sample_move = dy / cos_pretilt
        else:
            # Use dz calculation when sin component is larger
            y_sample_move = -dz / sin_pretilt

        # Reverse: expected_y = y_sample_move * cos(stage_tilt + perspective_tilt_adjustment)
        expected_y = y_sample_move * np.cos(stage_tilt + perspective_tilt_adjustment)

        # Apply compustage correction if needed
        if stage_is_compustage:
            expected_y *= compustage_sign

        return expected_y


def generate_grid_positions(
    ncols: int, nrows: int, fov_x: float, fov_y: float, overlap: float = 0.1
) -> list[tuple[float, float]]:
    """Generate a grid of positions, centered around the origin, for acquiring tiles.

    Creates a regular grid of (x, y) positions that are properly centered around the origin
    (0, 0) for both odd and even numbers of columns and rows. The spacing between positions
    accounts for the specified field of view and tile overlap.

    Args:
        ncols: Number of columns in the grid (must be positive)
        nrows: Number of rows in the grid (must be positive)
        fov_x: Horizontal field of view size in meters (physical dimension of each tile)
        fov_y: Vertical field of view size in meters (physical dimension of each tile)
        overlap: Fraction of overlap between adjacent tiles (0.0 to 1.0)

    Returns:
        List of (x, y) tuples representing grid positions in meters, centered around origin

    Example:
        >>> # 3x3 grid with 10μm x 8μm FOV and 10% overlap
        >>> positions = generate_grid_positions(3, 3, 10e-6, 8e-6, 0.1)
        >>> len(positions)
        9
        >>> positions[4]  # Center position
        (0.0, 0.0)

        >>> # 4x4 grid is also properly centered
        >>> positions = generate_grid_positions(4, 4, 10e-6, 10e-6, 0.0)
        >>> import numpy as np
        >>> np.mean([pos[0] for pos in positions])  # Mean x should be ~0
        0.0
    """
    positions = []
    for i in range(nrows):
        for j in range(ncols):
            x = (j - (ncols - 1) / 2) * (fov_x * (1 - overlap))
            y = -(i - (nrows - 1) / 2) * (fov_y * (1 - overlap))
            positions.append((x, y))

    return positions

def convert_grid_positions_to_stage_positions(
    microscope: 'FibsemMicroscope',
    positions: list[tuple[float, float]],
    beam_type: BeamType = BeamType.ELECTRON,
    base_position: FibsemStagePosition | None = None,
) -> list[FibsemStagePosition]:
    """Convert grid positions to stage positions using microscope projection.

    Takes a list of (x, y) grid positions and converts them to FibsemStagePosition
    objects using the microscope's project_stable_move method. This accounts for
    the microscope's coordinate system and current stage configuration.

    Args:
        microscope: The FibsemMicroscope instance to use for projection
        positions: List of (x, y) tuples representing grid positions in meters
        beam_type: Beam type to use for projection (default: ELECTRON)
        base_position: Base stage position to project from (default: current position)

    Returns:
        List of FibsemStagePosition objects representing the projected stage positions

    Example:
        >>> # Generate grid positions
        >>> positions = generate_grid_positions(3, 3, 10e-6, 0.1)
        >>> # Convert to stage positions
        >>> stage_positions = convert_grid_positions_to_stage_positions(
        ...     microscope, positions, BeamType.ELECTRON
        ... )
        >>> len(stage_positions)
        9
    """
    if base_position is None:
        base_position = microscope.get_stage_position()

    stage_positions = []
    for pos in positions:
        x, y = pos
        stage_position = microscope.project_stable_move(
            dx=x, dy=y, beam_type=beam_type, base_position=base_position
        )
        stage_positions.append(stage_position)
    return stage_positions


# TODO:
# - restore initial state, always even on error - DONE
# - support overlap
# - support non-square tilesizes - DONE
# - migrate/create to acquisition_progress_signal for progress updates 
# - pass stop event as arg
# - add auto-focus, and options for focus strategy (per-tile, z-focus-stack, etc)

##### TILED ACQUISITION
# def tiled_image_acquisition_v2(
#     microscope: FibsemMicroscope,
#     image_settings: ImageSettings,
#     nrows: int,
#     ncols: int,
#     tile_size: float,
#     overlap: float = 0.1,
#     cryo: bool = True,
#     stop_event: Optional[threading.Event] = None,
# ) -> dict:
#     """Tiled image acquisition.
#     Args:
#         microscope: The microscope connection.
#         image_settings: The image settings.
#         nrows: The number of rows in the grid.
#         ncols: The number of columns in the grid.
#         tile_size: The size of the tiles (field of view in x-direction).
#         overlap: The overlap between tiles in pixels. Currently not supported.
#         cryo: Whether to use cryo mode (histogram equalisation).
#         stop_event: Optional threading.Event to cancel acquisition.
#     Returns:
#         A dictionary containing the acquisition details for stitching."""

#     image_width, image_height = image_settings.resolution
#     fov_x = tile_size
#     fov_y = tile_size * (image_height / image_width)
#     step_x, step_y = fov_x*(1-overlap), fov_y*(1-overlap)

#     total_fov_x = ncols * (fov_x * (1-overlap))  # total fov_x
#     total_fov_y = nrows * (fov_y * (1-overlap))  # total fov_y

#     # fixed image settings
#     image_settings.hfw = tile_size
#     image_settings.autogamma = False
#     image_settings.autocontrast = False # required for cryo
#     image_settings.save = True

#     # we save all intermediates into a folder with the same name as the filename, then save the stitched image into the parent folder
#     prev_path = image_settings.path
#     prev_label = image_settings.filename
#     image_settings.path = os.path.join(image_settings.path, image_settings.filename)
#     os.makedirs(image_settings.path, exist_ok=True)

#     # TOP LEFT CORNER START
#     start_move_x = (ncols * step_x) / 2 - step_x / 2
#     start_move_y = (nrows * step_y) / 2 - step_y / 2

#     start_offset_x = (ncols - 1) * step_x / 2
#     start_offset_y = (nrows - 1) * step_y / 2

#     assert np.isclose(start_move_x, start_offset_x), f"start_move_x: {start_move_x}, start_offset_x: {start_offset_x}"
#     assert np.isclose(start_move_y, start_offset_y), f"start_move_y: {start_move_y}, start_offset_y: {start_offset_y}"

#     # stitched image dimensions
#     effective_tile_width = max(1, int(round(image_width * (1 - overlap))))
#     effective_tile_height = max(1, int(round(image_height * (1 - overlap))))

#     mosaic_width = effective_tile_width * (ncols - 1) + image_width
#     mosaic_height = effective_tile_height * (nrows - 1) + image_height
#     arr = np.zeros((mosaic_height, mosaic_width), dtype=np.uint8)
#     total_tiles = nrows*ncols

#     logging.info(f"Tiled Acquisition: {nrows} rows x {ncols} cols, Total Tiles: {total_tiles}")
#     logging.info(f"Tile FoV: {fov_x*1e6:.0f} um x {fov_y*1e6:.0f} um, Overlap: {overlap*100:.1f} %")
#     logging.info(f"Total FoV: {total_fov_x*1e6:.0f} um x {total_fov_y*1e6:.0f} um")
#     logging.info(f"Tile size: {image_height}x{image_width}, Mosaic size: {mosaic_height}x{mosaic_width}")
#     logging.info(f"Stitched image shape: {arr.shape}, Non-overlap grid shape: {image_height*nrows}x{image_width*ncols}")

#     logging.info(f"-----------------Starting tiled acquisition---------------")
#     try:
#         # start in the middle of the grid
#         start_state = microscope.get_microscope_state()
    
#         # move to top-left corner
#         logging.info(f"Moving to top-left corner: dx: {-start_offset_x*1e6:.0f} um, dy: {start_offset_y*1e6:.0f} um")
#         microscope.stable_move(dx=-start_offset_x, dy=start_offset_y, beam_type=image_settings.beam_type)
#         start_position = microscope.get_stage_position() # top-left corner

#         n_tiles_acquired = 0
#         images: list[list[FibsemImage]] = []

#         step_y *= -1 # need to invert y-axis

#         for i in range(nrows):

#             microscope.safe_absolute_stage_movement(start_position)

#             img_row: list[FibsemImage] = []
#             microscope.stable_move(dx=0, dy=i*step_y, beam_type=image_settings.beam_type) 
#             # NOTE: this will be slow for large nrows, esp on arctis -> migrate to abs move + projection

#             for j in range(ncols):
#                 image_settings.filename = f"tile_{i}_{j}"
#                 stage_position = microscope.stable_move(dx=step_x*(j!=0),  dy=0, beam_type=image_settings.beam_type) # dont move on the first tile?

#                 if stop_event and stop_event.is_set():
#                     raise Exception("User Stopped Acquisition")

#                 logging.info(f"Acquiring Tile {i}, {j} at {stage_position.pretty}")
#                 image = acquire.acquire_image(microscope, image_settings)

#                 # stitch image
#                 start_y = i * effective_tile_height
#                 start_x = j * effective_tile_width
#                 arr[start_y:start_y + image_height, start_x:start_x + image_width] = image.data

#                 n_tiles_acquired += 1
#                 microscope.tiled_acquisition_signal.emit(
#                     {
#                         "msg": "Tile Collected",
#                         "i": i,
#                         "j": j,
#                         "n_rows": nrows,
#                         "n_cols": ncols,
#                         "image": arr,
#                         "counter": n_tiles_acquired,
#                         "total": total_tiles,
#                     }
#                 )
#                 if isinstance(microscope, DemoMicroscope):
#                     time.sleep(1)

#                 img_row.append(image)
#             images.append(img_row)
#     except Exception as e:
#         logging.error(f"Tiled acquisition failed: {e}")
#         raise
#     finally:
#         logging.info(f"Tiled acquisition complete, restoring initial position: {start_state.stage_position.pretty}")
#         microscope.set_microscope_state(start_state)
#     logging.info(f"-----------------Finished tiled acquisition---------------")

#     image_settings.path = prev_path

#     ddict = {"total_fov": total_fov_x, "tile_size": tile_size, "n_rows": nrows, "n_cols": ncols, 
#             "image_settings": image_settings, 
#             "dx": step_x, "dy": step_y, "cryo": cryo,
#             "start_state": start_state, "prev-filename": prev_label, 
#             "start_move_x": start_move_x, "start_move_y": start_move_y, 
#             "dxg": start_move_x, "dyg": start_move_y,
#             "images": images, "big_image": None, "stitched_image": arr}

#     return ddict