from typing import Tuple

import numpy as np

from fibsem.structures import FibsemImage, Point


def image_to_microscope_image_coordinates_px(
    coord: Point, image_shape: Tuple[int, int], subpixel_precision: bool
) -> Point:
    """
    Convert an image pixel coordinate to a microscope image coordinate, in pixels.

    The microscope image coordinate system is centered on the image with positive Y-axis pointing upwards,
    so y is mirrored about the centre where x is only shifted. This is the single source of that convention:
    the two metres-returning functions below are this plus a pixel-size multiply, and the correlation module
    reprojects points through it (FIB-362).

    ``subpixel_precision`` has no default on purpose. Flooring the centre only differs from the true centre
    on an odd image dimension, and then by half a pixel -- small enough to go unnoticed and large enough to
    matter, so each caller says which it wants.

    Args:
        coord (Point): A Point object representing the pixel coordinates in the original image.
        image_shape (Tuple[int, int]): The shape of the image (height, width).
        subpixel_precision (bool): Floors the centre to a whole pixel if False.

    Returns:
        Point: A Point object representing the corresponding microscope image coordinates, in pixels.
    """
    # convert from image pixel coord (0, 0) top left to microscope image (0, 0) mid
    centre_px = np.asarray(image_shape) / 2
    if not subpixel_precision:
        centre_px = np.asarray(image_shape) // 2
    cy, cx = centre_px

    # distance from centre?
    return Point(
        x=float(coord.x - cx),   # neg = left
        y=float(cy - coord.y),   # neg = down
    )

def image_to_microscope_image_coordinates(
    coord: Point, image: np.ndarray, pixelsize: float, subpixel_precision: bool = False
) -> Point:
    """
    Convert an image pixel coordinate to a microscope image coordinate.

    The microscope image coordinate system is centered on the image with positive Y-axis pointing upwards.

    Args:
        coord (Point): A Point object representing the pixel coordinates in the original image.
        image (np.ndarray): A numpy array representing the image.
        pixelsize (float): The pixel size in meters.
        subpixel_precision (bool): Rounds to the nearest pixel if False (default)

    Returns:
        Point: A Point object representing the corresponding microscope image coordinates in meters.
    """
    point_px = image_to_microscope_image_coordinates_px(
        coord, image.shape, subpixel_precision
    )

    return convert_point_from_pixel_to_metres(point_px, pixelsize)

def image_to_microscope_image_coordinates2(
    coord: Point, image_shape: Tuple[int, int], pixelsize: float, subpixel_precision: bool = False
) -> Point:
    """
    Convert an image pixel coordinate to a microscope image coordinate.

    The microscope image coordinate system is centered on the image with positive Y-axis pointing upwards.

    Args:
        coord (Point): A Point object representing the pixel coordinates in the original image.
        image (np.ndarray): A numpy array representing the image.
        pixelsize (float): The pixel size in meters.
        subpixel_precision (bool): Rounds to the nearest pixel if False (default)

    Returns:
        Point: A Point object representing the corresponding microscope image coordinates in meters.
    """
    if len(image_shape) != 2:
        raise ValueError("image_shape must be a tuple of two integers (height, width)")

    point_px = image_to_microscope_image_coordinates_px(
        coord, image_shape, subpixel_precision
    )

    return point_px._to_metres(pixel_size=pixelsize)

def microscope_image_to_image_coordinates(point: Point, image_shape: Tuple[int, int], pixel_size: float) -> Point:
    """
    Convert a microscope image coordinate to an image pixel coordinate. 
    The microscope image coordinate system is centered on the image with positive Y-axis pointing upwards.
    Args:
        point (Point): A Point object representing the microscope image coordinates in metres.
        image_shape (Tuple[int, int]): A tuple representing the shape of the image (height, width).
        pixel_size (float): The pixel size in metres.
    Returns:
        Point: A Point object representing the corresponding pixel coordinates in the original image.
    """
    # position in metres from image centre
    mx, my = point.x, point.y
    pmx, pmy = mx / pixel_size, my / pixel_size

    # convert to image coordinates
    cy, cx = image_shape[0] // 2, image_shape[1] // 2
    px = cx + pmx
    py = cy - pmy

    return Point(x=px, y=py)

def get_lamella_size_in_pixels(
    img: FibsemImage, protocol: dict, use_trench_height: bool = False
) -> Tuple[int]:
    """Get the relative size of the lamella in pixels based on the hfw of the image.

    Args:
        img (FibsemImage): A reference image.
        protocol (dict): A dictionary containing the protocol information.
        use_trench_height (bool, optional): If True, returns the height of the trench instead of the lamella. Default is False.

    Returns:
        Tuple[int]: A tuple containing the height and width of the lamella in pixels.
    """
    # get real size from protocol
    lamella_width = protocol["lamella_width"]
    lamella_height = protocol["lamella_height"]

    total_height = lamella_height
    if use_trench_height:
        trench_height = protocol["stages"][0]["trench_height"]
        total_height += 2 * trench_height

    # convert to m
    pixelsize = img.metadata.pixel_size.x
    width, height = img.metadata.image_settings.resolution
    vfw = convert_pixels_to_metres(height, pixelsize)
    hfw = convert_pixels_to_metres(width, pixelsize)

    # lamella size in px (% of image)
    lamella_height_px = int((total_height / vfw) * height)
    lamella_width_px = int((lamella_width / hfw) * width)

    return (lamella_height_px, lamella_width_px)


def convert_metres_to_pixels(distance: float, pixelsize: float) -> int:
    """
    Convert a distance in metres to pixels based on a given pixel size.

    Args:
        distance (float): The distance to convert, in metres.
        pixelsize (float): The size of a pixel in metres.

    Returns:
        int: The distance converted to pixels, as an integer.
    """
    return int(distance / pixelsize)


def convert_pixels_to_metres(pixels: int, pixelsize: float) -> float:
    """
    Convert a distance in pixels to metres based on a given pixel size.

    Args:
        pixels (int): The number of pixels to convert.
        pixelsize (float): The size of a pixel in metres.

    Returns:
        float: The distance converted to metres.
    """
    return float(pixels * pixelsize)


def distance_between_points(p1: Point, p2: Point) -> Point:
    """
    Calculate the Euclidean distance between two points, returning a Point object representing the result.

    Args:
        p1 (Point): The first point.
        p2 (Point): The second point.

    Returns:
        Point: A Point object representing the distance between the two points.
    """

    return Point(x=(p2.x - p1.x), y=(p2.y - p1.y))


def convert_point_from_pixel_to_metres(point: Point, pixelsize: float) -> Point:
    """
    Convert a Point object from pixel coordinates to metre coordinates, based on a given pixel size.

    Args:
        point (Point): The Point object to convert.
        pixelsize (float): The size of a pixel in metres.

    Returns:
        Point: The converted Point object, with its x and y values in metre coordinates.
    """
    point_m = Point(
        x=convert_pixels_to_metres(point.x, pixelsize),
        y=convert_pixels_to_metres(point.y, pixelsize),
    )

    return point_m


def convert_point_from_metres_to_pixel(point: Point, pixelsize: float) -> Point:
    """
    Convert a Point object from metre coordinates to pixel coordinates, based on a given pixel size.

    Args:
        point (Point): The Point object to convert.
        pixelsize (float): The size of a pixel in metres.

    Returns:
        Point: The converted Point object, with its x and y values in pixel coordinates.
    """
    point_px = Point(
        x=convert_metres_to_pixels(point.x, pixelsize),
        y=convert_metres_to_pixels(point.y, pixelsize),
    )
    return point_px


def is_inside_image_bounds(coords: Tuple[float, float], shape: Tuple[int, int]) -> bool:
    """Check if the coordinates are inside the image bounds.

    Args:
        coords (Tuple[float, float]): y, x coordinates
        shape (Tuple[int, int]): image shape (y, x)

    Returns:
        bool: True if inside image bounds, False otherwise.

    NOTE: the lower bound is exclusive, so row/column 0 reports as outside even
    though it is a real pixel. This is intentional -- callers use it to decide
    whether to draw a marker, and widening the region would change what they
    render. It reads like an off-by-one, so it is pinned by test to stop it being
    tidied away.
    """
    ycoord, xcoord = coords

    if (ycoord > 0 and ycoord < shape[0]) and (
        xcoord > 0 and xcoord < shape[1]
    ):
        return True

    return False
