"""The analysis tools: measurements on a saved image in the experiment, in its
own values, each with a figure to look at and cite (`fibsem.imaging.analysis`).
Nothing moves and no beam fires."""

from typing import List

from fibsem.ai.agent.catalog.spec import ToolSpec, arg

_CHANNEL = arg(
    "optional: the layer to measure, a fluorescence channel by name (default the first); a beam image has one",
)

TOOLS = (
    ToolSpec(
        name="analysis_measure_intensity",
        returns="ImageMeasured",
        description="The signal at points of a saved image (a fluorescence stack or a beam image), per plane, in the file's own values: for each point the peak and mean in a disc of radius_px, the median of the ring 2 to 3 radii out as its background, and signal (mean above background); best_plane is where signal is highest, saturated whether the peak there reaches the file's full value. reference (a point's index) adds each point's signal as a ratio to it: a bead against the target, a lamella's spot against the bulk. Saves a figure beside the image as drawn/<image>-intensity.png, a row per point: a crop of each plane about it at one set of limits, the disc and ring on the best plane, and its signal and background against plane. Cite that path as evidence. Nothing moves.",
        method="POST",
        path="/app/experiment/analysis/intensity",
        level="watch",
        router="app",
        params={
            "path": "a saved image file path inside the experiment folder",
            "points": arg(
                "[[x, y], ...] in the image's full-resolution pixels", List[List[float]]
            ),
            "radius_px": arg(
                "optional: the disc's radius in pixels (default 3)", float
            ),
            "channel": _CHANNEL,
            "reference": arg(
                "optional: the index of the point the others are a ratio to", int
            ),
            "planes": arg(
                "optional: the planes to measure, 0 the first (default all)", List[int]
            ),
        },
        optional=("radius_px", "channel", "reference", "planes"),
    ),
    ToolSpec(
        name="analysis_measure_profile",
        returns="ImageMeasured",
        description="A saved image's values along a line from start to end ([x, y] pixels), a sample per pixel, each the mean across width_px, on one plane of a fluorescence stack (plane) or its max projection. Answers the samples and their distance (in metres too when the pixel size is known), and the full width at half maximum of the profile's peak (its dip with dip=true) above the baseline, the median of the profile's outer fifths: fwhm_px (fwhm_m), with where it crosses the half level. fwhm is null when the profile doesn't fall to half on both sides: draw a longer line. For a spot's size, a lamella's edge, a bar's width. Saves a figure beside the image as drawn/<image>-profile-<x0>-<y0>-<x1>-<y1>.png: the line on the image and the profile with its FWHM. Cite that path as evidence. Nothing moves.",
        method="POST",
        path="/app/experiment/analysis/profile",
        level="watch",
        router="app",
        params={
            "path": "a saved image file path inside the experiment folder",
            "start": arg("[x, y] in the image's full-resolution pixels", List[float]),
            "end": arg("[x, y] in the image's full-resolution pixels", List[float]),
            "width_px": arg(
                "optional: the band averaged across the line, in pixels (default 1)",
                float,
            ),
            "channel": _CHANNEL,
            "plane": arg(
                "optional: a fluorescence stack's plane, 0 the first (default the max projection)",
                int,
            ),
            "dip": arg(
                "optional: measure the profile's dip, not its peak (default false)",
                bool,
            ),
        },
        optional=("width_px", "channel", "plane", "dip"),
    ),
    ToolSpec(
        name="analysis_measure_quality",
        returns="ImageMeasured",
        description="Whether a saved image is good enough to go on with. A beam image is judged as auto brightness/contrast judges one: the fraction of pixels clipped black and white (only areas wider than a few pixels count), the centre and spread of the bulk of its histogram on the file's range, problems in words (empty when it meets the targets), and unusable when it is mostly black, white or flat. A fluorescence channel has no such targets: its percentiles, saturated (% of pixels at the file's full value) and range_used (the 99.9th percentile as a fraction of it). focus is the mean Laplacian: only comparable between images of one scene at one magnification. For a fluorescence stack, each plane's sharpness relative to the sharpest, and which that is. Measured on plane, else a stack's max projection. Nothing is saved; nothing moves.",
        method="GET",
        path="/app/experiment/analysis/quality",
        level="watch",
        router="app",
        params={
            "path": "a saved image file path inside the experiment folder",
            "channel": _CHANNEL,
            "plane": arg(
                "optional: a fluorescence stack's plane, 0 the first (default the max projection)",
                int,
            ),
        },
        optional=("channel", "plane"),
    ),
)
