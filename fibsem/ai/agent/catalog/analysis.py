"""The analysis tools: they read and measure saved image files and nothing
else (`fibsem.analysis`): crop and draw an image, map its pixels to the stage
and back, and measure it in its own values, with a figure to look at and cite.
Nothing moves and no beam fires."""

from typing import List, Union

from fibsem.ai.agent.catalog.spec import (
    BOX_PARAM,
    LOOKS_PARAM,
    MARKS_PARAM,
    STACK_PARAMS,
    ToolSpec,
    arg,
)

_CHANNEL = arg(
    "optional: the layer to measure, a fluorescence channel by name (default the first); a beam image has one",
)

TOOLS = (
    ToolSpec(
        name="analysis_crop_image",
        returns="FileRead",
        description="One region of an image in the open experiment, at native resolution (downscaled only when wider than 768 px): full detail where it matters, without the whole file. Give box in source pixels or fraction (0 to 1, as read off a preview), each [x, y, width, height]; or cells, up to 8 cells by name (B3, as experiment_read_file with cells lists them), each answered with its own preview and centre stage position; what every crop's preview shares (width, height, hfw, preview_pixelsize) is said once, in crop_preview. A crop's preview carries its region and its own scale. A point read off a crop's picture goes to analysis_map_pixel_to_stage and marks with crop (the cell's name or the box), so you never offset it yourself.",
        method="GET",
        path="/app/experiment/files/{path}",
        level="watch",
        router="analysis",
        params={
            "path": "an image path from experiment_list_files, or one an image answer gave",
            "box": arg(
                "[x, y, width, height] in source pixels (instead of fraction)", list
            ),
            "fraction": arg(
                "[left, top, width, height] as fractions of the image, 0 to 1 (instead of box)",
                list,
            ),
            "cells": arg(
                'up to 8 cell names, e.g. ["B3", "C3"] (instead of box)', List[str]
            ),
            **STACK_PARAMS,
            "experiment": "optional: a past experiment to read instead of the open one, by its relative_path from experiment_list_all (or its path from experiment_list_recent); only one the operator named. Every read of it goes on the record",
        },
        optional=("box", "fraction", "cells", "z", "channels", "experiment"),
    ),
    ToolSpec(
        name="analysis_draw_image",
        returns="ImageDrawn",
        description="Draw marks on any saved image in the experiment (a beam image, or a fluorescence stack's max projection or one plane: plane, else the plane your marks' z share) and look at them before you act or cite them: the same marks as evidence takes, drawn as the operator sees them, in your purple, each label in a box placed clear of the others. crosshair adds the centre crosshair; the scale bar and metadata bar are under it. Saved beside the image as drawn/<image>.png (drawn/<image>-plane-<n>.png for a plane; replaced when drawn again): cite that path as evidence ({image: path}). looks sets each layer's contrast limits, gamma, opacity and colour; box draws one region. Answers the path, the marks (each one's label, a measured line's length, and the stage position of each place when the image says where it was taken; an unlabelled one also where it is), a preview, and per layer the look it was drawn with (auto limits as numbers) and its stats: min, p1, median, p99, max, clipped_low / clipped_high (% of pixels at the limits) and saturated (% at the file's full value). Too dark or saturated: set clim from those numbers and draw again.",
        method="POST",
        path="/app/experiment/images/drawn",
        level="watch",
        router="analysis",
        params={
            "path": "an image file path inside the experiment folder",
            "marks": arg(MARKS_PARAM, dict),
            "crosshair": arg(
                "optional: draw the centre crosshair (default false)", bool
            ),
            "metadata_bar": arg(
                "optional: the metadata bar under the image (default true)", bool
            ),
            "channels": arg(
                "optional: for a fluorescence stack, the channels to show, by name (default all)",
                List[str],
            ),
            "plane": arg(
                "optional: for a fluorescence stack, the plane to show (0 the first; default the plane your marks' z share, else the max projection): marks on other planes are left out",
                int,
            ),
            "looks": arg(LOOKS_PARAM, dict),
            "box": arg(BOX_PARAM, List[float]),
        },
        optional=(
            "marks",
            "crosshair",
            "metadata_bar",
            "channels",
            "plane",
            "looks",
            "box",
        ),
    ),
    ToolSpec(
        name="analysis_draw_pair",
        returns="ImagePairDrawn",
        description="Two saved images of the experiment side by side at one scale, each with its own scale bar and metadata bar: before and after a mill, before and after an alignment, SEM next to FIB of one site, two planes of one fluorescence stack (before_plane, after_plane: the target's plane next to the surface's, each with only its own marks). marks (in before's pixels) are drawn on both, at the same place of each frame, so a shift or a missing feature shows. box (before's pixels) draws that region of each, the same place on after. looks applies to both; share_looks (default true) draws after with before's limits wherever looks leaves a layer on auto, so a fade shows as a fade. Saved beside before as drawn/<before>-<after>.png: cite that path as evidence. Answers the path, the marks, a preview and each half's layers (look and stats, as analysis_draw_image).",
        method="POST",
        path="/app/experiment/images/pair",
        level="watch",
        router="analysis",
        params={
            "before": "an image file path inside the experiment folder, drawn on the left",
            "after": "an image file path inside the experiment folder, drawn on the right",
            "marks": arg(MARKS_PARAM, dict),
            "metadata_bar": arg(
                "optional: the metadata bar under each image (default true)", bool
            ),
            "channels": arg(
                "optional: for fluorescence stacks, the channels to show, by name (default all)",
                List[str],
            ),
            "before_plane": arg(
                "optional: before's plane, when a fluorescence stack (0 the first; default as analysis_draw_image picks)",
                int,
            ),
            "after_plane": arg(
                "optional: after's plane, when a fluorescence stack (0 the first; default as analysis_draw_image picks)",
                int,
            ),
            "looks": arg(LOOKS_PARAM, dict),
            "box": arg(BOX_PARAM + "; in before's pixels", List[float]),
            "share_looks": arg(
                "optional: draw after with before's limits wherever looks leaves a layer on auto (default true)",
                bool,
            ),
        },
        optional=(
            "marks",
            "metadata_bar",
            "channels",
            "before_plane",
            "after_plane",
            "looks",
            "box",
            "share_looks",
        ),
    ),
    ToolSpec(
        name="analysis_get_image_metadata",
        returns="SavedImageMetadata",
        description="The full saved metadata of one image file in the experiment folder: the image settings, pixel size, microscope state (stage, both beams, detectors) and the item and task it was taken for. The image tools already carry a short block of this; use this for the rest. Paths outside the experiment are refused.",
        method="GET",
        path="/app/experiment/images/metadata",
        level="watch",
        router="analysis",
        params={
            "path": "an image file path inside the experiment folder, e.g. the path an image answer gives"
        },
    ),
    ToolSpec(
        name="analysis_map_pixel_to_stage",
        returns="SavedImagePosition",
        description="The stage position under a point of any saved image in the experiment, beam or FM, from the image's own metadata, so it holds wherever the stage is now. Give x, y in source pixels, or fx, fy from 0 to 1 for a point read off a preview; with crop, x, y as read off that crop's picture (or fx, fy of it), converted for you. Nothing moves.",
        method="GET",
        path="/app/experiment/images/stage_position",
        level="watch",
        router="analysis",
        params={
            "path": "an image file path inside the experiment folder",
            "x": arg("source-pixel x (with y)", float),
            "y": arg("source-pixel y, +y down (with x)", float),
            "fx": arg("fraction across the image, 0-1 (with fy)", float),
            "fy": arg("fraction down the image, 0-1 (with fx)", float),
            "crop": arg(
                "optional: the crop the point was read off: a cell's name (B3) from analysis_crop_image, or its box [x, y, width, height]",
                Union[str, List[float]],
            ),
        },
        optional=("x", "y", "fx", "fy", "crop"),
    ),
    ToolSpec(
        name="analysis_map_stage_to_pixel",
        returns="SavedImagePixels",
        description="Where stage positions fall on any saved image in the experiment, beam or FM: source pixels for each (and whether inside the image), with the image's scale block. A position the image can't place answers why. Computed from the image's own metadata; never re-derive stage geometry. Nothing moves.",
        method="POST",
        path="/app/experiment/images/pixels",
        level="watch",
        router="analysis",
        params={
            "path": "an image file path inside the experiment folder",
            "stage_positions": arg(
                "stage position dicts (x, y, z, r, t, coordinate_system)", list
            ),
        },
    ),
    ToolSpec(
        name="analysis_measure_intensity",
        returns="ImageMeasured",
        description="The signal at points of a saved image (a fluorescence stack or a beam image), per plane, in the file's own values: for each point the peak and mean in a disc of radius_px, the median of the ring 2 to 3 radii out as its background, and signal (mean above background); best_plane is where signal is highest, saturated whether the peak there reaches the file's full value. reference (a point's index) adds each point's signal as a ratio to it: a bead against the target, a lamella's spot against the bulk. Saves a figure beside the image as drawn/<image>-intensity.png, a row per point: a crop of each plane about it at one set of limits, the disc and ring on the best plane, and its signal and background against plane. Cite that path as evidence. Nothing moves.",
        method="POST",
        path="/app/experiment/analysis/intensity",
        level="watch",
        router="analysis",
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
        description="A saved image's values along a line from start to end ([x, y] pixels), a sample per pixel, each the mean across width_px, on one plane of a fluorescence stack (plane) or its max projection. Answers the line's length (in metres too when the pixel size is known), min, max, the extremum and where it is, and the full width at half maximum of the profile's peak (its dip with dip=true) above the baseline, the median of the profile's outer fifths: fwhm_px (fwhm_m), with where it crosses the half level. fwhm is null when the profile doesn't fall to half on both sides: draw a longer line. The samples themselves are left out; points=N adds the profile resampled to N samples (distance_px, values). For a spot's size, a lamella's edge, a bar's width. Saves a figure beside the image as drawn/<image>-profile-<x0>-<y0>-<x1>-<y1>.png: the line on the image and the profile with its FWHM. Cite that path as evidence. Nothing moves.",
        method="POST",
        path="/app/experiment/analysis/profile",
        level="watch",
        router="analysis",
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
            "points": arg(
                "optional: also answer the profile resampled to this many samples, 2 to 10000 (default none; a line shorter than that comes per pixel)",
                int,
            ),
        },
        optional=("width_px", "channel", "plane", "dip", "points"),
    ),
    ToolSpec(
        name="analysis_measure_coincidence_offset",
        returns="ImageMeasured",
        description="SEM/FIB coincidence from features you picked yourself in a saved SEM image and a saved FIB image taken at one stage position: one feature as sem_point and fib_point, or 2 to 10 as pairs, each {sem: [x, y], fib: [x, y], label?} on the same feature, in each image's full-resolution pixels; points read off a crop's picture take that crop (sem_crop, fib_crop), converted for you. Converts with the images' own pixel sizes and the pair's stage geometry (the geometry alignment_check_coincidence uses). Answers per pair its offsets from each image centre (x right, y up) and the dy and dx it alone gives; vertical_move: the dy to hand stage_vertical_move with beam_type ION, the median of the pairs that agree, with from_pairs naming them, and direction in words; dz, the height error as alignment_check_coincidence reports it; stage_move_m, the raw stage move that dy makes; per pair after_fib_px, where its FIB point lands (the SEM view is unchanged; a feature lands at its SEM offset foreshortened, so one centred in the SEM ends centred in both); spread_m across pairs. A pair whose dy or dx is off the others' median by more than tolerance_m (2 um or 4 pixels unless you give tolerance_um) carries disagrees: it may be another feature, or the sample may not be flat; you decide. Two pairs that disagree, or no majority agreeing, give no vertical_move: each pair's own dy stands, for you to choose from. dx is what no stage move changes: one pair's differing dx marks a wrong match; a dx common to all (above 5 um, in warnings) is a beam offset. Refuses images that are not one electron and one ion image, or that lack pixel size or geometry; images whose recorded stage positions differ are measured, with a warning saying by how much. Saves a picture beside the SEM image as drawn/<sem>-<fib>-coincidence.png: every pair labelled in both images, and where each lands in the FIB. Check it before moving. Nothing moves.",
        method="POST",
        path="/app/experiment/analysis/coincidence",
        level="watch",
        router="analysis",
        params={
            "sem_path": "the saved SEM (electron) image's path inside the experiment folder",
            "fib_path": "the saved FIB (ion) image of the same place, taken without moving the stage",
            "sem_point": arg(
                "one feature's [x, y] in the SEM image's full-resolution pixels (with fib_point; or pairs)",
                List[float],
            ),
            "fib_point": arg(
                "the same feature's [x, y] in the FIB image's full-resolution pixels",
                List[float],
            ),
            "pairs": arg(
                "instead of sem_point and fib_point, up to 10 features: [{sem: [x, y], fib: [x, y], label: optional short name}, ...]",
                list,
            ),
            "sem_crop": arg(
                "optional: the crop the SEM points were read off: a cell's name (B3) from analysis_crop_image, or its box [x, y, width, height]",
                Union[str, List[float]],
            ),
            "fib_crop": arg(
                "optional: the crop the FIB points were read off, as sem_crop",
                Union[str, List[float]],
            ),
            "tolerance_um": arg(
                "optional: how far, in um, a pair's dy or dx may be off the others' before it is flagged (default 2 um or 4 pixels)",
                float,
            ),
        },
        optional=(
            "sem_point",
            "fib_point",
            "pairs",
            "sem_crop",
            "fib_crop",
            "tolerance_um",
        ),
    ),
    ToolSpec(
        name="analysis_measure_quality",
        returns="ImageMeasured",
        description="Measures of a saved image, to weigh with how it looks: whether it is good enough for what you need is yours to say. A beam image is measured as auto brightness/contrast measures one: the fraction of pixels clipped black and white (only areas wider than a few pixels count), and the centre and spread of the bulk of its histogram on the file's range. problems and unusable are what that routine would say against its own targets, which were set on its probe images, not on your sample: a hint, where an image can serve with a problem named or fail you with none. A fluorescence channel has no such targets: its percentiles, saturated (% of pixels at the file's full value) and range_used (the 99.9th percentile as a fraction of it). focus is the mean Laplacian: only comparable between images of one scene at one magnification. For a fluorescence stack, each plane's sharpness relative to the sharpest, and which that is. Measured on plane, else a stack's max projection. Nothing is saved; nothing moves.",
        method="GET",
        path="/app/experiment/analysis/quality",
        level="watch",
        router="analysis",
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
    ToolSpec(
        name="analysis_measure_histogram",
        returns="ImageMeasured",
        description="The histogram of a saved image's values (a beam image, or a fluorescence channel on plane, else its max projection), in bins equal bins over the file's range (0 to its full value when known, else its lowest to highest value), and only in box when given. Answers the edges and counts, the percentiles (min, p1, p5, median, p95, p99, max), mean and std, clipped_low / clipped_high (% of pixels at the range's ends: crushed blacks, saturated whites) and range_used (the fraction of the range the values span). Before you set contrast limits, judge a detector's gain, or compare two images' exposure. Saves a figure beside the image as drawn/<image>-histogram.png (-plane-<n>, -box-<x>-<y>-<w>-<h> when given): the counts on a log scale with p1, median and p99 marked. Cite that path as evidence. Nothing moves.",
        method="POST",
        path="/app/experiment/analysis/histogram",
        level="watch",
        router="analysis",
        params={
            "path": "a saved image file path inside the experiment folder",
            "channel": _CHANNEL,
            "plane": arg(
                "optional: a fluorescence stack's plane, 0 the first (default the max projection)",
                int,
            ),
            "bins": arg("optional: the number of bins, 2 to 1024 (default 64)", int),
            "box": arg(
                "optional: only this region, [x, y, width, height] in the image's pixels",
                List[float],
            ),
        },
        optional=("channel", "plane", "bins", "box"),
    ),
    ToolSpec(
        name="analysis_measure_power_spectrum",
        returns="ImageMeasured",
        description="The power spectrum of a saved image (a beam image, or a fluorescence channel on plane, else its max projection), only in box when given: the region cut to a square, windowed, and its 2D Fourier power averaged in rings. Answers the noise_floor (median power of the outermost tenth), cutoff (the highest frequency still twice the floor) as finest_period_px (and finest_period_m with the pixel size): the finest detail the image holds, which grows coarser with defocus, drift or a worn probe. anisotropy is the power along x over along y in the middle frequencies: near 1 for a round probe, far from it with astigmatism, drift or scan lines. The radial average itself is left out; points=N adds it resampled to N frequencies (frequency in cycles per pixel, 0 to 0.5, and radial_power). Compare images of one scene at one magnification; a featureless area has no detail to find. Saves a figure beside the image as drawn/<image>-spectrum.png (-plane-<n>, -box-... when given): the log power with the cutoff circle, and the radial average. Cite that path as evidence. Nothing moves.",
        method="POST",
        path="/app/experiment/analysis/spectrum",
        level="watch",
        router="analysis",
        params={
            "path": "a saved image file path inside the experiment folder",
            "channel": _CHANNEL,
            "plane": arg(
                "optional: a fluorescence stack's plane, 0 the first (default the max projection)",
                int,
            ),
            "box": arg(
                "optional: only this region, [x, y, width, height] in the image's pixels; at least 16 px on its short side",
                List[float],
            ),
            "points": arg(
                "optional: also answer the radial average resampled to this many frequencies, 2 to 1000 (default none)",
                int,
            ),
        },
        optional=("channel", "plane", "box", "points"),
    ),
    ToolSpec(
        name="analysis_measure_thickness",
        returns="Thickness",
        description="A lamella's thickness, read off any saved FIB image (full resolution) with no item needed: the bright band between the trenches, its width at half its peak brightness column by column, in a window width_um along the lamella by height_um across it (default 10 x 4 um) centred at centre ([x, y] in the image's pixels; default the image's centre). For an item's lamella, item_measure_thickness places the window from the item's POI and trench patterns. Answers what item_measure_thickness answers: thickness in nm (mean, median, min, max, by thirds), wedge, angle_deg, straightness, centre_offset_nm, continuity, contrast, other_bands, resolution, uncertainty_nm (one pixel) and warnings (check the image before using the numbers), plus a plot. A top-down estimate. Nothing moves; the result (data.json, window.tif, plot.png) is saved in a thickness folder beside the image, at saved_to, and replaced when the same image is measured again.",
        method="POST",
        path="/app/experiment/analysis/thickness",
        level="watch",
        router="analysis",
        params={
            "path": "a FIB image file path inside the experiment folder",
            "width_um": arg(
                "optional: window along the lamella, um (default 10)", float
            ),
            "height_um": arg(
                "optional: window across the lamella, um (default 4)", float
            ),
            "centre": arg(
                "optional: the window's centre, [x, y] in the image's pixels (default: the image's centre)",
                List[float],
            ),
        },
        optional=("width_um", "height_um", "centre"),
    ),
    ToolSpec(
        name="analysis_measure_shift",
        returns="ImageShift",
        description="The shift of one saved beam image from another, as alignment_align_to_reference measures it: metres and pixels, a score (higher is better) and whether it cleared the minimum. compare also measures it by every method and says how far apart they are in pixels and whether they agree, the check alignment_align_to_reference validates with: a disagreement means the measurement is not to be trusted. Both paths inside the experiment, and both images one size. Works with no microscope connected. Nothing moves.",
        method="POST",
        path="/app/experiment/analysis/shift",
        level="watch",
        router="analysis",
        params={
            "reference": "the saved reference image's path inside the experiment folder",
            "image": "the saved image to measure against it",
            "method": arg(
                "optional: cross-correlation (default), phase-correlation or skimage-phase-correlation",
            ),
            "compare": arg(
                "optional: also measure by every method and say whether they agree (default false)",
                bool,
            ),
            "agreement_px": arg(
                "optional: with compare, the most the methods may differ and agree, in pixels (default 2)",
                float,
            ),
        },
        optional=("method", "compare", "agreement_px"),
    ),
    ToolSpec(
        name="analysis_find_grid_squares",
        returns="ImageMeasured",
        description="A fit of the grid-bar lattice on a saved image (SEM, a tilted FIB view, or a stitched overview), to help with where the bars and square centres are; check it against its overlay and the image. Answers lattice (per direction a, b: step_px between square centres, pitch, bar width, bar angle; px and m in the image plane, so a tilted view's are foreshortened), quality (score, explained, shifts, and the cutoffs that shaped the answer) and warnings. When the fit is not accepted, lattice is null with reason and the best candidate is under candidate, same shape, drawn on the figure: weigh it with your own reading of the image. squares: the limit squares nearest the first point (else the image centre), each with id (r<row>c<col>, only for this answer), centre px and stage_position, in_view (all its film in the image), match (how clearly its bars show, 0 to 1) and extrapolated (outside box). Each of points ([x, y] px, a candidate site) answers bar_distance (to the nearest bar edge, negative on a bar), offset from its square's centre, px and m, and its square. box fits only that region (a tile, the part on the grid); pitch_um prefers candidates whose longer step is near it and says if the fit disagrees. Saves drawn/<image>-grid.png with every square's fitted outline. Nothing moves.",
        method="POST",
        path="/app/experiment/analysis/grid_squares",
        level="watch",
        router="analysis",
        params={
            "path": "a saved image file path inside the experiment folder",
            "points": arg(
                "optional: [[x, y], ...] candidate sites in the image's full-resolution pixels",
                List[List[float]],
            ),
            "limit": arg(
                "optional: how many squares to list, 0 to 200 (default 9)", int
            ),
            "box": arg(
                "optional: fit only this region, [x, y, width, height] in the image's pixels; answers stay in the whole image's pixels",
                List[float],
            ),
            "pitch_um": arg(
                "optional: the expected bar pitch in um if known (about 127 for 200 mesh), compared with the longer step",
                float,
            ),
        },
        optional=("points", "limit", "box", "pitch_um"),
    ),
)
