"""Render an image for export: the picture, its scalebar and legend, and a metadata bar.

The export dialog's preview and the file it saves both come from :func:`render_export`,
so what the dialog shows is what lands on disk. Qt-free for the same reason as
``fibsem.imaging.drawing``, which draws the scalebar and crosshair: it can be scripted,
and tested on CI, where PyQt5 is not installed.

Two steps. :func:`load_export_image` (or one of the ``from_*`` constructors) reads an
image once into an :class:`ExportImage` -- display pixels plus the metadata already
formatted as text. :func:`render_export` then draws it under an :class:`ExportOptions`,
cheaply enough to re-run on every toggle in the dialog.

A value the file does not record is left out rather than shown as "N/A": an image
acquired outside a workflow has no lamella, and a bar that says so is noise.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fibsem.constants import MU_SYMBOL
from fibsem.fm.composite import FMLayer, auto_clim, composite_fm_layers, tint_rgb
from fibsem.fm.preview import is_fluorescence_image, projection_layers
from fibsem.fm.structures import FluorescenceImage
from fibsem.imaging.drawing import _get_font, draw_crosshair, draw_scalebar
from fibsem.structures import BeamType, FibsemImage
from fibsem.util.timestamps import acquisition_datetime_of, format_time

RGB = Tuple[int, int, int]

# The values of PANEL_COLOR, BORDER_COLOR, TEXT_COLOR, TEXT_STRONG_COLOR and
# TEXT_MUTED_COLOR in fibsem/ui/tokens.py, and of the canvas overlays' _TEXT_BG plate.
# Copied rather than imported: importing anything under fibsem.ui imports the widget
# layer, and with it PyQt5, which this module must not need. tests/ui checks they agree.
BAR_COLOR: RGB = (0x1E, 0x20, 0x27)
BORDER_COLOR: RGB = (0x3D, 0x42, 0x51)
TEXT_COLOR: RGB = (0xD6, 0xD6, 0xD6)
TEXT_STRONG_COLOR: RGB = (0xF0, 0xF1, 0xF2)
TEXT_MUTED_COLOR: RGB = (0x86, 0x8E, 0x93)
PLATE_COLOR: RGB = (20, 20, 20)
PLATE_ALPHA = 0.7
# tokens.CURRENT_POSITION_COLOUR: the canvas's "the stage is here" yellow.
CROSSHAIR_COLOR: RGB = (0xFF, 0xEE, 0x58)

# Labelled values on the bar's first row. The detector or objective sits unlabelled
# before them and does not count: past five the row wraps at normal export widths.
MAX_FIELDS = 5

SCALEBAR_LOCATIONS = ("lower right", "lower left")


@dataclass
class ExportField:
    """One value the bar can show, already formatted.

    ``key`` is stable, for options and tests. ``name`` is what the dialog lists it as.
    ``label`` is the short form the bar prints before the value; empty for the
    detector or objective, which read as themselves.
    """

    key: str
    name: str
    label: str
    value: str


@dataclass
class ExportChannel:
    name: str
    color: RGB


@dataclass
class ExportImage:
    """An image ready to render: display pixels and its metadata as text."""

    rgb: np.ndarray  # (H, W, 3) uint8, as acquired
    pixel_size: Optional[float]  # metres; None when the file does not say
    kind: str  # "SEM", "FIB", "FM", or "Image" when the file does not say
    fields: List[ExportField] = field(default_factory=list)
    provenance: List[ExportField] = field(default_factory=list)
    channels: List[ExportChannel] = field(default_factory=list)
    path: Optional[str] = None
    # The greyscale data scaled to [0, 1], for contrast and gamma. None for an image
    # that is colour already -- a fluorescence composite is contrasted per channel,
    # and one min/max over the blend would clip a dim channel to show a bright one.
    gray: Optional[np.ndarray] = None
    # A fluorescence image's channels, each max-projected over z: kept so a hidden
    # channel is re-blended out without reading the stack again.
    layers: List[FMLayer] = field(default_factory=list)

    @property
    def adjustable(self) -> bool:
        return self.gray is not None

    def field_keys(self) -> List[str]:
        return [f.key for f in self.fields]

    def provenance_keys(self) -> List[str]:
        return [f.key for f in self.provenance]


@dataclass
class ExportOptions:
    scalebar: bool = True
    scalebar_location: str = "lower right"
    crosshair: bool = False
    legend: bool = True  # fluorescence only
    fields: Sequence[str] = ()  # keys of ExportImage.fields, in display order
    # Keys of ExportImage.provenance; empty hides the row.
    provenance: Sequence[str] = ()
    scale: int = 1  # 1 or 2: each pixel repeated, never interpolated
    # Display contrast, as the canvas's ContrastGammaControl: limits on the [0, 1]
    # data, then gamma. The defaults leave the image as acquired.
    contrast_min: float = 0.0
    contrast_max: float = 1.0
    gamma: float = 1.0
    # Fluorescence channels left out of the blend and the legend, by index.
    hidden_channels: Sequence[int] = ()

    def contrast_is_default(self) -> bool:
        return (self.contrast_min, self.contrast_max, self.gamma) == (0.0, 1.0, 1.0)


# Which values a new export starts with. Working distance and dwell time are offered
# but off: they matter to whoever reproduces the image, not to whoever reads it.
DEFAULT_FIELDS = (
    "detector",
    "objective",
    "hfw",
    "pixel_size",
    "voltage",
    "current",
    "z",
)
DEFAULT_PROVENANCE = ("item", "task", "date")

# What a short label stands for, where its name alone does not say: the canvas bar
# shows it on hover. A key missing here reads as its field's name.
FIELD_TITLES = {
    "hfw": "Horizontal field width",
    "voltage": "Accelerating voltage",
    "current": "Beam current",
    "detector": "Detector · mode",
    "objective": "Objective · numerical aperture",
}


def default_options(image: ExportImage) -> ExportOptions:
    """Options a new export starts from: the defaults this image has values for."""
    keys = image.field_keys()
    return ExportOptions(fields=[k for k in DEFAULT_FIELDS if k in keys])


# ---------------------------------------------------------------------------
# Formatting


_PREFIXES = {-12: "p", -9: "n", -6: MU_SYMBOL, -3: "m", 0: "", 3: "k", 6: "M"}


def format_si(value: float, unit: str) -> str:
    """Three significant figures and the nearest SI prefix: `150 µm`, `97.7 nm`, `2 kV`.

    Not ``fibsem.utils.format_value``, whose fixed decimal places print `150.00 µm`:
    a figure caption wants the figures that mean something and no more.
    """
    if value == 0:
        return f"0 {unit}"
    exponent = int(math.floor(math.log10(abs(value)) / 3) * 3)
    exponent = max(-12, min(6, exponent))
    text = f"{value / 10**exponent:.3g}"
    if "e" in text and exponent < 6:  # 999.96 rounds up to 1e+03
        exponent += 3
        text = f"{value / 10**exponent:.3g}"
    return f"{text} {_PREFIXES[exponent]}{unit}"


def _to_uint8(data: np.ndarray) -> np.ndarray:
    """Display pixels: (H, W, 3) uint8. A wider type is stretched over its own range."""
    data = np.asarray(data)
    if data.ndim == 3 and data.shape[2] in (3, 4):
        data = data[..., :3]
    if data.dtype != np.uint8:
        data = data.astype(np.float64)
        lo, hi = float(data.min()), float(data.max())
        scale = 255.0 / (hi - lo) if hi > lo else 0.0
        data = ((data - lo) * scale).astype(np.uint8)
    if data.ndim == 2:
        data = np.stack([data] * 3, axis=2)
    return np.ascontiguousarray(data)


def _normalize(data: np.ndarray) -> Optional[np.ndarray]:
    """Greyscale data scaled to [0, 1], as ContrastGammaControl.normalize; None for
    colour data."""
    data = np.asarray(data)
    if data.ndim != 2:
        return None
    f = data.astype(np.float32)
    lo, hi = float(f.min()), float(f.max())
    return (f - lo) / (hi - lo) if hi > lo else np.zeros_like(f)


def adjust_contrast(
    norm: np.ndarray, lo: float = 0.0, hi: float = 1.0, gamma: float = 1.0
) -> np.ndarray:
    """Clip to [lo, hi], rescale to [0, 1], then gamma: the canvas's contrast.

    The same arithmetic as ``ContrastGammaControl.apply``, which this module cannot
    import -- it is a Qt widget. tests/ui checks the two agree, so an image adjusted
    to look the same in both places is the same.
    """
    out = np.clip(norm, lo, hi)
    if hi > lo:
        out = (out - lo) / (hi - lo)
    if gamma != 1.0:
        out = np.power(out, gamma)  # skimage's adjust_gamma, for data in [0, 1]
    return out


def auto_contrast_limits(image: ExportImage) -> Tuple[float, float]:
    """Limits at the 1st and 99th percentiles, as the fluorescence canvas's auto."""
    if image.gray is None:
        return 0.0, 1.0
    return auto_clim(image.gray)


def _visible_channels(image: ExportImage, options: ExportOptions):
    hidden = set(options.hidden_channels)
    return [c for i, c in enumerate(image.channels) if i not in hidden]


def _display_rgb(image: ExportImage, options: ExportOptions) -> np.ndarray:
    hidden = set(options.hidden_channels)
    if image.layers and hidden:
        # Set on every render, so the layers never carry one render's choice into
        # the next. Each layer keeps its own contrast cache across these.
        for i, layer in enumerate(image.layers):
            layer.visible = i not in hidden
        try:
            return composite_fm_layers(image.layers, shape=image.rgb.shape[:2])
        finally:
            for layer in image.layers:
                layer.visible = True
    if image.gray is None or options.contrast_is_default():
        return image.rgb
    lo, hi = options.contrast_min, options.contrast_max
    out = adjust_contrast(image.gray, lo, hi, options.gamma)
    gray = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return np.stack([gray] * 3, axis=2)


# Detector modes as the microscope state panel abbreviates them
# (microscope_state_widget._detector, which this module cannot import; tests/ui
# checks they agree). A mode not listed is shown as reported.
DETECTOR_MODE_ABBREVIATIONS = {
    "SecondaryElectrons": "SE",
    "BackscatteredElectrons": "BSE",
    "SecondaryIons": "SI",
}


# Every field either bar can show: its key, the name a checklist lists it under, and
# the short label printed before its value (empty for those that read as themselves).
# One table, so the export dialog, the canvas bar and its field picker all name a
# field the same way.
FIELD_CATALOGUE: Dict[str, Tuple[str, str]] = {
    "detector": ("Detector", ""),
    "objective": ("Objective", ""),
    "hfw": ("HFW", "HFW"),
    "pixel_size": ("Pixel size", "px"),
    "voltage": ("Voltage", "HV"),
    "current": ("Current", "I"),
    "working_distance": ("Working distance", "WD"),
    "dwell_time": ("Dwell time", "Dwell"),
    "z": ("Z-stack", "Z"),
    "experiment": ("Experiment", ""),
    "item": ("Item", ""),
    "task": ("Task", ""),
    "date": ("Date", ""),
    "instrument": ("Instrument", ""),
    "user": ("User", ""),
    "version": ("Version", ""),
}

# The fields each kind of image can record, in the order a checklist offers them.
BEAM_FIELD_KEYS = (
    "detector",
    "hfw",
    "pixel_size",
    "voltage",
    "current",
    "working_distance",
    "dwell_time",
)
FM_FIELD_KEYS = ("objective", "hfw", "pixel_size", "z")


def field_keys_for(kind: str) -> Tuple[str, ...]:
    """The fields an image of *kind* ("SEM", "FIB", "FM") can record."""
    return FM_FIELD_KEYS if kind == "FM" else BEAM_FIELD_KEYS


def _add(fields: List[ExportField], key: str, value) -> None:
    """Append a field only when there is a value to show."""
    if value is None or value == "":
        return
    name, label = FIELD_CATALOGUE[key]
    fields.append(ExportField(key=key, name=name, label=label, value=str(value)))


def _provenance(experiment, date, instrument, user, version) -> List[ExportField]:
    out: List[ExportField] = []
    _add(out, "experiment", getattr(experiment, "name", None))
    # A lamella in a lamella task, a grid in a grid task. "Item", as the record calls
    # it: the file does not say which kind it is.
    _add(out, "item", getattr(experiment, "item_name", None))
    _add(out, "task", getattr(experiment, "task_name", None))
    _add(out, "date", format_time(date))
    _add(out, "instrument", instrument)
    _add(out, "user", user)
    _add(out, "version", f"fibsem-os {version}" if version else None)
    return out


def _instrument(model: Optional[str], serial: Optional[str]) -> Optional[str]:
    model = (model or "").strip()
    serial = (serial or "").strip()
    if model and serial:
        # No " · " inside: the provenance row uses that between values.
        return f"{model} #{serial}"
    return model or None


# ---------------------------------------------------------------------------
# Reading


@dataclass
class ImageFields:
    """What an image's metadata says, formatted: the bar without the picture.

    The export renders it under the image; the canvas shows it under the view. Built
    from the metadata and the array's shape alone, never its pixels, so it is cheap
    enough to rebuild on every image the canvas is handed.
    """

    kind: str  # "SEM", "FIB", "FM", or "Image" when the file does not say
    fields: List[ExportField] = field(default_factory=list)
    provenance: List[ExportField] = field(default_factory=list)


def image_fields(image) -> ImageFields:
    """The fields of a beam or fluorescence image, without touching its pixels."""
    if isinstance(image, FluorescenceImage):
        return fluorescence_image_fields(image)
    return fibsem_image_fields(image)


def fibsem_image_fields(image: FibsemImage) -> ImageFields:
    """An SEM or FIB image's fields, with whatever its metadata records."""
    md = image.metadata
    if md is None:
        return ImageFields(kind="Image")

    is_ion = md.image_settings.beam_type is BeamType.ION
    state = md.microscope_state
    beam = None
    detector = None
    if state is not None:
        beam = state.ion_beam if is_ion else state.electron_beam
        detector = state.ion_detector if is_ion else state.electron_detector

    pixel_size = md.pixel_size.x if md.pixel_size is not None else None
    fields: List[ExportField] = []
    if detector is not None:
        mode = DETECTOR_MODE_ABBREVIATIONS.get(detector.mode, detector.mode)
        parts = [p for p in (detector.type, mode) if p and p != "Unknown"]
        _add(fields, "detector", " · ".join(parts))
    if pixel_size:
        # From the pixel size, not image_settings.hfw: that is what was asked for,
        # this is what the image is (FIB-482).
        width = np.shape(image.data)[1]
        _add(fields, "hfw", format_si(pixel_size * width, "m"))
        _add(fields, "pixel_size", format_si(pixel_size, "m"))
    if beam is not None:
        if beam.voltage:
            _add(fields, "voltage", format_si(beam.voltage, "V"))
        if beam.beam_current:
            _add(fields, "current", format_si(beam.beam_current, "A"))
        if beam.working_distance:
            # Fixed in mm, not three figures: at the ion beam's 16.5 mm that is a
            # 100 µm step, coarser than focusing moves it.
            wd = f"{beam.working_distance * 1e3:.2f} mm"
            _add(fields, "working_distance", wd)
    if md.image_settings.dwell_time:
        dwell = format_si(md.image_settings.dwell_time, "s")
        _add(fields, "dwell_time", dwell)

    system = md.system_info
    provenance = _provenance(
        md.experiment,
        acquisition_datetime_of(md),
        _instrument(system.model, system.serial_number) if system else None,
        md.user.name if md.user is not None else None,
        system.fibsem_version if system else None,
    )
    return ImageFields(
        kind="FIB" if is_ion else "SEM", fields=fields, provenance=provenance
    )


def z_value(slices: int, step: float, plane: Optional[int] = None) -> str:
    """The Z field: `MIP · 21 × 568 nm`, or `11 of 21 × 568 nm` for one plane.

    *plane* is 0-based, as the FM canvas counts it; None is the max projection.
    """
    step_text = format_si(step, "m")
    if plane is None:
        return f"MIP · {slices} × {step_text}"
    return f"{plane + 1} of {slices} × {step_text}"


def fluorescence_image_fields(image: FluorescenceImage) -> ImageFields:
    """A fluorescence stack's fields. Z reads as the projection the export draws."""
    md = image.metadata
    fields: List[ExportField] = []
    first = md.channels[0] if md.channels else None
    if first is not None and first.objective_magnification:
        objective = f"{first.objective_magnification:g}×"
        if first.objective_numerical_aperture:
            objective += f" · {first.objective_numerical_aperture:g} NA"
        _add(fields, "objective", objective)
    shape = np.shape(image.data)
    if md.pixel_size_x:
        hfw = format_si(md.pixel_size_x * shape[-1], "m")
        _add(fields, "hfw", hfw)
        _add(fields, "pixel_size", format_si(md.pixel_size_x, "m"))
    # Slices counted from the data, not z_positions: some files list a position per
    # channel per slice -- 132 for a 4-channel, 33-slice stack on a real one.
    slices = shape[-3] if len(shape) >= 4 else 1  # (C, Z, Y, X) or (T, C, Z, Y, X)
    if slices > 1 and md.pixel_size_z:
        _add(fields, "z", z_value(slices, md.pixel_size_z))

    system = md.system_info or {}
    provenance = _provenance(
        md.experiment,
        acquisition_datetime_of(md),
        _instrument(system.get("model"), system.get("serial_number")),
        None,
        system.get("fibsem_version"),
    )
    return ImageFields(kind="FM", fields=fields, provenance=provenance)


def from_fibsem_image(image: FibsemImage, path: Optional[str] = None) -> ExportImage:
    """An SEM or FIB image, with whatever its metadata records."""
    rgb = _to_uint8(image.data)
    gray = _normalize(image.data)
    if image.metadata is None:
        return ExportImage(rgb=rgb, pixel_size=None, kind="Image", path=path, gray=gray)
    md = image.metadata
    info = fibsem_image_fields(image)
    return ExportImage(
        rgb=rgb,
        pixel_size=md.pixel_size.x if md.pixel_size is not None else None,
        kind=info.kind,
        fields=info.fields,
        provenance=info.provenance,
        path=path,
        gray=gray,
    )


def from_fluorescence_image(
    image: FluorescenceImage, path: Optional[str] = None
) -> ExportImage:
    """A fluorescence stack: each channel max-projected over z and blended by colour."""
    layers = projection_layers(image)
    rgb = composite_fm_layers(layers)
    if rgb is None:
        raise ValueError("fluorescence image has no displayable channels")

    channels = []
    for layer in layers:
        r, g, b = tint_rgb(layer.color)
        rgb_255 = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
        channels.append(ExportChannel(name=layer.name, color=rgb_255))

    info = fluorescence_image_fields(image)
    return ExportImage(
        rgb=rgb,
        pixel_size=image.metadata.pixel_size_x or None,
        kind=info.kind,
        fields=info.fields,
        provenance=info.provenance,
        channels=channels,
        layers=layers,
        path=path,
    )


def load_export_image(path: str) -> ExportImage:
    """Read an SEM/FIB ``.tif`` or a fluorescence ``.ome.tiff`` for export."""
    path = os.fspath(path)
    if is_fluorescence_image(path):
        return from_fluorescence_image(FluorescenceImage.load(path), path)
    return from_fibsem_image(FibsemImage.load(path), path)


# ---------------------------------------------------------------------------
# Rendering

_MONO_FONT_CANDIDATES = [
    "Menlo.ttc",  # macOS
    "consola.ttf",  # Windows
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Ubuntu/Debian
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",  # RHEL/Fedora
]


@functools.lru_cache(maxsize=16)
def _mono_font(size: int) -> ImageFont.FreeTypeFont:
    """The bar's numbers in a monospace face, like NUMBER_FONT on screen."""
    for path in _MONO_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return _get_font(size)


def _sizes(width: int) -> Dict[str, int]:
    """Annotation sizes as fractions of the image width, so a 6144 px export looks
    like a 768 px one rather than carrying the same 11 px text."""
    font = max(11, round(width * 0.0125))
    return {
        "font": font,
        "margin": max(8, round(width * 0.012)),
        "bar_height": max(3, round(width * 0.004)),
        "line": max(1, round(width / 1000)),
    }


def _text_width(font, text: str) -> int:
    return int(math.ceil(font.getlength(text)))


def _draw_text(draw, x: int, cy: int, text: str, font, color: RGB) -> None:
    """Text vertically centred on ``cy`` by the font's metrics, not the glyphs': so
    "px" and "HFW" sit on one line instead of each centring its own ink."""
    draw.text((x, cy), text, font=font, fill=color, anchor="lm")


def _draw_legend(rgb: np.ndarray, channels: List[ExportChannel], sizes) -> np.ndarray:
    """Channel swatches and names on a plate in the top-right corner."""
    font = _get_font(sizes["font"])
    pad = max(4, sizes["font"] // 2)
    swatch = max(6, round(sizes["font"] * 0.7))
    row = round(sizes["font"] * 1.5)
    text_w = max(_text_width(font, c.name) for c in channels)
    plate_w = pad + swatch + pad + text_w + pad
    plate_h = pad + row * len(channels) + pad // 2

    h, w = rgb.shape[:2]
    x0 = w - sizes["margin"] - plate_w
    y0 = sizes["margin"]

    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Square, like the scalebar's plate from fibsem.imaging.drawing.
    plate = (x0, y0, x0 + plate_w, y0 + plate_h)
    draw.rectangle(plate, fill=(*PLATE_COLOR, int(PLATE_ALPHA * 255)))
    for i, channel in enumerate(channels):
        cy = y0 + pad + row * i + row // 2
        sx = x0 + pad
        swatch_box = (sx, cy - swatch // 2, sx + swatch, cy + swatch // 2)
        draw.rounded_rectangle(swatch_box, max(1, swatch // 4), fill=channel.color)
        _draw_text(draw, sx + swatch + pad, cy, channel.name, font, TEXT_STRONG_COLOR)
    return np.array(Image.alpha_composite(base, overlay).convert("RGB"))


def _bar_items(image: ExportImage, options: ExportOptions):
    """The first row as (kind, text) pieces, in order."""
    by_key = {f.key: f for f in image.fields}
    chosen = [by_key[k] for k in options.fields if k in by_key]
    unlabelled = [f for f in chosen if not f.label]
    labelled = [f for f in chosen if f.label]

    items = [("strong", image.kind)]
    items += [("value", f.value) for f in unlabelled]
    if unlabelled and labelled:
        items.append(("divider", ""))
    items += [("pair", f) for f in labelled]
    return items


def _draw_bar(
    image: ExportImage, options: ExportOptions, width: int, sizes
) -> Image.Image:
    """The metadata bar as its own image, the export's width, to stack under it."""
    fs = sizes["font"]
    strong = _get_font(fs, bold=True)
    label_font = _get_font(fs)
    mono = _mono_font(fs)
    pad_x = fs
    gap = fs
    inner = max(3, round(fs * 0.35))
    line_h = round(fs * 1.9)

    # Lay out first, as (x, line, draw-call) placements, so the height is known
    # before anything is drawn and a long row can wrap rather than run off the edge.
    placements = []
    x, line = pad_x, 0

    def place(w: int, draw_fn) -> None:
        nonlocal x, line
        if x > pad_x and x + w > width - pad_x:
            x, line = pad_x, line + 1
        placements.append((x, line, draw_fn))
        x += w + gap

    def text_fn(text, font, color):
        def fn(draw, x0, cy):
            _draw_text(draw, x0, cy, text, font, color)

        return fn

    for kind, payload in _bar_items(image, options):
        if kind == "strong":
            place(
                _text_width(strong, payload),
                text_fn(payload, strong, TEXT_STRONG_COLOR),
            )
        elif kind == "value":
            place(_text_width(mono, payload), text_fn(payload, mono, TEXT_COLOR))
        elif kind == "divider":

            def divider(draw, x0, cy):
                half = round(fs * 0.6)
                draw.line(
                    [(x0, cy - half), (x0, cy + half)],
                    fill=BORDER_COLOR,
                    width=sizes["line"],
                )

            place(sizes["line"], divider)
        else:
            lw = _text_width(label_font, payload.label)
            vw = _text_width(mono, payload.value)

            def pair(draw, x0, cy, f=payload, lw=lw):
                text_fn(f.label, label_font, TEXT_MUTED_COLOR)(draw, x0, cy)
                text_fn(f.value, mono, TEXT_COLOR)(draw, x0 + lw + inner, cy)

            place(lw + inner + vw, pair)

    by_key = {f.key: f for f in image.provenance}
    chosen = [by_key[k] for k in options.provenance if k in by_key]
    if chosen:
        # Values separated by " · ": names with spaces in them ("my great grid",
        # "SEM Overview") run together when only a gap tells one from the next.
        # A separator that would start a wrapped line is dropped.
        sep = "·"
        sep_w = _text_width(label_font, sep)
        half = gap // 2
        x, line = pad_x, line + 1
        for i, f in enumerate(chosen):
            w = _text_width(label_font, f.value)
            if i > 0:
                if x - half + sep_w + gap + w > width - pad_x:
                    x, line = pad_x, line + 1
                else:
                    x -= half  # the separator sits mid-gap, not a full gap out
                    placements.append(
                        (x, line, text_fn(sep, label_font, TEXT_MUTED_COLOR))
                    )
                    x += sep_w + half + 1
            placements.append((x, line, text_fn(f.value, label_font, TEXT_MUTED_COLOR)))
            x += w + gap

    lines = line + 1
    height = lines * line_h + round(fs * 0.3)
    bar = Image.new("RGB", (width, height), BAR_COLOR)
    draw = ImageDraw.Draw(bar)
    draw.line([(0, 0), (width, 0)], fill=BORDER_COLOR, width=sizes["line"])
    for x0, ln, fn in placements:
        cy = sizes["line"] + ln * line_h + line_h // 2
        fn(draw, x0, cy)
    return bar


def render_export(image: ExportImage, options: ExportOptions) -> np.ndarray:
    """The exported picture: (H, W, 3) uint8, the bar (if any) stacked below the image."""
    rgb = _display_rgb(image, options)
    pixel_size = image.pixel_size
    if options.scale > 1:
        rgb = np.repeat(np.repeat(rgb, options.scale, axis=0), options.scale, axis=1)
        pixel_size = pixel_size / options.scale if pixel_size else pixel_size

    sizes = _sizes(rgb.shape[1])
    if options.crosshair:
        # Arms 2.5% of the width each way: the canvas's 5%-of-the-image crosshair.
        rgb = draw_crosshair(
            rgb,
            color=CROSSHAIR_COLOR,
            alpha=0.8,
            size_ratio=0.025,
            thickness=sizes["line"],
        )
    visible = _visible_channels(image, options)
    if options.legend and visible:
        rgb = _draw_legend(rgb, visible, sizes)
    if options.scalebar and pixel_size:
        rgb = draw_scalebar(
            rgb,
            pixel_size,
            location=options.scalebar_location,
            bg_color=PLATE_COLOR,
            bg_alpha=PLATE_ALPHA,
            margin=sizes["margin"],
            bar_height=sizes["bar_height"],
            font_scale=sizes["font"] / 30,
        )

    if not options.fields and not options.provenance:
        return rgb
    bar = _draw_bar(image, options, rgb.shape[1], sizes)
    return np.concatenate([rgb, np.array(bar)], axis=0)


def export_shape(image: ExportImage, options: ExportOptions) -> Tuple[int, int]:
    """(height, width) of what :func:`render_export` would return, without drawing
    the image -- only the bar, whose height depends on its text."""
    h, w = image.rgb.shape[:2]
    h, w = h * options.scale, w * options.scale
    if options.fields or options.provenance:
        h += _draw_bar(image, options, w, _sizes(w)).height
    return h, w


def save_export(rgb: np.ndarray, path: str) -> str:
    """Write a rendered export. The format follows the extension: .png or .tif(f)."""
    path = os.fspath(path)
    Image.fromarray(rgb).save(path)
    return path


def default_export_name(image: ExportImage, extension: str = ".png") -> str:
    """`<source>_export.png`, beside the source; `export.png` with no source."""
    if image.path is None:
        return f"export{extension}"
    directory, name = os.path.split(image.path)
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return os.path.join(directory, f"{name}_export{extension}")
