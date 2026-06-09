"""
fibsemOS MCP Server

Exposes fibsem microscope control as Claude-callable tools via the
Model Context Protocol (MCP). Run with:

    python -m fibsem.mcp.server

or reference the file directly in Claude Code settings.json.
"""

import logging
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import FibsemImage, MicroscopeSettings

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger("fibsem.mcp")

mcp = FastMCP("fibsem")

# Global microscope state — persists across tool calls within a server session
_microscope: Optional["FibsemMicroscope"] = None
_settings: Optional["MicroscopeSettings"] = None
_last_image: Optional["FibsemImage"] = (
    None  # last acquired FibsemImage, used by pick_point
)


def _check_microscope() -> Optional[str]:
    """Return an error string if no microscope is connected, else None."""
    if _microscope is None:
        return "Error: no microscope connected — call connect_microscope first."
    return None


def _bt(beam_type: str):
    """Convert 'electron'/'ion' string to BeamType enum."""
    from fibsem.structures import BeamType

    return BeamType.ELECTRON if beam_type.lower() == "electron" else BeamType.ION


def _save_preview_png(image, path: str, filename: str) -> str:
    """Save a normalised uint8 PNG alongside the TIFF and return the PNG path."""
    import numpy as np
    from PIL import Image as PILImage

    data = image.data.copy()
    if data.dtype != np.uint8:
        d_min, d_max = int(data.min()), int(data.max())
        if d_max > d_min:
            data = ((data.astype(np.float32) - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)
    png_path = str(Path(path) / f"{filename}.png")
    PILImage.fromarray(data, mode="L" if data.ndim == 2 else "RGB").save(png_path)
    return png_path


def _image_to_b64_jpeg(image, max_width: int = 768) -> str:
    """Convert a FibsemImage to a base64 JPEG string (downsampled to max_width)."""
    import base64
    import io
    import numpy as np
    from PIL import Image as PILImage

    data = image.data.copy()
    if data.dtype != np.uint8:
        d_min, d_max = int(data.min()), int(data.max())
        if d_max > d_min:
            data = ((data.astype(np.float32) - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)
    pil = PILImage.fromarray(data, mode="L" if data.ndim == 2 else "RGB")
    if pil.width > max_width:
        scale = max_width / pil.width
        pil = pil.resize((max_width, int(pil.height * scale)), PILImage.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def _image_to_jpeg_content(image):
    """Convert a FibsemImage to an MCP ImageContent block."""
    from mcp.types import ImageContent
    return ImageContent(type="image", data=_image_to_b64_jpeg(image), mimeType="image/jpeg")


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


@mcp.tool()
def connect_microscope(
    manufacturer: str = "Demo", ip_address: str = "localhost"
) -> str:
    """Connect to a FIB/SEM microscope (in-process).

    Args:
        manufacturer: Microscope manufacturer — 'Demo' (no hardware needed),
            'ThermoFisher', or 'TESCAN'
        ip_address: IP address of the microscope PC (ignored for Demo)
    """
    global _microscope, _settings
    from fibsem import utils

    log.info("Connecting to %s at %s", manufacturer, ip_address)
    _microscope, _settings = utils.setup_session(
        manufacturer=manufacturer, ip_address=ip_address
    )
    return f"Connected to {manufacturer} microscope at {ip_address}."


@mcp.tool()
def connect_microscope_remote(host: str = "localhost", port: int = 8001) -> str:
    """Connect to a FIB/SEM microscope via a remote FibsemServer (REST API).
    Use this when the microscope PC is running FibsemServer and Claude Code
    is on a different machine.

    Args:
        host: IP address or hostname of the machine running FibsemServer
        port: Port FibsemServer is listening on (default 8001)
    """
    global _microscope, _settings
    from fibsem.server.client import FibsemClient

    log.info("Connecting to remote FibsemServer at %s:%d", host, port)
    _microscope = FibsemClient(host=host, port=port)
    _settings = None
    manufacturer = (
        _microscope.system.info.manufacturer if _microscope.system else "unknown"
    )
    dashboard = _microscope.dashboard_url if isinstance(_microscope, FibsemClient) else ""
    return f"Connected to remote FibsemServer at {host}:{port} (manufacturer: {manufacturer}).\n  Dashboard: {dashboard}"


# ---------------------------------------------------------------------------
# State observation
# ---------------------------------------------------------------------------


@mcp.tool()
def get_stage_position() -> str:
    """Get the current stage position (x, y, z in mm; rotation and tilt in degrees)."""
    if err := _check_microscope():
        return err

    pos = _microscope.get_stage_position()
    lines = [
        "Stage position:",
        f"  x        = {pos.x * 1e3:.4f} mm"
        if pos.x is not None
        else "  x        = unknown",
        f"  y        = {pos.y * 1e3:.4f} mm"
        if pos.y is not None
        else "  y        = unknown",
        f"  z        = {pos.z * 1e3:.4f} mm"
        if pos.z is not None
        else "  z        = unknown",
        f"  rotation = {math.degrees(pos.r):.2f}°"
        if pos.r is not None
        else "  rotation = unknown",
        f"  tilt     = {math.degrees(pos.t):.2f}°"
        if pos.t is not None
        else "  tilt     = unknown",
    ]
    return "\n".join(lines)


@mcp.tool()
def get_microscope_state() -> str:
    """Get the full microscope state: stage position, electron and ion beam settings,
    and detector settings for both beams."""
    if err := _check_microscope():
        return err

    state = _microscope.get_microscope_state()

    def _beam_lines(label: str, b, d) -> list:
        lines = [f"\n{label} beam:"]
        if b is not None:
            lines += [
                f"  voltage          = {b.voltage / 1e3:.1f} kV"
                if b.voltage is not None
                else "  voltage          = unknown",
                f"  current          = {b.beam_current * 1e9:.2f} nA"
                if b.beam_current is not None
                else "  current          = unknown",
                f"  working_distance = {b.working_distance * 1e3:.3f} mm"
                if b.working_distance is not None
                else "  working_distance = unknown",
                f"  hfw              = {b.hfw * 1e6:.2f} µm"
                if b.hfw is not None
                else "  hfw              = unknown",
            ]
        if d is not None:
            lines += [
                f"  detector_type    = {d.type}",
                f"  detector_mode    = {d.mode}",
                f"  brightness       = {d.brightness:.2f}",
                f"  contrast         = {d.contrast:.2f}",
            ]
        return lines

    pos = state.stage_position
    lines = [
        "Stage position:",
        f"  x        = {pos.x * 1e3:.4f} mm"
        if pos.x is not None
        else "  x        = unknown",
        f"  y        = {pos.y * 1e3:.4f} mm"
        if pos.y is not None
        else "  y        = unknown",
        f"  z        = {pos.z * 1e3:.4f} mm"
        if pos.z is not None
        else "  z        = unknown",
        f"  rotation = {math.degrees(pos.r):.2f}°"
        if pos.r is not None
        else "  rotation = unknown",
        f"  tilt     = {math.degrees(pos.t):.2f}°"
        if pos.t is not None
        else "  tilt     = unknown",
    ]
    lines += _beam_lines("Electron", state.electron_beam, state.electron_detector)
    lines += _beam_lines("Ion", state.ion_beam, state.ion_detector)
    return "\n".join(lines)


@mcp.tool()
def get_beam_settings(beam_type: str = "electron") -> str:
    """Get the current beam settings for the electron or ion beam.

    Args:
        beam_type: 'electron' or 'ion'
    """
    if err := _check_microscope():
        return err

    b = _microscope.get_beam_settings(_bt(beam_type))
    lines = [
        f"{beam_type.upper()} beam settings:",
        f"  voltage          = {b.voltage / 1e3:.1f} kV"
        if b.voltage is not None
        else "  voltage          = unknown",
        f"  current          = {b.beam_current * 1e9:.2f} nA"
        if b.beam_current is not None
        else "  current          = unknown",
        f"  working_distance = {b.working_distance * 1e3:.3f} mm"
        if b.working_distance is not None
        else "  working_distance = unknown",
        f"  hfw              = {b.hfw * 1e6:.2f} µm"
        if b.hfw is not None
        else "  hfw              = unknown",
        f"  scan_rotation    = {math.degrees(b.scan_rotation):.2f}°"
        if b.scan_rotation is not None
        else "  scan_rotation    = unknown",
        f"  shift            = ({b.shift.x * 1e6:.2f}, {b.shift.y * 1e6:.2f}) µm"
        if b.shift is not None
        else "  shift            = unknown",
        f"  stigmation       = ({b.stigmation.x:.4f}, {b.stigmation.y:.4f})"
        if b.stigmation is not None
        else "  stigmation       = unknown",
    ]
    return "\n".join(lines)


@mcp.tool()
def get_stage_orientation() -> str:
    """Get the current stage orientation — one of: 'SEM', 'FIB', 'MILLING', 'FM', 'NONE'."""
    if err := _check_microscope():
        return err

    orientation = _microscope.get_stage_orientation()
    return f"Stage orientation: {orientation}"


@mcp.tool()
def get_milling_angle() -> str:
    """Get the current milling angle in degrees based on stage tilt, pretilt, and ion column tilt."""
    if err := _check_microscope():
        return err

    angle = _microscope.get_current_milling_angle()
    return f"Current milling angle: {angle:.2f}°"


@mcp.tool()
def get_milling_state() -> str:
    """Get the current milling state (IDLE, RUNNING, PAUSED, etc.)."""
    if err := _check_microscope():
        return err

    state = _microscope.get_milling_state()
    return f"Milling state: {state.name}"


# ---------------------------------------------------------------------------
# Imaging
# ---------------------------------------------------------------------------


@mcp.tool()
def acquire_image(
    beam_type: str = "electron",
    hfw_um: float = 150.0,
    resolution_x: int = 1536,
    resolution_y: int = 1024,
    dwell_time_us: float = 1.0,
    autocontrast: bool = False,
    save: bool = False,
    path: str = "",
    filename: str = "mcp_image",
) -> "str | list":
    """Acquire a microscope image.

    Args:
        beam_type: 'electron' for SEM or 'ion' for FIB/ion beam
        hfw_um: Horizontal field width in micrometers — smaller = higher
            magnification (e.g. 10 µm for high-mag, 500 µm for overview)
        resolution_x: Image width in pixels (e.g. 1536, 3072)
        resolution_y: Image height in pixels (e.g. 1024, 2048)
        dwell_time_us: Dwell time per pixel in microseconds — higher = less
            noise but slower (typical range 0.1–10 µs)
        autocontrast: Auto-adjust brightness/contrast before capture
        save: Save image to disk as a TIFF file
        path: Directory path for saving (required if save=True)
        filename: Base filename without extension (beam suffix added automatically)
    """
    if err := _check_microscope():
        return err

    from fibsem import acquire
    from fibsem.structures import ImageSettings

    image_settings = ImageSettings(
        resolution=(resolution_x, resolution_y),
        dwell_time=dwell_time_us * 1e-6,
        hfw=hfw_um * 1e-6,
        beam_type=_bt(beam_type),
        autocontrast=autocontrast,
        save=save,
        path=Path(path) if path else None,
        filename=filename,
    )

    log.info(
        "Acquiring %s image: hfw=%.1f µm, res=%dx%d",
        beam_type,
        hfw_um,
        resolution_x,
        resolution_y,
    )
    image = acquire.acquire_image(_microscope, image_settings)

    global _last_image
    _last_image = image

    px = image.metadata.pixel_size if image.metadata else None
    px_str = f"{px.x * 1e9:.2f} nm" if px else "unknown"
    save_str = f", saved to {path}/{filename}" if save else ""
    text = (
        f"Acquired {beam_type.upper()} image: shape={image.data.shape}, "
        f"hfw={hfw_um} µm, pixel_size={px_str}{save_str}"
    )

    from mcp.types import ImageContent, TextContent

    if save:
        png_path = _save_preview_png(image, path, f"{filename}_preview")
        return f"{text}\npreview: {png_path}"
    b64 = _image_to_b64_jpeg(image)
    return [TextContent(type="text", text=text), ImageContent(type="image", data=b64, mimeType="image/jpeg")]


@mcp.tool()
def acquire_both_beams(
    hfw_um: float = 150.0,
    resolution_x: int = 1536,
    resolution_y: int = 1024,
    dwell_time_us: float = 1.0,
    autocontrast: bool = False,
    save: bool = False,
    path: str = "",
    filename: str = "mcp_reference",
) -> "str | list":
    """Acquire a reference pair: one SEM (electron) and one FIB (ion) image.

    Args:
        hfw_um: Horizontal field width in micrometers
        resolution_x: Image width in pixels
        resolution_y: Image height in pixels
        dwell_time_us: Dwell time per pixel in microseconds
        autocontrast: Auto-adjust brightness/contrast before capture
        save: Save images to disk
        path: Directory path for saving
        filename: Base filename (beam suffix _eb/_ib added automatically)
    """
    if err := _check_microscope():
        return err

    from fibsem import acquire
    from fibsem.structures import ImageSettings
    from mcp.types import TextContent

    image_settings = ImageSettings(
        resolution=(resolution_x, resolution_y),
        dwell_time=dwell_time_us * 1e-6,
        hfw=hfw_um * 1e-6,
        autocontrast=autocontrast,
        save=save,
        path=Path(path) if path else None,
        filename=filename,
    )

    eb, ib = acquire.take_reference_images(_microscope, image_settings)

    from mcp.types import ImageContent

    suffixes = {"SEM (electron)": "_eb", "FIB (ion)": "_ib"}
    content = []
    for label, img in [("SEM (electron)", eb), ("FIB (ion)", ib)]:
        if img is not None:
            px = img.metadata.pixel_size if img.metadata else None
            px_str = f"{px.x * 1e9:.2f} nm" if px else "unknown"
            if save:
                beam_filename = f"{filename}{suffixes[label]}"
                png_path = _save_preview_png(img, path, f"{beam_filename}_preview")
                text = f"{label}: shape={img.data.shape}, pixel_size={px_str}, saved to {path}/{beam_filename}\npreview: {png_path}"
            else:
                text = f"{label}: shape={img.data.shape}, pixel_size={px_str}"
            content.append(TextContent(type="text", text=text))
            if not save:
                b64 = _image_to_b64_jpeg(img)
                content.append(ImageContent(type="image", data=b64, mimeType="image/jpeg"))
        else:
            content.append(TextContent(type="text", text=f"{label}: not acquired"))

    return content


@mcp.tool()
def auto_focus(beam_type: str = "electron") -> str:
    """Auto-focus the beam by adjusting working distance.

    Args:
        beam_type: 'electron' or 'ion'
    """
    if err := _check_microscope():
        return err

    _microscope.auto_focus(_bt(beam_type))
    wd = _microscope.get_working_distance(_bt(beam_type))
    return f"Auto-focus complete. Working distance: {wd * 1e3:.3f} mm"


@mcp.tool()
def autocontrast(beam_type: str = "electron") -> str:
    """Auto-adjust brightness and contrast for the beam.

    Args:
        beam_type: 'electron' or 'ion'
    """
    if err := _check_microscope():
        return err

    _microscope.autocontrast(_bt(beam_type))
    return f"Autocontrast complete for {beam_type.upper()} beam."


# ---------------------------------------------------------------------------
# Beam control
# ---------------------------------------------------------------------------


@mcp.tool()
def set_field_of_view(hfw_um: float, beam_type: str = "electron") -> str:
    """Set the horizontal field width (magnification) for a beam.

    Args:
        hfw_um: Horizontal field width in micrometers
        beam_type: 'electron' or 'ion'
    """
    if err := _check_microscope():
        return err

    _microscope.set_field_of_view(hfw_um * 1e-6, _bt(beam_type))
    actual = _microscope.get_field_of_view(_bt(beam_type))
    return f"{beam_type.upper()} field of view set to {actual * 1e6:.2f} µm"


@mcp.tool()
def beam_on_off(action: str, beam_type: str = "electron") -> str:
    """Turn a beam on or off.

    Args:
        action: 'on' to turn the beam on, 'off' to turn it off
        beam_type: 'electron' or 'ion'
    """
    if err := _check_microscope():
        return err

    bt = _bt(beam_type)
    if action.lower() == "on":
        _microscope.turn_on(bt)
        state = "on"
    else:
        _microscope.turn_off(bt)
        state = "off"
    return f"{beam_type.upper()} beam turned {state}."


@mcp.tool()
def blank_unblank(action: str, beam_type: str = "electron") -> str:
    """Blank or unblank a beam to protect the sample between acquisitions.

    Args:
        action: 'blank' to blank the beam, 'unblank' to restore it
        beam_type: 'electron' or 'ion'
    """
    if err := _check_microscope():
        return err

    bt = _bt(beam_type)
    if action.lower() == "blank":
        _microscope.blank(bt)
        state = "blanked"
    else:
        _microscope.unblank(bt)
        state = "unblanked"
    return f"{beam_type.upper()} beam {state}."


@mcp.tool()
def reset_beam_shifts() -> str:
    """Reset beam shifts to zero for both electron and ion beams."""
    if err := _check_microscope():
        return err

    _microscope.reset_beam_shifts()
    return "Beam shifts reset to zero for both beams."


# ---------------------------------------------------------------------------
# Stage movement
# ---------------------------------------------------------------------------


@mcp.tool()
def move_stage(
    x_mm: Optional[float] = None,
    y_mm: Optional[float] = None,
    z_mm: Optional[float] = None,
    rotation_deg: Optional[float] = None,
    tilt_deg: Optional[float] = None,
) -> str:
    """Move the stage to an absolute position. Omit any axis to leave it unchanged.

    Args:
        x_mm: Target X position in millimeters
        y_mm: Target Y position in millimeters
        z_mm: Target Z position in millimeters
        rotation_deg: Target rotation in degrees
        tilt_deg: Target tilt in degrees
    """
    if err := _check_microscope():
        return err

    from fibsem.structures import FibsemStagePosition

    current = _microscope.get_stage_position()
    target = FibsemStagePosition(
        x=x_mm * 1e-3 if x_mm is not None else current.x,
        y=y_mm * 1e-3 if y_mm is not None else current.y,
        z=z_mm * 1e-3 if z_mm is not None else current.z,
        r=math.radians(rotation_deg) if rotation_deg is not None else current.r,
        t=math.radians(tilt_deg) if tilt_deg is not None else current.t,
        coordinate_system=current.coordinate_system,
    )

    log.info(
        "Moving stage to x=%.4f y=%.4f z=%.4f mm",
        target.x * 1e3,
        target.y * 1e3,
        target.z * 1e3,
    )
    result = _microscope.move_stage_absolute(target)
    return (
        f"Stage moved to: x={result.x * 1e3:.4f} mm, y={result.y * 1e3:.4f} mm, "
        f"z={result.z * 1e3:.4f} mm"
    )


@mcp.tool()
def move_stage_relative(
    dx_mm: float = 0.0,
    dy_mm: float = 0.0,
    dz_mm: float = 0.0,
    drotation_deg: float = 0.0,
    dtilt_deg: float = 0.0,
) -> str:
    """Move the stage by a relative offset from its current position.

    Args:
        dx_mm: X offset in millimeters
        dy_mm: Y offset in millimeters
        dz_mm: Z offset in millimeters
        drotation_deg: Rotation offset in degrees
        dtilt_deg: Tilt offset in degrees
    """
    if err := _check_microscope():
        return err

    from fibsem.structures import FibsemStagePosition

    delta = FibsemStagePosition(
        x=dx_mm * 1e-3,
        y=dy_mm * 1e-3,
        z=dz_mm * 1e-3,
        r=math.radians(drotation_deg),
        t=math.radians(dtilt_deg),
    )

    result = _microscope.move_stage_relative(delta)
    return (
        f"Stage moved to: x={result.x * 1e3:.4f} mm, y={result.y * 1e3:.4f} mm, "
        f"z={result.z * 1e3:.4f} mm"
    )


@mcp.tool()
def move_to_milling_angle(milling_angle_deg: float = 15.0) -> str:
    """Move the stage to achieve a target milling angle (moves tilt axis).

    Args:
        milling_angle_deg: Target milling angle in degrees (default 15°, typical range 7-15° for cryo-lamella)
    """
    if err := _check_microscope():
        return err

    success = _microscope.move_to_milling_angle(math.radians(milling_angle_deg))
    actual = _microscope.get_current_milling_angle()
    status = "OK" if success else "WARNING: may not be at target angle"
    return f"Moved to milling angle. Actual: {actual:.2f}° ({status})"


@mcp.tool()
def move_flat_to_beam(beam_type: str = "electron") -> str:
    """Tilt the stage so the sample surface is flat/perpendicular to the beam.

    Args:
        beam_type: 'electron' (flat to SEM) or 'ion' (flat to FIB)
    """
    if err := _check_microscope():
        return err

    _microscope.move_flat_to_beam(_bt(beam_type))
    pos = _microscope.get_stage_position()
    tilt_str = f"{math.degrees(pos.t):.2f}°" if pos.t is not None else "unknown"
    return f"Stage moved flat to {beam_type.upper()} beam. Tilt: {tilt_str}"


@mcp.tool()
def link_stage() -> str:
    """Link the stage Z to the working distance (synchronises focus with stage height).
    Should be called when at eucentric height before milling."""
    if err := _check_microscope():
        return err

    _microscope.link_stage()
    return "Stage linked to working distance."


# ---------------------------------------------------------------------------
# Milling
# ---------------------------------------------------------------------------


@mcp.tool()
def setup_milling(
    milling_current_na: float = 0.02,
    milling_voltage_kv: float = 30.0,
    application_file: str = "Si",
    patterning_mode: str = "Serial",
) -> str:
    """Configure the ion beam for milling. Call this before draw_rectangle and run_milling.

    Args:
        milling_current_na: Ion beam current in nanoamps (e.g. 0.02 = 20 pA for fine milling,
            1.0 nA for fast bulk removal)
        milling_voltage_kv: Ion beam voltage in kilovolts (default 30 kV)
        application_file: Milling application file — 'Si' for silicon, 'C' for carbon
            (ThermoFisher only; ignored on TESCAN)
        patterning_mode: 'Serial' (one pattern at a time) or 'Parallel' (simultaneous)
    """
    if err := _check_microscope():
        return err

    from fibsem.structures import FibsemMillingSettings

    mill_settings = FibsemMillingSettings(
        milling_current=milling_current_na * 1e-9,
        milling_voltage=milling_voltage_kv * 1e3,
        application_file=application_file,
        patterning_mode=patterning_mode,
    )
    _microscope.setup_milling(mill_settings)
    return (
        f"Milling configured: current={milling_current_na} nA, "
        f"voltage={milling_voltage_kv} kV, application={application_file}, "
        f"mode={patterning_mode}"
    )


@mcp.tool()
def draw_rectangle(
    width_um: float = 10.0,
    height_um: float = 5.0,
    depth_um: float = 1.0,
    centre_x_um: float = 0.0,
    centre_y_um: float = 0.0,
    rotation_deg: float = 0.0,
    cleaning_cross_section: bool = False,
) -> str:
    """Draw a rectangle milling pattern on the ion beam. Call setup_milling first.
    Positions are offsets from the image centre.

    Args:
        width_um: Pattern width in micrometers
        height_um: Pattern height in micrometers
        depth_um: Milling depth in micrometers
        centre_x_um: X offset from image centre in micrometers (positive = right)
        centre_y_um: Y offset from image centre in micrometers (positive = down)
        rotation_deg: Pattern rotation in degrees
        cleaning_cross_section: If True, use cleaning cross-section mode (polishing pass)
    """
    if err := _check_microscope():
        return err

    from fibsem.structures import FibsemRectangleSettings

    pattern = FibsemRectangleSettings(
        width=width_um * 1e-6,
        height=height_um * 1e-6,
        depth=depth_um * 1e-6,
        centre_x=centre_x_um * 1e-6,
        centre_y=centre_y_um * 1e-6,
        rotation=math.radians(rotation_deg),
        cleaning_cross_section=cleaning_cross_section,
    )
    _microscope.draw_patterns([pattern])
    return (
        f"Rectangle drawn: {width_um} × {height_um} µm, depth={depth_um} µm, "
        f"centre=({centre_x_um}, {centre_y_um}) µm, rotation={rotation_deg}°"
    )


@mcp.tool()
def draw_line(
    start_x_um: float = 0.0,
    start_y_um: float = -5.0,
    end_x_um: float = 0.0,
    end_y_um: float = 5.0,
    depth_um: float = 1.0,
) -> str:
    """Draw a line milling pattern on the ion beam. Call setup_milling first.
    Positions are offsets from the image centre.

    Args:
        start_x_um: Start X position in micrometers (offset from image centre)
        start_y_um: Start Y position in micrometers (offset from image centre)
        end_x_um: End X position in micrometers (offset from image centre)
        end_y_um: End Y position in micrometers (offset from image centre)
        depth_um: Milling depth in micrometers
    """
    if err := _check_microscope():
        return err

    from fibsem.structures import FibsemLineSettings

    pattern = FibsemLineSettings(
        start_x=start_x_um * 1e-6,
        start_y=start_y_um * 1e-6,
        end_x=end_x_um * 1e-6,
        end_y=end_y_um * 1e-6,
        depth=depth_um * 1e-6,
    )
    _microscope.draw_patterns([pattern])
    return (
        f"Line drawn: ({start_x_um}, {start_y_um}) → ({end_x_um}, {end_y_um}) µm, "
        f"depth={depth_um} µm"
    )


@mcp.tool()
def draw_circle(
    centre_x_um: float = 0.0,
    centre_y_um: float = 0.0,
    radius_um: float = 5.0,
    depth_um: float = 1.0,
    start_angle_deg: float = 0.0,
    end_angle_deg: float = 360.0,
    thickness_um: float = 0.0,
) -> str:
    """Draw a circle (or arc/annulus) milling pattern on the ion beam. Call setup_milling first.
    Positions are offsets from the image centre.

    Args:
        centre_x_um: Circle centre X in micrometers (offset from image centre)
        centre_y_um: Circle centre Y in micrometers (offset from image centre)
        radius_um: Circle radius in micrometers
        depth_um: Milling depth in micrometers
        start_angle_deg: Start angle in degrees (0 = right, 90 = up)
        end_angle_deg: End angle in degrees — set to <360 for an arc
        thickness_um: Annulus thickness in micrometers (0 = solid disk)
    """
    if err := _check_microscope():
        return err

    from fibsem.structures import FibsemCircleSettings

    pattern = FibsemCircleSettings(
        centre_x=centre_x_um * 1e-6,
        centre_y=centre_y_um * 1e-6,
        radius=radius_um * 1e-6,
        depth=depth_um * 1e-6,
        start_angle=start_angle_deg,
        end_angle=end_angle_deg,
        thickness=thickness_um * 1e-6,
    )
    _microscope.draw_patterns([pattern])
    return (
        f"Circle drawn: centre=({centre_x_um}, {centre_y_um}) µm, "
        f"radius={radius_um} µm, depth={depth_um} µm, "
        f"arc={start_angle_deg}°→{end_angle_deg}°"
    )


@mcp.tool()
def clear_patterns() -> str:
    """Clear all milling patterns from the ion beam without restoring beam settings.
    Use this to reset patterns before redrawing. Use finish_milling to also restore
    imaging beam conditions."""
    if err := _check_microscope():
        return err

    _microscope.clear_patterns()
    return "All milling patterns cleared."


@mcp.tool()
def run_milling(
    milling_current_na: float = 0.02,
    milling_voltage_kv: float = 30.0,
    asynch: bool = False,
) -> str:
    """Run the ion beam milling with currently drawn patterns. Blocks until complete unless asynch=True.

    Args:
        milling_current_na: Ion beam current in nanoamps — should match setup_milling
        milling_voltage_kv: Ion beam voltage in kilovolts
        asynch: If True, return immediately without waiting for milling to finish
    """
    if err := _check_microscope():
        return err

    log.info(
        "Running milling: %.3f nA, %.1f kV", milling_current_na, milling_voltage_kv
    )
    _microscope.run_milling(
        milling_current=milling_current_na * 1e-9,
        milling_voltage=milling_voltage_kv * 1e3,
        asynch=asynch,
    )
    return f"Milling complete ({milling_current_na} nA, {milling_voltage_kv} kV)."


@mcp.tool()
def finish_milling(
    imaging_current_na: float = 0.1,
    imaging_voltage_kv: float = 2.0,
) -> str:
    """Finalise milling: clear patterns and restore the ion beam to imaging settings.
    Call this after run_milling.

    Args:
        imaging_current_na: Ion beam current to restore for imaging, in nanoamps
        imaging_voltage_kv: Ion beam voltage to restore for imaging, in kilovolts
    """
    if err := _check_microscope():
        return err

    _microscope.finish_milling(
        imaging_current=imaging_current_na * 1e-9,
        imaging_voltage=imaging_voltage_kv * 1e3,
    )
    return f"Milling finished. Beam restored to {imaging_current_na} nA, {imaging_voltage_kv} kV."


@mcp.tool()
def stop_milling() -> str:
    """Stop any currently running milling operation."""
    if err := _check_microscope():
        return err

    _microscope.stop_milling()
    return "Milling stopped."


@mcp.tool()
def estimate_milling_time() -> str:
    """Estimate the time remaining for the current milling patterns."""
    if err := _check_microscope():
        return err

    t = _microscope.estimate_milling_time()
    return f"Estimated milling time: {t:.1f} s ({t / 60:.1f} min)"


@mcp.tool()
def run_milling_task(config_json: str) -> str:
    """Run a complete milling task from a JSON configuration.
    Handles setup, pattern drawing, milling, and cleanup in one call.
    Supports multiple stages with optional beam-shift alignment.

    Args:
        config_json: JSON string matching the FibsemMillingTaskConfig schema.
            Required keys: "name" (str), "stages" (list of stage dicts).
            Each stage dict has "milling" (FibsemMillingSettings fields) and
            "pattern" (pattern type + dimensions).
            Optional keys: "field_of_view" (float, metres), "channel" ("ION"/"ELECTRON"),
            "alignment" (alignment settings), "acquisition" (imaging settings).

    Example config_json for a polish pass:
        {
          "name": "Polish",
          "field_of_view": 60e-6,
          "stages": [{
            "milling": {"milling_current": 60e-12, "application_file": "Si-ccs"},
            "pattern": {
              "name": "Trench",
              "width": 9e-6, "depth": 1e-6,
              "upper_trench_height": 0.7e-6, "lower_trench_height": 0.7e-6,
              "spacing": 0.3e-6
            }
          }]
        }
    """
    if err := _check_microscope():
        return err

    import json
    from fibsem.milling.tasks import (
        FibsemMillingTaskConfig,
        run_milling_task as _run_task,
    )

    try:
        config_dict = json.loads(config_json)
    except json.JSONDecodeError as exc:
        return f"Error: invalid JSON — {exc}"

    try:
        config = FibsemMillingTaskConfig.from_dict(config_dict)
    except Exception as exc:
        return f"Error: could not parse milling config — {exc}"

    n_stages = len(config.stages)
    est = config.estimated_time if hasattr(config, "estimated_time") else None
    est_str = f", estimated time: {est:.0f} s" if est else ""
    log.info(
        "Running milling task '%s' with %d stage(s)%s", config.name, n_stages, est_str
    )

    try:
        _run_task(_microscope, config)
    except Exception as exc:
        return f"Error during milling task '{config.name}': {exc}"

    return f"Milling task '{config.name}' complete ({n_stages} stage(s){est_str})."


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


@mcp.tool()
def load_image(path: str) -> "str | list":
    """Load a FibsemImage from a TIFF file on disk and return it for inspection.
    Works without a connected microscope — use this to review previously saved images.

    Args:
        path: Absolute or relative path to a .tif / .tiff file saved by fibsem
    """
    from fibsem.structures import FibsemImage
    from mcp.types import TextContent
    import math

    try:
        image = FibsemImage.load(path)
    except FileNotFoundError:
        return f"Error: file not found — {path}"
    except Exception as exc:
        return f"Error loading image: {exc}"

    md = image.metadata
    lines = [
        f"Loaded image from: {path}",
        f"  shape      = {image.data.shape}",
        f"  dtype      = {image.data.dtype}",
    ]

    if md is not None:
        px = md.pixel_size
        if px is not None:
            lines.append(f"  pixel_size = ({px.x * 1e9:.2f}, {px.y * 1e9:.2f}) nm")
        ims = md.image_settings
        if ims is not None:
            lines.append(
                f"  beam_type  = {ims.beam_type.name if ims.beam_type else 'unknown'}"
            )
            if ims.hfw is not None:
                lines.append(f"  hfw        = {ims.hfw * 1e6:.2f} µm")
            if ims.dwell_time is not None:
                lines.append(f"  dwell_time = {ims.dwell_time * 1e6:.2f} µs")
        try:
            acq_date = md.acquisition_date
            lines.append(f"  acquired   = {acq_date.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception:
            pass
        sp = md.stage_position
        if sp is not None and all(v is not None for v in [sp.x, sp.y, sp.z, sp.t]):
            lines.append(
                f"  stage      = x={sp.x * 1e3:.4f} mm, y={sp.y * 1e3:.4f} mm, "
                f"z={sp.z * 1e3:.4f} mm, tilt={math.degrees(sp.t):.2f}°"
            )
    else:
        lines.append("  metadata   = not available")

    return [TextContent(type="text", text="\n".join(lines)), _image_to_jpeg_content(image)]


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@mcp.tool()
def get_dashboard_url() -> str:
    """Return the URL of the live fibsemOS dashboard.
    Only available in remote mode (when connected via connect_microscope_remote).
    Open this in VSCode Simple Browser: View → Open Simple Browser → paste URL."""
    from fibsem.server.client import FibsemClient
    if _microscope is None:
        return "No microscope connected."
    if not isinstance(_microscope, FibsemClient):
        return (
            "Dashboard is only available in remote mode (connect_microscope_remote). "
            "In direct mode, images are returned inline in the chat."
        )
    return f"fibsemOS dashboard: {_microscope.dashboard_url}\nOpen in VSCode Simple Browser: View → Open Simple Browser → paste URL."


# ---------------------------------------------------------------------------
# Interactive point selection
# ---------------------------------------------------------------------------


@mcp.tool()
def approve_milling(
    prompt: str = "Approve milling?",
    milling_current_na: float = 0.0,
    milling_voltage_kv: float = 30.0,
    timeout: int = 120,
) -> str:
    """Show an Approve/Cancel prompt on the live dashboard and wait for the user to respond.
    Patterns already drawn will be visible as overlays on the FIB image.
    Returns 'approved' or 'cancelled'.
    Requires remote mode (connect_microscope_remote) and the dashboard open in a browser.

    Args:
        prompt: Message shown in the approval banner (e.g. 'Rough Mill 01 — proceed?')
        milling_current_na: Milling current in nanoamps — shown alongside the prompt
        milling_voltage_kv: Milling voltage in kV — shown alongside the prompt
        timeout: Seconds to wait before auto-cancelling (default 120)
    """
    from fibsem.server.client import FibsemClient

    if _microscope is None:
        return "Error: no microscope connected."
    if not isinstance(_microscope, FibsemClient):
        return (
            "approve_milling requires remote mode (connect_microscope_remote). "
            "In direct mode, ask the user for confirmation in the chat instead."
        )

    params_str = ""
    if milling_current_na > 0:
        params_str = f"current={milling_current_na} nA   voltage={milling_voltage_kv} kV"

    try:
        _microscope.request_approval(prompt=prompt, params=params_str)
        approved = _microscope.wait_for_approval(timeout=timeout)
    except TimeoutError:
        return f"Approval timed out after {timeout} s — milling NOT run."
    except Exception as exc:
        return f"Error: {exc}"

    return "approved" if approved else "cancelled"


@mcp.tool()
def pick_point(title: str = "Click to select a point") -> str:
    """Show the last acquired image in a window and wait for the user to click a point.
    Returns the clicked point as an offset from image center in micrometers.
    The offset can be passed directly to move_stage_relative (x_um / 1000, y_um / 1000).

    Args:
        title: Window title shown to the user
    """
    if _last_image is None:
        return "Error: no image acquired yet — call acquire_image first."

    import matplotlib

    try:
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.close()
    except Exception:
        return (
            "Error: pick_point requires a graphical display ($DISPLAY) — "
            "not available in this environment. Use move_stage_relative with "
            "known offsets instead."
        )
    import numpy as np

    data = _last_image.data.copy()
    if data.dtype != np.uint8:
        d_min, d_max = int(data.min()), int(data.max())
        if d_max > d_min:
            data = ((data.astype(np.float32) - d_min) / (d_max - d_min) * 255).astype(
                np.uint8
            )
        else:
            data = np.zeros_like(data, dtype=np.uint8)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(data, cmap="gray", origin="upper")
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    pts = fig.ginput(n=1, timeout=60, show_clicks=True)
    plt.close(fig)

    if not pts:
        return "No point selected (timed out or window closed)."

    px_x, px_y = pts[0]
    h, w = _last_image.data.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    px_size = _last_image.metadata.pixel_size if _last_image.metadata else None
    if px_size:
        dx_um = (px_x - cx) * px_size.x * 1e6
        dy_um = (px_y - cy) * px_size.y * 1e6
        return (
            f"Selected point: pixel=({px_x:.1f}, {px_y:.1f}), "
            f"offset from center=({dx_um:+.3f} µm, {dy_um:+.3f} µm)"
        )
    return f"Selected point: pixel=({px_x:.1f}, {px_y:.1f}) — no pixel size metadata available"


@mcp.tool()
def pick_point_web(title: str = "Click to select a point", timeout: int = 60) -> str:
    """Show a click prompt on the live dashboard and wait for the user to click the FIB image.
    Returns the clicked point as an offset from image centre in micrometers.
    Requires remote mode (connect_microscope_remote) and dashboard open in browser.

    Args:
        title: Prompt text shown on the dashboard overlay
        timeout: Seconds to wait for a click before giving up (default 60)
    """
    from fibsem.server.client import FibsemClient
    if _microscope is None:
        return "Error: no microscope connected."
    if not isinstance(_microscope, FibsemClient):
        return (
            "pick_point_web requires remote mode (connect_microscope_remote). "
            "In direct mode, use pick_point instead."
        )
    if _microscope.dashboard_url is None:
        return "Error: dashboard not available."

    try:
        _microscope.request_click(title)
        result = _microscope.wait_for_click(timeout=timeout)
        info = _microscope.get_dashboard_info()
        hfw_um = info.get("hfw_um", 150.0)
        aspect = info.get("fib_aspect", 0.667)
    except TimeoutError:
        return f"No point selected (timed out after {timeout} s). Dashboard: {_microscope.dashboard_url}"
    except Exception as exc:
        return f"Error: {exc}"

    x_frac = result.get("x_frac", 0.5)
    y_frac = result.get("y_frac", 0.5)
    dx_um = (x_frac - 0.5) * hfw_um
    dy_um = (y_frac - 0.5) * hfw_um * aspect
    return (
        f"Selected point: offset from centre = ({dx_um:+.3f} µm, {dy_um:+.3f} µm)\n"
        f"To move stage: move_stage_relative(dx_mm={dx_um/1000:.6f}, dy_mm={dy_um/1000:.6f})"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n_tools = len(mcp._tool_manager._tools)
    log.info("fibsemOS MCP server starting — %d tools registered", n_tools)
    log.info("Call connect_microscope() or connect_microscope_remote() to begin.")
    mcp.run(transport="stdio")
