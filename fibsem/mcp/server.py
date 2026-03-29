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
from typing import Optional

from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger("fibsem.mcp")

mcp = FastMCP("fibsem")

# Global microscope state — persists across tool calls within a server session
_microscope = None
_settings = None
_last_image = None  # last acquired FibsemImage, used by pick_point


def _check_microscope() -> Optional[str]:
    """Return an error string if no microscope is connected, else None."""
    if _microscope is None:
        return "Error: no microscope connected — call connect_microscope first."
    return None


def _bt(beam_type: str):
    """Convert 'electron'/'ion' string to BeamType enum."""
    from fibsem.structures import BeamType
    return BeamType.ELECTRON if beam_type.lower() == "electron" else BeamType.ION


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@mcp.tool()
def connect_microscope(manufacturer: str = "Demo", ip_address: str = "localhost") -> str:
    """Connect to a FIB/SEM microscope.

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
    return f"Connected to {manufacturer} microscope at {ip_address}"


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
        f"  x        = {pos.x * 1e3:.4f} mm" if pos.x is not None else "  x        = unknown",
        f"  y        = {pos.y * 1e3:.4f} mm" if pos.y is not None else "  y        = unknown",
        f"  z        = {pos.z * 1e3:.4f} mm" if pos.z is not None else "  z        = unknown",
        f"  rotation = {math.degrees(pos.r):.2f}°" if pos.r is not None else "  rotation = unknown",
        f"  tilt     = {math.degrees(pos.t):.2f}°" if pos.t is not None else "  tilt     = unknown",
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
                f"  voltage          = {b.voltage / 1e3:.1f} kV" if b.voltage is not None else "  voltage          = unknown",
                f"  current          = {b.beam_current * 1e9:.2f} nA" if b.beam_current is not None else "  current          = unknown",
                f"  working_distance = {b.working_distance * 1e3:.3f} mm" if b.working_distance is not None else "  working_distance = unknown",
                f"  hfw              = {b.hfw * 1e6:.2f} µm" if b.hfw is not None else "  hfw              = unknown",
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
        f"  x        = {pos.x * 1e3:.4f} mm" if pos.x is not None else "  x        = unknown",
        f"  y        = {pos.y * 1e3:.4f} mm" if pos.y is not None else "  y        = unknown",
        f"  z        = {pos.z * 1e3:.4f} mm" if pos.z is not None else "  z        = unknown",
        f"  rotation = {math.degrees(pos.r):.2f}°" if pos.r is not None else "  rotation = unknown",
        f"  tilt     = {math.degrees(pos.t):.2f}°" if pos.t is not None else "  tilt     = unknown",
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
        f"  voltage          = {b.voltage / 1e3:.1f} kV" if b.voltage is not None else "  voltage          = unknown",
        f"  current          = {b.beam_current * 1e9:.2f} nA" if b.beam_current is not None else "  current          = unknown",
        f"  working_distance = {b.working_distance * 1e3:.3f} mm" if b.working_distance is not None else "  working_distance = unknown",
        f"  hfw              = {b.hfw * 1e6:.2f} µm" if b.hfw is not None else "  hfw              = unknown",
        f"  scan_rotation    = {math.degrees(b.scan_rotation):.2f}°" if b.scan_rotation is not None else "  scan_rotation    = unknown",
        f"  shift            = ({b.shift.x * 1e6:.2f}, {b.shift.y * 1e6:.2f}) µm" if b.shift is not None else "  shift            = unknown",
        f"  stigmation       = ({b.stigmation.x:.4f}, {b.stigmation.y:.4f})" if b.stigmation is not None else "  stigmation       = unknown",
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

    log.info("Acquiring %s image: hfw=%.1f µm, res=%dx%d", beam_type, hfw_um, resolution_x, resolution_y)
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

    import base64
    import io
    import numpy as np
    from mcp.types import TextContent, ImageContent
    from PIL import Image as PILImage

    data = image.data
    if data.dtype != np.uint8:
        d_min, d_max = int(data.min()), int(data.max())
        if d_max > d_min:
            data = ((data.astype(np.float32) - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)
    pil = PILImage.fromarray(data, mode="L" if data.ndim == 2 else "RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return [
        TextContent(type="text", text=text),
        ImageContent(type="image", data=b64, mimeType="image/png"),
    ]


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
) -> str:
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
    lines = []
    for label, img in [("SEM (electron)", eb), ("FIB (ion)", ib)]:
        if img is not None:
            px = img.metadata.pixel_size if img.metadata else None
            px_str = f"{px.x * 1e9:.2f} nm" if px else "unknown"
            lines.append(f"{label}: shape={img.data.shape}, pixel_size={px_str}")
        else:
            lines.append(f"{label}: not acquired")

    return "\n".join(lines)


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

    log.info("Moving stage to x=%.4f y=%.4f z=%.4f mm",
             target.x * 1e3, target.y * 1e3, target.z * 1e3)
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
def move_to_milling_angle(milling_angle_deg: float) -> str:
    """Move the stage to achieve a target milling angle (moves tilt axis).

    Args:
        milling_angle_deg: Target milling angle in degrees (e.g. 5–15° for cryo-lamella)
    """
    if err := _check_microscope():
        return err

    success = _microscope.move_to_milling_angle(milling_angle_deg)
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


# ---------------------------------------------------------------------------
# Interactive point selection
# ---------------------------------------------------------------------------

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
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = _last_image.data.copy()
    if data.dtype != np.uint8:
        d_min, d_max = int(data.min()), int(data.max())
        if d_max > d_min:
            data = ((data.astype(np.float32) - d_min) / (d_max - d_min) * 255).astype(np.uint8)
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
