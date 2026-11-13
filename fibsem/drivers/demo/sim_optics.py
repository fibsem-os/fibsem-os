"""How the beam's settings show in a simulated SEM or FIB image.

Three things blur an image, and they add in quadrature:

- **focus**: the beam converges at half-angle ``ANGLE``, so a working distance
  ``d`` away from focus spreads it over ``ANGLE * d``
- **stigmation**: off zero, the beam focuses at two heights ``SPLIT * |s|``
  either side of the working distance, one per direction, so the blur is
  stretched one way (and the other way through focus)
- **current**: more current is a bigger probe

Focus is where the sample is: a surface above or below the eucentric point is
that much nearer or further along the beam, and across a tilted view one edge
is nearer than the other (`Optics.tilt`), so the blur changes down the image.

Current and dwell time set the noise: the counts in a pixel go with both.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

UM = 1e6
ANGLE = (4e-3, 1e-3)  # rad, SEM and FIB: the beam's convergence half-angle
SPLIT = 1e-3  # m of focus between the two directions, per unit of stigmation
# the probe's size at a reference current, and how it grows: (um, A, power)
PROBE = ((0.002, 100e-12, 0.4), (0.012, 20e-12, 0.4))
DOSE = (0.25, 8.0)  # the least and most a current changes the counts
DWELL = (1e-6, 0.5, 4.0)  # s for the usual counts; the least and most it changes them
LEVELS = 6  # the most different blurs drawn across one tilted view
SHARP = 0.15  # px; a blur under this is not drawn
WIDEST = 60.0  # px; and no blur is wider


@dataclass(frozen=True)
class Optics:
    """A beam as it images: how far its working distance is from focus at the
    view's centre (m), how that changes across the image (m per m of x and of
    y, down), its stigmation, its current (A) and dwell time (s). A current or
    dwell of None leaves the probe and the noise alone."""

    defocus: float = 0.0
    stigmation: Tuple[float, float] = (0.0, 0.0)
    current: Optional[float] = None
    dwell: Optional[float] = None
    tilt: Tuple[float, float] = (0.0, 0.0)

    def spread(self, which: int, defocus: Optional[float] = None):
        """The blur as (sigma along, sigma across, angle): um and rad, at the
        view's centre or at `defocus`."""
        defocus = self.defocus if defocus is None else defocus
        sx, sy = self.stigmation
        split = SPLIT * float(np.hypot(sx, sy))
        probe = 0.0
        if self.current:
            size, ref, power = PROBE[which]
            probe = size * (abs(self.current) / ref) ** power
        along = np.hypot(ANGLE[which] * (defocus + split) * UM, probe)
        across = np.hypot(ANGLE[which] * (defocus - split) * UM, probe)
        return float(along), float(across), 0.5 * float(np.arctan2(sy, sx))

    def dose(self, which: int) -> float:
        """The counts in a pixel, against those at the reference current and
        the usual dwell."""
        dose = 1.0
        if self.current:
            dose *= float(np.clip(abs(self.current) / PROBE[which][1], *DOSE))
        if self.dwell:
            dose *= float(np.clip(self.dwell / DWELL[0], *DWELL[1:]))
        return dose


def blur(img: np.ndarray, optics: Optics, which: int, pix: float) -> np.ndarray:
    """`img` as the beam draws it, at `pix` um a pixel. The probe at the
    reference current is what the renderer already draws, so only what a
    setting adds to it is blurred here."""
    height, width = img.shape
    tx, ty = optics.tilt
    # how far from focus the corners are: the most the view's tilt adds
    reach = (abs(tx) * width + abs(ty) * height) * 0.5 * pix / UM
    step = SHARP * pix / (ANGLE[which] * UM)  # the defocus that shows
    if reach < step:
        return _blur(img, optics, which, pix, optics.defocus)
    # a tilted view: a few blurs across its range, each pixel between two
    n = int(np.clip(np.ceil(2 * reach / (4 * step)) + 1, 2, LEVELS))
    levels = optics.defocus + np.linspace(-reach, reach, n)
    x = (np.arange(width) - width / 2) * pix / UM
    y = (np.arange(height) - height / 2) * pix / UM
    at = (optics.defocus + tx * x[None, :] + ty * y[:, None] - levels[0]) / (
        levels[1] - levels[0]
    )
    at = np.clip(at, 0, n - 1)
    out = np.zeros(img.shape, dtype=np.float32)
    for j, level in enumerate(levels):
        weight = np.clip(1 - np.abs(at - j), 0, 1)
        if weight.any():
            out += weight * _blur(img, optics, which, pix, float(level))
    return out.astype(img.dtype)


def _blur(img, optics: Optics, which: int, pix: float, defocus: float):
    """`img` blurred as the beam is `defocus` from focus."""
    along, across, angle = optics.spread(which, defocus)
    base = PROBE[which][0]
    a = np.sqrt(max(along**2 - base**2, 0.0)) / pix
    b = np.sqrt(max(across**2 - base**2, 0.0)) / pix
    a, b = min(a, WIDEST), min(b, WIDEST)
    if max(a, b) < SHARP:
        return img
    if abs(a - b) < SHARP:
        return ndi.gaussian_filter(img, 0.5 * (a + b), mode="nearest")
    # stretched one way: a gaussian with its own axes, as a filter in
    # frequency, over a border wide enough that the edges do not wrap
    pad = int(np.ceil(3 * max(a, b)))
    wide = np.pad(img, pad, mode="edge")
    fy = np.fft.fftfreq(wide.shape[0])[:, None]
    fx = np.fft.rfftfreq(wide.shape[1])[None, :]
    c, s = np.cos(angle), np.sin(angle)
    # image rows run down, so the angle turns the other way on screen
    fu, fv = fx * c - fy * s, fx * s + fy * c
    gain = np.exp(-2 * np.pi**2 * ((a * fu) ** 2 + (b * fv) ** 2))
    out = np.fft.irfft2(np.fft.rfft2(wide) * gain, s=wide.shape)
    return out[pad:-pad, pad:-pad].astype(img.dtype)
