"""How the beam's settings show in a simulated SEM or FIB image.

Three things blur an image, and they add in quadrature:

- **focus**: the beam converges at half-angle ``ANGLE``, so a working distance
  ``d`` away from focus spreads it over ``ANGLE * d``
- **stigmation**: off zero, the beam focuses at two heights ``SPLIT * |s|``
  either side of the working distance, one per direction, so the blur is
  stretched one way (and the other way through focus)
- **current**: more current is a bigger probe

Current also sets the noise: the counts in a pixel go with it.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi

UM = 1e6
ANGLE = (4e-3, 2e-3)  # rad, SEM and FIB: the beam's convergence half-angle
SPLIT = 1e-3  # m of focus between the two directions, per unit of stigmation
# the probe's size at a reference current, and how it grows: (um, A, power)
PROBE = ((0.002, 100e-12, 0.4), (0.012, 20e-12, 0.4))
DOSE = (0.25, 8.0)  # the least and most a current changes the counts
SHARP = 0.15  # px; a blur under this is not drawn
WIDEST = 60.0  # px; and no blur is wider


@dataclass(frozen=True)
class Optics:
    """A beam as it images: how far its working distance is from focus (m),
    its stigmation, and its current (A; None leaves probe and noise alone)."""

    defocus: float = 0.0
    stigmation: Tuple[float, float] = (0.0, 0.0)
    current: float = None

    def spread(self, which: int) -> Tuple[float, float, float]:
        """The blur as (sigma along, sigma across, angle): um and rad."""
        sx, sy = self.stigmation
        split = SPLIT * float(np.hypot(sx, sy))
        probe = 0.0
        if self.current:
            size, ref, power = PROBE[which]
            probe = size * (abs(self.current) / ref) ** power
        along = np.hypot(ANGLE[which] * (self.defocus + split) * UM, probe)
        across = np.hypot(ANGLE[which] * (self.defocus - split) * UM, probe)
        return float(along), float(across), 0.5 * float(np.arctan2(sy, sx))

    def dose(self, which: int) -> float:
        """The counts in a pixel, against those at the reference current."""
        if not self.current:
            return 1.0
        return float(np.clip(abs(self.current) / PROBE[which][1], *DOSE))


def blur(img: np.ndarray, optics: Optics, which: int, pix: float) -> np.ndarray:
    """`img` as the beam draws it, at `pix` um a pixel. The probe at the
    reference current is what the renderer already draws, so only what a
    setting adds to it is blurred here."""
    along, across, angle = optics.spread(which)
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
