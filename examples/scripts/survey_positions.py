"""Acquire an ion image at every lamella position."""

# Declaring this hands the script ctx.microscope and moves it onto a worker
# thread, so the window stays responsive and the Run button becomes Stop. Nothing
# validates what the script then does -- no limits, no interlocks. Without the
# flag ctx.microscope is None, on purpose.
uses_microscope = True

from fibsem import acquire
from fibsem.structures import BeamType, ImageSettings


def run(ctx):
    # acquire.acquire_image, NOT ctx.microscope.acquire_image. Only this one
    # applies autocontrast and auto-gamma and honours save=True; the microscope
    # method silently ignores it and writes nothing to disk.
    settings = ImageSettings(hfw=80e-6, beam_type=BeamType.ION, save=True)

    lamellae = ctx.experiment.positions
    imaged = 0

    for lamella in lamellae:
        # Stopping is cooperative. A Python thread cannot be killed, so Stop only
        # sets a flag -- if the script never asks, the button does nothing and the
        # run continues to the end. Ask once per iteration, before the slow part.
        ctx.raise_if_cancelled()

        ctx.log(f"moving to {lamella.name}")
        ctx.microscope.safe_absolute_stage_movement(lamella.stage_position)

        # Each lamella has its own directory, so images land beside that lamella.
        settings.path = lamella.path
        settings.filename = "survey"
        acquire.acquire_image(ctx.microscope, settings)
        imaged += 1

    # Reading ctx.experiment from a microscope script is fine. *Writing* to it is
    # not: this runs off the GUI thread, and the experiment is psygnal-evented, so
    # assigning to it here would drive widget updates from the wrong thread. Do
    # the hardware work here and the bookkeeping in a separate `writes` script.
    return f"imaged {imaged} of {len(lamellae)} positions"
