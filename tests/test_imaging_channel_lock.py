"""Setting the beam channel and grabbing the frame must be one locked step (FIB-542).

The FM and the beams are one connection with one active view and one active device, so
whoever sets it last owns it. `grab_frame` reads the active view's buffer, which means a
`set_channel` that is no longer in force when the grab lands returns whoever took the
channel in between -- and returns it silently, because the metadata is built from the
requested `ImageSettings` rather than from what actually came back.

That is FIB-517's failure mode on the beam side: an FM property getter fired from a
GUI-thread stage poll while a workflow task acquired on a worker. FIB-517 fixed the FM
half, so a getter now hands the channel back, but anything that holds the channel for the
length of its own operation -- a deliberate FM acquisition, a live stream -- can still
land inside this window.

Structural, over the real source. `ThermoMicroscope` cannot be constructed without the
AutoScript SDK, which is absent off the microscope, and the simulator never reads
`active_view`/`active_device` at all (FIB-518), so there is no harness here that could
observe the race. What is pinned is the discipline: the pair is locked, it is locked
together, and the locked region stays narrow.
"""
import ast
from pathlib import Path

import pytest

import fibsem


def _thermo_class() -> ast.ClassDef:
    """The `ThermoMicroscope` class body, parsed from source."""
    source = (Path(fibsem.__file__).parent / "microscope.py").read_text()
    return next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == "ThermoMicroscope"
    )


def _calls_named(node: ast.AST, name: str) -> list:
    """Every `something.<name>(...)` call anywhere under `node`."""
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == name
    ]


def _locked_blocks(node: ast.AST) -> list:
    """Every `with self._threading_lock:` block anywhere under `node`."""
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.With)
        and any("_threading_lock" in ast.dump(item.context_expr) for item in child.items)
    ]


@pytest.fixture(scope="module")
def thermo() -> ast.ClassDef:
    return _thermo_class()


def test_every_grab_is_locked(thermo):
    """The defect itself, and the guard against a fourth copy of the pair appearing.

    `_acquire_image2` was exactly that -- a second, unlocked copy with no callers, kept
    around long enough to be a template. It was deleted with this fix.
    """
    grabs = _calls_named(thermo, "grab_frame")
    assert grabs, "no grab_frame call found -- the probe missed, not the code"

    locked = {id(call) for block in _locked_blocks(thermo) for call in _calls_named(block, "grab_frame")}
    unlocked = [call.lineno for call in grabs if id(call) not in locked]
    assert unlocked == [], (
        f"grab_frame runs without the lock at line(s) {unlocked}: whatever took the "
        f"shared channel after set_channel is what this returns"
    )


def test_the_channel_is_set_inside_the_same_block(thermo):
    """A lock that covers only the grab guards nothing.

    The whole point is that the channel is still ours when the grab lands, so the
    `set_channel` has to be under the same lock -- not merely before it.
    """
    for block in _locked_blocks(thermo):
        if not _calls_named(block, "grab_frame"):
            continue
        assert _calls_named(block, "set_channel"), (
            f"the block at line {block.lineno} locks the grab but not the set_channel "
            f"that precedes it, so the channel can still be taken in between"
        )


def test_the_locked_region_stays_narrow(thermo):
    """`_threading_lock` is a class attribute, shared by every caller in the process.

    Held across the metadata reads or the `get_microscope_state` fetch, an acquisition
    would block the milling monitor, a Stop click and every FM channel scope for the
    length of a frame. Frame-long is already the cost of the grab itself; making it
    frame-plus-a-full-state-read is not. Pinned so a later "while we're here" widening
    is caught rather than merged.
    """
    for block in _locked_blocks(thermo):
        if not _calls_named(block, "grab_frame"):
            continue
        widened = sorted(
            {
                call.func.attr
                for call in _calls_named(block, "get_microscope_state")
                + _calls_named(block, "get_imaging_settings")
                + _calls_named(block, "_set_additional_metadata")
            }
        )
        assert widened == [], (
            f"the block at line {block.lineno} holds the process-wide lock across "
            f"{widened}, which blocks every other caller for longer than the frame"
        )
