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

Structural, over the real source: `ThermoMicroscope` cannot be constructed without the
AutoScript SDK, which is absent off the microscope. What is pinned is the discipline --
the pair is locked, it is locked together, and the locked region stays narrow.

The race itself is now observable on the simulator, which models the shared channel on
both sides since FIB-518; `tests/fm/test_simulated_shared_channel.py` runs it. That does
not replace this file, which is about the class that cannot be built here.
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


"""The other view-dependent actions (FIB-569, FIB-545).

Sweeping the class for the same shape found three more. Each is documented by the vendor
as acting on the active view, so each needs the channel to still be its own when it runs:

* `imaging.get_image` — "Retrieves a microscope image currently present in the active
  view". `last_image` is FIB-542's pair on the retrieval path.
* `auto_functions.run_auto_cb` — "optimizes contrast and brightness of the active
  detector in the active view".
* `auto_functions.run_auto_focus` — "Runs the automatic focus routine in the active
  view".

The autofunctions are the ones that matter most, and they are not covered by
`test_the_locked_region_stays_narrow` on purpose: the routine *is* the thing that needs
the channel, so there is no narrower correct scope and the hold is deliberately as long
as the routine. That exception is the reason the narrowness test keys on `grab_frame`
rather than applying to every locked block.
"""

VIEW_DEPENDENT_ACTIONS = ["get_image", "run_auto_cb", "run_auto_focus"]


@pytest.mark.parametrize("action", VIEW_DEPENDENT_ACTIONS)
def test_every_view_dependent_action_is_locked(thermo, action):
    calls = _calls_named(thermo, action)
    assert calls, f"no {action} call found -- the probe missed, not the code"

    locked = {
        id(call)
        for block in _locked_blocks(thermo)
        for call in _calls_named(block, action)
    }
    unlocked = [call.lineno for call in calls if id(call) not in locked]
    assert unlocked == [], (
        f"{action} runs without the lock at line(s) {unlocked}: it acts on the active "
        f"view, so whatever took the shared channel is what it acts on"
    )


@pytest.mark.parametrize("action", ["run_auto_cb", "run_auto_focus"])
def test_the_autofunctions_claim_the_channel_in_the_same_block(thermo, action):
    """Setting the channel before the lock would leave the window open."""
    for block in _locked_blocks(thermo):
        if not _calls_named(block, action):
            continue
        assert _calls_named(block, "set_channel"), (
            f"the block at line {block.lineno} locks {action} but not the set_channel "
            f"that precedes it, so the channel can still be taken in between"
        )


def test_the_autofunctions_hold_the_reduced_area_too(thermo):
    """The pair is a triple when a reduced area is given.

    A reduced-area write left outside the lock lets the routine run on the right view
    with someone else's scan region -- the channel is guarded and the result is still
    wrong.
    """
    for action in ("run_auto_cb", "run_auto_focus"):
        blocks = [b for b in _locked_blocks(thermo) if _calls_named(b, action)]
        assert blocks, f"no locked {action} block found"
        for block in blocks:
            assert _calls_named(block, "set_reduced_area_scanning_mode"), (
                f"the block at line {block.lineno} runs {action} without the "
                f"reduced-area write inside it"
            )


def test_the_chamber_camera_puts_the_view_back(thermo):
    """FIB-545. Not a race -- a channel taken and abandoned.

    `acquire_chamber_image` points the connection at view 4 / device 3. Before this it
    never pointed it back, so a glance at the chamber stranded whatever had the
    microscope for the rest of the session. The restore has to be in a `finally`: a
    chamber camera that does not answer would otherwise strand it just the same.
    """
    chamber = next(
        node
        for node in ast.walk(thermo)
        if isinstance(node, ast.FunctionDef) and node.name == "acquire_chamber_image"
    )

    assert _calls_named(chamber, "get_active_view"), (
        "acquire_chamber_image never reads the view it is about to replace, so it has "
        "nothing to put back"
    )

    tries = [node for node in ast.walk(chamber) if isinstance(node, ast.Try)]
    restored = [
        call
        for node in tries
        for handler in node.finalbody
        for call in _calls_named(handler, "set_active_view")
    ]
    assert restored, (
        "acquire_chamber_image does not restore the active view in a `finally`, so a "
        "failed grab leaves the connection on the chamber camera"
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


"""The detector property pairs (FIB-544).

The same defect on the property path, and the most exposed instance of it. `_get` and
`_set` each did `set_channel(beam_type)` and then reached for `connection.detector.…`,
which resolves against the *active device* — so a channel taken in between makes the read
answer from the other column, silently, and makes the write land on the other column's
detector and stay there.

More exposed than the acquisition pair for two reasons. `get_detector_settings` is four
of these back to back, and `get_microscope_state()` calls it once per beam — so one state
read with no `beam_type` is eight pairs. And that path is polled from the GUI thread by
the minimap's stage refresh and the coincidence viewer, which is exactly how FIB-517
stopped a workflow task.

Attribute accesses rather than calls, so these key on `self.connection.detector.…`
instead of going through `_calls_named`.
"""


def _attr_chain(node: ast.AST) -> list:
    """`self.connection.detector.type.value` -> the parts, outermost last."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return list(reversed(parts))


def _detector_accesses(node: ast.AST) -> list:
    """Every `self.connection.detector.<something>` access anywhere under `node`.

    Deduplicated by line: walking an attribute chain yields the outer node and each of
    its prefixes, so one access would otherwise be reported several times.
    """
    lines = {}
    for child in ast.walk(node):
        if not isinstance(child, ast.Attribute):
            continue
        chain = _attr_chain(child)
        if chain[:3] == ["self", "connection", "detector"] and len(chain) > 3:
            lines[child.lineno] = child
    return list(lines.values())


def test_every_detector_access_is_locked(thermo):
    """The defect. A read or a write against a channel that may no longer be ours."""
    accesses = _detector_accesses(thermo)
    assert accesses, "no connection.detector access found -- the probe missed, not the code"

    locked = {
        access.lineno
        for block in _locked_blocks(thermo)
        for access in _detector_accesses(block)
    }
    unlocked = sorted({a.lineno for a in accesses} - locked)
    assert unlocked == [], (
        f"connection.detector is reached without the lock at line(s) {unlocked}: it "
        f"resolves against the active device, so whatever took the shared channel after "
        f"set_channel is the detector this reads or writes (FIB-544)"
    )


def test_the_detector_pairs_set_the_channel_in_the_same_block(thermo):
    """A lock around the access alone guards nothing — the pair has to be together."""
    for block in _locked_blocks(thermo):
        if not _detector_accesses(block):
            continue
        assert _calls_named(block, "set_channel"), (
            f"the block at line {block.lineno} locks a detector access but not the "
            f"set_channel that precedes it, so the channel can still be taken in between"
        )


def test_get_detector_settings_holds_the_lock_across_the_group(thermo):
    """Four locked pairs still let the channel move between them.

    Each value would be correct on its own and the four could still describe two
    different states, and each pair re-claims the channel — so a contended read flips the
    active view four times rather than once. `_threading_lock` is an RLock, so holding it
    across the group costs the pairs inside nothing.
    """
    override = next(
        (
            node
            for node in ast.walk(thermo)
            if isinstance(node, ast.FunctionDef) and node.name == "get_detector_settings"
        ),
        None,
    )
    assert override is not None, (
        "ThermoMicroscope no longer overrides get_detector_settings, so its four reads "
        "are four independent claims on the shared channel"
    )
    assert _locked_blocks(override), (
        "get_detector_settings does not hold the lock across its four reads"
    )
