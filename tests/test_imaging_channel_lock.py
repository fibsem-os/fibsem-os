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
AutoScript SDK, which is absent off the microscope. Its imaging runs in its beam devices
(`AutoscriptBeam`), whose `claim_channel()` is the same lock with the beam's channel
selected inside it, so both classes are read. What is pinned is the discipline --
the pair is locked, it is locked together, and the locked region stays narrow.

The race itself is now observable on the simulator, which models the shared channel on
both sides since FIB-518; `tests/fm/test_simulated_shared_channel.py` runs it. That does
not replace this file, which is about the class that cannot be built here.
"""

import ast
from pathlib import Path

import pytest

import fibsem


def _class(module: str, name: str) -> ast.ClassDef:
    """A class body, parsed from source."""
    source = (Path(fibsem.__file__).parent / module).read_text(encoding="utf-8")
    return next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _thermo_class() -> ast.ClassDef:
    """The `ThermoMicroscope` class body, parsed from source."""
    return _class("microscopes/autoscript.py", "ThermoMicroscope")


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
    """Every `with self._threading_lock:` or `with self.claim_channel():` block
    anywhere under `node`."""
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.With)
        and any(
            "_threading_lock" in ast.dump(item.context_expr)
            or "claim_channel" in ast.dump(item.context_expr)
            for item in child.items
        )
    ]


def _sets_the_channel(block: ast.With) -> bool:
    """A block that selects the channel inside the lock: a `set_channel` call, or a
    device's `claim_channel()`, which selects its own."""
    return bool(_calls_named(block, "set_channel")) or any(
        "claim_channel" in ast.dump(item.context_expr) for item in block.items
    )


@pytest.fixture(scope="module")
def thermo() -> ast.Module:
    """`ThermoMicroscope` and its beam device, where its imaging runs."""
    return ast.Module(
        body=[
            _thermo_class(),
            _class("devices/drivers/autoscript.py", "AutoscriptBeam"),
        ],
        type_ignores=[],
    )


def test_every_grab_is_locked(thermo):
    """The defect itself, and the guard against a fourth copy of the pair appearing.

    `_acquire_image2` was exactly that -- a second, unlocked copy with no callers, kept
    around long enough to be a template. It was deleted with this fix.
    """
    grabs = _calls_named(thermo, "grab_frame")
    assert grabs, "no grab_frame call found -- the probe missed, not the code"

    locked = {
        id(call)
        for block in _locked_blocks(thermo)
        for call in _calls_named(block, "grab_frame")
    }
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
        assert _sets_the_channel(block), (
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
        assert _sets_the_channel(block), (
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
            # the microscope's method, or the beam's own scan command
            assert _calls_named(block, "set_reduced_area_scanning_mode") or (
                _calls_named(block, "reduced_area")
            ), (
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
    """`_threading_lock` is shared by every caller on the microscope, devices included.

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


# The detector property pairs (FIB-544). Setting the channel and then reaching for
# `connection.detector.…`, which resolves against the active device, has to be one
# locked step: a channel taken in between reads, or writes, the other column's
# detector. `get_detector_settings` is four of these, and `get_microscope_state` reads
# it per beam from GUI-thread polls. The beam device reaches the detector through its
# `_detector` property; its detector parameters (`needs_channel`) are read and
# written under the device's channel claim, which the core takes around the
# `read_`/`write_` call, so those count as locked. Attribute accesses rather than
# calls, so these key on `self.connection.detector.…` and `self._detector.…`.


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
    """Every `self.connection.detector.<something>` or `self._detector.<something>`
    access under `node`, one per line (walking a chain also yields each of its
    prefixes)."""
    lines = {}
    for child in ast.walk(node):
        if not isinstance(child, ast.Attribute):
            continue
        chain = _attr_chain(child)
        if (chain[:3] == ["self", "connection", "detector"] and len(chain) > 3) or (
            chain[:2] == ["self", "_detector"] and len(chain) > 2
        ):
            lines[child.lineno] = child
    return list(lines.values())


def _claimed_methods(node: ast.AST) -> list:
    """The `read_<name>`/`write_<name>` methods of the parameters a device class
    declares in `needs_channel`, which the core runs under the channel claim."""
    methods = []
    for cls in (c for c in ast.walk(node) if isinstance(c, ast.ClassDef)):
        names = set()
        for stmt in cls.body:
            if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "needs_channel"
                for t in stmt.targets
            ):
                names = {
                    c.value
                    for c in ast.walk(stmt.value)
                    if isinstance(c, ast.Constant) and isinstance(c.value, str)
                }
        methods += [
            stmt
            for stmt in cls.body
            if isinstance(stmt, ast.FunctionDef)
            and stmt.name.split("_", 1)[0] in ("read", "write")
            and stmt.name.split("_", 1)[1] in names
        ]
    return methods


def test_every_detector_access_is_locked(thermo):
    accesses = _detector_accesses(thermo)
    assert accesses, "no detector access found -- the probe missed"

    locked = {
        access.lineno
        for block in _locked_blocks(thermo) + _claimed_methods(thermo)
        for access in _detector_accesses(block)
    }
    # the property itself only names the detector; its callers are what is locked
    named = {
        access.lineno
        for fn in ast.walk(thermo)
        if isinstance(fn, ast.FunctionDef) and fn.name == "_detector"
        for access in _detector_accesses(fn)
    }
    unlocked = sorted({a.lineno for a in accesses} - locked - named)
    assert unlocked == [], (
        f"the detector is reached without the lock at line(s) {unlocked}: "
        f"whatever took the channel after set_channel is the detector this reaches"
    )


def test_the_detector_pairs_set_the_channel_in_the_same_block(thermo):
    """A lock around the access alone guards nothing; the pair has to be together."""
    for block in _locked_blocks(thermo):
        if not _detector_accesses(block):
            continue
        assert _sets_the_channel(block), (
            f"the block at line {block.lineno} locks a detector access but not the "
            f"set_channel before it"
        )


def test_get_detector_settings_holds_the_lock_across_the_group(thermo):
    """Four locked pairs still let the channel move between them, so the four values
    could describe two states. The lock is re-entrant, so the pairs inside cost
    nothing more."""
    override = next(
        (
            node
            for node in _thermo_class().body
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_detector_settings"
        ),
        None,
    )
    assert override is not None, (
        "ThermoMicroscope no longer overrides get_detector_settings"
    )
    assert _locked_blocks(override)
