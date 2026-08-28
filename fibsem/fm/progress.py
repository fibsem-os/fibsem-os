"""What a fluorescence acquisition says about itself while it runs.

`FluorescenceMicroscope.acquisition_progress_signal` used to carry a bare dict with no
declared shape, so every consumer worked out what it had received from which keys
happened to be present -- and a key an emitter forgot was indistinguishable from one
legitimately absent.

This signal is **fluorescence only**, by construction rather than by convention: it is
declared on `FluorescenceMicroscope`, every subscriber reaches it as
`microscope.fm.acquisition_progress_signal`, and every emit site is in `fibsem/fm`. That
is why nothing here carries a modality, unlike `imaging.tiling.progress.TiledProgress` --
that signal hangs off `FibsemMicroscope` and genuinely has two producers, so it needs a
field to tell them apart. This one can only ever have one.

The types are named for the modality anyway, because `acquisition_progress_signal` is
*also* the name of an unrelated `pyqtSignal` on `FibsemImageSettingsWidget` carrying the
beam side's `{"msg"} / {"finished"}`. The two are distinguishable only by whether you
reach them through `.fm.` or `.image_widget.`, which is a poor thing to rely on when
reading a call site.

# One flat record, not a union

The same reasoning as `TiledProgress`, and it applies harder here. A union keyed on the
acquisition routine looks natural and does not match how either consumer branches: both
ask first *is there a z count?* --

    if zlevel and total_zlevels:

-- and **both the z-stack and the autofocus sweep carry one**. A `ZStackProgress` type
would miss autofocus, so the check becomes `isinstance(e, (ZStackProgress,
AutofocusProgress))`: the multi-class tuple whose failure mode is silent, since an
omitted class is a bar that stops moving rather than an error. The union would cut
across the axis the consumers actually use.

What is left after that is presence checks on optional fields, which is what this is.

# One status, not `state` plus `operation`

Those two fields were separate axes with only four live combinations between them, and
the overlap is what made autofocus signal itself two different ways -- a `state:
"autofocus"` at sweep start and an `operation: "autofocus"` per step, both live at once.
A single closed enum makes that unrepresentable.

`zlevel` and `channel_index` are 1-based on the wire, unlike the tile indices on
`TiledProgress`. They are counts of work done, not indices into anything -- nothing
subscripts by them -- so the number a reader sees is the number the producer sends, and
there is no offset to apply anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = [
    "FluorescenceAcquisitionStatus",
    "FluorescenceAcquisitionProgress",
]


class FluorescenceAcquisitionStatus(str, Enum):
    """What a fluorescence acquisition is doing.

    Four members for what used to be a `state` field crossed with an `operation` field.
    The three `ACQUIRING_*` members differ in *what is being counted*, which is the only
    thing a consumer needs them for: channels, z planes, or focus steps.

    A `str` mixin so a member compares equal to its own value. These are persisted
    nowhere, but they reach log lines, and `"acquiring-zstack"` reads better than
    `FluorescenceAcquisitionStatus.ACQUIRING_ZSTACK`.
    """

    # A plain multi-channel acquisition, counting channels.
    ACQUIRING_CHANNELS = "acquiring-channels"
    # A z-stack, counting planes within the channel named on the report.
    ACQUIRING_ZSTACK = "acquiring-zstack"
    # A focus sweep, counting objective positions. Distinct from `ACQUIRING_ZSTACK`
    # because a sweep steps the objective through a search range rather than acquiring a
    # stack, and calling its positions "Z-level" names something that is not running.
    ACQUIRING_AUTOFOCUS = "acquiring-autofocus"
    # The acquisition is over. Carries no counts: there is nothing left to count.
    FINISHED = "finished"


@dataclass(frozen=True)
class FluorescenceAcquisitionProgress:
    """One report from a fluorescence acquisition.

    Most fields are absent on most reports. `status` is the only one every report
    carries, and the only required field -- which is also what keeps this constructible
    on Python 3.8, where `kw_only` does not exist and required fields must come first.

    Equality is left generated, unlike `TiledProgress`. Nothing on this signal carries a
    numpy array, so there is no field whose `__eq__` returns an array and blows up the
    comparison -- which makes these usable in `assert emitted == [...]` and in a set.
    """

    status: FluorescenceAcquisitionStatus
    # Which channel this report is about. Absent on `FINISHED`, and empty rather than
    # absent on a focus sweep run without channel settings.
    channel: Optional[str] = None
    # Channels done and to do. Only a plain multi-channel acquisition counts these; a
    # z-stack reports them too, but its bar counts planes instead.
    channel_index: Optional[int] = None
    total_channels: Optional[int] = None
    # Planes, or focus-sweep positions, done and to do. Carried by both
    # `ACQUIRING_ZSTACK` and `ACQUIRING_AUTOFOCUS` -- which is exactly why a type per
    # acquisition routine would not have worked.
    zlevel: Optional[int] = None
    total_zlevels: Optional[int] = None
    # Which pass of a multi-pass focus sweep. Without it a coarse sweep followed by a
    # fine one looks like one bar inexplicably starting over.
    pass_index: Optional[int] = None
    total_passes: Optional[int] = None
