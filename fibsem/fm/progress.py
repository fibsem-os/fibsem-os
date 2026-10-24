"""What a fluorescence acquisition says about itself while it runs.

`FluorescenceMicroscope.acquisition_progress_signal` carries a
`FluorescenceAcquisitionProgress`. The signal is fluorescence only: it is declared on
`FluorescenceMicroscope` and every emit site is in `fibsem/fm`, so nothing here carries
a modality, unlike `imaging.tiling.progress.TiledProgress`. The types are named for the
modality anyway, because `FibsemImageSettingsWidget` has an unrelated
`acquisition_progress_signal` for the beam side.

One flat record rather than a type per acquisition routine: both consumers first ask
whether there is a z count, and both the z-stack and the autofocus sweep carry one. One
closed status rather than separate state and operation fields, so autofocus cannot
signal itself two ways at once.

`zlevel` and `channel_index` are 1-based on the wire, unlike the tile indices on
`TiledProgress`: they are counts of work done, not indices into anything, so there is no
offset to apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fibsem.progress import ProgressStatus, ends

__all__ = [
    "FluorescenceAcquisitionStatus",
    "FluorescenceAcquisitionProgress",
]


class FluorescenceAcquisitionStatus(ProgressStatus):
    """What a fluorescence acquisition is doing.

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


ends(FluorescenceAcquisitionStatus.FINISHED)


@dataclass(frozen=True)
class FluorescenceAcquisitionProgress:
    """One report from a fluorescence acquisition.

    Most fields are absent on most reports. `status` is the only one every report
    carries, and the only required field.

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
    # `ACQUIRING_ZSTACK` and `ACQUIRING_AUTOFOCUS`.
    zlevel: Optional[int] = None
    total_zlevels: Optional[int] = None
    # Which pass of a multi-pass focus sweep. Without it a coarse sweep followed by a
    # fine one looks like one bar inexplicably starting over.
    pass_index: Optional[int] = None
    total_passes: Optional[int] = None
