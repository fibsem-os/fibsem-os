"""Base classes for instruments co-located inside the FIBSEM vacuum chamber.

Usage pattern
-------------
Any secondary device inside the FIBSEM chamber (FM, Raman, EBSD, …) inherits
``ChamberDevice`` and gets coordinate transforms, accessibility checks, and
orientation checks for free.  Device-specific behaviour (e.g. sample-plane
movement that depends on an insertion angle) is implemented on the subclass:

    from fibsem.devices.base import ChamberDevice, ChamberDeviceGeometry
    from fibsem.structures import FibsemStagePosition, RangeLimit

    geometry = ChamberDeviceGeometry(
        offset=FibsemStagePosition(x=48.8e-3, y=0, z=0),
        available_range={"x": RangeLimit(40e-3, 60e-3), ...},
        default_orientation=FibsemStagePosition(r=..., t=...),
        available_orientations=["FM", "FIB"],
    )

    class RamanProbe(ChamberDevice):
        def __init__(self, parent, geometry):
            super().__init__(geometry=geometry, parent=parent)

    probe.transform_to_device_frame(fibsem_pos)
    probe.move_to_device_position(fibsem_pos)
    probe.is_accessible(fibsem_pos)
    probe.has_valid_orientation()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import FibsemStagePosition, RangeLimit


@dataclass
class ChamberDeviceGeometry:
    """Physical geometry of an instrument mounted inside the FIBSEM vacuum chamber.

    Captures *where* the device is relative to the FIBSEM, not how it views
    the sample.  Device-specific viewing angles (e.g. ``column_tilt`` for an
    angled objective) belong on the subclass.

    Attributes:
        offset: Position of the device in the lab frame relative to the FIBSEM
            eucentric point (metres). Only x/y/z are used; r/t are ignored.
            Defined with the stage at ``default_orientation``.
        available_range: Per-axis stage position limits within which the device
            can operate safely, keyed by ``"x"``, ``"y"``, ``"z"``.
        default_orientation: Stage r/t at which the device is used. The x/y/z
            fields are ignored; only r and t matter.
        available_orientations: Named orientations (e.g. ``["FM", "FIB"]``) at
            which this device may be operated. Same keys as
            ``FibsemMicroscope.orientations``.
    """

    offset: "FibsemStagePosition"
    available_range: Dict[str, "RangeLimit"]
    default_orientation: "FibsemStagePosition"
    available_orientations: List[str]

    def to_dict(self) -> dict:
        return {
            "offset": self.offset.to_dict(),
            "available_range": {
                k: v.to_dict() for k, v in self.available_range.items()
            },
            "default_orientation": self.default_orientation.to_dict(),
            "available_orientations": list(self.available_orientations),
        }

    @staticmethod
    def from_dict(d: dict) -> "ChamberDeviceGeometry":
        from fibsem.structures import FibsemStagePosition, RangeLimit

        return ChamberDeviceGeometry(
            offset=FibsemStagePosition.from_dict(d["offset"]),
            available_range={
                k: RangeLimit.from_dict(v) for k, v in d["available_range"].items()
            },
            default_orientation=FibsemStagePosition.from_dict(d["default_orientation"]),
            available_orientations=list(d["available_orientations"]),
        )


class ChamberDevice:
    """Base class for instruments co-located inside the FIBSEM vacuum chamber.

    Provides coordinate transforms, accessibility checks, and orientation checks
    for any device mounted at a fixed offset from the FIBSEM eucentric point.

    Sample-plane movement (``stable_move``) raises ``NotImplementedError`` on
    the base class because the correct decomposition depends on the device's
    viewing geometry (e.g. insertion tilt angle), which varies per device type.
    Subclasses that need it should override ``stable_move``.

    Attributes:
        geometry: Physical geometry of the device mount. ``None`` means the device
            is inline with the FIBSEM (no offset transform needed).
        parent: The parent :class:`~fibsem.microscope.FibsemMicroscope` used for
            stage access and system settings.
    """

    def __init__(
        self,
        geometry: "ChamberDeviceGeometry",
        parent: "FibsemMicroscope",
    ) -> None:
        self.geometry = geometry
        self.parent = parent

    # ------------------------------------------------------------------
    # Coordinate transform
    # ------------------------------------------------------------------

    def transform_to_device_frame(
        self, fibsem_pos: "FibsemStagePosition"
    ) -> "FibsemStagePosition":
        """Convert a FIBSEM-frame stage position to the equivalent position for
        this device to image the same physical point.

        Uses ``FibsemMicroscope.get_target_position`` to convert the input
        position to the device orientation (``"FM"``), then adds
        ``geometry.offset`` to account for the physical separation between the
        FIBSEM column and this device.

        Note: the target orientation is currently hardcoded to ``"FM"``.
        Subclasses for other device types (Raman, EBSD, …) should override this
        method and pass the appropriate orientation key once it is registered in
        ``FibsemMicroscope.orientations``.

        Args:
            fibsem_pos: Stage position at which the feature was observed under
                the FIBSEM.

        Returns:
            Stage position to move to so that the device sees the same feature.

        Raises:
            ValueError: If ``geometry`` is not configured.
        """
        if self.geometry is None:
            raise ValueError("ChamberDeviceGeometry is not configured on this device.")

        # move to target orientation to get the correct stage rotation for the transform
        target_position = self.parent.get_target_position(
            fibsem_pos, target_orientation="FM"
        )
        device_position = target_position + self.geometry.offset

        return device_position

    def transform_to_fibsem_frame(
        self, device_pos: "FibsemStagePosition"
    ) -> "FibsemStagePosition":
        """Convert a device-frame stage position to the equivalent FIBSEM-frame
        position.

        Inverse of ``transform_to_device_frame``.

        Args:
            device_pos: Stage position under this device.

        Returns:
            Stage position at which the same physical point would be observed under
            the FIBSEM.

        Raises:
            ValueError: If ``geometry`` is not configured.
        """
        if self.geometry is None:
            raise ValueError("ChamberDeviceGeometry is not configured on this device.")

        # move to target orientation to get the correct stage rotation for the transform
        target_position = device_pos - self.geometry.offset
        target_position = self.parent.get_target_position(
            target_position, target_orientation="SEM"
        )

        return target_position

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def stable_move(self, dx: float, dy: float) -> "FibsemStagePosition":
        """Move the stage in the sample plane as seen by this device.

        Not implemented on the base class — the correct Y/Z decomposition
        depends on the device's viewing geometry (e.g. insertion tilt angle).
        Subclasses that need sample-plane movement should override this method.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement stable_move. "
            "Override this method on the subclass."
        )

    def move_to_device_position(
        self, fibsem_pos: "FibsemStagePosition"
    ) -> "FibsemStagePosition":
        """Move the stage so that this device images the same point that was at
        ``fibsem_pos`` in the FIBSEM frame.

        Args:
            fibsem_pos: Stage position recorded while imaging under the FIBSEM.

        Returns:
            Actual stage position after the move.
        """
        device_pos = self.transform_to_device_frame(fibsem_pos)
        self.parent.safe_absolute_stage_movement(device_pos)
        return self.parent.get_stage_position()

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def is_accessible(self, fibsem_pos: "FibsemStagePosition") -> bool:
        """Return True if the device can reach the point at ``fibsem_pos``.

        Transforms the position to the device frame and checks it against
        ``geometry.available_range``.

        Raises:
            ValueError: If ``geometry`` is not configured.
        """
        device_pos = self.transform_to_device_frame(fibsem_pos)
        return device_pos.is_within_limits(self.geometry.available_range)

    def has_valid_orientation(
        self, stage_position: Optional["FibsemStagePosition"] = None
    ) -> bool:
        """Return True if the current (or given) stage orientation is in
        ``geometry.available_orientations``.

        Returns True unconditionally when ``geometry`` is not configured.
        """
        if self.geometry is None:
            return True
        orientation = self.parent.get_stage_orientation(stage_position)
        return orientation in self.geometry.available_orientations
