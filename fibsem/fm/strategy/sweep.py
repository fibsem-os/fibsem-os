from typing import ClassVar, Optional

from pydantic import Field

from fibsem.fm.strategy.base import AutoFocusStrategy, AutoFocusStrategyConfig
from fibsem.fm.structures import AutoFocusResult, FocusMethod, ZParameters


class SweepAutoFocusConfig(AutoFocusStrategyConfig):
    """Config for a single z-sweep autofocus."""
    name: ClassVar[str] = "Sweep"

    range: float = Field(default=20e-6, json_schema_extra={
        "label": "Range", "unit": "m", "scale": 1e6,
        "minimum": 1.0, "maximum": 2000.0, "step": 1.0, "decimals": 1,
        "tooltip": "Total z search range, centred on current position.",
    })
    step: float = Field(default=1e-6, json_schema_extra={
        "label": "Step", "unit": "m", "scale": 1e6,
        "minimum": 0.1, "maximum": 50.0, "step": 0.1, "decimals": 2,
        "tooltip": "Step size between z positions.",
    })
    method: FocusMethod = Field(default=FocusMethod.LAPLACIAN, json_schema_extra={
        "label": "Method", "items": list(FocusMethod),
        "tooltip": "Focus quality metric used to score each z position.",
    })


class SweepAutoFocusStrategy(AutoFocusStrategy[SweepAutoFocusConfig]):
    """Sweep through a z-range and pick the sharpest position."""
    config_class = SweepAutoFocusConfig

    def run(
        self,
        microscope,
        channel_settings=None,
        roi=None,
        stop_event=None,
    ) -> Optional[AutoFocusResult]:
        from fibsem.fm.calibration import run_autofocus

        z_params = ZParameters(
            zmin=-self.config.range / 2,
            zmax=self.config.range / 2,
            zstep=self.config.step,
        )
        return run_autofocus(
            microscope=microscope,
            channel_settings=channel_settings,
            z_parameters=z_params,
            method=self.config.method.value,
            stop_event=stop_event,
            roi=roi,
        )
