from typing import ClassVar, Optional

from pydantic import Field

from fibsem.fm.strategy.base import AutoFocusStrategy, AutoFocusStrategyConfig
from fibsem.fm.structures import AutoFocusSettings, FocusMethod


class CoarseFineAutoFocusConfig(AutoFocusStrategyConfig):
    """Config for a two-stage coarse-then-fine autofocus."""
    name: ClassVar[str] = "CoarseFine"

    coarse_range: float = Field(default=50e-6, json_schema_extra={
        "label": "Coarse Range", "unit": "m", "scale": 1e6,
        "minimum": 1.0, "maximum": 2000.0, "step": 5.0, "decimals": 1,
        "tooltip": "Total range for the coarse sweep.",
    })
    coarse_step: float = Field(default=5e-6, json_schema_extra={
        "label": "Coarse Step", "unit": "m", "scale": 1e6,
        "minimum": 0.1, "maximum": 100.0, "step": 1.0, "decimals": 1,
        "tooltip": "Step size for the coarse sweep.",
    })
    fine_range: float = Field(default=10e-6, json_schema_extra={
        "label": "Fine Range", "unit": "m", "scale": 1e6,
        "minimum": 0.5, "maximum": 2000.0, "step": 1.0, "decimals": 1,
        "tooltip": "Total range for the fine sweep around the coarse optimum.",
    })
    fine_step: float = Field(default=1e-6, json_schema_extra={
        "label": "Fine Step", "unit": "m", "scale": 1e6,
        "minimum": 0.05, "maximum": 20.0, "step": 0.1, "decimals": 2,
        "tooltip": "Step size for the fine sweep.",
    })
    method: FocusMethod = Field(default=FocusMethod.LAPLACIAN, json_schema_extra={
        "label": "Method", "items": list(FocusMethod),
        "tooltip": "Focus quality metric used to score each z position.",
    })


class CoarseFineAutoFocusStrategy(AutoFocusStrategy[CoarseFineAutoFocusConfig]):
    """Coarse sweep followed by a fine sweep around the coarse optimum."""
    config_class = CoarseFineAutoFocusConfig

    def run(
        self,
        microscope,
        channel_settings=None,
        roi=None,
        stop_event=None,
    ) -> Optional[float]:
        from fibsem.fm.calibration import run_coarse_fine_autofocus

        af_settings = AutoFocusSettings(
            coarse_range=self.config.coarse_range,
            coarse_step=self.config.coarse_step,
            coarse_enabled=True,
            fine_range=self.config.fine_range,
            fine_step=self.config.fine_step,
            fine_enabled=True,
            method=self.config.method,
        )
        return run_coarse_fine_autofocus(
            microscope=microscope,
            autofocus_settings=af_settings,
            channel_settings=channel_settings,
            roi=roi,
        )
