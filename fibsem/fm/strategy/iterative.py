import logging
from typing import ClassVar, Optional

from pydantic import Field

from fibsem.fm.strategy.base import AutoFocusStrategy, AutoFocusStrategyConfig
from fibsem.fm.structures import FocusMethod, ZParameters


class IterativeAutoFocusConfig(AutoFocusStrategyConfig):
    """Config for an iterative autofocus that narrows the search range each iteration."""
    name: ClassVar[str] = "Iterative"

    initial_range: float = Field(default=50e-6, json_schema_extra={
        "label": "Initial Range", "unit": "m", "scale": 1e6,
        "minimum": 1.0, "maximum": 2000.0, "step": 5.0, "decimals": 1,
        "tooltip": "Total z search range for the first iteration.",
    })
    initial_step: float = Field(default=5e-6, json_schema_extra={
        "label": "Initial Step", "unit": "m", "scale": 1e6,
        "minimum": 0.1, "maximum": 100.0, "step": 1.0, "decimals": 1,
        "tooltip": "Step size for the first iteration.",
    })
    num_iterations: int = Field(default=3, json_schema_extra={
        "label": "Iterations", "type": int,
        "minimum": 1, "maximum": 10, "step": 1, "decimals": 0,
        "tooltip": "Number of focus iterations.",
    })
    reduction_factor: float = Field(default=0.5, json_schema_extra={
        "label": "Reduction Factor",
        "minimum": 0.1, "maximum": 0.9, "step": 0.05, "decimals": 2,
        "tooltip": "Range and step are multiplied by this factor each iteration.",
    })
    method: FocusMethod = Field(default=FocusMethod.LAPLACIAN, json_schema_extra={
        "label": "Method", "items": list(FocusMethod),
        "tooltip": "Focus quality metric used to score each z position.",
    })


class IterativeAutoFocusStrategy(AutoFocusStrategy[IterativeAutoFocusConfig]):
    """Repeated sweeps with a shrinking range and step, converging on best focus."""
    config_class = IterativeAutoFocusConfig

    def run(
        self,
        microscope,
        channel_settings=None,
        roi=None,
        stop_event=None,
    ) -> Optional[float]:
        from fibsem.fm.calibration import run_autofocus

        current_range = self.config.initial_range
        current_step = self.config.initial_step
        best_z: Optional[float] = None

        for i in range(self.config.num_iterations):
            logging.info(
                f"Iterative autofocus [{i + 1}/{self.config.num_iterations}]: "
                f"range={current_range * 1e6:.1f} μm, step={current_step * 1e6:.2f} μm"
            )
            z_params = ZParameters(
                zmin=-current_range / 2,
                zmax=current_range / 2,
                zstep=current_step,
            )
            result = run_autofocus(
                microscope=microscope,
                channel_settings=channel_settings,
                z_parameters=z_params,
                method=self.config.method.value,
                stop_event=stop_event,
                roi=roi,
            )
            if result is None:
                return best_z

            best_z = result
            current_range *= self.config.reduction_factor
            current_step *= self.config.reduction_factor

        return best_z
