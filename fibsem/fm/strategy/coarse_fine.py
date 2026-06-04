import logging
import threading
from typing import ClassVar, Optional

from pydantic import Field

from fibsem.fm.strategy.base import AutoFocusStrategy, AutoFocusStrategyConfig
from fibsem.fm.structures import AutoFocusResult, FocusMethod, ZParameters


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
        stop_event: Optional[threading.Event] = None,
    ) -> Optional[AutoFocusResult]:
        from fibsem.fm.calibration import run_autofocus

        initial_position = microscope.objective.position
        logging.info(f"Starting two-stage autofocus from {initial_position * 1e6:.1f} μm")

        # Stage 1: Coarse search
        logging.info(
            f"Coarse search: {self.config.coarse_range * 1e6:.1f} μm, "
            f"step {self.config.coarse_step * 1e6:.1f} μm"
        )
        coarse_result = run_autofocus(
            microscope=microscope,
            channel_settings=channel_settings,
            z_parameters=ZParameters(
                zmin=-self.config.coarse_range / 2,
                zmax=self.config.coarse_range / 2,
                zstep=self.config.coarse_step,
            ),
            method=self.config.method.value,
            stop_event=stop_event,
            roi=roi,
        )
        if coarse_result is None:
            logging.warning("Coarse autofocus cancelled or failed")
            return None

        # Move to coarse best position for fine search
        microscope.objective.move_absolute(coarse_result.best_z)

        # Stage 2: Fine search
        logging.info(
            f"Fine search around {coarse_result.best_z * 1e6:.1f} μm: "
            f"{self.config.fine_range * 1e6:.1f} μm, step {self.config.fine_step * 1e6:.1f} μm"
        )
        fine_result = run_autofocus(
            microscope=microscope,
            channel_settings=channel_settings,
            z_parameters=ZParameters(
                zmin=-self.config.fine_range / 2,
                zmax=self.config.fine_range / 2,
                zstep=self.config.fine_step,
            ),
            method=self.config.method.value,
            stop_event=stop_event,
            roi=roi,
        )
        if fine_result is None:
            logging.warning("Fine autofocus cancelled or failed")
            return coarse_result

        total_adjustment = fine_result.best_z - initial_position
        logging.info(
            f"Final position {fine_result.best_z * 1e6:.1f} μm "
            f"(total adjustment: {total_adjustment * 1e6:.1f} μm)"
        )
        return fine_result
