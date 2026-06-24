from fibsem.autofunctions.gamma import auto_gamma, apply_gamma, apply_clahe
from fibsem.autofunctions.metrics import (
    laplacian_focus_measure,
    sobel_focus_measure,
    variance_focus_measure,
    tenengrad_focus_measure,
    get_focus_measure_function,
    find_best_focus_plane,
    calculate_focus_quality,
)
from fibsem.autofunctions.stacking import (
    pixel_based_focus_selection,
    block_based_focus_selection,
    create_focus_stack_from_selection,
    create_pixel_based_focus_stack,
    create_block_based_focus_stack,
)
from fibsem.autofunctions.integration import frame_integration, adaptive_frame_integration
from fibsem.autofunctions.charge_neutralisation import auto_charge_neutralisation
from fibsem.autofunctions.acb import (
    AutoContrastBrightnessSettings,
    AutoContrastBrightnessIteration,
    AutoContrastBrightnessResult,
    run_auto_contrast_brightness,
)
from fibsem.autofunctions.autofocus import (
    AutoFocusSettings,
    AutoFocusIteration,
    AutoFocusResult,
    FocusMethod,
    FocusSweepPass,
    run_auto_focus,
)

def __getattr__(name):
    if name == "percentile_stretch":
        from fibsem.imaging.utils import percentile_stretch
        return percentile_stretch
    if name == "ImageStats":
        from fibsem.structures import ImageStats
        return ImageStats
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ImageStats",
    "percentile_stretch",
    "AutoContrastBrightnessSettings",
    "AutoContrastBrightnessIteration",
    "AutoContrastBrightnessResult",
    "run_auto_contrast_brightness",
    "AutoFocusSettings",
    "AutoFocusIteration",
    "AutoFocusResult",
    "FocusMethod",
    "FocusSweepPass",
    "run_auto_focus",
    "auto_gamma",
    "apply_gamma",
    "apply_clahe",
    "laplacian_focus_measure",
    "sobel_focus_measure",
    "variance_focus_measure",
    "tenengrad_focus_measure",
    "get_focus_measure_function",
    "find_best_focus_plane",
    "calculate_focus_quality",
    "pixel_based_focus_selection",
    "block_based_focus_selection",
    "create_focus_stack_from_selection",
    "create_pixel_based_focus_stack",
    "create_block_based_focus_stack",
    "frame_integration",
    "adaptive_frame_integration",
    "auto_charge_neutralisation",
]
