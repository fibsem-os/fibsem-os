


TOOLTIPS_PROTOCOL = {
    "turn_beams_off": "Turn off the beams after the workflow is complete.",
    "supervision": "Enabling supervision will pause the workflow before each step, and allow the user to confirm the step before proceeding.",
    "use_fiducial": "Mill a fiducial marker (X) for each lamella, to be used for alignment.",
    "use_microexpansion": "Use microexpansion joints (stress relief cuts), to reduce risk of lamella breakage.",
    "use_notch": "",
    "alignment_attempts": "The maximum number of re-alignment attempts",
    "alignment_at_milling_current": "Acquire alignment images at the milling current, rather than the imaging current.",
    "checkpoint": "The machine learning model to be used during the workflow. Models are automatically downloaded from huggingface.com",
    "milling_angle": "The tilt angle at which the lamella will be milled. This is the angle between the grid and the ion beam.",
    "undercut_tilt_angle": "The tilt between each undercut milling step. The number of steps is specified in the milling protocol",
    "take_final_reference_images": "Acquire reference images after each workflow stage (Recommended)",
}

TOOLTIPS = {
    "protocol": TOOLTIPS_PROTOCOL,
}