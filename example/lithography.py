import os
from pprint import pprint

import numpy as np
from PIL import Image

from fibsem import utils
from fibsem.structures import FibsemBitmapSettings, FibsemMillingSettings

# from fibsem.ui import windows

BASE_PATH = os.path.dirname(__file__)


def save_profile_to_bmp(arr: np.ndarray, fname: str = "profile.bmp"):

    # scale values to int
    arr = (arr / np.max(arr)) * 255
    arr = arr.astype(np.uint8)

    # save profile
    Image.fromarray(arr).convert("RGB").save(fname)


def main():

    PROTOCOL_PATH = os.path.join(BASE_PATH, "protocol_lithography.yaml")
    microscope, settings = utils.setup_session(protocol_path=PROTOCOL_PATH)

    # move to the milling angle
    # stage_position = FibsemStagePosition(
    #     r=np.deg2rad(settings.protocol["stage_rotation"]),
    #     t=np.deg2rad(settings.protocol["stage_tilt"]),
    # )
    # microscope.safe_absolute_stage_movement(stage_position)

    microscope.move_to_orientation("FIB")

    # eucentric, select position
    # windows.ask_user_movement(
    #     microscope,
    #     settings,
    #     msg_type="eucentric",
    #     msg="Select a position to mill the pattern.",
    # )

    # lens profile files
    npy_path = os.path.join(BASE_PATH, settings.protocol["profile"])
    bmp_path = os.path.join(BASE_PATH, "profile.bmp")

    # load milling properties
    profile = np.load(npy_path)
    pixel_size = settings.protocol["pixelsize"]
    lens_height = settings.protocol["milling"]["height"]
    lens_width = settings.protocol["milling"]["width"]

    # save profile to bmp
    save_profile_to_bmp(profile, bmp_path)

    # milling setup
    microscope.setup_milling(
        FibsemMillingSettings(
            milling_current=settings.protocol["milling"]["milling_current"],
            application_file=settings.protocol["application_file"],
        )
    )

    # surface milling, then the lens itself, from the same bitmap
    for depth in (
        settings.protocol["milling"]["initial_depth"],
        settings.protocol["milling"]["milling_depth"],
    ):
        microscope.draw_pattern(
            FibsemBitmapSettings(
                width=lens_width,
                height=lens_height,
                depth=depth,
                centre_x=0,
                centre_y=0,
                path=bmp_path,
            )
        )
    microscope.run_milling()
    microscope.finish_milling()


if __name__ == "__main__":
    main()
