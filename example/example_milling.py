import logging

from fibsem import utils
from fibsem.milling import FibsemMillingStage, MillingAlignment
from fibsem.milling.patterning.patterns import RectanglePattern
from fibsem.structures.settings import (
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemRectangleSettings,
    Point,
)

"""
This script demonstrates how to use the milling module to mill a rectangle and two lines.

The script will:
    - connect to the microscope
    - setup milling
    - draw a rectangle and two lines
    - run milling
    - finish milling (restore the ion beam)
    - run a whole milling stage through the milling service

"""


def main():

    # connect to microscope
    microscope, settings = utils.setup_session(
        manufacturer="Demo", ip_address="localhost"
    )

    # setup milling stage (settings and alignment)
    stage = FibsemMillingStage(
        milling=FibsemMillingSettings(
            milling_current=2.0e-9,
            milling_voltage=30.0e3,
            hfw=100e-6,
            application_file="Si",
            patterning_mode="Serial",
        ),
        alignment=MillingAlignment(enabled=False),
    )

    # rectangle
    rectangle_shape = FibsemRectangleSettings(
        width=10.0e-6,
        height=10.0e-6,
        depth=2.0e-6,
        rotation=0.0,
        centre_x=0.0,
        centre_y=0.0,
    )

    # circle
    circle_shape = FibsemCircleSettings(
        radius=10.0e-6,
        depth=2.0e-6,
        centre_x=10e-6,
        centre_y=10e-6,
    )

    # line pattern
    line_shape = FibsemLineSettings(
        start_x=0.0,
        start_y=0.0,
        end_x=10.0e-6,
        end_y=10.0e-6,
        depth=1.0e-6,
    )

    logging.info("""\nMilling Pattern Example: """)
    logging.info(f"The current milling settings are: \n{stage.milling}")
    logging.info(f"The current rectangle pattern is \n{rectangle_shape}")
    logging.info(f"The current circle pattern ins is \n{circle_shape}")
    logging.info(f"The current line pattern is \n{line_shape}")
    logging.info(
        "---------------------------------- Milling ----------------------------------\n"
    )
    # setup patterns in a list
    patterns = [rectangle_shape, circle_shape, line_shape]

    # setup milling: the beam conditions and patterning defaults
    microscope.setup_milling(stage.milling)

    # draw patterns
    microscope.draw_patterns(patterns)

    # run milling, returning when the mill ends
    microscope.run_milling()

    # finish milling: clear the patterns, put the ion beam back as setup found it
    microscope.finish_milling()

    # or run a whole milling stage (setup, draw, mill, finish) through the milling service
    rect = RectanglePattern(
        width=10e-6, height=10e-6, depth=2e-6, rotation=0, point=Point(0, 0)
    )
    stage.pattern = rect

    microscope.milling.run_stage(stage)


if __name__ == "__main__":
    main()
