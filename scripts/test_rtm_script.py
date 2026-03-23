from matplotlib import pyplot as plot
from IPython.display import clear_output
import numpy

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import *
from autoscript_sdb_microscope_client.structures import *

# microscope = SdbMicroscopeClient()
# microscope.connect("localhost")

# Define a helper function to convert RTM data to an image
def get_images_from_rtm_data(rtm_data, rtm_positions):
    result = []

    for pattern in rtm_positions:
        try:
            # Find pattern in data list for the pattern in position list
            data_set = next(x for x in rtm_data if x.pattern_id == pattern.pattern_id)
        except Exception:
            continue

        # Prepare image for drawing
        x_min = min(pattern.positions, key=lambda p: p[0])[0]
        x_max = max(pattern.positions, key=lambda p: p[0])[0]
        y_min = min(pattern.positions, key=lambda p: p[1])[1]
        y_max = max(pattern.positions, key=lambda p: p[1])[1]

        arr = numpy.full((y_max - y_min + 1, x_max - x_min + 1), -1, dtype=int)
        for i in range(1, len(pattern.positions)):
            arr[pattern.positions[i][1] - y_min, pattern.positions[i][0] - x_min] = data_set.values[i]

        # Take only lines and rows, where at least one pixel is valid
        arr = arr[numpy.any(arr > -1, axis=1), :]
        arr = arr[:, numpy.any(arr > -1, axis=0)]
        result.append(arr)

    return result

from fibsem import utils
from fibsem.structures import FibsemRectangleSettings
from fibsem.milling.patterning import RectanglePattern

m1, settings = utils.setup_session()
microscope = m1.connection

m1.clear_patterns()
m1.draw_rectangle(FibsemRectangleSettings(width=10e-6, height=10e-6, depth=1e-6, centre_x=0, centre_y=0, cross_section=CrossSection))

# m1.draw_rectangle(FibsemRectangleSettings(width=10e-6, height=5e-6, depth=1e-6, centre_x=-5e-6, centre_y=15e-6))
microscope.patterning.mode = "Parallel"

input()

position_settings = GetRtmPositionSettings(None, RtmCoordinateSystem.IMAGE_PIXELS)

print("Setting RTM mode to low resolution mode...")
microscope.patterning.real_time_monitor.mode = RtmMode.LOW_RESOLUTION

print("Starting RTM acquisition...")
microscope.patterning.real_time_monitor.restart()

print("Starting patterning...")
microscope.patterning.start()

rtm_positions = []
try:
    while microscope.patterning.state != PatterningState.IDLE:

        # Read pattern point intensities
        rtm_data = microscope.patterning.real_time_monitor.get_data()

        try:
            dset = rtm_data[0]
            print(dset.pattern_id)
            print(rtm_positions[0].pattern_id, rtm_positions[0].coordinate_system)
        except:
            pass

        if len(rtm_positions) == 0:
            # Read pattern point positions only once
            rtm_positions = microscope.patterning.real_time_monitor.get_positions(position_settings)

        # If both positions and intensities are present, process them
        if len(rtm_positions) > 0 and len(rtm_data) > 0:
            images = get_images_from_rtm_data(rtm_data, rtm_positions)
            for image in images:
                # Plot the image
                clear_output(wait=True)
                plot.imshow(image, cmap='gray')
                plot.show()
finally:
    microscope.patterning.stop()
    microscope.patterning.real_time_monitor.stop()