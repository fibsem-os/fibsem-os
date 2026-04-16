# do contrast test polishing
protocol = {
    "milling": {
        "polish": {
            "stages": [{
        "application_file": "autolamella",
        "cross_section": "CleaningCrossSection",
        "hfw": 80e-6,
        "height": 6.0e-07,
        "width": 6.0e-06,
        "depth": 4.0e-07,
        "milling_current": 6.0e-11,
        "milling_voltage": 3.0e3,
        "type": "Rectangle",
            }
        ],
        "point": {
            "x": 0.0,
            "y": 5e-6,}
        }
    }
}

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings, BeamType
from fibsem import utils, acquire, milling, alignment
from fibsem.structures import ImageSettings, BeamType
from fibsem.ui.utils import _draw_milling_stages_on_image
from fibsem.milling import get_milling_stages
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint
import logging
import copy

def adaptive_mill_polishing(microscope: FibsemMicroscope, settings: MicroscopeSettings, protocol: dict, parent_ui=None):
    """Adaptive lamella polishing using contrast detection
        
    # ref: https://pure.mpg.de/rest/items/item_2415133/component/file_3148891/content
    # ref: https://www.biorxiv.org/content/10.1101/2024.02.21.581285v1.full.pdf
    # target acquisition
    # set up imaging (voltage = 3kv, line integration= 20, dwell time 100ns, res [3072x2188])

    # constrast test
    # set electron imaging (voltage = 3kv, line integration = 1, dwell time 200ns)
    # set up top milling pattern (rectangule) (stepsize = 100nm)

    # steps:
    # do standard rough milling
    # loop until target is found

    """
    logging.info({"msg": "adaptive_mill_polishing", "protocol": protocol})

    stages = get_milling_stages("polish", protocol["milling"])
    point = stages[0].pattern.point

    # beam settings
    initial_voltage = microscope.get("voltage", BeamType.ELECTRON)
    microscope.set("voltage", 3000, BeamType.ELECTRON)

    # imaging settings
    settings.image.hfw = stages[0].milling.hfw
    settings.image.save = True
    settings.image.beam_type = BeamType.ION

    # contrast detection protocol
    experimental_protocol = protocol["options"].get("experimental", {})
    contrast_protocol = experimental_protocol.get("adaptive_polishing", {})

    # contrast image settings
    contrast_image_settings = copy.deepcopy(settings.image)
    contrast_image_settings.resolution = contrast_protocol.get("image_resolution", (3072, 2188))
    contrast_image_settings.line_integration = contrast_protocol.get("image_line_integration", 20)
    contrast_image_settings.dwell_time = contrast_protocol.get("image_dwell_time", 100e-9)
    contrast_image_settings.beam_type = BeamType.ELECTRON

    # contrast test settings
    threshold = contrast_protocol.get("threshold", 100)
    step_size = contrast_protocol.get("step_size", 100e-9)
    step_limit = contrast_protocol.get("step_limit", 100)
    n_step = 0
    while True:

        stages = get_milling_stages("polish", protocol["milling"], point=point)

        from pprint import pprint
        pprint(stages[0].pattern)

        # draw rectangle (milling pattern ion beam)
        image = acquire.acquire_image(microscope, settings.image)
        _draw_milling_stages_on_image(image, stages)    

        # mill pattern
        milling.mill_stages(microscope, stages)

        # acquire image (contrast detection)
        contrast_image_settings.filename = f"thickness-dependent-contrast-test-{n_step}"
        image = acquire.acquire_image(microscope, contrast_image_settings)

        # measure brightness and contrast
        brightness = np.mean(image.data)
        contrast = np.std(image.data)
        logging.info({"msg": "thickness-dependent-contrast-test",  
                    "brightness": brightness, "contrast": contrast, "threshold": threshold, 
                    "n_step": n_step, "step_size": step_size, "step_limit": step_limit})

        if brightness < threshold:
            break
        
        # manual validation
        if parent_ui is None:
            
            plt.title(f"Step {n_step}/{step_limit} - Brightness: {brightness:.2f}, Contrast: {contrast:.2f}, Threshold: {threshold}")
            plt.imshow(image.data, cmap="gray")
            plt.show()
        
            response = input(f"Has the contrast faded? (y/n)")

            if response == 'y':
                break

        # ml classifier
        # TODO: implement a classifier to determine if the contrast has faded

        n_step += 1
        if n_step > step_limit:
            break

        # move pattern down by step
        point.y -= step_size

    logging.info({"msg": "adaptive_mill_polishing_finished", 
                  "n_step": n_step, "step_size": step_size, "step_limit": step_limit, 
                  "threshold": threshold, "brightness": brightness, "contrast": contrast})

    # restore beam settings
    microscope.set("voltage", initial_voltage, BeamType.ELECTRON)

    return