

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from fibsem import acquire, alignment, utils
from fibsem.applications.autolamella.config import PROTOCOL_PATH
from fibsem.milling.base import get_milling_stages
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    MicroscopeState,
)


@dataclass
class Lamella:
    state: MicroscopeState
    reference_image: FibsemImage
    path: Path
    num: int

def main():
    
    microscope, settings = utils.setup_session(protocol_path=PROTOCOL_PATH)

    # take a reference image    
    settings.image.filename = "grid_reference"
    settings.image.beam_type = BeamType.ION
    settings.image.hfw = 900e-6
    settings.image.save = True
    acquire.take_reference_images(microscope, settings.image)

    # select positions
    experiment: List[Lamella] = []
    lamella_no = 1
    settings.image.hfw = 80e-6
    base_path = settings.image.path

    while True:
        response = input(f"""Move to the desired position. 
        Do you want to select another lamella? [y]/n {len(experiment)} selected so far.""")

        # store lamella information
        if response.lower() in ["", "y", "yes"]:
            
            # set filepaths
            path = os.path.join(base_path, f"{lamella_no:02d}")
            settings.image.path = path
            settings.image.filename = "ref_lamella"
            acquire.take_reference_images(microscope, settings.image)

            lamella = Lamella(
                state=microscope.get_microscope_state(),
                reference_image=acquire.new_image(microscope, settings.image),
                path=path,
                num=lamella_no
            )
            experiment.append(lamella)
            lamella_no += 1
        else:
            break

    # sanity check
    if len(experiment) == 0:
        logging.info("No lamella positions selected. Exiting.")
        return

    # mill (rough, polish)
    workflow_stages = ["mill_rough", "mill_polishing"]
    for stage_no, stage_name in enumerate(workflow_stages):
        
        logging.info(f"Starting milling stage {stage_no}")

        for lamella in experiment:

            logging.info(f"Starting lamella {lamella.num:02d}")

            # return to lamella
            microscope.set_microscope_state(lamella.state)

            # realign
            alignment.beam_shift_alignment_v2(microscope, lamella.reference_image)
                       
            if stage_no == 0:
                microexpansion_stage = get_milling_stages("microexpansion", settings.protocol["milling"])
                microexpansion_stage[0].run(microscope)

            # get trench milling pattern, and mill
            milling_stages = get_milling_stages(stage_name, settings.protocol["milling"])
            for milling_stage in milling_stages:
                logging.info(f"Running milling stage {milling_stage.name} for lamella {lamella.num:02d}")
                milling_stage.run(microscope)

            # retake reference image
            settings.image.path = lamella.path
            settings.image.filename = f"ref_mill_stage_{stage_no:02d}"
            lamella.reference_image = acquire.new_image(microscope, settings.image)

            if stage_no == len(workflow_stages) - 1:
                # take final reference images
                settings.image.filename = "ref_final"
                acquire.take_reference_images(microscope, settings.image)

    logging.info(f"Finished autolamella example. {len(experiment)} lamellas processed.")


if __name__ == "__main__":
    main()
