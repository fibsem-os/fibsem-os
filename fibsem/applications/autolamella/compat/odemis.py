import os
from copy import deepcopy
from datetime import datetime
from typing import List, Dict, Optional

from fibsem.microscopes.odemis_microscope import add_odemis_path
from fibsem.structures import FibsemImage, FibsemStagePosition
from fibsem.utils import save_yaml

from autolamella.protocol.validation import (
    MICROEXPANSION_KEY,
    MILL_POLISHING_KEY,
    MILL_ROUGH_KEY,
)
from autolamella.structures import AutoLamellaStage, Experiment, Lamella, LamellaState
from autolamella.workflows.core import log_status_message

add_odemis_path()

from odemis.acq.feature import CryoFeature, read_features
from odemis.acq.move import FM_IMAGING, MILLING, SEM_IMAGING
from odemis.acq.milling.tasks import MillingTaskSettings  # noqa: E402

def create_lamella_from_feature(feature: CryoFeature, 
                                path: str,
                                num: int,
                                reference_image_path: str,
                                workflow_stage: AutoLamellaStage = AutoLamellaStage.SetupLamella) -> Lamella:
    """Create a Lamella object from a CryoFeature object."""

    # sem_position = feature.posture_positions[SEM_IMAGING]
    feature_position = feature.posture_positions[MILLING]

    pos = FibsemStagePosition.from_odemis_dict(feature_position)

    # get the microscope state from the reference image
    image: FibsemImage = FibsemImage.load_odemis_image(reference_image_path)
    state = deepcopy(image.metadata.microscope_state)

    # create the lamella with the correct state
    lamella = Lamella(
            petname=feature.name.value,
            path=path,
            number=num,
            state=LamellaState(
                stage=workflow_stage,
                microscope_state=state,
                start_timestamp=datetime.timestamp(datetime.now()),
            ),
        )

    # add the milling protocol
    lamella.protocol = convert_milling_tasks_to_milling_protocol(feature.milling_tasks)

    log_status_message(lamella, "STARTED") # for logging

    # update the position
    lamella.state.microscope_state.stage_position = deepcopy(pos)
    lamella.state.microscope_state.stage_position.name = lamella.petname

    # save the reference image in required format and location
    save_reference_image(reference_image_path, lamella.path)

    return lamella

def save_reference_image(odemis_image_path: str, path: str, filename: str = "ref_alignment_ib") -> None:
    """Save odemis reference image as fibsem image for use in autolamella alignment."""
    
    # open odemis image, and convert to fibsem
    image = FibsemImage.load_odemis_image(odemis_image_path)
    
    # adjust md
    image.metadata.image_settings.save = True
    image.metadata.image_settings.path = path
    image.metadata.image_settings.filename = filename

    # save
    image.save()


def create_experiment_from_odemis(path: str, protocol: dict, name: str = "AutoLamella", program: str = "Odemis", method: str = "autolamella-on-grid") -> Experiment:
    """Create an experiment from an Odemis project folder."""

    experiment_name = f"{name}-{os.path.basename(path)}"
    experiment = Experiment(path=path, name=experiment_name, method=method)
    experiment.save()

    # save the protocol in the experiment folder
    save_yaml(os.path.join(experiment.path, "protocol.yaml"), protocol)

    return experiment

def load_experiment(path: str) -> Experiment:
    """Load an experiment from a path."""
    return Experiment.load(path)

def add_features_to_experiment(experiment: Experiment, features: List[CryoFeature]) -> Experiment:
    """Add features to an experiment."""
    odemis_project_path = os.path.dirname(experiment.path)

    for feature in features:
    
        # reference image path
        # reference_image_path = os.path.join(odemis_project_path, "test-image-FIBSEM-001.ome.tiff") 
        reference_image_path = os.path.join(odemis_project_path, f"{feature.name.value}-Reference-FIB.ome.tiff") # TODO: update for each feature

        # create lamella
        lamella = create_lamella_from_feature(feature,
                                        path=experiment.path, 
                                        num=len(experiment.positions) + 1, 
                                        reference_image_path=reference_image_path,
                                        workflow_stage=AutoLamellaStage.SetupLamella)
        
        experiment.positions.append(deepcopy(lamella))
        experiment.save()
    return experiment

def _add_features_from_odemis(path: str) -> List[FibsemStagePosition]:

    features = read_features(path)

    stage_positions = []
    for feature in features:
        feature_position = feature.posture_positions.get(MILLING, None)
        if feature_position is None:
            continue

        pos = FibsemStagePosition.from_odemis_dict(feature_position)
        pos.name = feature.name.value
        stage_positions.append(pos)
    return stage_positions

# convert odemis milling task to autolamella protocol
ODEMIS_TO_AUTOLAMELLA = {
    "trench": {
        "width": "lamella_width",
        "spacing": "lamella_height",
        "height": "trench_height",
        "depth": "depth",
        "current": "milling_current",
        "field_of_view": "hfw",
        "voltage": "milling_voltage",
        "name": "name",
        "pattern": "type",
        "mode": "patterning_mode", 
        "channel": "milling_channel", 
    },
    "microexpansion": {
        "spacing": "distance",
        "depth": "depth",
        "current": "milling_current",
        "field_of_view": "hfw",
        "voltage": "milling_voltage",
        "name": "name",
        "pattern": "type",
        "mode": "patterning_mode", 
        "channel": "milling_channel", 
    }
}

  
# remap the odemis milling task to autolamella protocol
def remap_milling_task(task: dict) -> dict:
    remap = ODEMIS_TO_AUTOLAMELLA[task["pattern"]] # TODO: make this more generic
    new_task = {}
    for key, value in task.items():
        if key in remap:
            new_key = remap[key]
            new_task[new_key] = value
        else:
            new_task[key] = value

    return new_task

def remap_milling_task_to_protocol(task: MillingTaskSettings) -> dict:

    task1 = task.to_json()
    task1.update(task1["milling"])
    task1.update(task1["patterns"][0]) # only support one pattern for now
    del task1["patterns"]
    del task1["milling"]
    
    protocol = remap_milling_task(task1)
    protocol["name"] = task.name
    
    return protocol

def _convert_to_stages_protocol(pprotocol) -> dict:

    rough_keys = [key for key in pprotocol if "rough" in key.lower()]
    polishing_keys = [key for key in pprotocol if "polishing" in key.lower()]
    microexpansion_keys = [key for key in pprotocol if "microexpansion" in key.lower()]
    
    # point is the same across all
    point = {"x": pprotocol[rough_keys[0]]["center_x"], 
            "y": pprotocol[rough_keys[0]]["center_y"]}

    protocol = {}
    protocol[MILL_ROUGH_KEY] = {
            "point": point,
            "stages": [pprotocol[key] for key in rough_keys]
    }
    protocol[MILL_POLISHING_KEY] = {
            "point": point,
            "stages": [pprotocol[key] for key in polishing_keys]
    }
    protocol[MICROEXPANSION_KEY] = {
            "point": point,
            "stages": [pprotocol[key] for key in microexpansion_keys]
    }
    return protocol

def convert_milling_tasks_to_milling_protocol(milling_tasks: Dict[str, MillingTaskSettings]) -> dict:
    tmp_protocol = {}
    for key in milling_tasks:
        tmp_protocol[key] = remap_milling_task_to_protocol(milling_tasks[key])
    protocol = _convert_to_stages_protocol(tmp_protocol)
    return protocol