"""Run CLEM acquisition for all lamellae in an AutoLamella experiment.yaml.
"""
from fibsem import microscope, utils
from fibsem.structures import FibsemStagePosition
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.fm.structures import ChannelSettings, ZParameters, FibsemRectangle, FocusMethod, AutoFocusSettings
from fibsem.fm.calibration import run_autofocus, run_coarse_fine_autofocus
import sys
from pathlib import Path
import numpy as np
import os
import yaml
#from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

EXPERIMENT_YAML = "C:\\Users\\User\\Desktop\\2026\\AutoLamella-2026-09-04-10-54\\experiment.yaml"
MILLING_ANGLE = 15.0
CHANNEL_SETTINGS =  [ChannelSettings(
                                        name="Channel-01",
                                        excitation_wavelength=650,
                                        emission_wavelength=None,
                                        exposure_time=0.005,
                                        power=0.01,
                                        gain=0.0,),
                    ChannelSettings(                    
                                        name="Channel-02",
                                        excitation_wavelength=550,
                                        emission_wavelength=550,
                                        exposure_time=0.5,
                                        power=0.1,
                                        gain=0.5),]

Z_STACK_SETTINGS = {"z_min": -2.0, "z_max": 2.0, "z_step": 0.5}
BINNING = 4


def read_milling_positions(experiment_yaml: Path) -> dict:
    """Return {lamella_name: stage_position} for every lamella with a MILLING pose."""
    with open(experiment_yaml) as f:
        data = yaml.safe_load(f)

    positions = {}
    for lamella in data.get("positions", []):
        name = lamella.get("petname") or lamella.get("name")
        defect_state = (lamella.get("defect") or {}).get("state", "NONE")
        if defect_state != "NONE":
            print(f"[INFO] Skipping {name}: defect state {defect_state}.")
            continue
        stage_position = lamella.get("poses", {}).get("MILLING", {}).get("stage_position")
        if name and stage_position:
            stage_position.setdefault("coordinate_system", "RAW")
            positions[name] = stage_position

    if not positions:
        raise ValueError(f"No MILLING stage positions found in {experiment_yaml}")
    return positions

def adjust_z_stage_offset(fixed_stage_z, lamella_z_position):
    delta_z = fixed_stage_z - lamella_z_position
    if MILLING_ANGLE == 15:
        shift_y = -0.22147 * delta_z
    elif MILLING_ANGLE == 18:
        shift_y = -0.264 * delta_z
    else:
        print("[WARNING] Milling angle is not setup.")
        shift_y = 0
    return shift_y

def detect_lamella(image_path, name_token=None):

    bonding_boxes = []
    model = YOLO('fibsem/czii_scripts/model_weights/lamella_best.pt')
    results = model(image_path)
    for result in results:
        bonding_boxes = result.boxes
        if bonding_boxes is None or len(bonding_boxes) == 0:
            print("No detections")
            return bonding_boxes
        else:
            result.save(filename=os.path.join(Path(image_path).parent, f"{name_token}_lamella_result.jpg"))
            return bonding_boxes


def run_clem(experiment_yaml: Path) -> None:

    fib_microscope, fib_settings = utils.setup_session(manufacturer="Thermo", config_path='fibsem/config/tfs-arctis-configuration.yaml')
    fl_microscope = FluorescenceMicroscope(parent=fib_microscope)
    print(fl_microscope.objective.position)
    input("Please insert objective and press ENTER to continue.")

    experiment_yaml = experiment_yaml.expanduser().resolve()
    experiment_dir = experiment_yaml.parent
    positions = read_milling_positions(experiment_yaml)
    print(f"[INFO] Loaded {len(positions)} lamella position(s) from {experiment_yaml}")

    for i, lamella in enumerate(positions):
        if i == 0:
            fixed_stage_z = positions.get(lamella, {}).get("z")
        stage_x = positions.get(lamella, {}).get("x")
        stage_y = positions.get(lamella, {}).get("y")
        lamella_stage_z = positions.get(lamella, {}).get("z")
        print(fl_microscope.objective.position)
        input(f"ENTER to continue with CLEM acquisition for lamella {lamella}. Press ENTER to continue.")
        if fl_microscope.objective.position < 0 and fib_microscope.get_stage_position().t != np.deg2rad(-180 + MILLING_ANGLE):
            fib_microscope.move_stage_absolute(FibsemStagePosition(x=stage_x, y=stage_y, z=fixed_stage_z, t=np.deg2rad(-180 + MILLING_ANGLE)))
            input("[ToDo] Insert objective using the SkyFluorescenceMicroscope. Press ENTER to continue.")         
        elif fl_microscope.objective.position >= 0 and fib_microscope.get_stage_position().t != np.deg2rad(-180 + MILLING_ANGLE):
            raise RuntimeError("Objective is inserted but stage is not at correct tilt. Please retract objective and move stage to correct tilt.")
        elif fl_microscope.objective.position < 0 and fib_microscope.get_stage_position().t == np.deg2rad(-180 + MILLING_ANGLE):
            fib_microscope.move_stage_absolute(FibsemStagePosition(x=stage_x, y=stage_y, z=fixed_stage_z, t=np.deg2rad(-180 + MILLING_ANGLE)))
            input("[ToDo] Insert objective using the SkyFluorescenceMicroscope. Press ENTER to continue.")           
        else:
            fib_microscope.move_stage_absolute(FibsemStagePosition(x=stage_x, y=stage_y, z=fixed_stage_z))

        #shift_y = adjust_z_stage_offset(lamella_z_position=lamella_stage_z, fixed_stage_z=fixed_stage_z)
        #if shift_y != 0 and not None:
        #    fib_microscope.move_stage_relative(FibsemStagePosition(y=shift_y))

        fl_microscope.objective.move_absolute(0.0080)

        focus_method = FocusMethod.TENENGRAD
        roi = FibsemRectangle(left=0.2, top=0.3, width=0.6, height=0.4)
  
        autofocus_settings = AutoFocusSettings.from_coarse_fine(
                                                                coarse_range=200e-6,
                                                                coarse_step=10e-6,
                                                                fine_range=25e-6,
                                                                fine_step=2e-6,
                                                                method=FocusMethod.TENENGRAD,
                                                                channel_name="Channel_01",
                                                            )

        autofocus_result = run_coarse_fine_autofocus(fl_microscope, autofocus_settings=autofocus_settings, channel_settings=CHANNEL_SETTINGS[0], roi=roi)
        bonding_boxes = []
        delta_focus = -20.0e-6
        while delta_focus < 20.0e-6 and len(bonding_boxes) == 0:
            fl_microscope.objective.move_absolute(autofocus_result.best_z + delta_focus)
            image_after = fl_microscope.acquire_image(fl_microscope, CHANNEL_SETTINGS[0])
            image_after.save(os.path.join(experiment_dir, lamella, "FM_image.tif"))
            bonding_boxes = detect_lamella(image_path=os.path.join(experiment_dir, lamella, "FM_image.tif"), name_token=lamella)
            delta_focus += 2.0e-6

        if len(bonding_boxes) > 0:
            center_x, center_y, width, height = bonding_boxes[0].xywhn[0].tolist()
            start_x = center_x - width / 2
            start_y = center_y - height / 2
            roi = FibsemRectangle(left=start_x, top=start_y, width=width, height=height)
            autofocus_parameters = ZParameters(zmin=-25e-6, zmax=25e-6, zstep=2e-6,)
            best_focus_fl = run_autofocus(microscope=fl_microscope, channel_settings=CHANNEL_SETTINGS[0], z_parameters=autofocus_parameters, method=focus_method)
            lamella_focus_position = best_focus_fl.best_z
        else:
            input("Auto-Lamella-Detection failed! Please manually focus on the lamella. Press OK when done.")
            lamella_focus_position = fl_microscope.objective.position

        fl_microscope.set_binning(BINNING)
        fl_microscope.objective.move_absolute(lamella_focus_position)
        zparams = ZParameters(zmin=Z_STACK_SETTINGS['z_min'], zmax=Z_STACK_SETTINGS['z_max'], zstep=Z_STACK_SETTINGS['z_step'])
        zstack = acquire_image(fl_microscope, channel_settings=CHANNEL_SETTINGS, z_parameters=zparams)
        zstack.save(os.path.join(experiment_dir, lamella, f"{lamella}_clem_data.tif"))
        

        



if __name__ == "__main__":
    run_clem(Path(sys.argv[1] if len(sys.argv) > 1 else EXPERIMENT_YAML))