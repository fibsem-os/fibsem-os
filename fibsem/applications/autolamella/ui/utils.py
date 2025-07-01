import datetime
import logging
import os

from fibsem.ui import utils as fui
from PyQt5 import QtWidgets

from autolamella import config as cfg
from autolamella.structures import Experiment, create_new_experiment


def setup_experiment_ui_v2(
    parent_ui: QtWidgets.QMainWindow, 
    new_experiment: bool = True
):
    """Helper dialog to create or load an experiment."""

    # new_experiment
    if new_experiment:
        experiment = create_experiment_ui(parent_ui)
        return experiment

    experiment = load_experiment_ui(parent_ui)
    if experiment is not None:
        logging.info(f"Experiment {experiment.name}. {len(experiment.positions)} lamella loaded from {experiment.path}")

    return experiment

def load_experiment_ui(parent: QtWidgets.QMainWindow) -> Experiment:
    """Helper function to load an experiment via a dialog"""
    
    experiment_path = fui.open_existing_file_dialog(
        msg="Select an experiment file", 
        path=cfg.LOG_PATH, 
        parent=parent
    )

    if experiment_path == "":
        logging.info("No path selected")
        return

    return Experiment.load(experiment_path)


def create_experiment_ui(parent: QtWidgets.QMainWindow) -> Experiment:
    """Helper function to create a new experiment via a dialog:
    """
    
    # get experiment path
    experiment_path = fui.open_existing_directory_dialog(
        msg="Select a directory to save the experiment",
        path=cfg.LOG_PATH,
        parent=parent,
    )
    if experiment_path == "":
        logging.warning("No path selected")
        return
    
    if not os.path.exists(experiment_path):
        logging.warning(f"Path {experiment_path} does not exist")
        return

    # get experiment name
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    default_experiment_name = f"{cfg.EXPERIMENT_NAME}-{current_date}"
    experiment_name, okPressed = fui.open_text_input_dialog(
        msg="Enter a name for the experiment",
        title="Experiment Name",
        default=default_experiment_name,
        parent=parent,
    )

    if experiment_name == "" or not okPressed:
        logging.debug(f"Invalid experiment name {experiment_name}, or dialog cancelled")
        return

    # create new experiment
    experiment = create_new_experiment(experiment_path, experiment_name)
    return experiment
