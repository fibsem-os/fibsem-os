"""An `AppSession` holding a microscope and an experiment, as a window's does."""

from fibsem.workflows.session import AppSession


def app_session(microscope=None, experiment=None) -> AppSession:
    session = AppSession()
    session.microscope = microscope
    session.experiment = experiment
    return session
