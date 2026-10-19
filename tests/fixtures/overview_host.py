"""A window's stand-in for the overview tabs: what they reach for on their host."""

from fibsem.workflows.session import AppSession


class OverviewHost:
    """A microscope and an experiment, the microscope on an `AppSession` as the
    window holds it, so a tab gets its overview from `session.overview`."""

    def __init__(self, microscope=None, experiment=None):
        self.session = AppSession()
        self.session.microscope = microscope
        self.experiment = experiment

    @property
    def microscope(self):
        return self.session.microscope

    @microscope.setter
    def microscope(self, microscope):
        self.session.microscope = microscope
