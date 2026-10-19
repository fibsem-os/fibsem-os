"""A window's stand-in for the overview tabs: what they reach for on their host."""

from types import SimpleNamespace

from fibsem.workflows.session import AppSession


class OverviewHost:
    """A microscope and an experiment on an `AppSession`, as the window holds
    them, so a tab gets its overview from `session.overview`. A tab is built
    with `session` and `add_lamella`."""

    def __init__(self, microscope=None, experiment=None):
        self.session = AppSession()
        self.session.microscope = microscope
        self.experiment = experiment
        # Where the window finds the add, for the window's own wiring; no
        # lamella list of its own.
        self.panel = SimpleNamespace(
            experiment_tab=SimpleNamespace(
                add_new_lamella=self.add_lamella, lamella_list=None
            ),
            sample_widget=None,
        )

    @property
    def microscope(self):
        return self.session.microscope

    @microscope.setter
    def microscope(self, microscope):
        self.session.microscope = microscope

    @property
    def experiment(self):
        return self.session.experiment

    @experiment.setter
    def experiment(self, experiment):
        self.session.experiment = experiment

    def add_lamella(self, *args, **kwargs):
        """Looked up on each call, so a test can give the host its own
        `add_new_lamella` after the tab is built."""
        return self.add_new_lamella(*args, **kwargs)

    def add_new_lamella(self, *args, **kwargs):
        raise NotImplementedError("a test that adds lamellae gives the host its own")


def tab_args(host):
    """What the window builds an overview tab with: the session, and the add."""
    return host.session, host.add_lamella
