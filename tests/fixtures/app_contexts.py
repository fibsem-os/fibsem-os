"""App contexts for server tests that need one without an app session."""

from typing import Optional

from fibsem.server.app_context import AppContext


class ExperimentContext(AppContext):
    """An app context with no session, answering only its experiment folder."""

    def __init__(self, experiment_path: Optional[str] = None):
        super().__init__(session=None)
        self._path = experiment_path

    @property
    def experiment_path(self) -> Optional[str]:
        return self._path

    @property
    def experiment_sample(self) -> Optional[str]:
        return None
