"""A host for the FM control widget outside AutoLamella."""

from PyQt5.QtWidgets import QWidget


class FMHost(QWidget):
    """A host with a quad view. Pass ``view_controller=lambda: host.view_controller``
    to the widget, or the FM canvas stays empty."""

    def __init__(self):
        from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

        super().__init__()
        self.view_controller = MicroscopeViewController(parent=self)
