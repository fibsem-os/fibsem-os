import napari
import fibsem
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI

def main():
    autolamella_ui = AutoLamellaUI(viewer=napari.Viewer())
    autolamella_ui.viewer.window.add_dock_widget(
        widget=autolamella_ui,
        area="right",
        add_vertical_stretch=True,
        name=f"AutoLamella v{fibsem.__version__}",
    )
    napari.run()

if __name__ == "__main__":
    main()
