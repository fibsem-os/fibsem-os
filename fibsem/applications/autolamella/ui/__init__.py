import logging

try:
    from fibsem.applications.autolamella.ui import utils
    from fibsem.applications.autolamella.ui.qt import AutoLamellaUI as AutoLamellaMainUI
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
except ImportError as e:
    logging.info(f"Error importing autolamella.ui: {e}, using dummy instead.")

    AutoLamellaUI = None
