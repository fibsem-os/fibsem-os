"""An executor that runs a call on the GUI thread, as the window's does: for
tests of an experiment with no window, whose edits must still land there."""

from concurrent.futures import Future

from PyQt5.QtCore import QCoreApplication, QThread
from superqt import ensure_main_thread


def on_the_gui_thread(call) -> Future:
    outcome: Future = Future()

    def run():
        try:
            outcome.set_result(call())
        except Exception as exc:  # noqa: BLE001 - the requester owns the failure
            outcome.set_exception(exc)

    if QThread.currentThread() is QCoreApplication.instance().thread():
        run()
    else:
        ensure_main_thread(await_return=False)(run)()
    return outcome
