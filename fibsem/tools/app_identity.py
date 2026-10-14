"""Make the operating system show a Qt app under its own name and icon.

The applications run as a plain Python interpreter, not as an installed app, so
by default the OS shows them as that interpreter: "python3.11" in the macOS menu
bar, and Python's icon on the Windows taskbar. Both fixes have to run before the
QApplication is created, and both are cosmetic, so a failure is logged and
startup carries on.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import sys


def set_app_identity(name: str, windows_app_id: str) -> None:
    """Show this process as ``name`` with its own taskbar identity.

    Call before creating the QApplication. ``windows_app_id`` follows Windows'
    ``Company.Product`` convention, e.g. ``"fibsemOS.AutoLamella"``.
    """
    if sys.platform == "darwin":
        _set_macos_bundle_name(name)
    elif sys.platform == "win32":
        _set_windows_app_id(windows_app_id)


def _set_windows_app_id(app_id: str) -> None:
    # Windows groups a window on the taskbar by the executable that owns it,
    # python.exe here, so the taskbar shows Python's icon whatever setWindowIcon
    # says. An explicit AppUserModelID makes it a separate app with our icon.
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except (AttributeError, OSError) as e:
        logging.warning("Could not set the Windows taskbar app id: %s", e)


def bring_to_front() -> None:
    """Ask the OS to put this app in front of the one the user is in.

    Call once the QApplication exists. macOS otherwise leaves a newly started
    app behind the one it was launched from (a terminal, Finder, an editor), so
    a splash screen shown while the app loads opens out of sight. Windows and
    Linux bring a launched app's first window forward themselves.
    """
    if sys.platform != "darwin":
        return
    try:
        send = _objc_sender("AppKit")
        app = send(b"NSApplication", b"sharedApplication")
        send(app, b"activateIgnoringOtherApps:", ctypes.c_bool(True))
    except (AttributeError, OSError, TypeError) as e:
        logging.warning("Could not bring the app to the front: %s", e)


def _objc_sender(framework: str):
    """``send(receiver, selector, *args)`` on the Objective-C runtime.

    ``receiver`` is an object pointer, or a class name as bytes. Plain ctypes
    rather than pyobjc, which is not a dependency.
    """
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    ctypes.cdll.LoadLibrary(ctypes.util.find_library(framework))
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]

    def send(receiver, selector: bytes, *args):
        if isinstance(receiver, bytes):
            receiver = objc.objc_getClass(receiver)
        msg_send = objc.objc_msgSend
        msg_send.restype = ctypes.c_void_p
        msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [type(a) for a in args]
        return msg_send(receiver, objc.sel_registerName(selector), *args)

    return send


def _set_macos_bundle_name(name: str) -> None:
    # The menu-bar title is the main bundle's CFBundleName, and an interpreter
    # has no bundle of its own, so macOS falls back to the executable name.
    # Writing the key into the bundle's info dictionary before Qt builds its
    # menu bar renames it.
    try:
        send = _objc_sender("Foundation")

        def ns_string(text: str):
            return ctypes.c_void_p(
                send(
                    b"NSString",
                    b"stringWithUTF8String:",
                    ctypes.c_char_p(text.encode()),
                )
            )

        bundle = send(b"NSBundle", b"mainBundle")
        info = send(bundle, b"infoDictionary")
        if not info:
            raise OSError("the main bundle has no info dictionary")
        send(info, b"setObject:forKey:", ns_string(name), ns_string("CFBundleName"))
    except (AttributeError, OSError, TypeError) as e:
        logging.warning("Could not set the macOS application name: %s", e)
