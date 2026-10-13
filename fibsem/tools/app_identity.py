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


def _set_macos_bundle_name(name: str) -> None:
    # The menu-bar title is the main bundle's CFBundleName, and an interpreter
    # has no bundle of its own, so macOS falls back to the executable name.
    # Writing the key into the bundle's info dictionary before Qt builds its
    # menu bar renames it. Plain ctypes on the Objective-C runtime rather than
    # pyobjc, which is not a dependency.
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]

        def send(receiver, selector: bytes, *args):
            msg_send = objc.objc_msgSend
            msg_send.restype = ctypes.c_void_p
            msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [
                type(a) for a in args
            ]
            return msg_send(receiver, objc.sel_registerName(selector), *args)

        def ns_string(text: str):
            return ctypes.c_void_p(
                send(
                    objc.objc_getClass(b"NSString"),
                    b"stringWithUTF8String:",
                    ctypes.c_char_p(text.encode()),
                )
            )

        bundle = send(objc.objc_getClass(b"NSBundle"), b"mainBundle")
        info = send(bundle, b"infoDictionary")
        if not info:
            raise OSError("the main bundle has no info dictionary")
        send(info, b"setObject:forKey:", ns_string(name), ns_string("CFBundleName"))
    except (AttributeError, OSError, TypeError) as e:
        logging.warning("Could not set the macOS application name: %s", e)
