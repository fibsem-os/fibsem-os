"""The remote driver: a device on another computer, reached through its device server.

It is no microscope, so it has no ``DRIVER`` record: a device entry names it with
``driver: remote``, and the registry takes its builders from ``devices.py``
(``DEVICE_BUILDERS``). ``fibsem.server.devices`` is the server side.
"""
