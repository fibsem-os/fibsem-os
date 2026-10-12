"""A placeholder for the Zeiss driver, through the Zeiss Python API.

Nothing is here yet: the driver is to be migrated from SerialFIB. It has no ``DRIVER``
record and is not in the registry's built-ins, so connecting to a Zeiss microscope still
raises ``NotImplementedError``. Build it in this package, in the shape of the others
(``fibsem.drivers.tescan`` is the vendor SDK example): the ``DRIVER`` record here, then
``devices.py``, ``services.py`` and ``microscope.py``, and list the record in
``fibsem.drivers.registry._BUILT_IN``.
"""
