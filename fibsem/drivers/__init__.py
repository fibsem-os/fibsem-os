"""The drivers: one package per driver API, holding all of that driver's code.

``fibsem.drivers.<driver>`` (``autoscript``, ``tescan``, ``odemis``, ``demo``,
``remote``) has its ``DRIVER`` record in ``__init__.py``, its devices and their builders
in ``devices.py``, its services in ``services.py`` and, for a microscope, its
``FibsemMicroscope`` in ``microscope.py``. ``registry.py`` is how fibsem finds a driver
by name, built in or from a plugin package of the same shape.

``fibsem.devices`` and ``fibsem.services`` are vendor-neutral and never import from
here, and a driver does not import another driver: the one exception is
``OdemisTescanMicroscope``, kept for an external consumer. Each driver module may import
its vendor SDK, guarded, so a driver loads on a computer without it.
"""
