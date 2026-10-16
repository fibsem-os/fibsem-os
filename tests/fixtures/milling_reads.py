"""Which settings a milling service's ``setup`` reads.

A driver's ``settings_class`` says which settings it mills with, and the milling
form shows only those. `fields_setup_reads` runs ``setup`` with the driver's
settings recording each field it is asked for, so a test can hold the class to the
code.
"""

from typing import Set

from fibsem.structures import FibsemMillingSettings


def fields_setup_reads(milling, settings: FibsemMillingSettings) -> Set[str]:
    read: Set[str] = set()
    names = set(milling.setting_names)
    driver = milling.driver_settings(settings)

    class Recording(type(driver)):
        def __getattribute__(self, name):
            if name in names:
                read.add(name)
            return object.__getattribute__(self, name)

    recording = Recording(**{name: getattr(driver, name) for name in names})
    read.clear()
    # the service hands ``setup`` the recording in place of the settings it reads
    milling.driver_settings = lambda _settings: recording
    try:
        milling.setup(settings)
    finally:
        del milling.driver_settings
    return read
