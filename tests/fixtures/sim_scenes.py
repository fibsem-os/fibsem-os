"""The simulator's sample with the options it had before RPE cells and a holey
film became the default: mammalian cells on a continuous film. The SEM to FIB
alignment tests were written against it; they ask for it by name.
"""

from fibsem.drivers.demo.sim_scene import SampleScene

MAMMALIAN_SCENE = {"cell_type": "mammalian", "film": "continuous"}


def mammalian_scene(**options) -> SampleScene:
    return SampleScene(**{**MAMMALIAN_SCENE, **options})
