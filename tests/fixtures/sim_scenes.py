"""The simulator's sample as it was before the 3D sample became the default.

The SEM to FIB alignment gates and the flat and relief renderers' own tests
were written against it; they ask for it by name.
"""

from fibsem.drivers.demo.sim_scene import SampleScene

FLAT_SCENE = {"renderer": "flat", "cell_type": "mammalian", "film": "continuous"}


def flat_scene(**options) -> SampleScene:
    return SampleScene(**{**FLAT_SCENE, **options})


def make_flat(scene: SampleScene) -> SampleScene:
    """Turn a scene built from the configuration into the flat one, in place."""
    for key, value in FLAT_SCENE.items():
        setattr(scene, key, value)
    return scene
