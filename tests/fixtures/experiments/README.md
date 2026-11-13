# Experiments as each release wrote them

One small AutoLamella experiment per released version, written by that release:
its `experiment.yaml` and `protocol.yaml`, without the images. This version must
open every one of them, save it and open it again; `tests/test_release_experiments.py`
checks that.

| Folder | Written by | What |
| -- | -- | -- |
| `v0_5_0` | fibsem 0.5.0 from PyPI | three lamellae on the Demo microscope; Setup Lamella Position, Mill Fiducial and Rough Milling run on two, one untouched |
| `v0_5_3` | fibsem 0.5.3 from PyPI | the same |

`make_experiment.py` wrote them, run with the release installed in its own
environment: `python make_experiment.py <folder>`. Paths in the files were then
rewritten to `/data/...`. Don't regenerate them with a later version: the point is
the file the old code wrote. When a version is released, add its folder.

0.4.x is not here: AutoLamella was a separate package then.
