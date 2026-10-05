# Legacy fluorescence image fixtures

Small OME-TIFFs written by the `FluorescenceImage.save` FibsemOS had before FIB-279
(commit `179bbeff7`), so the loader is tested against the bytes old code actually wrote
rather than against the layout the current code believes in.

That writer never changed its byte layout between #90 (May 2026) and the fix: every
version wrote the planes channel by channel, z fastest, mapped each plane correctly in
the OME, and declared the order `XYCZT` (channel fastest), which is the reverse. tifffile
follows the per-plane mapping, so these files read back as `ZCYX`. The loader before the
fix guessed the order from sizes and swapped channels and z when there were as many of
each; `c2z2`, `c3z3` and `c4z4` are that case.

| File | Channels × z |
| -- | -- |
| `c1z4.ome.tiff` | 1 × 4 |
| `c4z1.ome.tiff` | 4 × 1 |
| `c2z3.ome.tiff` / `c3z2.ome.tiff` | unequal, both ways round |
| `c2z2.ome.tiff`, `c3z3.ome.tiff`, `c4z4.ome.tiff` | equal: what FIB-279 got wrong |
| `tile.ome.tiff` | a 2D image, as an overview tile is saved |

Every plane holds `(10 * c + z + 1) * 97` plus a small ramp, so a swap or a shuffle of
planes shows; `make_fixtures.py` has the formula and how to regenerate the files. The
headers carry no paths, hostnames or serial numbers.
