# Image files saved before 0.60

Small images written by the savers fibsem had before the 0.60 image format
(`fibsem/image_format.py`): a beam image's metadata as JSON in the TIFF description,
FM stacks contiguous or zlib, no small copies. 0.60 must read every one of them as it
was saved, pixels and metadata both; `tests/test_image_files.py` checks that against
`expected.json`, written alongside the files by the same code.

| File | What |
| -- | -- |
| `fib_v0_electron_uint8.tif` | SEM image, 8-bit, full metadata from the demo microscope |
| `fib_v0_ion_uint16.tif` | FIB image, 16-bit |
| `fib_v0_no_metadata.tif` | a FibsemImage saved with no metadata |
| `fm_v0_stack.ome.tiff` | 2 channels × 3 planes, contiguous |
| `fm_v0_stack_zlib.ome.tiff` | the same, zlib |
| `fm_v0_single.ome.tiff` | 1 channel, 1 plane |

`make_fixtures.py` wrote them, run on main before the format changed (49e50f8):
`python make_fixtures.py tests/fixtures/image_files_v0`. Don't regenerate them with a
later version: the point is the bytes the old code wrote.
