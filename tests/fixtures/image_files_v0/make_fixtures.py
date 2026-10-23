"""Write image files with today's (pre-0.60) savers, plus what they must load as."""
import hashlib, json, os, sys
import numpy as np
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
from fibsem import utils, acquire
from fibsem.structures import BeamType, FibsemImage, ImageSettings
from fibsem.fm.structures import FluorescenceImage, FluorescenceImageMetadata, FluorescenceChannelMetadata

microscope, settings = utils.setup_session(manufacturer="Demo", ip_address="localhost")
rng = np.random.default_rng(60)
expected = {}

def record(name, data, md):
    expected[name] = {"shape": list(data.shape), "dtype": data.dtype.name,
                      "sha256": hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest(),
                      "metadata": md}

for name, beam, dtype in [("fib_v0_electron_uint8.tif", BeamType.ELECTRON, np.uint8),
                          ("fib_v0_ion_uint16.tif", BeamType.ION, np.uint16)]:
    s = ImageSettings(resolution=(96, 64), hfw=80e-6, beam_type=beam, save=False)
    image = acquire.acquire_image(microscope, s)
    hi = 255 if dtype == np.uint8 else 65535
    image = FibsemImage(data=rng.integers(0, hi, (64, 96), dtype=dtype), metadata=image.metadata)
    path = image.save(os.path.join(out, name))
    record(name, image.data, image.metadata.to_dict())

bare = FibsemImage(data=rng.integers(0, 255, (64, 96), dtype=np.uint8), metadata=None)
bare.save(os.path.join(out, "fib_v0_no_metadata.tif"))
record("fib_v0_no_metadata.tif", bare.data, None)

def fm(nc, nz):
    chans = [FluorescenceChannelMetadata(name=n, excitation_wavelength=w, power=0.5, exposure_time=0.1,
             gain=1.0, offset=0.0, color="gray", objective_magnification=100,
             objective_numerical_aperture=0.75) for n, w in [("Reflection", 550), ("GFP", 488)][:nc]]
    md = FluorescenceImageMetadata(acquisition_datetime="2026-10-23T03:40:00", pixel_size_x=110e-9,
             pixel_size_y=110e-9, pixel_size_z=500e-9 if nz > 1 else None, channels=chans,
             z_positions=[i * 500e-9 for i in range(nz)] if nz > 1 else None, filename="fixture")
    return FluorescenceImage(data=rng.integers(0, 4000, (nc, nz, 48, 64), dtype=np.uint16), metadata=md)

for name, nc, nz, comp in [("fm_v0_stack.ome.tiff", 2, 3, None), ("fm_v0_stack_zlib.ome.tiff", 2, 3, "zlib"),
                           ("fm_v0_single.ome.tiff", 1, 1, None)]:
    image = fm(nc, nz)
    image.save(os.path.join(out, name), compression=comp)
    record(name, image.data, image.metadata.to_dict())

with open(os.path.join(out, "expected.json"), "w") as f:
    json.dump(expected, f, indent=1, sort_keys=True, default=str)
print(sorted(os.listdir(out)))
