# FibsemMicroscope Server

A simple HTTP server that exposes a `FibsemMicroscope` instance over the network, allowing remote control from any Python client.

## Installation

```bash
pip install -e .[server]
```

## Starting the Server

On the machine connected to the microscope:

```python
from fibsem.server import FibsemServer

server = FibsemServer.from_session(manufacturer="ThermoFisher", ip_address="192.168.1.1", port=8001)
server.run()
```

Or with the Demo microscope for development:

```bash
python -m fibsem.server.server --manufacturer Demo --ip-address localhost --port 8001
```

The Swagger UI is available at `http://<host>:8001/docs`.

---

## Using the Client

```python
from fibsem.server import FibsemClient

microscope = FibsemClient(host="192.168.1.100", port=8001)
```

### Image Acquisition

```python
from fibsem.structures import BeamType

# Acquire a fresh image (uses current microscope settings)
image = microscope.acquire_image(BeamType.ELECTRON)
image = microscope.acquire_image(BeamType.ION)

# Get the last acquired image without triggering a new acquisition
image = microscope.last_image(BeamType.ELECTRON)

# Chamber overview image
image = microscope.acquire_chamber_image()

# image.data is a numpy array; image.metadata contains full FibsemImageMetadata
print(image.data.shape, image.data.dtype)
```

### Stage Movement

```python
from fibsem.structures import FibsemStagePosition, BeamType

# Get current position
pos = microscope.get_stage_position()

# Move to an absolute position (metres / radians)
pos = microscope.move_stage_absolute(
    FibsemStagePosition(x=0.001, y=0.0, z=0.004, r=0.0, t=0.0, coordinate_system="RAW")
)

# Move by a relative offset (None fields are ignored)
pos = microscope.move_stage_relative(FibsemStagePosition(x=50e-6, y=-20e-6))

# Beam-corrected lateral move (compensates for stage tilt)
pos = microscope.stable_move(dx=10e-6, dy=5e-6, beam_type=BeamType.ELECTRON)

# Vertical move (along the beam axis)
pos = microscope.vertical_move(dy=5e-6)

# Safe move with collision avoidance
microscope.safe_absolute_stage_movement(FibsemStagePosition(x=0.0, y=0.0, z=0.004))

# Orient sample flat to beam
microscope.move_flat_to_beam(BeamType.ELECTRON)
microscope.move_flat_to_beam(BeamType.ION)
```

### Milling

Milling follows a setup → draw → run → finish sequence.

```python
from fibsem.structures import FibsemMillingSettings, FibsemRectangleSettings

# 1. Configure mill settings (current, voltage, application file, etc.)
mill_settings = FibsemMillingSettings(
    milling_current=20e-12,
    milling_voltage=30000,
)
microscope.setup_milling(mill_settings)

# 2. Draw patterns onto the ion beam
rect = FibsemRectangleSettings(
    width=10e-6,
    height=5e-6,
    depth=1e-6,
    centre_x=0.0,
    centre_y=0.0,
)
microscope.draw_patterns([rect])

# Supported pattern types: FibsemRectangleSettings, FibsemLineSettings,
#                          FibsemCircleSettings, FibsemBitmapSettings, FibsemPolygonSettings

# 3. Run milling (blocks until complete)
microscope.run_milling(milling_current=20e-12, milling_voltage=30000)

# 4. Restore imaging conditions
microscope.finish_milling(imaging_current=1e-12, imaging_voltage=2000)

# --- Control ---
microscope.start_milling()           # start without blocking
microscope.pause_milling()
microscope.resume_milling()
microscope.stop_milling()
microscope.clear_patterns()

# --- State ---
state   = microscope.get_milling_state()    # MillingState enum
seconds = microscope.estimate_milling_time() # float
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status and microscope type |
| `POST` | `/acquire_image` | Acquire a fresh image |
| `POST` | `/last_image` | Get last acquired image |
| `POST` | `/acquire_chamber_image` | Chamber overview image |
| `GET` | `/stage_position` | Current stage position |
| `POST` | `/move_stage_absolute` | Absolute stage move |
| `POST` | `/move_stage_relative` | Relative stage move |
| `POST` | `/stable_move` | Beam-corrected lateral move |
| `POST` | `/vertical_move` | Vertical stage move |
| `POST` | `/safe_absolute_stage_movement` | Safe move with collision avoidance |
| `POST` | `/move_flat_to_beam` | Orient flat to beam |
| `POST` | `/setup_milling` | Configure milling settings |
| `POST` | `/draw_patterns` | Draw milling patterns |
| `POST` | `/run_milling` | Run milling (blocking) |
| `POST` | `/start_milling` | Start milling (non-blocking) |
| `POST` | `/stop_milling` | Stop milling |
| `POST` | `/pause_milling` | Pause milling |
| `POST` | `/resume_milling` | Resume milling |
| `POST` | `/finish_milling` | Restore imaging conditions |
| `POST` | `/clear_patterns` | Clear all patterns |
| `GET` | `/milling_state` | Current milling state |
| `GET` | `/estimate_milling_time` | Estimated milling time (seconds) |
