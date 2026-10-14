# The App Server

The HTTP server AutoLamella hosts for agents and the dashboard (the App
Server), and `FibsemClient`, a `FibsemMicroscope` that talks to it. The user
guide is [docs/app-server.md](../../docs/app-server.md).

## Installation

```bash
pip install -e .[server]
```

## Starting the Server

The App Server runs inside AutoLamella: turn it on in Preferences → App
Server, and Tools → App Server shows its address and the dashboard. With
*Serve the microscope* on, it also serves the `/microscope` routes below. While it runs it writes `~/.fibsem/app-server.json` (url and
token), so local clients connect without copying anything.

The Swagger UI is available at `http://<host>:8001/docs`.

### Authentication and the agent level

Every request needs the server's bearer token, sent as an
`Authorization: Bearer <token>` header. What the token may do is the agent
level the operator sets from the Agent chip: reads at Watch, acting at Drive;
at Off everything is refused. A refusal is a structured 403
naming why (`level`, `ask_first`, `agent_off`, `agent_stopped`, `not_allowed`).
Separate from the level, an allowlist (`fibsem/server/allowlist.py`) names what
the agent may never write or run, at any level and in an approved request too:
a beam's `on`, `angular_correction`, `tilt_correction` and `plasma_gas`; the
stage's `home` and `link`; every manipulator write and command; the objective's
`limit_position`. `GET /microscope/devices` and `/microscope/services` mark those
`agent: false`. A command's `path` argument must be inside the open experiment. A hardware
command takes control of the microscope while it runs: when a run, a script or
another command is in control it gets `409 microscope_busy`, naming who.
A stop (`POST /microscope/services/milling/commands/stop`, a device's `stop`,
`stop_live` or `cancel`, or `POST /app/workflow/stop`) never takes control. The
agent stops its own action (its command holds control) at any level; anything
else needs Drive, or a request approved at Assist. Every stop needs a `reason`
in its body (`422 missing_reason` without one), recorded as an `agent_stop`
event. `GET /capabilities` reports the
level.

The server binds `127.0.0.1` by default. Exposing it on the LAN is an
explicit choice: the token then travels in cleartext HTTP, which keeps
strangers out but is not sniff-proof — see the AutoLamella App Server design
doc for the trust model.

### App route layout

The `/app` routes sort by what they act on:

- `/app/experiment`: the open experiment. `GET /app/experiment` summarises it;
  under it are `recent`, `protocol`, `items/{item}` (with its `task_config`
  and `outputs`), `grids/{grid}`, `images` and `task_history`.
- `/app/workflow`: the run. `status`, `queue`, `run_summary`, `start`,
  `stop`, `schedule`, `options`, and the questions the run asks: `prompt`,
  `prompt/answer`, `prompt/suggest`, `decide`, `reviews`.
- `/app/agent`: the agent's own records and settings: `notes`, `requests`,
  `supervision`.
- `events` and `log` stay at `/app`: they cover the whole session.

Exactly one experiment is open per session, so `/app/experiment` needs no
experiment name in the path.

---

## Using the Client

```python
from fibsem.server import FibsemClient

microscope = FibsemClient(host="192.168.1.100", port=8001, token="<token from Connect an agent…>")
```

### Image Acquisition

```python
from fibsem.structures import BeamType

# Acquire a fresh image (uses current microscope settings)
image = microscope.acquire_image(BeamType.ELECTRON)
image = microscope.acquire_image(BeamType.ION)

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

# Move to a named orientation ("SEM", "FIB", "MILLING", or "FM" on a compustage)
microscope.move_to_orientation("SEM")
microscope.move_to_orientation("FIB")
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

### Devices and beam settings

Every device the microscope built (`microscope.devices`: `electron`, `ion`,
`stage`, `manipulator`, FM parts) but the chamber is served generically, the same
routes the device server (`fibsem/server/devices.py`) serves. The chamber is not
served: pumping and venting are the operator's, at the microscope.

```python
microscope.list_devices()                               # parameters + commands per device
microscope.get_parameter("ion", "current")              # live read
microscope.set_parameter("electron", "hfw", 150e-6)     # checked server-side, returns the value written
microscope.parameter_metadata("ion", "current")         # limits, choices, settable
microscope.call_command("ion", "blank")
image = microscope.call_command("electron", "acquire")  # a FibsemImage

# FibsemMicroscope's wrappers are kept, over the same routes
microscope.set_beam_current(1e-9, BeamType.ION)         # PUT /microscope/devices/ion/current
microscope.get_beam_settings(BeamType.ION)              # GET /microscope/devices/ion/values
```

The settings groups (beam, imaging, detector) are put together by the client
from one `values` read, and written parameter by parameter. Stage moves and
milling go through the `stage_movement` and `milling` services.

Reads work at Watch. Parameter writes and commands need Drive and take control
of the microscope, except the stops (`stop`, `stop_live`, `cancel`), which follow
the stop rule above and never take control.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status |
| `GET` | `/capabilities` | API version, routers served, the agent level |
| `GET` | `/microscope` | Who the microscope is: manufacturer, model, serial number, software, fitted parts, its configuration |
| `GET`/`PUT` | `/microscope/state` | The whole microscope state |
| `GET` | `/microscope/devices` | Every device: parameters (type, unit, limits, choices, settable) and commands |
| `GET` | `/microscope/devices/{device}` | One device |
| `GET` | `/microscope/devices/{device}/values` | Every parameter's value in one read; the ones unavailable now listed apart |
| `GET` | `/microscope/devices/{device}/{parameter}` | Live read of a parameter |
| `PUT` | `/microscope/devices/{device}/{parameter}` | Write a parameter (`{"value": ...}`) |
| `GET` | `/microscope/devices/{device}/{parameter}/metadata` | Limits, choices, settable |
| `POST` | `/microscope/devices/{device}/commands/{command}` | Run a command (`{"kwargs": {...}}`); an image answers as its data (TIFF, npy), or with `{"answer": "preview"}` as `{"result": {"summary", "path", "preview"}}`, the shape agents get for every image (a beam image, a camera frame, a z-stack) |
| `POST` | `/microscope/devices/{device}/queries/{query}` | Ask a query (`{"kwargs": {...}}`), read-only |
| `GET` | `/microscope/services` | Every service (`imaging`, `milling`, `stage_movement`, ...) and what it offers |
| `GET` | `/microscope/services/{service}/{parameter}` | A service's state, e.g. `milling/state` |
| `POST` | `/microscope/services/{service}/commands/{command}` | Run a service command, e.g. `stage_movement/commands/move_relative`, `milling/commands/run`, `milling/commands/stop` |
| `POST` | `/microscope/services/{service}/queries/{query}` | Ask a service query, e.g. `milling/queries/estimate` |
