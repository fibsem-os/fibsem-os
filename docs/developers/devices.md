# Devices

A microscope in fibsemOS is a coordinator that holds devices: the electron and ion
beams, the stage, the chamber, the manipulator and, where there is one, the
fluorescence microscope and its parts. Each device describes itself. Its parameters
carry a type, a unit, and the limits and choices the instrument reports; its commands
are plain methods that a UI, the agent server or a remote client can list; and both
emit events when something changes.

This page covers using devices and services from a script or from the application,
configuring which devices a microscope builds, and adding a device or a driver. The code examples
run against the Demo microscope, and the test suite executes them
(`tests/test_devices_guide.py`), so they stay correct.

The string-key `microscope.get("key", beam_type)` and `microscope.set(...)` calls
are deprecated and will be removed in the next minor release.
[From get/set to devices](#from-getset-to-devices) lists where each key went. The
named methods (`get_beam_current`, `move_stage_absolute`,
`get_microscope_state`, ...) are not deprecated; they now read and write the devices.

## Why the API changed

Before devices, a `FibsemMicroscope` subclass per vendor did everything, and most
settings went through two string-keyed methods, `get(key, beam_type)` and
`set(key, value, beam_type)`. An inventory of that API in September 2026 (141 public
methods, 56 keys, four backends) found these limits:

- **Nothing said what a backend supported.** Only 27 of the 56 keys were handled by
  all four backends (ThermoFisher 46, Demo 47, Odemis 46, Tescan 31). An unsupported
  key looked the same as a supported one: an unknown key, including a typo, logged a
  warning and returned `None`; on some backends setting the resolution, dwell time or
  stigmation only logged; and 30 abstract methods across the backends were silent
  no-ops. A caller found out by reading the backend's code, or on the instrument.
- **What a value could be was scattered.** Choices came from a second key chain,
  `get_available_values`. Limits came from four places: `voltage_limits` keys on two
  backends, a module-level table on another, clipping inline in two setters, and the
  stage's axis limits. Only the field of view was checked against its limits when
  set. A widget had to ask the instrument to fill a drop-down, and could not show a
  range it did not know.
- **Nothing reported a change.** There were no events, so the UI polled, and a change
  made in the vendor's own software was invisible until the next read.
- **Shared behaviour was copied or borrowed.** The Demo and Odemis backends called
  ThermoFisher's stage moves and milling as `ThermoMicroscope.method(self, ...)`, and
  code shared by every backend branched on the stage type. A fix to one backend's
  moves changed another's, and a new backend had to imitate ThermoFisher's internals.
- **The instrument was one object.** Every device came from the microscope's own
  driver, and a device the vendor class did not know about meant new methods and
  branches on that class. A configuration could not switch a device off, drive one
  device with another driver (a METEOR FM on its own computer beside ThermoFisher
  beams), or add a device from a plugin.
- **One lock guarded everything,** so devices that the hardware allows in parallel
  waited on each other.

The device API answers each of these. A device class declares every parameter it may
have, with its type and unit, and a parameter the backend does not bind is absent, so
`in device.parameters` says what is supported. Limits and choices are read once at
connect and kept on the parameter, and `set_value` checks against them. Every change
emits `changed`. Behaviour shared by every backend lives once, in the vendor-neutral
classes and services, and each driver implements only the instrument calls. Devices
are built from configuration entries, each with its own driver, into one flat map,
and named resources replace the single lock where a backend allows it.

## Finding a device

```python
from fibsem import utils
from fibsem.structures import BeamType

microscope, settings = utils.setup_session(manufacturer="Demo")

list(microscope.devices)
# ['electron', 'ion', 'stage', 'chamber', 'manipulator', ...]

sem = microscope.devices["electron"]
assert sem is microscope.beams[BeamType.ELECTRON]
assert microscope.devices["stage"] is microscope.stage
```

`microscope.devices` is the one map of every device the microscope built, by name, in
the order they were built. It is read-only. A device the configuration switched off,
or that a backend does not have, is not in it. The typed attributes are views of the
same map:

| Attribute | Device | Name in `devices` |
| -- | -- | -- |
| `microscope.beams[BeamType.ELECTRON]` | `Beam` | `electron` |
| `microscope.beams[BeamType.ION]` | `Beam` | `ion` |
| `microscope.stage` | `Stage` | `stage` |
| `microscope.chamber_device` | `Chamber` | `chamber` |
| `microscope.manipulator_device` | `Manipulator` | `manipulator` |
| `microscope.fm_devices` | `FM` and its parts | `fm`, `camera`, `light_source`, `filter_set`, `objective` |

`microscope.stage_device` is an older name for `microscope.stage` and is kept as an
alias. The vendor's own stage object is private to each backend.

The device classes are in `fibsem.devices` (`Beam`, `Stage`, `Chamber`,
`Manipulator`) and `fibsem.devices.fm` (`FM`, `Camera`, `LightSource`, `FilterSet`,
`Objective`). Each backend subclasses them in `fibsem/devices/drivers/<driver>.py`.

## Parameters

A device class declares every parameter it may have. A backend binds the ones its
hardware has. `device.parameters` lists the bound ones:

```python
sorted(sem.parameters)
# ['blanked', 'current', 'detector_brightness', ..., 'voltage', 'working_distance']

"preset" in sem.parameters    # False: the Demo has no presets
```

Reading a declared parameter the backend did not bind raises `ParameterUnavailable`,
which is an `AttributeError`, so `hasattr(sem, "preset")` is also False. Check
`in device.parameters` before using a parameter that not every instrument has.

### Reading

```python
sem.current.get_value()    # a live read from the instrument
sem.current.cached         # the last known value, with no instrument call
sem.current.value          # the same as get_value()
```

Use `get_value()` where the current value matters, such as a check before a move,
and `cached` for displays. `cached` reads live only the first time, before anything
was read or written.

### What a parameter allows

```python
sem.current.unit           # 'A'
sem.current.type           # float
sem.current.choices        # the currents the instrument offers
sem.current.limits         # a RangeLimit, or None
sem.current.settable       # False for a read-only parameter
sem.describe()["hfw"]      # all of it as plain data, for a UI or a remote client
```

These come from the instrument once, when the device connects, and are cached; reading
them makes no instrument call. They are refreshed when a parameter they depend on
changes (the ion beam's current choices depend on its plasma gas). Units are SI
throughout: metres, amps, volts, seconds, radians. `limits` on a composite value such
as the stage position is one `RangeLimit` per field.

`choices` replaces `microscope.get_available_values(key, beam_type)`.

### Writing

```python
written = sem.hfw.set_value(150e-6)   # checked, then written; returns what was written
sem.hfw.value = 150e-6                # the same, without the return value
```

`set_value` checks the value before writing it:

- a value of the wrong type raises `TypeError`;
- a value that is not one of the `choices` raises `ValueError`;
- a number outside the `limits` is clipped to them, with a logged warning, and the
  clipped value is what `set_value` returns;
- a read-only parameter raises `ParameterReadOnly`.

The write is then made, the value cached, and `changed` emitted. A parameter that needs
the imaging channel (on Thermo Fisher, most beam settings) claims it and selects its
beam for the write, so two threads cannot interleave a channel switch.

`write_through(value)` is the path the deprecated `set` takes. It skips the checks
above so that old callers behave as they did. Do not use it in new code.

### Events

```python
from fibsem.structures import Point

seen = []
sem.hfw.changed.connect(seen.append)                          # this parameter
sem.changed.connect(lambda name, value: seen.append(name))    # any on the device

sem.hfw.set_value(200e-6)
sem.shift.set_value(Point(0, 0))
```

`changed` fires after every change the device knows about: a `set_value`, a write
through the old API, a live read that found a different value, or a change the
backend reports from elsewhere (the vendor's own UI, another client). The value before
the change is in `parameter.previous`. `metadata_changed` fires with the new
`ParameterMetadata` when a parameter's limits or choices were refreshed. The signals
are [psygnal](https://psygnal.readthedocs.io) signals; in a Qt widget, connect them to
a slot that runs on the GUI thread.

## Commands

A command is a method on the device:

```python
image = sem.acquire()      # with the beam's current settings
sem.blank()
sem.unblank()

sorted(sem.commands)
# ['acquire', 'auto_focus', 'autocontrast', 'blank', 'full_frame', ...]
sem.commands["blank"].available
sem.commands["acquire"].signature    # its arguments, as a string
```

`device.commands` lists every command the class has, with its signature, its
docstring, and whether it is available on this instrument right now. Check
`available` before calling a command that not every instrument has.

The beam's commands are `acquire`, `last_image`, `autocontrast`, `auto_focus`,
`start_live`/`stop_live` (each frame on `beam.live_frame`), `blank`/`unblank`, and the
scan modes `spot(point)`, `reduced_area(rectangle)` and `full_frame()`.
`beam.acquire()` returns the frame as the instrument gives it. To apply
autocontrast and gamma and save the image, use `fibsem.acquire.acquire_image`, as
before.

## The stage

```python
from fibsem.devices import StageLimitError
from fibsem.structures import FibsemStagePosition

stage = microscope.stage
stage.position.get_value()     # a FibsemStagePosition, in the raw stage frame
list(stage.axes)               # ['x', 'y', 'z', 'r', 't']; a compustage has no r
stage.axes.t.limits            # radians, from the instrument
stage.axes.t.cached            # no instrument call

stage.move_relative(FibsemStagePosition(x=10e-6))
try:
    stage.move_absolute(FibsemStagePosition(x=1.0))
except StageLimitError as e:
    print(e)                   # names each axis that is out of its limits
```

The position is read-only: moves are commands. `move_absolute` and `move_relative`
check the target against each axis's limits and raise `StageLimitError` rather than
clip. Axes left `None` do not move. `home()` and `link()` are available where the
stage has them.

`microscope.move_stage_absolute` and `move_stage_relative` keep their old behaviour
and do not check limits. `microscope.safe_absolute_stage_movement` is still the move
for workflow code: it handles the order of the axes and the stage's poses, which the
device's moves do not.

`stage.position.changed` fires after every move. Each axis has its own `changed`,
which fires only when that axis moved.

## The chamber and the manipulator

```python
chamber = microscope.chamber_device
chamber.state.get_value()     # ChamberState
chamber.pressure.get_value()  # Pa, where the instrument reports it

needle = microscope.manipulator_device
needle.state.get_value()      # InsertableDeviceState
needle.named_positions()      # the named positions the driver reports
needle.axes()                 # ("x", "y", "z"), with "r" or "t" if the arm rotates or tilts
```

The chamber's commands are `pump()` and `vent()`. The manipulator's are
`insert(name)`, `retract()`, `move_absolute`, `move_relative`, `move_to_offset` and
`stop()`. `stop` does not wait for the move it stops; a driver that cannot stop its
needle raises `NotImplementedError`. Whether the arm rotates
(`is_available("manipulator_rotation")`) comes from `axes()`.

## The fluorescence microscope

An FM is several devices: a `Camera`, a `LightSource`, a `FilterSet` and an
`Objective`, and an `FM` device that uses all four to acquire channels, frames and
z-stacks. The four are the FM's roles (see [Roles](#roles)), and each is also in
`microscope.devices` under its own name.

```python
import os

from fibsem import config

microscope, settings = utils.setup_session(
    config_path=os.path.join(config.CONFIG_PATH, "sim-arctis-configuration.yaml")
)
fm = microscope.devices["fm"]
assert fm.camera is microscope.devices["camera"]

fm.camera.exposure_time.limits             # seconds
fm.filter_set.excitation_wavelength.choices
fm.light_source.power.set_value(0.2)       # a fraction of the source's maximum
fm.light_source.power.metadata.native_max  # what 1.0 is, where the driver knows it
sorted(fm.commands)
# ['acquire_channel', 'acquire_frame', 'acquire_z_stack', 'cancel', 'start_live', 'stop_live']
```

`microscope.fm` is the older `FluorescenceMicroscope` interface. It drives the same
hardware objects, so the two always agree.

## Roles

A role is a place on a device that another device fills. The device class declares
it with the interface the filler must have:

<!-- not run -->
```python
class FM(Device):
    camera = Role(Camera)
    light_source = Role(LightSource)
```

A builder fills the roles with `device.fill_roles(camera=..., ...)` before
`connect()`. The device uses its roles only through their interface, so it does not
matter which driver built the filler: the METEOR's FM, for example, can use a camera
served from another computer. `device.roles` lists the filled roles. A required role
left empty is an error at `connect()`, and reading an unfilled optional role raises
`RoleUnfilled`.

Roles are references between devices in the flat `microscope.devices` map, not a
tree: every device keeps its own unique name.

### Binding a role from the configuration

A beam has an optional `scanner` role (`fibsem.devices.scanner.Scanner`). Left
empty, the beam images through the vendor's scan, as it always has. Bound to a scan
generator, `beam.acquire()`, `acquire_image` and live view scan through it instead:
the beam keeps `resolution`, `dwell_time` and `hfw`, the scanner returns the frame,
and the beam builds the `FibsemImage`. The binding sits on the beam's entry:

```yaml
hardware:
  devices:
    - name: scan_generator
      type: scan_generator
      driver: demo
    - name: electron
      roles: {scanner: scan_generator}
```

The Demo has a simulated scan generator (frames read "SG"), so a binding can be
tried without hardware; only the Demo binds roles so far. Connecting fails
(`RoleBindingError`) for a role the device doesn't have, a device that isn't the
role's interface, a name no entry has, or bindings that form a cycle. A bound device
that is switched off (`enabled: false`) leaves the role empty, with a warning.

`Scanner` is minimal until a real unit's SDK says more: `acquire(resolution,
dwell_time)` returns one frame, and `stop()` ends a frame in flight.

## Services

A service is something the instrument does with its devices over time, such as
milling. It has the same shape as a device (parameters, commands, roles and change
events) but is not hardware, so it is not in `microscope.devices`. The microscope holds
each service as an attribute, built by its driver as devices are. One beam's own
operation is a command on the beam (imaging, the scan modes); anything that
coordinates devices or runs steps over time is a service. The base class is
`fibsem.services.Service`.

### Milling

`microscope.milling` is the milling service. It reaches the beams through its roles,
`ion` and, on a dual beam, `electron`:

```python
from fibsem.structures import FibsemMillingSettings, FibsemRectangleSettings

microscope, settings = utils.setup_session(manufacturer="Demo")
milling = microscope.milling
ion = microscope.beams[BeamType.ION]
imaging_current = ion.current.get_value()

milling.prepare(
    FibsemMillingSettings(milling_current=2e-9),
    [FibsemRectangleSettings(width=10e-6, height=5e-6, depth=1e-6, centre_x=0, centre_y=0)],
)
milling.estimate()            # seconds
milling.start()               # starts, and returns
milling.state.get_value()     # MillingState.RUNNING
milling.stop()

milling.clear()
milling.restore()             # the ion beam back as the first setup found it
assert ion.current.get_value() == imaging_current
```

`setup(settings)` applies the recipe, `draw(patterns)` adds patterns, and `prepare`
does both. `start`, `pause`, `resume` and `stop` run it, `state` says where it is,
and `clear` removes the patterns. The first `setup` saves the milling beam's preset,
voltage, current and field of view, and `restore` writes them back, so the beam ends
as milling found it on every backend.

Workflow code mills through `fibsem.milling` (milling stages, strategies and
`fibsem.milling.tasks.run_milling_task`), as before. The microscope's named milling methods (`setup_milling`,
`draw_rectangle`, `start_milling`, `run_milling`, `finish_milling`, ...) go to the
service; `finish_milling` restores the beam. `run_milling(stop_event=None)` is the
service's `run`: it mills what is drawn with the beam conditions `setup_milling`
applied, reports `progress`, and returns when the mill ends; a set `stop_event` stops
the beam. To start a mill and return at once, use `start_milling`. With the ion column
disabled, `microscope.milling` is `None` and those methods raise.

A driver adds milling by subclassing `fibsem.services.milling.Milling` and
implementing its hooks (`_setup`, `_draw`, `_start`, `_stop`, `_pause`, `_resume`,
`_estimate`, `_clear` and `read_state`) the way its instrument mills. The driver also
applies the recipe's beam conditions, since backends use different recipe fields.
`bind_milling(MyMilling, microscope)` builds it over the microscope's beams.
`fibsem/services/drivers/demo.py` is the reference.

## From get/set to devices

`microscope.get("key", beam_type)` and `microscope.set("key", value, beam_type)` now
raise a `DeprecationWarning` when called from outside the microscope classes. The
warning names the replacement. They will be removed in the next minor release; the
named methods stay.

| Old call | Device |
| -- | -- |
| `get("current", bt)` / `set("current", v, bt)` | `microscope.beams[bt].current` |
| `get("voltage", bt)`, `"hfw"`, `"working_distance"`, `"scan_rotation"`, `"resolution"`, `"dwell_time"`, `"stigmation"`, `"shift"`, `"plasma_gas"`, `"preset"`, `"scanning_mode"` | the beam parameter of the same name |
| `"detector_type"`, `"detector_mode"`, `"detector_contrast"`, `"detector_brightness"` | the beam parameter of the same name |
| `get("blanked", bt)`, `get("on", bt)` | `beam.blanked`, `beam.on` |
| `"angular_correction_angle"`, `"angular_correction_tilt_correction"` | `beam.angular_correction`, `beam.tilt_correction` |
| `set("spot_mode", point, bt)`, `set("reduced_area", rect, bt)`, `set("full_frame", ..., bt)` | `beam.spot(point)`, `beam.reduced_area(rect)`, `beam.full_frame()` |
| `get("stage_position")`, `"stage_homed"`, `"stage_linked"` | `microscope.stage.position`, `.homed`, `.linked` |
| `set("stage_home", True)`, `set("stage_link", True)` | `microscope.stage.home()`, `.link()` |
| `get_available_values(key, bt)` | the parameter's `choices` |

`fibsem.devices.BEAM_ROUTES` and `STAGE_ROUTES` are the complete, current list. A key
in neither table has not moved to a device yet; until it has, use the named method
rather than `get`/`set`. `tests/test_no_direct_get_set.py` keeps new code inside
`fibsem` off direct `get`/`set` calls.

## Configuring devices

Which devices a microscope builds is set in the microscope configuration, under
`hardware.devices` (configuration version 2):

```yaml
hardware:
    devices:
      - name: stage
        type: stage
        rotation_reference: 0
      - name: electron
        type: beam
        column_tilt: 0
        eucentric_height: 7.0e-3
      - name: ion
        type: beam
        column_tilt: 52
        eucentric_height: 16.5e-3
      - name: manipulator
        enabled: false
      - name: fm
        enabled: true
        driver: remote
        address: 192.168.0.20
        port: 8765
```

**The list is an overlay.** A backend builds the devices its instrument always has.
An entry changes one of them, or adds one the backend cannot find by itself; a device
the file does not name is built as it would be without the entry.

Each entry has:

| Key | Meaning |
| -- | -- |
| `name` | Unique in the file, and the device's name in `microscope.devices`. Defaults to the type. A beam is named `electron` or `ion`. |
| `type` | The device type: `beam`, `stage`, `chamber`, `manipulator`, `fm`, or a type a plugin adds. May be left out where the name is a type (`name: fm`). |
| `enabled` | Absent: the backend's default. `false`: never built, and its driver never touches it. |
| `driver` | The driver that builds it, by its registry name. Absent: the driver for `info.manufacturer`. `remote` is a device on another computer, at `address` and `port`. |
| `required` | `true`: connecting fails if the device cannot be built. Otherwise a device that fails to build is logged and left out. |
| `roles` | Binds a role of this device to another entry by name (`{scanner: scan_generator}`). The Demo binds it; see [Binding a role from the configuration](#binding-a-role-from-the-configuration). |

Every other key belongs to the device or its driver (`column_tilt`, `address`,
`port`, ...) and sits beside these in the entry.

Version 1 files, with one block per device under `hardware:`, still load, and are
written back as version 2 when the configuration is saved.

## Adding a device type

A device class declares its parameters and commands. A driver implements each
parameter with methods named after it, which `connect()` binds:

```python
from fibsem.devices import Device, Parameter, ParameterMetadata, command
from fibsem.structures import RangeLimit


class Knife(Device):
    """A device type a plugin adds."""

    angle = Parameter(float, unit="rad")

    def __init__(self, name, parent=None):
        super().__init__(name, parent)
        self._angle = 0.0    # stands in for the vendor connection

    def read_angle(self):
        return self._angle

    def write_angle(self, value):
        self._angle = value

    def metadata_angle(self):
        return ParameterMetadata(limits=RangeLimit(min=0.0, max=0.5))

    @command
    def cut(self) -> None:
        """Make one cut."""


knife = Knife("knife").connect()
knife.angle.set_value(1.0)    # clipped to 0.5, with a warning
assert knife.angle.cached == 0.5
```

For each declared parameter, a device subclass may define:

- `read_<name>()`: required for the parameter to exist on this device;
- `write_<name>(value)`: without it, the parameter is read-only;
- `metadata_<name>()`: the limits, choices and settable flag the instrument reports;
- `available_<name>()`: False leaves the parameter out on this instance.

A method named after a parameter the class does not declare is an error when the
class is defined, so a typo cannot silently hide a parameter. A subclass may not
change a declared parameter's type or unit: what differs between backends goes in the
metadata. `needs_channel` names the parameters whose reads and writes claim the
imaging channel. The vendor-neutral classes live in `fibsem/devices/` and never import
a driver; each driver's classes live in `fibsem/devices/drivers/<driver>.py`.

## Adding a driver

A driver is a `DriverEntry` in the registry (`fibsem/microscopes/registry.py`). Its
`devices` map says how it builds each device type from a configuration entry:

<!-- not run -->
```python
from fibsem.microscopes.registry import DeviceBuilder, DriverEntry


def build_knife(entry, context):
    """Build the device an entry names. entry is a DeviceEntry; its own keys are in
    entry.options. context.microscope is the microscope being connected, and
    context.built the devices built before this one."""
    knife = Knife(entry.name, context.microscope)
    return knife.connect()


def driver():
    return DriverEntry(
        manufacturer="KnifeCo",
        devices={"knife": DeviceBuilder("my_plugin.devices:build_knife")},
    )
```

A package outside fibsemOS registers `driver` through the `fibsem.drivers` entry
point (see [Plugins](extending.md#plugins)), or calls
`fibsem.microscopes.registry.register_driver(driver())` at run time. A configuration
then adds the device with an entry naming the driver:

```yaml
      - name: knife
        type: knife
        driver: KnifeCo
```

The builder is imported only when an entry needs it. A builder that raises leaves the
device out, or fails the connect if the entry is `required`.
`BuildContext.shared` is scratch space the builders of one connect share, such as one
connection per address.

A driver like this one, that only builds devices, has no `microscope_class`. It is
never offered as a manufacturer, and connecting with it as `info.manufacturer` is
refused. Its name matches a `driver:` in any case (`knifeco` finds `KnifeCo`).

A driver for a whole microscope also names its `FibsemMicroscope` subclass in
`microscope_class`. [Supporting a microscope](extending.md#supporting-a-microscope)
covers that.

## Devices on another computer

A device server makes devices on one computer available to the microscope on
another, as the METEOR's FM is on the computer that drives the beams:

```bash
python -m fibsem.server.devices --serve odemis-fm --host 0.0.0.0 --port 8765
```

On the microscope's computer, the `fm` entry names `driver: remote` and the server's
`address` and `port`, as in the example under
[Configuring devices](#configuring-devices). The FM's devices are then built from what
the server has, backed by HTTP (`fibsem/devices/drivers/remote.py`), and parameters,
metadata and commands behave as they do locally. The FM is the only device a
configuration can put on another computer today; `connect_remote_beams` in the same
module connects to served beams from a script. `get_value()` raises `RemoteDeviceUnreachable` when the server cannot
be reached, rather than return a stale value, and `cached` is kept current by the
server's event stream. `fibsem/server/devices.py` documents the server's endpoints.
INSTALLATION.md covers setting up a METEOR.

The server has no authentication yet. Keep it on the microscope's private network.
