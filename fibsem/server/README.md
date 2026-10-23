# The App Server

The HTTP server AutoLamella hosts for agents and the event log (the App
Server). Its client is `fibsem.client.FibsemClient`. The user guide is
[docs/app-server.md](../../docs/app-server.md).

## Installation

```bash
pip install -e .[server]
```

## Starting the Server

The App Server runs inside AutoLamella: turn it on in Tools → App Server,
which also shows its address and the event log. With
*Serve the microscope* on, it also serves the `/microscope` routes below. While it runs it writes `~/.fibsem/app-server.json` (url and
a connect token), so local clients find it without copying anything.

The Swagger UI is available at `http://<host>:8001/docs`.

### Authentication and the agent level

Every request needs the server's bearer token, sent as an
`Authorization: Bearer <token>` header. What the token may do is the agent
level the operator sets from the Agent chip: reads at Watch, asking first at
Assist (a request the operator approves), acting at Drive; at Off everything is
refused. A refusal is a structured 403
naming why (`level`, `ask_first`, `agent_off`, `agent_stopped`, `not_allowed`).
Separate from the level, a blocklist (`fibsem/ai/agent/blocklist.py`) names what
the agent may never write or run, at any level and in an approved request too:
a beam's `on`, `angular_correction`, `tilt_correction` and `plasma_gas`; the
stage's `home`, `link` and raw `move_absolute` (stage_movement's too: the agent moves with `safe_absolute_stage_movement`); every chamber and manipulator write and command; the
objective's `limit_position`; the FM's `cancel` (the agent stops a z-stack with
`imaging`'s `stop`). `GET /microscope/devices` and `/microscope/services` mark those
`agent: false`. A command's `path` argument must be inside the open experiment. A hardware
command takes control of the microscope while it runs: when a workflow, a script or
another command is in control it gets `409 microscope_busy`, naming who.
The **operator token** (Tools → App Server → Copy operator token; one per
install, kept in `~/.fibsem/operator-token.json`) is the operator from outside
the window. It reads what the dashboard reads and reaches the operator's routes,
whatever the level: `GET /app/agent` (level, last heard, the waiting request),
`POST /app/agent/messages` `{"text"}`, `PUT /app/agent/level` `{"level"}`,
`POST /app/agent/stop`, and `POST /app/agent/requests/{id}/approve` or
`/reject` `{"reason"}`. The agent's token gets `403 operator_only` there.
It also reaches every agent route as the operator, at any level (Off too): the
operator's app on another computer (`fibsem.client.remote_microscope`). Its acts
take control as the operator and a stop needs no reason; the blocklist still
applies (venting, the manipulator and the GIS stay at the microscope).
To connect from another computer: open an SSH tunnel to the App Server port
(`ssh -N -L 8001:localhost:8001 <pc>`), copy the operator token at the PC
(Tools → App Server → Copy operator token), and pass it in:
`RemoteMicroscope.connect("http://localhost:8001", token)`, or set
`FIBSEM_SERVER_URL` and `FIBSEM_SERVER_TOKEN`. The remote computer keeps no copy;
"New operator token" revokes it.
`fibsem-cli` is such a client: on the PC it finds the App Server and reads the
operator token itself (`resolve_operator`); elsewhere it takes `--url` and
`--token`, or `FIBSEM_SERVER_URL` and `FIBSEM_OPERATOR_TOKEN`.
Control can be held across calls, so a remote run's calls never interleave with
anyone's: `POST /microscope/control` `{"action"}` (Drive) takes it and answers
`{"lease", "idle_s"}`, or `409 microscope_busy`. Calls with the `X-Fibsem-Control`
header run inside the lease; anyone else is refused, naming it. `PUT
/microscope/control/{lease}` renews it (a lease idle for `idle_s` is given up),
`POST`/`DELETE .../{lease}/lend` lends it while a run waits on a question, and
`DELETE /microscope/control/{lease}` gives it up at any level. `GET
/microscope/control` says who is in control. `RemoteMicroscope.control` does all
of this: a remote run holds the microscope as a local run does.
Or **pair** the computer, with no token to copy: Tools → App Server → *Pair a
computer* (or the web app's Settings page) shows a 4-digit code for 5
minutes, and `RemoteMicroscope.pair("1234")` sends it (to the tunnel's end,
`http://localhost:8001`, unless given a url), keeps the token in
`~/.fibsem/remote/` and connects; later `RemoteMicroscope.connect()` finds it. A
browser pairs from the web app the same way. Each paired computer has its own
token, an operator token, kept hashed in `~/.fibsem/paired.json`; *Unpair* cuts off
one. The routes: `POST /pair` `{"code", "name"}` answers `{"token"}` (no token
needed; 5 wrong codes close pairing), and the operator's `GET`/`POST`/`DELETE
/pairing` and `DELETE /pairing/computers/{id}`. The headless server prints a
code only when the web app asks for one (`POST /pair/request`). An agent pairs
the same way: *Pair an agent* (`POST /pairing` `{"role": "agent"}`) gives a code
for `fibsem-mcp pair`, and that computer's token is an agent's, gated by the
level, with its name on the agent's records.

An agent on this PC asks instead (`fibsem/server/connections.py`): the
discovery file's connect token reaches only `POST /connect` (anything else is
`403 connect_first`), which waits up to 25 s for the operator and answers
`{"state": "waiting" | "allowed" | "denied"}`. The card shows the ask (`asking`
on `GET /app/agent`); the operator's `POST
/app/agent/connections/{id}/allow` pairs it as an agent, and `/deny`
`{"reason"}` refuses it for a minute. The client keeps the token in
`~/.fibsem/agent/` (`fibsem.client.pairing.ask_to_connect`); the transport asks
by itself on `connect_first`, and drops a kept token the server refuses.
A long service command (marked `long: true` in the listing: milling's `run`,
`run_task` and `run_stage`; imaging's `acquire_focus_stack`, `acquire_overview`,
`fm_auto_focus`, `fm_acquire_image` and `fm_acquire_overview`; alignment's
`align_to_reference`; spot burn's `run`) answers at once with its run (`run_id`, `state: running`)
and runs on in control of the microscope; `GET /microscope/services/{service}/run`
reads it, waiting up to `?wait=` seconds (at most 60) for the end, and the end is
a `command_ended` event. The client's `Run.wait` waits for it. The run is the
service's (`fibsem/devices/runs.py`), so the same route reads a run a workflow
task or the operator started: a milling task is a run under its
`milling_run_id`.
A stop (`POST /microscope/services/milling/commands/stop`, a device's or a
service's `stop` or `stop_live`, or `POST /app/workflow/stop`) never takes control. The
agent stops its own action (its command holds control) at Assist or Drive; anything
else needs Drive, or a request approved at Assist. Every stop needs a `reason`
in its body (`422 missing_reason` without one), recorded as an `action`
event naming what it `stops`. `GET /capabilities` reports the
level.

The server binds `127.0.0.1` by default. Exposing it on the LAN is an
explicit choice: the token then travels in cleartext HTTP, which keeps
strangers out but is not sniff-proof — see the AutoLamella App Server design
doc for the trust model.

### App route layout

The `/app` routes sort by what they act on:

- `/app/experiment`: the open experiment. `GET /app/experiment` summarises it;
  under it are `recent`, `protocol` (with `supervision`), `items/{item}` (with
  its `task_config` and `outputs`), `grids/{grid}`, `images`, `notes`,
  `evidence` and `task_history`. Evidence (`fibsem/workflows/evidence.py`) is
  kept under `<experiment>/evidence/`; a call cites it by id in the
  `X-Agent-Evidence` header.
- `/app/workflow`: the workflow. `status`, `queue`, `summary`, `start`,
  `stop`, `stop_task`, `schedule`, `options`, `grids`, and the proposals the
  run is stopped on: `proposals` (with `suggest`) and `decide`.
- `/app/agent`: the agent's own requests: `requests`.
- `events` and `log` stay at `/app`: they cover the whole session.

Exactly one experiment is open per session, so `/app/experiment` needs no
experiment name in the path.

---

## Using the Client

`fibsem.client.FibsemClient` reaches the devices and services by name, from what
the server describes. A new parameter, command or query on the server needs no
client change. Values and results come back as the types the server declares.

```python
from fibsem.client import FibsemClient
from fibsem.structures import BeamType, FibsemStagePosition

# With no url and token, the App Server on this machine (~/.fibsem/app-server.json),
# else FIBSEM_SERVER_URL and FIBSEM_SERVER_TOKEN.
with FibsemClient("http://192.168.1.100:8001", "<token>") as client:
    electron = client.devices["electron"]
    electron.get("hfw")                          # a live read, as a float
    electron.get("stigmation")                   # a Point
    electron.set("hfw", 150e-6)                  # checked server-side; the value written
    electron.values()                            # every parameter, one read
    electron.metadata("hfw")                     # limits, choices, settable
    image = electron.acquire()                   # a FibsemImage

    stage_movement = client.services["stage_movement"]
    stage_movement.move_relative(delta=FibsemStagePosition(x=10e-6))  # a FibsemStagePosition
    stage_movement.project_stable_move(
        dx=10e-6, dy=0, beam_type=BeamType.ELECTRON,
        base_position=client.devices["stage"].get("position"),
    )

    # A long command answers at once with its run, which runs on.
    run = client.services["milling"].run()
    run.wait(timeout=3600)                       # its result; a failed run raises
    client.services["milling"].stop(reason="stopped from a script")  # a stop needs a reason

    client.analysis["correlation"]               # the analysis services (/services)
    client.state()                               # MicroscopeState
    client.system                                # SystemSettings

    # The app the server hosts: the agent's tools, by the parts it serves.
    client.app.parts                             # ["items", "poses", "grids", ...]
    client.app.app_get_status()
    client.app.item_get(item_name="01-rested-goblin")
    client.app.app_get_events(since=0)

    client.follow()                              # WS /microscope/events
    client.devices["stage"].changed.connect(print)
```

A command or query is a method (`electron.acquire(...)`), or `call(name, ...)` and
`ask(name, ...)`. A refusal raises the exception it names (`StageLimitError`,
`ResourceBusy`, `DeviceFault`, ...), else `ServerError` with the status and the
server's `error_type`. An unreachable server raises `ServerUnreachable`.

Reads work at Watch. Parameter writes and commands need Drive, or a request the
operator approved at Assist, and take control of the microscope, except the stops,
which follow the stop rule above.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status |
| `GET` | `/capabilities` | API version, routers served, the agent level |
| `GET` | `/microscope` | Who the microscope is: manufacturer, model, serial number, software, fitted parts, its configuration |
| `GET`/`PUT` | `/microscope/state` | The whole microscope state |
| `GET` | `/microscope/devices` | Every device: parameters (type and its `type_path`, unit, limits, choices, settable), and commands and queries (arguments, what they return) |
| `GET` | `/microscope/devices/{device}` | One device |
| `GET` | `/microscope/devices/{device}/values` | Every parameter's value in one read; the ones unavailable now listed apart |
| `GET` | `/microscope/devices/{device}/{parameter}` | Live read of a parameter |
| `PUT` | `/microscope/devices/{device}/{parameter}` | Write a parameter (`{"value": ...}`) |
| `GET` | `/microscope/devices/{device}/{parameter}/metadata` | Limits, choices, settable |
| `POST` | `/microscope/devices/{device}/commands/{command}` | Run a command (`{"kwargs": {...}}`); an image answers as its data (TIFF, npy), or with `{"answer": "preview"}` as `{"result": {"summary", "path", "preview"}}`, the shape agents get for every image (a beam image, a camera frame, a z-stack) |
| `POST` | `/microscope/devices/{device}/queries/{query}` | Ask a query (`{"kwargs": {...}}`), read-only |
| `GET` | `/microscope/services` | Every service (`imaging`, `milling`, `stage_movement`, ...) and what it offers |
| `GET` | `/microscope/services/{service}` | One service |
| `GET` | `/microscope/services/{service}/values` | Every parameter's value in one read |
| `GET` | `/microscope/services/{service}/run` | A long command's latest run (`?run_id=`, `?wait=` up to 60 s) |
| `GET` | `/microscope/services/{service}/{parameter}` | A service's state, e.g. `milling/state` |
| `PUT` | `/microscope/services/{service}/{parameter}` | Write a service parameter, e.g. `stage_movement/milling_angle` |
| `GET` | `/microscope/services/{service}/{parameter}/metadata` | Limits, choices, settable |
| `POST` | `/microscope/services/{service}/commands/{command}` | Run a service command, e.g. `stage_movement/commands/move_relative`, `milling/commands/run`, `milling/commands/stop` |
| `POST` | `/microscope/services/{service}/queries/{query}` | Ask a service query, e.g. `milling/queries/estimate_time` |
| `WS` | `/microscope/events` | Every device's and service's changes as they happen: `{"device" or "service", "parameter", "kind", "value"}`, kind `changed`, `metadata` or `signal` (a service's `remote_signals`, such as `milling`'s `notable_reading` and `sample_handling`'s `holder_changed`). The agent token in the header at Watch; a browser `Origin` is refused, and the stream closes when the level drops to Off |
