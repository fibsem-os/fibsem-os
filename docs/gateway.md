# The Gateway

The gateway (`fibsem-gateway`) is a small, always-on program on a microscope PC. It starts
and stops the app there, so a hub on another computer, or a browser, can do it without
anyone at the PC. The app is either **off** (nothing holds the microscope) or **running**:
the app's host, with its window at this PC (as a desktop launch opens it) or without one
(as `fibsem-autolamella-server` runs it).

It never touches the microscope itself; the app it starts does. Without the gateway
installed, nothing changes: desktop launches work as they always have.

A PC runs one App Server at a time, and the gateway runs one app at a time. Clients on the
PC (`fibsem-agent`, `fibsem-cli`, the `fibsem-mcp` MCP server program, the gateway itself) find it
through its discovery file, `~/.fibsem/app-server.json`. A second app's App Server is
refused while a live one runs, and says which one (its pid and address).

## Installing it

```
fibsem-gateway install
```

On Windows this puts a "Agent Gateway" shortcut in your Startup folder, so the gateway
starts when you log on, with its window minimised and no console. It needs no admin rights.
`fibsem-gateway uninstall` removes it. To run it once by hand: `fibsem-gateway run`.

To stop the gateway, close its window, or run `fibsem-gateway stop`.
`fibsem-gateway restart` stops it and starts it again with the same flags. A gateway that doesn't answer is ended by its process id.

The app starts with no microscope. Connect one in the window, or on the gateway's page once
the app is up (**Connection**, with the configurations saved on this PC). To have the app
connect as it starts, give connection flags to `install` or `run` (`--manufacturer`,
`--ip-address`, `--config`): every start then connects with them.

The gateway listens on this PC only (`http://127.0.0.1:8010`). Another computer reaches it
over an SSH tunnel: see [Running the agent on another computer](ssh-tunnel.md).

The gateway's port also serves the App Server's routes: a call to `/app/...`,
`/microscope/...`, `/services/...` or `/capabilities` on 8010 goes to the App Server running
on this PC, and its answer comes back unchanged (status, headers and body, streamed). The
caller's token goes on as it came and the App Server checks it as ever, so one tunnel to 8010
is enough for a client. With no app running the gateway answers 503 `no_app`; a client reads
that, and an app that stops answering mid-call (504 `app_lost`), as the App Server not
answering, as it would calling the App Server directly. A stop (the workflow's, a task's, the
agent's, or a device's stop or halt command) goes through connections of its own, so it never
waits behind other calls.

The gateway is the one place this PC pairs, and it serves the web app (`/ui`). The App
Server serves neither: it checks each call's token against the paired computers the gateway
keeps (`~/.fibsem/paired.json`) and the operator's token, and nothing else. Its own port
stays for now, for a program on this PC that names it.

### Pairing

Every browser, computer, agent and hub pairs at the gateway, with a 4-digit code from
**Pairing → Pair a computer…** on its page (or **Pair a computer** in its window, or Tools →
Host in AutoLamella, which opens the page). Pick what pairs: **Operator** (a browser or
another computer, which acts as you), **Agent** (`fibsem-mcp pair <code>` on the agent's
computer; one agent per microscope, so it replaces the one paired before) or **Hub**. The
same list unpairs one.

An agent on this PC asks instead: a program here reads the connect token in the gateway's
`gateway.json` (readable by this PC's user only) and asks to connect. The ask shows on the
gateway's Overview and on the Agent page, with **Allow** and **Deny**; Allow pairs it as the
agent, listed with the rest. **Always allow agents logged in to this PC**, under Pairing,
allows an ask from this PC at once.

One agent holds the microscope's slot at a time: the one the gateway started, or one paired
another way. Taking the slot from an agent in use is always the operator's choice. Start
asks before it disconnects an agent paired another way. The Allow card says when allowing
ends the agent the gateway started, and Allow then ends it as its End does. Always allow
never replaces an agent in use (the gateway's running agent, or one the app heard from in
the last two minutes): the ask waits on the card instead. Every Allow and Deny, by the
operator or by the setting, goes on the History.

An App Server run alone, with no gateway (tests, live evals, scripts), pairs nothing: its
agent token is the one its launcher sets in `FIBSEM_SERVER_TOKEN`, sent by the client the
same way.

## The window and the page

The gateway runs in a small window, on the taskbar (the Dock on a Mac). It shows the app, with
**Start** and **Stop**; the microscope's connection; and the agent, with **End**. **Pair a
computer** shows a code to pair another computer, an agent or a hub. **Open page** opens the gateway's page,
and **Restart** and **Stop** at the bottom restart or stop the gateway. The desktop shortcut brings the window to
the front, starting the gateway first if none runs.
Its icon's badge shows the app: grey for off, blue with its window, purple without one. Closing the window asks,
then stops the gateway. Stopping it also stops an app without its window that isn't running a
workflow; an app with its window keeps running. A restart leaves the app running, and the
new gateway takes it back as it starts, so Stop and the app's state work again.
`--no-window` runs the gateway without it.

The page (`/ui/gateway`) is a launcher: it starts and stops the app and the agent on this
PC. What a running agent is doing, and its questions, are on the app's Agent page. The list
on the left has four pages:

- **Overview.** While nothing runs: the **Microscope** (a configuration saved on this PC,
  the default first, and **Connect when the app starts**) and the **App** (an app, an
  experiment folder and **Show the window on this PC**), with **Start the app**. An app
  shows only what its start takes: an app for an instrument that isn't a FIB-SEM may open
  no experiment, take no microscope connection, have no window of its own (it then runs
  without one), and have settings of its own (its server, a folder), passed to it as
  `--<name> <value>`. While it
  runs: one row for the microscope, with Connect or Disconnect, and one for the app, with
  **Open the app** and **Stop** (Stop asks first in a dialog; with its window, it closes the window).
  Under them, the **Agent** card (below).
- **History.** Every start and stop of the app, with who made it and from where, and the
  agent sessions, newest first. **All**, **App** and **Agent** filter it, and **Show older**
  reads further back, so every session on this PC can be opened and resumed. A session
  opens to show its playbook, spend, how it ended and what it last said, and **Resume**.
- **Pairing.** The paired computers, agents and hubs, and a code to pair another
  (see [Pairing](#pairing)).
- **Customise.** The agent's setup: each row opens the Agent page's **Knowledge** window on
  that part. Through the gateway, its playbooks, skills, subagents, tools and connectors work
  with no app running (the gateway's Agent Server serves them); its entries need the app.
  - **Playbooks, Skills, Subagents:** what fibsemOS ships and the lab's own, each with its
    form or its file. Editing a shipped one saves the lab's copy. Editing is the operator's,
    and each save, new one or removal made through the gateway is in History.
  - **Connectors:** the lab's MCP servers (`~/.fibsem/agent/mcp.yaml`), each with
    **Test** (the tools it offers, asked as the agent asks), **Edit** and **Remove**, and
    its token's key. A connector is checked as the agent reads it, so one that wouldn't
    load can't be saved; the file before is kept as `mcp.yaml.bak`. A command connector
    runs a program on this PC at each agent start: it is added, changed, removed and
    tested only at this PC, and a new or changed command is saved only once you confirm
    its command line.
- **Settings.**
  - **Keys:** whether each key fibsemOS reads (and each connector's token) is set on this
    PC, and where from (the environment or the key store). Keys are write-only: the
    operator at this PC sets, replaces or removes one, and the page never shows it back.
    A computer paired as operator, a hub or an agent never changes a key; it gets
    **Copy command** (the `fibsem-cli key set` command to run on the PC). A connector's
    token is stored with where it may go (the scheme and host of its address, or its
    command line), and is sent nowhere else: edit a connector's address and its token
    must be set again.
  - **This PC:** whether a new app session lets agents use the microscope, whether an
    agent on this PC connects without your click (set at this PC only), the gateway's
    port (used from its next start), and the update check.
  - **Defaults:** what Overview fills in for the microscope, the app and the agent, kept in
    `~/.fibsem/gateway-defaults.json`, for each app: choosing an app shows its own, so one
    app's playbook and settings don't replace another's. Only the operator saves them.
  - **Gateway:** the version, **Restart** and **Stop**. Stopping the gateway from the page
    stops the app and any agent run. Start it again at the computer.

### Health

The gateway asks the App Server how it is every 5 seconds (its `/health`, timed), and how
its microscope is every 15 seconds and as soon as the app comes up or goes (its
`/microscope/health`: the connection, a Safety Halt, and when each device and service was
last heard from). Neither reaches the hardware: both are what the App Server already holds,
so asking never takes the microscope's locks or switches its imaging channel. "Last heard
from" is kept from the calls a run already makes: when one last came back, the last that
failed in the driver (a refusal such as a value out of range doesn't count), and the calls
running now. A call running over 2 minutes is slow and over 10 minutes not answering (a long
command, such as a mill, never is); a driver error in the last 30 minutes with no call back
since is a fault. Noting a call never raises into it or holds it up. Each answers in one shape, at `GET /gateway/health` and in `GET /gateway`'s
`health`: a status (`ok`, `slow`, `not_answering`, `stopped`, `starting`, and for the
microscope also `not_connected`, `connecting`, `halted`, `fault` or `unknown` while the
app doesn't answer), why in a line, how long the answer took, and since when. A hub reads
the same. A piece that isn't well turns its dot amber on Overview, with the reason and for
how long under its row. The gateway only reports: it restarts nothing.

An app the gateway started is `starting` from its launch until its App Server first
answers (up to 3 minutes, for a slow first start), not `stopped` or `not_answering`; an
agent's Start in that time says the app is still starting.

Every fibsem program answers `GET /health` the same way, open to anyone: `{"status": "ok",
"program", "version", "api_version", "state"}`. The program is `fibsem-gateway`,
`fibsem-app-server` or `fibsem-agent-server`; `version` is fibsemOS's; `api_version` is the
version of that program's own routes (the gateway's are versioned apart from the App
Server's it passes on), so a hub can tell a gateway it is too old or too new for.

The bottom of the list says when there is newer code: "fibsemOS 0.6.3 available" when a
newer release is on PyPI, or "Update available" for a source checkout behind GitHub's
`main` (hover it for how many changes). The gateway asks when it starts and once a day after. Offline, or with a commit that
was never pushed, it shows nothing.

### The Device Server

On a PC with a `~/.fibsem/device-server.yaml` (a METEOR's or an AFM's PC, or a microscope PC
whose device needs a vendor's Python), the gateway also runs that PC's Device Server, as it
starts. It runs it with the Python the file names under `server.python` (another environment
on this PC, for a vendor's SDK), else its own:

```yaml
server:
  port: 8765
  python: C:\vendor\envs\sdk\python.exe
```

Overview shows it under Microscope, in Devices: a line with how many Device Servers and
devices, which opens to each Device Server (running, starting or stopped, its address and
version) and whether each device it serves answers (from its `/health`, read with this install's Device
Server token). The list opens by itself when one is not well.
One that ends is not started again by itself: it shows stopped with its last
log lines, until Start. Stop ends it by its stop file, so it switches its devices off as it
ends. Only the operator starts and stops it; a hub reads it at `GET /gateway/device-server`.
Its log is in `~/.fibsem/logs/gateway/device-server-<time>.log`. Without the file, nothing
changes.


### Device Servers on other PCs

The microscope reaches a device on another PC (a METEOR's FM, an AFM) through that PC's
Device Server, named by a `remote: <address>:<port>` entry in the microscope configuration.
The gateway reads those entries from the configuration the app connects with (the app's own,
else the one chosen on Overview, else the default), so there is no second list, and asks
each Device Server how it is every 10 seconds, with the token this PC was paired with there.
Overview lists them under Microscope, in Devices, beside this PC's own: running, not
answering, not paired, another version (its routes are another minor version's), or tunnel
down, and whether each of its devices answers. `GET /gateway` has the same list in
`device_servers`, which is what a hub reads.

An entry can name an SSH tunnel to its PC, and the gateway opens and keeps it open:

```yaml
hardware:
  devices:
    - name: fm
      driver: Odemis
      remote: 127.0.0.1:8765
      tunnel: meteor@meteor-pc
```

The Device Server then stays on its own PC's 127.0.0.1, and the gateway brings its port to
the same port here. The gateway runs this PC's own `ssh` as the same user, with their SSH key
and settings, and never asks for a password: setting the key up is a one-time step by a
person (`ssh-keygen`, then `ssh-copy-id meteor@meteor-pc`, then `ssh meteor@meteor-pc` once to
accept its host key). The gateway stores no secret and makes no key; when ssh can't connect,
the row says what ssh said. A port that already answers here, such as a tunnel opened by
hand, is used as it is.

New, beside Devices, points one device at a Device Server (its name, the address and port, and
the tunnel if any), and Remove on a Device Server's row takes it off. Each changes only that
device's entry in the configuration (its `remote:` and `tunnel:`), keeping the file as it was
beside it the first time (`<name>.yaml.bak`), and counts from the app's next connect. Only the
operator adds and removes them.

## The agent

The gateway can run the agent on this PC, next to the app. The agent reaches the app
through the gateway's port. The **Agent** card on the page
starts it with:

- a **playbook** (the default in Settings, else `fibsem-agent`, the general one);
- a **level** (Watch, Assist or Drive): the default in Settings, else the playbook's
  session `level`, else Watch;
- an optional **budget**, in place of the playbook's (the default in Settings is picked
  first);
- a **goal** for the session.

Under the goal, the playbook's `description` says what it is for.

**Show configuration** lists what the playbook runs with: the model, budget, skills,
reminders and rules.

Start needs:

- **The app running**, with its window or without, for the agent to act on the microscope.
  A microscope needn't be connected: without one the agent can still read and review an
  experiment. With no app, Start runs the agent on the gateway's Agent Server (below): it can
  talk, plan and keep its review list, and reaches the microscope once the app runs.
- **An API key**, stored once with `fibsem-cli key set ANTHROPIC_API_KEY` (see below).
- **A playbook that doesn't run code.** Windows has no sandbox for the agent's code.

If another agent is already connected, such as Claude Code from another computer, Start asks
first. **Disconnect and start** replaces it.

While it runs, the card shows the goal, and the playbook, level, spend and start time on one
line, with **Open the Agent page** and **End**. It says so when the agent is waiting for the
app or has restarted.

- **If the app restarts,** the agent waits for it and resumes the same session. That's at
  most 3 times an hour after the app restarts, and once an hour after a crash. The gateway
  sets the level again each time, because the app starts at Watch: the level you last set
  while the agent ran (**Stop agent** counts as Watch), else the Start level. If the app
  kept running, its level is left as it is.
- **If the gateway closes,** the agent isn't started again by itself. A gateway that was
  closed on purpose (Quit) leaves its unfinished session for the operator: the Agent card
  shows "Last session ...: did not finish (the gateway closed) · Resume", and Resume
  offers the level you last set in that session. The same goes for a session not heard
  from for more than 15 minutes before the gateway started. After a gateway crash or a
  restart from its page, a session that was running a moment ago carries on.
- **End** asks first, then stops the agent and keeps the app, back at Watch. The app goes
  back to Watch whenever the gateway's agent ends: End, replaced by an **Allow**, or
  restarts used up. The session keeps the level you last set, for its Resume.
- **Start with no app running** starts the agent at once on the Agent Server, at the level you
  picked. When the app answers, the agent's calls go to it and the gateway sets the app to
  the level the agent runs at; nothing restarts. Resume works with no app the same way.
- **Stopping the app** from the gateway (Stop or a new start) leaves the agent running on the
  Agent Server, still paired, at the level you last set. When the same app and experiment
  answer again, it carries on there. A new start of another app or experiment ends it
  (the gateway compares the app and experiment it started; one opened by hand in the app
  isn't seen). End ends it. Quit leaves it for Resume, as above.
- **With no Agent Server** (it couldn't start), Start with no app leaves the agent waiting, and
  it starts once the app answers; stopping the app stops the agent first and leaves it
  waiting, carrying on once the next app answers, never on the app being stopped. A start
  still waiting for its app ends on Quit, since it never ran.

The page has **End** while one runs. A hub with "may start and stop apps" can start and
end the agent too. Each session's record and log are in
`~/.fibsem/agent/sessions/<date>-<session id>/`: its plan, transcript, review list and
handoff notes go in its `agent/` folder there, not in the experiment, so the experiment
holds only what the experiment made. The two name each other: `agent/experiment.json` in
the session's folder, and an `agent_session_joined` event on the experiment's record.
History lists the latest sessions, each with its status (running, waiting for its app,
ended, left unfinished when the gateway closed, or stopped without saying why) and the
experiment it worked on, and the Agent card shows the last one with **Resume**. The list
comes from `fibsem.records`, which reads the session folders and keeps no copy.

Running `fibsem-agent` by hand still works as before. The gateway doesn't need to be running
for it.

While the gateway runs, an agent never goes to the App Server directly: an App Server address
(an old `FIBSEM_SERVER_URL`, an MCP setting, a token kept for port 8001) is sent to the gateway
instead, so the one-agent slot, the Allow card and History apply to every agent. Without a
gateway, a client still finds an App Server running alone (tests, scripts).

The App Server enforces it too. As it starts it makes a secret and writes it into its
`app-server.json` (this user's alone); the gateway adds it to every call it passes on. While
a gateway runs, an agent's call without it is refused (`401 through_gateway`, naming the
gateway). An app started by hand or by a desktop launch works the same way. The operator's
token and the App Server's own (an App Server run alone: tests, evals, scripts) never need
it, and giving control up is never refused. Only the App Server with this PC's usual
`app-server.json` makes one, so a second App Server with its own file (an eval) is never
refused, and websockets (which the gateway doesn't pass on yet) aren't checked. Any program
running as this user can read the file, so the secret guards against a mistake (an old
address or token), not against a program set on getting around the gateway.

The gateway runs `fibsem-agent-server` itself, with no window, from its start until it closes,
and starts it again if it ends (its log is in `~/.fibsem/logs/gateway/agent-host-*.log`). A
host already started by hand is used as it is and never stopped by the gateway. The host
serves the agent's own routes: its events, session, plan, transcript, messages, level and
Stop. While no App Server runs the gateway passes calls to it, so the gateway's agent, and the
Agent page, work with no app; a call only the app serves (the microscope's, a workflow's) is
answered 503 `no_app`. The host keeps its own level, which the gateway sets as it does the
App Server's.

An agent with no App Server (no instrument, or an instrument served by something else) runs
against the same host. It sits behind the gateway the same way: it serves no pages and
pairs nothing, checks tokens against the gateway's `paired.json`, and the gateway passes calls
to it while no App Server runs. `fibsem-agent --host` asks to connect at the gateway, and the
operator allows it on the gateway's Agent page. Agents allowed on a host before this have to
ask again.

### The API key

Store it once on this PC, on the gateway's page (**Settings**, **Keys**, **Set…**) or with:

```
fibsem-cli key set ANTHROPIC_API_KEY
```

It asks for the key without showing it and keeps it in Windows Credential Manager (the
Keychain on a Mac), encrypted to this Windows login. The gateway reads it at each Start, so
there's nothing to restart, and passes it only to the agent, never to the app.
`fibsem-agent` run by hand reads it too. It's never shown, logged or sent anywhere else.
`fibsem-cli key status ANTHROPIC_API_KEY` says whether one is stored, and `fibsem-cli key
remove ANTHROPIC_API_KEY` removes it. An `ANTHROPIC_API_KEY` environment variable, if set,
is used instead.

Anyone using this Windows login can read the key, so use a key for this microscope alone, in
its own Console workspace with a monthly spend limit. To retire a key, disable it in the
Console first: that stops it everywhere. Then remove it here. To replace one, store the new
key, check the agent starts, then disable the old one.

## The desktop shortcut

```
fibsem-gateway shortcut
```

This puts a "Agent Gateway" shortcut on the desktop; AutoLamella's **Tools → Create Desktop
Shortcut → Agent Gateway...** makes the same one. It starts the gateway if it isn't running
and opens its page signed in as the operator, so this browser needs no pairing. Nothing else
starts: pick the microscope, the app and the agent on the page. `fibsem-gateway open` does the
same from a terminal.

## Pairing a hub

Pair a hub with a 4-digit code from `fibsem-gateway pair --hub --may-switch`, or pick
**Hub** under **Pairing → Pair a computer…** on the page. With **May start and stop apps** a
hub can also connect and disconnect the microscope. Without that grant a hub can only read
the gateway. Untick **May start and stop apps** on the page to take it back, or unpair the
hub. A hub can never force a stop past a running workflow; only the operator at the PC
can.

## At the scope

Double-click the AutoLamella desktop shortcut as usual.

- With the gateway off, the window opens and the gateway shows the app running with its
  window until you close it.
- With the app running without its window, you're asked first: **Take over** stops it (and
  any workflow it is running), then opens the window; **Cancel** leaves it running.

Only the operator closes a window: on the gateway's page, or at the PC. A hub
can't. The window closes as its own close button does, so it saves and disconnects first. If
it asks a question as it closes, the gateway leaves it open to be answered at the PC and says
so. It is never killed.

Two things never hold the microscope at once: while a host the gateway didn't start is
running (a hand-run `fibsem-autolamella-server`), the gateway refuses every start.
