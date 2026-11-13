# The Gateway

The gateway (`fibsem-gateway`) is a small, always-on program on a microscope PC. It starts
and stops the app there, so a hub on another computer, or a browser, can do it without
anyone at the PC. The app is either **off** (nothing holds the microscope) or **running**:
the app's host, with its window at this PC (as a desktop launch opens it) or without one
(as `fibsem-autolamella-server` runs it).

It never touches the microscope itself; the app it starts does. Without the gateway
installed, nothing changes: desktop launches work as they always have.

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
over an SSH tunnel, as it reaches the App Server: see
[Running the agent on another computer](ssh-tunnel.md).

## The window and the page

The gateway runs in a small window, on the taskbar (the Dock on a Mac). It shows the app, with
**Start** and **Stop**; the microscope's connection; and the agent, with **End**. **Pair a
computer** shows a code for another computer's browser. **Open page** opens the gateway's page,
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
  experiment folder and **Show the window on this PC**), with **Start the app**. While it
  runs: one row for the microscope, with Connect or Disconnect, and one for the app, with
  **Open the app** and **Stop** (Stop asks first in a dialog; with its window, it closes the window).
  Under them, the **Agent** card (below).
- **History.** Every start and stop of the app, with who made it and from where, and the
  latest agent sessions, newest first. **All**, **App** and **Agent** filter it. A session
  opens to show its playbook, spend, how it ended and what it last said, and **Resume**.
- **Pairing.** The paired computers, and a code to pair another.
- **Settings.**
  - **Secrets:** whether each key fibsemOS reads is set on this PC, and where from (the
    environment or the key store). The page never shows or takes a key. **Copy command**
    copies the `fibsem-cli key set` command to run on the PC.
  - **Defaults:** what Overview fills in for the microscope, the app and the agent, kept in
    `~/.fibsem/gateway-defaults.json`. Only the operator saves them.
  - **Playbooks:** the playbooks on this PC, read-only.
  - **Gateway:** the version, **Restart** and **Stop**. Stopping the gateway from the page
    stops the app and any agent run. Start it again at the computer.

The bottom of the list says when there is newer code: "fibsemOS 0.6.3 available" when a
newer release is on PyPI, or "Update available" for a source checkout behind GitHub's
`main` (hover it for how many changes). The gateway asks when it starts and once a day after. Offline, or with a commit that
was never pushed, it shows nothing.

## The agent

The gateway can run the agent on this PC, next to the app. The **Agent** card on the page
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

- **The app running**, with its window or without. The agent acts through it. A microscope
  needn't be connected: without one the agent can still read and review an experiment.
- **An API key**, stored once with `fibsem-cli key set ANTHROPIC_API_KEY` (see below).
- **A playbook that doesn't run code.** Windows has no sandbox for the agent's code.

If another agent is already connected, such as Claude Code from another computer, Start asks
first. **Disconnect and start** replaces it.

While it runs, the card shows the goal, and the playbook, level, spend and start time on one
line, with **Open the Agent page** and **End**. It says so when the agent is waiting for the
app or has restarted.

- **If the app restarts,** the agent waits for it and resumes the same session. That's at
  most 3 times an hour after the app restarts, and once an hour after a crash. The gateway
  sets the Start level again each time, because the app starts at Watch.
- **End** asks first, then stops the agent and keeps the app.
- **Stopping the app** from the gateway (Stop, a new start, or Quit) ends the agent first.

The page has **End** while one runs. A hub with "may start and stop apps" can start and
end the agent too. Each session's record and log are in
`~/.fibsem/agent/sessions/<date>-<session id>/`. History lists the latest sessions, and the
Agent card shows the last one with **Resume**.

Running `fibsem-agent` by hand still works as before. The gateway doesn't need to be running
for it.

### The API key

Store it once on this PC:

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

Pair a hub with a 4-digit code from `fibsem-gateway pair --hub --may-switch`. The
window's **Pair a computer** pairs a browser, not a hub. With it a
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
