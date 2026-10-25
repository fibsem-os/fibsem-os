# The Gateway

The gateway (`fibsem-gateway`) is a small, always-on program on a microscope PC. It runs
one **mode** at a time, so a hub on another computer, or a browser, can start and stop the
app there without anyone at the PC:

| Mode | What runs |
|---|---|
| **Off** | Nothing holds the microscope. |
| **Window** | The app's window at this PC, as a desktop launch opens it. |
| **Headless app** | The app's App Server with no window (`fibsem-autolamella-server`). |

It never touches the microscope itself; the app it starts does. Without the gateway
installed, nothing changes: desktop launches work as they always have.

## Installing it

```
fibsem-gateway install
```

On Windows this puts a "fibsemOS Gateway" shortcut in your Startup folder, so the gateway
starts when you log on, with a tray icon and no console. It needs no admin rights.
`fibsem-gateway uninstall` removes it. To run it once by hand: `fibsem-gateway run` (add
`--no-tray` with no desktop).

A headless app connects with one of the microscope configurations saved on this PC, the
ones the window offers: the default unless the page picks another. Connection flags given
to `install` or `run` (`--manufacturer`, `--ip-address`) override the configuration's.

The gateway listens on this PC only (`http://127.0.0.1:8010`). Another computer reaches it
over an SSH tunnel, as it reaches the App Server.

## The tray and the page

The tray icon's dot shows the mode: grey for off, blue for a window, purple for a headless
app. Its menu does the following:

- opens a window;
- runs an app headless, with the last experiment or one you choose;
- stops it;
- ends the agent, while one runs;
- pairs a computer;
- opens the gateway's page. Quitting the gateway
stops a headless app that isn't running a workflow.

The page (`/ui/gateway`) shows:

- the same modes;
- the agent;
- the paired computers;
- every switch, with who made it and from where.

## The agent

The gateway can run the agent on this PC, next to the app. The **Agent** card on the page
starts it with:

- a **profile** (`default` is picked first);
- a **level** (Watch, Assist or Drive);
- an optional **budget**, in place of the profile's;
- a **goal** for the session.

**Show configuration** lists what the profile runs with: the model, budget, skills and
reminders.

Start needs:

- **The app running**, in Window or Headless mode. The agent acts through it.
- **An API key** in `ANTHROPIC_API_KEY`, set for this PC's user. Restart the gateway after
  setting it. The key stays in the environment; it's never written to a file or a log.
- **A profile that doesn't run code.** Windows has no sandbox for the agent's code.

If another agent is already connected, such as Claude Code from another computer, Start asks
first. **Disconnect and start** replaces it.

While it runs, the card shows the goal, profile, level and spend.

- **If the app restarts,** the agent waits for it and resumes the same session. That's at
  most 3 times an hour after the app restarts, and once an hour after a crash. The gateway
  sets the Start level again each time, because the app starts at Watch.
- **End** stops the agent and keeps the app. **End and stop the app** asks first, then stops
  both.
- **Stopping the app** from the gateway (Stop, a switch, or Quit) ends the agent first.

The tray has **End agent** while one runs. A hub with "may start and stop apps" can start and
end the agent too. Each session's record and log are in
`~/.fibsem/agent/sessions/<date>-<session id>/`, and the page lists the latest sessions.

Running `fibsem-agent` by hand still works as before. The gateway doesn't need to be running
for it.

## Pairing a hub

Pair a hub as you pair a browser, with a 4-digit code: **Pair a computer → A hub that may
start and stop apps** in the tray, or `fibsem-gateway pair --hub --may-switch`. Without
that grant a hub can only read the gateway. Untick **May start and stop apps** on the page
to take it back, or unpair the hub. A hub can never force a switch past a running
workflow; only the operator at the PC can.

## At the scope

Double-click the desktop shortcut as usual.

- With the gateway off, the window opens and the gateway shows Window mode until you close
  it.
- With the app running headless, you're asked first: **Take over** stops it (and any
  workflow it is running), then opens the window; **Cancel** leaves it running.

A window is only ever closed at the PC: the gateway refuses to switch away from it.

Two things never hold the microscope at once: while an App Server the gateway didn't start
is running (a hand-run `fibsem-autolamella-server`), the gateway refuses every switch.
