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
fibsem-gateway install --manufacturer Thermo --ip-address 192.168.0.1
```

On Windows this puts a "fibsemOS Gateway" shortcut in your Startup folder, so the gateway
starts when you log on, with a tray icon and no console. It needs no admin rights.
`fibsem-gateway uninstall` removes it. The connection flags are the ones
`fibsem-autolamella-server` takes, and the gateway passes them to the headless app. To run
it once by hand: `fibsem-gateway run` (add `--no-tray` with no desktop).

The gateway listens on this PC only (`http://127.0.0.1:8010`). Another computer reaches it
over an SSH tunnel, as it reaches the App Server.

## The tray and the page

The tray icon's dot shows the mode: grey for off, blue for a window, purple for a headless
app. Its menu opens a window, runs an app headless (the last experiment, or one you
choose), stops it, pairs a computer, and opens the gateway's page. Quitting the gateway
stops a headless app that isn't running a workflow.

The page (`/ui/gateway`) shows the same modes, the paired computers, and every switch
with who made it and from where.

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
