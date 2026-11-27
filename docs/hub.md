# The Hub

The Hub is a program for a PC away from the microscopes. It holds a lab's list of
instruments, reaches each one over SSH, and shows what each is doing. One Hub serves a lab; a
PC that looks after a single microscope from another room is a Hub with one instrument.

It is a different program from the [gateway](gateway.md), and a PC runs one or the other:

| | Gateway (`fibsem-gateway`) | Hub (`agent-hub`) |
| --- | --- | --- |
| Runs on | each microscope PC | one PC away from the microscopes |
| Connections | accepts them, and decides who gets in | only connects out |
| Starts and stops the app | yes | asks that PC's gateway |
| Holds the instrument list | no | yes |

The microscope PC decides. Its gateway pairs the Hub, says what the Hub may do there, and its
operator's Stop always wins. Nothing on a microscope PC calls the Hub, and no run depends on
it: with the Hub off, every microscope PC works as it does without one.

## Running it

```
agent-hub            # run the Hub, on 127.0.0.1:8020
agent-hub open       # its page, in this PC's browser
agent-hub status     # the instruments as last read
agent-hub pair       # a code that signs a browser on another computer in
agent-hub stop
```

The Hub listens on its own PC only. Its sign in is its own: the install's operator token, and
browsers paired with a code from `agent-hub pair`. They are kept apart from a gateway's
paired computers.

## The page

`agent-hub open` opens the Hub's page (`/ui/hub`), signed in as this PC's operator. The
instruments are listed down the side, each with a dot: green when it is read and well, amber
when it needs someone (not paired, a microscope or app that is unwell, an agent waiting), grey
when it can't be reached. **+** adds one, by name and where ssh connects to its PC.

Picking an instrument shows it as its own gateway does: the microscope, the app, and the
agent. The microscope is read only. The app card is the gateway's own: while nothing runs,
pick the app, its experiment and whether its window shows on the microscope PC, and start it;
stop one that runs with no window. From the Hub an app that can run with no window starts
without one, because a window is closed only at its own PC: the gateway never lets a Hub close
a window or force a stop. The agent card is the gateway's own too: start an agent
with a playbook, level, budget and goal, resume the last session, or end the one running.
These are asked of the instrument's gateway, which decides. The agent runs on the microscope
PC, and the gateway refuses a Hub that was not given "May start and stop apps" when it was
paired. The agent's chat and plan stay on the Agent page at the microscope PC.

Nothing else of an instrument is reached through the Hub: not its app's own pages, its
microscope's connection, its settings, its keys or its pairing.

### Test

**Test**, in the New instrument window and on an instrument that is not reachable, tries
reaching it now and lists each step: SSH reaches its PC, its PC accepts this account's SSH
key, its gateway answers, its gateway is a version this Hub was written for, and (once it is
in the list) this Hub is paired with it. The first step that fails says what to do, with ssh's
own words, and the steps after it are not tried. Testing an address before adding it keeps
nothing: no instrument, no tunnel.

### The simulator

**Start a simulator** gives the Hub an instrument with no microscope: an ordinary
`fibsem-gateway` on the hub PC, with no window and the Demo microscope, listed as
"Simulator". Pair with it like any instrument: run `fibsem-gateway pair --hub --may-switch`
on the hub PC and type the code. A PC runs one gateway, so the simulator is refused while
another gateway runs there; add that one by its address (`http://127.0.0.1:8010`) instead.
**Stop the simulator** stops its gateway and an app it runs with no window, and stopping the
Hub does the same. It stays in the list, with its pairing, until it is removed.

## Instruments

The list is `~/.fibsem/hub.yaml`: each instrument's name and where SSH connects to it
(`user@host`). The Hub opens one tunnel per instrument, to its gateway and nothing else: the
gateway is the one way onto an instrument, so the Hub never talks to an App Server or a
microscope server directly. It picks the local port, and opens a tunnel again when it drops. SSH runs in batch mode: it never asks for a password, and a host
this PC has not met is refused until a person has connected to it by hand once. See
[SSH tunnel](ssh-tunnel.md) for setting up the key on a microscope PC. If your SSH configuration
gives that host its own forwards (`LocalForward`), give the Hub a host name that has none: ssh
opens those too, and a port your own session already holds ends the Hub's tunnel.

To add an instrument, pair the Hub with its gateway: on the microscope PC run
`fibsem-gateway pair --hub`, and give the Hub the 4-digit code it shows
(`POST /hub/instruments/{name}/pair`). The Hub keeps the token per instrument in
`~/.fibsem/hub/`. Every 5 seconds it reads from each gateway whether it answers, what it runs,
whether an agent runs there, and how its App Server and microscope are (the gateway's own
[health](gateway.md#health): a status, why, and since when). Whether the PC answers at all is
the Hub's own to say; everything else is as the gateway reports it. A slow answer about the
agent or the state never makes an instrument that answered show as unreachable.

A token is kept with the address it was issued for, and is never sent anywhere else. If you
change where an instrument is reached, the Hub drops its token: pair the Hub with it again.
Two names that differ only in case, spaces or punctuation (`Scope 1`, `scope-1`) are refused,
because they would share one token file.
