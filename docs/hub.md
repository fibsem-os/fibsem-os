# The hub

The hub is a program for a PC away from the microscopes. It holds a lab's list of
instruments, reaches each one over SSH, and shows what each is doing. One hub serves a lab; a
PC that looks after a single microscope from another room is a hub with one instrument.

It is a different program from the [gateway](gateway.md), and a PC runs one or the other:

| | Gateway (`fibsem-gateway`) | Hub (`agent-hub`) |
| --- | --- | --- |
| Runs on | each microscope PC | one PC away from the microscopes |
| Connections | accepts them, and decides who gets in | only connects out |
| Starts and stops the app | yes | asks that PC's gateway |
| Holds the instrument list | no | yes |

The microscope PC decides. Its gateway pairs the hub, says what the hub may do there, and its
operator's Stop always wins. Nothing on a microscope PC calls the hub, and no run depends on
it: with the hub off, every microscope PC works as it does without one.

## Running it

```
agent-hub            # run the hub, on 127.0.0.1:8020
agent-hub status     # the instruments as last read
agent-hub pair       # a code that signs a browser on another computer in
agent-hub stop
```

The hub listens on its own PC only. Its sign in is its own: the install's operator token, and
browsers paired with a code from `agent-hub pair`. They are kept apart from a gateway's
paired computers.

## Instruments

The list is `~/.fibsem/hub.yaml`: each instrument's name and where SSH connects to it
(`user@host`). The hub opens one tunnel per instrument, to its gateway and nothing else: the
gateway is the one way onto an instrument, so the hub never talks to an App Server or a
microscope server directly. It picks the local port, and opens a tunnel again when it drops. SSH runs in batch mode: it never asks for a password, and a host
this PC has not met is refused until a person has connected to it by hand once. See
[SSH tunnel](ssh-tunnel.md) for setting up the key on a microscope PC.

To add an instrument, pair the hub with its gateway: on the microscope PC run
`fibsem-gateway pair --hub`, and give the hub the 4-digit code it shows
(`POST /hub/instruments/{name}/pair`). The hub keeps the token per instrument in
`~/.fibsem/hub/`. Every 5 seconds it reads from each gateway whether it answers, what it runs,
whether an agent runs there, and how its App Server and microscope are (the gateway's own
[health](gateway.md#health): a status, why, and since when). Whether the PC answers at all is
the hub's own to say; everything else is as the gateway reports it.
