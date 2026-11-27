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
agent-hub shortcut   # an "Agent Hub" shortcut on the Desktop
agent-hub stop
```

The Hub runs in a small window, as a gateway does: how many instruments it has and which
need someone, the agents running on this PC, **Pair** for a code that signs a browser on
another computer in, and **Open page**. Closing the window asks first, then stops the Hub
and says what happens to its agents. `agent-hub run --no-window` runs it with no window,
until Ctrl-C or `agent-hub stop`, and so does a PC with no display. `--minimised` starts with
the window minimised. The desktop shortcut starts the Hub when none runs and brings its
window to the front.

The Hub listens on its own PC only. Its sign in is its own: the install's operator token, and
browsers paired with a code from `agent-hub pair`. They are kept apart from a gateway's
paired computers.

## The page

`agent-hub open` opens the Hub's page (`/ui/hub`), signed in as this PC's operator. The
instruments are listed down the side, each with a dot: green when it is read and well, amber
when it needs someone (not paired, a microscope or app that is unwell, an agent waiting), grey
when it can't be reached. **+** adds one, by name and where ssh connects to its PC.

Picking an instrument shows it as its own gateway does, in three parts. **Now** has the
microscope, the app and the agent. **History** is the gateway's own History: the starts and
stops of its app and the agent sessions that ran on the microscope PC, with their Resume.
**About** lists what its gateway says of itself: its version, how it connects the microscope
and the configurations it has, its apps, its experiments folder and its Device Servers. About
is read only: these are set at the microscope PC.

In **Now**, the microscope is read only. The app card is the gateway's own: while nothing runs,
pick the app, its experiment and whether its window shows on the microscope PC, and start it;
stop one that runs with no window. From the Hub an app that can run with no window starts
without one, because a window is closed only at its own PC: the gateway never lets a Hub close
a window or force a stop. The agent card is the gateway's own too: start an agent
with a playbook, level, budget and goal, resume the last session, or end the one running.
These are asked of the instrument's gateway, which decides. The agent runs on the microscope
PC, and the gateway refuses a Hub that was not given "May start and stop apps" when it was
paired.

Nothing else of an instrument is changed through the Hub: not its app's other pages, its
microscope's connection, its settings, its keys or its pairing.

### The Agent page

**Open the Agent page** on an instrument opens that instrument's Agent page at the Hub
(`/ui/hub/agent/<name>`): the same page the microscope PC serves, with the chat, the plan
and the review list. There is one conversation, kept at the microscope PC and seen in both
places. The browser talks to the Hub only, so it is not paired with the microscope PC. The
Hub passes each of the page's calls to the instrument's gateway with its own pairing token,
and the gateway decides:

- **Reading** is allowed to any paired Hub.
- **A message, an answer to the agent's question, and Stop agent** are allowed to a Hub that
  was given "May start and stop apps".
- **The level** can be lowered freely, and raised as far as the microscope PC allows.
- **Everything else** is refused with "This is done at the microscope PC": approving a
  request, a workflow, the microscope, the plan, the library.

What a Hub may do is one setting at the microscope PC, **Remote control from a Hub**
(Settings, This PC, on its gateway's page): off (read only), message, answer and Stop, or
those and raising the level to Assist or to Drive. It is changed only at that PC. What is
done from the Hub is written in the Hub's History and in that gateway's History, with the
Hub's name.

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
"Simulator". The Hub pairs with the simulator it started by itself, with "may start and stop
apps": it asks that gateway for a code as `fibsem-gateway pair --hub --may-switch` does on the
hub PC. It does this for its own simulator only; every other gateway is paired by a person
with a code. A PC runs one gateway, so the simulator is refused while
another gateway runs there; add that one by its address (`http://127.0.0.1:8010`) instead.
**Stop the simulator** stops its gateway and an app it runs with no window, and stopping the
Hub does the same. It stays in the list, with its pairing, until it is removed.

### Agents with no instrument

**Agents** lists the agents running on the hub PC, and **New agent** starts one. Pick an
instrument and the form is that instrument's own start form, with the agent running on
the hub PC. Leave it at **No instrument** and the agent acts on no microscope: you give it a
goal and a budget, and talk to it on its Agent page. It takes no instrument's slot and has
no level, and it needs no playbook: it starts on the shipped `plain` playbook
(`instrument: none`), which a lab's own copy can add connectors to.

Such an agent runs on the hub PC with the hub PC's API key. Its chat, plan and record are
kept in its session's folder there (`~/.fibsem/hub/free-sessions/`) and served by an Agent
Server of its own, which the Hub holds: **Open the Agent page** shows the same Agent page
with the microscope parts left out. The agent's token there is made at each start and kept
in memory only. The people signed in to the Hub are the operator on that page.

Up to 5 run at once. **Settings** changes that number, at the hub PC only. **End** stops
one. Stopping the Hub ends them all, and one that outlived a Hub that crashed is stopped
and carries its session on when the Hub starts again. Each session is listed in
**Sessions** under "No instrument", and **Resume** there carries it on in the same folder.

### A folder, connectors and code mode

An agent that runs on the hub PC, with an instrument or without one, can be given three
things beyond its own folder. They are off until **Settings** has "Offer a folder,
connectors and code mode in the start form" ticked, which is done only at the hub PC. Tick
it once code mode has been tried on that PC. Until then the start form shows none of it
and a start that asks for it is refused.

- **Folder**: one data folder the agent may read, picked from the **Data folders** listed
  in Settings. It never writes there. What it makes stays in its own work folder, in its
  session.
- **Connectors**: which of the Hub's connectors the agent may use. Their tokens stay in
  the hub PC's key store and are never shown.
- **Code mode**: the agent writes and runs its own code, always inside the sandbox. There
  is no unsandboxed choice, and a hub PC on Windows has no sandbox, so no code mode. The
  sandbox has no network unless **Network** is ticked, which lets its code reach the
  **Allowed sites** listed in Settings and nothing else. A local address (localhost, an IP
  address, a name inside the lab's network) is never allowed.

The two lists are changed only at the hub PC. Every start and every resume is checked
against them as they are then: a folder that isn't listed, a site that isn't allowed and a
local address are each refused by name. An agent's row, its Agent page and **Sessions** say
what it was given, such as "runs code, sandboxed".

### Sessions

**Sessions** lists the agent sessions of every instrument in one place: those that ran on
each microscope PC, as its gateway lists them, and those that ran on the hub PC. Each row
says which instrument, where it ran, who started it, how it ended and what it spent. The
card above totals the spend of the rows listed, per instrument and per day. An instrument
that can't be read now is named with why, and its sessions are left out until it answers.

Open a row for **Resume**. A session that ran on a microscope PC is resumed there by its
gateway. One that ran on the hub PC is resumed on the hub PC: the Hub asks the instrument's
gateway for the slot again, at the level you pick, and the agent carries on with the same
playbook, goal and folder.

### After the Hub restarts

An agent on the hub PC is its own process, so it keeps running if the Hub ends without
being stopped (a crash, a closed terminal, a restart of the PC's session). The Hub that
starts next finds it, asks the instrument's gateway for the same slot back and carries on
keeping it. If the gateway had already set Watch and freed the slot, it gives the slot back
to the same paired Hub for the same session, and the agent carries on at the level it last
ran at. The gateway still refuses if the level was changed at the microscope PC since, if
the agent was stopped there, or if another agent took the slot: the agent is then ended with
that reason, and Sessions has its Resume. Stopping the Hub
(`agent-hub stop`, Ctrl-C) still ends every agent it runs and gives the slots back. The
Hub's log says how long each carry-on took.

### History

**History** lists what was done through this Hub, newest first: agents started, resumed
and ended, apps started and stopped, instruments added, removed and paired, the simulator,
and changes to connectors, keys (by name, never a value), playbooks and skills. Each line
says who asked and whether it was refused. It is kept in `~/.fibsem/hub/history.jsonl`.

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
