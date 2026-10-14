# fibsem-mcp — connect an agent to the microscope

> New to the agent? Start with the guide: [docs/app-server.md](../../docs/app-server.md) — what the agent can see and do at each level, and the security model.

The MCP sidecar lets Claude (Code or Desktop) work through AutoLamella's App
Server: acquire and *look at* images, read state, move the stage, and stop a
mill. The server is the security boundary: the sidecar registers every tool the
server's mounted routers serve, whatever the agent's level, and the server
refuses what the level doesn't allow with a 403 that says so. Each tool's description names the level it needs. If the server
restarts with a new token, the sidecar re-reads the discovery file and carries
on.

## Install

```bash
pip install -e ".[server,mcp]"
```

## 1. Start the App Server

In AutoLamella, turn the App Server on in Preferences → App Server (the guide,
[docs/app-server.md](../../docs/app-server.md), walks through it). *Serve the
microscope's devices* there adds the microscope tools; the Agent chip sets what
the agent may do with them.

- While running, the server writes `~/.fibsem/app-server.json`
  (url + token) so local clients connect without copying anything, and
  refuses to start if another live server owns that file.
- It binds `127.0.0.1` by default; exposing it on the LAN is an explicit
  choice, and the token then travels as cleartext HTTP.

## 2. Register the sidecar

```bash
claude mcp add -s user fibsem -- fibsem-mcp
```

`-s user` registers it for all your projects. In AutoLamella, *Copy for your
agent* in the Agent chip's *Connect an agent…* gives the same setup with the
absolute path filled in, for the agent to run itself. Same-machine setups need
nothing more: the sidecar autodiscovers the running
server. For a server on another machine (or a non-default setup):

```bash
claude mcp add -s user fibsem -- fibsem-mcp --url http://hydra-support:8001 --token <token>
```

(`FIBSEM_SERVER_URL` / `FIBSEM_SERVER_TOKEN` environment variables also work.)

For Claude Desktop, merge this into the `mcpServers` already in
`claude_desktop_config.json` (don't replace the file):

```json
{
  "mcpServers": {
    "fibsem": { "command": "fibsem-mcp" }
  }
}
```

If `claude` runs outside the environment where fibsem is installed, register
the absolute path to the script (e.g. `<env>/bin/fibsem-mcp`).

## 3. Use it

Start a new Claude session (`/mcp` should show `fibsem` connected with 15
tools) and ask:

- "What's the stage position?"
- "Acquire an ion image and tell me what you see."
- "Tilt to a 15 degree milling angle and confirm the readback."
- "Does the last electron image look focused?"

## 4. Watch a run

MCP cannot wake an idle agent, so an agent that supervises or observes a run
keeps a watcher running in the background and wakes on each line it prints:

```bash
fibsem-mcp watch
```

It finds the server as the sidecar does (`--url`/`--token`, the environment
variables, the discovery file) and follows a restarted app the same way. It
prints `PROMPT`, `ANSWERED`, `PROMPT WITHDRAWN`, `QUESTION`,
`QUESTION RELEASED`, task `EVENT` lines, `GAP` (events were lost; re-read the
prompt and status) and `RUN ENDED`, and exits 0 when the run ends or 1 when
the app cannot be watched. A run ends completed, cancelled, stalled or failed;
one that stops without recording an end ends the watch too, as
`RUN ENDED (stopped without an end event)`. It needs only fibsem, not the `[mcp]` extra; where
`fibsem-mcp` is not on the PATH, `python -m fibsem.mcp.sidecar watch` is the
same thing.

## Order and troubleshooting

- **Order barely matters**: the sidecar waits up to 30 s for a server to
  appear (`--wait` tunes it), so starting the server a moment later is fine.
  If none appears in time it exits with guidance, which an MCP client shows
  as `CONNECTION_CLOSED` — start the server and reconnect (`/mcp` in the
  session). `fibsem-mcp --check` prints the connection status and exits;
  run it in a terminal to see what the sidecar sees.
- **Wrong server version / unknown arguments**: if you launched with
  `python -m` from inside another fibsem checkout, that directory shadows
  your installed copy. Run from the repo you installed, or any directory
  without a `fibsem/` folder.
- **Stale discovery file**: a hard-killed server can leave
  `~/.fibsem/app-server.json` behind. Readers reject it (dead pid), and the
  startup guard tells you the path — delete it if asked.

## Safety model, in one paragraph

Loopback bind keeps everything off the network by default; the per-session
bearer token keeps other users on the same machine out; the agent level the
operator sets from the Agent chip decides what the agent may do, and nothing
moves or images below Drive; one hardware command runs at a time (a second gets
`409 microscope_busy`); `stop_milling` works at Assist and Drive and never
waits.
