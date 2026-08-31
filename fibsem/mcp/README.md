# fibsem-mcp — connect an agent to the microscope

> New to the agent? Start with the guide: [docs/agent-server.md](../../docs/agent-server.md) — what the agent can see and do, permissions, and the security model.

The MCP sidecar lets Claude (Code or Desktop) drive a fibsem server: acquire
and *look at* images, read state, move the stage, and always stop a mill. The
server is the security boundary — the sidecar registers only the tools the
server's `/capabilities` reports, and the server enforces scopes regardless.

## Install

```bash
pip install -e ".[server,mcp]"    # sidecar needs Python 3.10+
```

## 1. Start the bench server

On the machine with the microscope connection (or anywhere, with the Demo
simulator):

```bash
python -m fibsem.server.server --manufacturer Demo --arm-hardware
```

- The bearer token is generated and **logged at startup**; pass `--token <t>`
  to choose your own.
- Without `--arm-hardware` the server is read-only: state and stop-milling
  work, anything that commands the hardware is refused with a structured 403.
- Binds `127.0.0.1` by default. `--host 0.0.0.0` exposes it on the LAN —
  an explicit choice; the token then travels as cleartext HTTP.
- While running, the server writes `~/.fibsem/agent-server.json`
  (url + token) so local clients connect without copying anything, and
  refuses to start if another live server owns that file.
- `--ip-address` / `--config` select a real microscope instead of Demo.

## 2. Register the sidecar

```bash
claude mcp add fibsem -- fibsem-mcp
```

Same-machine setups need nothing more: the sidecar autodiscovers the running
server. For a server on another machine (or a non-default setup):

```bash
claude mcp add fibsem -- fibsem-mcp --url http://hydra-support:8001 --token <token>
```

(`FIBSEM_SERVER_URL` / `FIBSEM_SERVER_TOKEN` environment variables also work.)

For Claude Desktop, add to `claude_desktop_config.json`:

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
  `~/.fibsem/agent-server.json` behind. Readers reject it (dead pid), and the
  startup guard tells you the path — delete it if asked.

## Safety model, in one paragraph

Loopback bind keeps everything off the network by default; the per-session
bearer token keeps other users on the same machine out; the `hardware` scope
must be armed before anything moves or images; one hardware command runs at a
time (a second gets `409 busy`); `stop_milling` is always allowed and never
waits. **Never run this beside a running AutoLamella session** — one
commander per microscope; agent access to a live AutoLamella session goes
*through* the app (embedded hosting, in development), not around it.
