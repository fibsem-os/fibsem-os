"""The App Server's routes for the agent and its operator, in one place.

- `session`: the agent's report of its session.
- `plan`, `transcript` and `files`: its plan.md and transcript, and where
  they are kept.
- `requests`: what it asks the operator to approve at Assist.
- `operator`: the operator's routes for the agent (message, level, stop,
  approve), behind the operator token.
- `setup`: the Connect an agent text.
- `customisation` and `library`: what the agent runs with: tools, skills,
  subagents and playbooks.

`routes.mount_agent_routes` mounts them. The level gate and the blocklist
stay in `fibsem.server.auth`: every route goes through them, not only these.
"""
