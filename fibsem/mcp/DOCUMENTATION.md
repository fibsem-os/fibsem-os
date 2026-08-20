# fibsemOS MCP — Programmatic Analysis

This document covers how to call AI-powered lamella analysis from Python, and how the current approach could be migrated to the Anthropic API in future.

---

## Analysing lamellas from Python

The `analyze_lamellas` function in `fibsem.applications.autolamella.tools.claude_analysis` runs the `analyze-lamellas` Claude Code skill as a subprocess and returns the result.

### How it works

```
Python process
  └── subprocess: claude -p --dangerously-skip-permissions "/analyze-lamellas <path> ..."
        └── Claude Code loads .claude/skills/analyze-lamellas/SKILL.md
              └── calls mcp__fibsem__load_image for each stage image
              └── visually assesses each image
              └── returns markdown report (and optionally a JSON block)
```

The `claude` binary is resolved automatically — first via `PATH`, then from the default Claude Code install location at `~/.claude/local/claude`.

The subprocess runs with `cwd` set to the fibsem repo root so that Claude Code finds `.claude/settings.json` and the fibsem MCP server configuration.

### Installation

Claude Code must be installed: https://claude.ai/code

The fibsem MCP server must be configured in `.claude/settings.json`:

```json
{
  "mcpServers": {
    "fibsem": {
      "command": "python",
      "args": ["-m", "fibsem.mcp.server"]
    }
  }
}
```

### Usage

```python
from fibsem.applications.autolamella.tools.claude_analysis import analyze_lamellas

experiment = "/path/to/AutoLamella-2026-04-08-10-55"

# Markdown report for all lamellas
report = analyze_lamellas(experiment)
print(report)

# Parsed dict — easier to work with programmatically
data = analyze_lamellas(experiment, json_output=True)
for lam in data["lamellas"]:
    print(lam["name"], lam["status"], lam["workflow_status"])
print(data["summary"])

# Analyse a subset by index, range, or name
data = analyze_lamellas(experiment, selectors=["1-3"], json_output=True)
data = analyze_lamellas(experiment, selectors=["vocal-walrus"], json_output=True)
```

### JSON schema

When `json_output=True` the returned dict has this structure:

```json
{
  "lamellas": [
    {
      "name": "01-hardy-baboon",
      "status": "Success",
      "workflow_status": "completed",
      "last_stage": "Polish",
      "key_finding": "Clean lamella, organelles visible in EB",
      "assessments": {
        "setup_ib": "Surface is clean with minimal ice contamination.",
        "rough_mill_ib": "Symmetric trenches, lamella ~1.5 µm thick.",
        "polish_ib": "Thin uniform lamella, straight edges.",
        "polish_eb": "Moderately bright band with visible membrane contrast."
      }
    }
  ],
  "summary": {
    "total": 3,
    "success": 2,
    "failed": 0,
    "incomplete": 1,
    "poor": 0
  }
}
```

`status` is one of `Success`, `Failed`, `Incomplete`, `Poor`.

`workflow_status` maps to the actionable state: `completed`, `failure`, `rework`, `continue`.

`assessments` values are strings, or `null` if the image was not found (stage not reached).

### Lamella selectors

| Selector | Matches |
|---|---|
| `"1"` or `"01"` | Lamella directory starting with `01-` |
| `"1-5"` | Indices 1 through 5 |
| `"vocal-walrus"` | Any directory whose name contains this substring |
| `"/path/to/dir"` | That specific directory directly |

### Function reference

```python
analyze_lamellas(
    experiment_path: str | Path,   # experiment root or single lamella dir
    selectors: list[str] | None,   # filter which lamellas to analyse (default: all)
    json_output: bool,             # return parsed dict instead of markdown (default: False)
    cwd: str | Path | None,        # override subprocess working directory
    timeout: int,                  # seconds before killing the subprocess (default: 300)
) -> str | dict
```

---

## Limitations of the current approach

The subprocess approach is simple and reuses the skill definition without duplicating logic, but has some practical constraints:

- **Blocks until complete** — the call takes 1–5 minutes depending on the number of lamellas; there is no streaming or progress callback.
- **Requires Claude Code** — the `claude` binary must be installed on the machine running the analysis.
- **Requires the MCP server** — the fibsem MCP server must be configured so Claude Code can call `mcp__fibsem__load_image` to load TIFFs as images.
- **Output parsing is fragile** — the JSON block is extracted with a regex. If the model deviates from the expected format the extraction fails.

---

## Future: migrating to the Anthropic API

The subprocess approach can be replaced with a direct call to the Anthropic API once a few small pieces are in place. The skill logic (SKILL.md) becomes the system prompt; the MCP tools are replaced with Python functions.

### What changes

| Current | API equivalent |
|---|---|
| `claude -p` subprocess | `anthropic.Anthropic().messages.create()` |
| `mcp__fibsem__load_image` MCP tool | Python function: load TIFF → base64 image block |
| `bash` (for contact sheet) | Python function: call `generate_contact_sheet()` directly |
| Skill SKILL.md | System prompt (read from file or hardcoded) |
| Regex JSON extraction | Ask the model to output JSON directly; use `response_format` |

### Sketch

```python
import anthropic
import base64
from pathlib import Path
from fibsem.structures import FibsemImage

client = anthropic.Anthropic()

def _load_image_tool(path: str) -> list:
    """Load a TIFF and return an Anthropic image block."""
    try:
        img = FibsemImage.load(path)
        # encode as JPEG for the API
        import cv2, numpy as np
        _, buf = cv2.imencode(".jpg", img.data)
        b64 = base64.standard_b64encode(buf).decode()
        return [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}]
    except Exception:
        return [{"type": "text", "text": f"[not found: {path}]"}]

skill = Path(".claude/skills/analyze-lamellas/SKILL.md").read_text()

tools = [
    {
        "name": "load_image",
        "description": "Load a FIB-SEM image from a file path.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
]

def _handle_tool(tool_name: str, tool_input: dict) -> list:
    if tool_name == "load_image":
        return _load_image_tool(tool_input["path"])
    return [{"type": "text", "text": f"Unknown tool: {tool_name}"}]

def analyze_lamellas_api(experiment_path: str) -> dict:
    messages = [{"role": "user", "content": f"Analyse {experiment_path} --json"}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            system=skill,
            messages=messages,
            tools=tools,
            max_tokens=8096,
        )

        if response.stop_reason == "end_turn":
            # extract text and parse JSON block
            text = next(b.text for b in response.content if b.type == "text")
            import re, json
            match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            return json.loads(match.group(1))

        # handle tool calls and loop
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result_content = _handle_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
```

### Benefits over subprocess

- **No Claude Code dependency** — runs anywhere the `anthropic` package is installed.
- **Streaming** — switch to `client.messages.stream()` for live output.
- **Prompt caching** — the skill system prompt can be cached with `cache_control`, cutting cost significantly for repeated calls.
- **Better error handling** — structured tool results rather than fragile output parsing.
- **Direct image loading** — images are passed as base64 blocks, no MCP server needed.

### When to migrate

The API approach is worth implementing when:
- Analysis needs to run on machines without Claude Code installed (e.g., a server or CI pipeline)
- You need streaming progress in a UI
- Cost matters and you want prompt caching
- The regex JSON extraction proves unreliable in practice
