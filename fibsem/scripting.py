"""Discovery and execution of user scripts.

Lets a user drop a plain ``.py`` file in a folder and run it, without packaging
anything. Application-agnostic: this module knows nothing about experiments,
microscopes or Qt. An application supplies the folder and the context object.

A script is a module exposing ``run(ctx)``:

    \"\"\"Export a summary to CSV.\"\"\"   # docstring -> menu tooltip

    def run(ctx):
        ...
        return path_or_dataframe_or_string

It declares its own contract with module-level flags, so the declaration cannot
drift from the code it governs:

    writes = True                 # the run mutates state the host should persist
    background = True             # safe to run off the GUI thread
    uses_microscope = True        # needs hardware, and whatever guarantees that implies
    on_workflow_completed = True  # host may also run it when a workflow finishes

Only ``writes`` is acted on here, via the context's optional ``save()`` hook. The
rest are parsed and exposed for the host to interpret — and no host acts on
``on_workflow_completed`` yet, so declaring it currently does nothing.
"""

import hashlib
import importlib.util
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, List, Optional

# Module-level names a script may declare. Anything else is ignored.
FLAG_WRITES = "writes"
FLAG_BACKGROUND = "background"
FLAG_USES_MICROSCOPE = "uses_microscope"
FLAG_ON_WORKFLOW_COMPLETED = "on_workflow_completed"

ENTRYPOINT = "run"


@dataclass
class DiscoveredScript:
    """A ``.py`` file found in the scripts folder, loaded or not."""

    path: Path
    name: str
    description: str = ""
    writes: bool = False
    background: bool = False
    uses_microscope: bool = False
    on_workflow_completed: bool = False
    error: Optional[str] = None  # set when the file could not be loaded
    _module: Optional[ModuleType] = field(default=None, repr=False)

    @property
    def is_runnable(self) -> bool:
        return self.error is None and self._module is not None

    @property
    def content_hash(self) -> str:
        """Short hash of the file as it was loaded.

        Users edit these in place, so "which version ran" is otherwise
        unanswerable after the fact.
        """
        try:
            return hashlib.sha256(self.path.read_bytes()).hexdigest()[:8]
        except OSError:
            return "unknown"


def _load_module(path: Path) -> ModuleType:
    """Import a script file under a unique synthetic name.

    Deliberately **not** inserted into ``sys.modules``: two scripts sharing a
    filename in different folders would collide, and stale modules would leak
    across reloads.
    """
    module_name = f"_fibsem_script_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create a module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_script(path: Path) -> DiscoveredScript:
    """Load one script file, capturing failure rather than raising.

    A file that cannot be loaded still comes back with ``error`` set, so the host
    can show it. Silently omitting it is the failure this avoids: the author sees
    nothing and assumes the folder is wrong.
    """
    script = DiscoveredScript(path=path, name=path.stem)
    try:
        module = _load_module(path)
    except Exception:
        script.error = traceback.format_exc(limit=3).strip().splitlines()[-1]
        logging.warning("Failed to load script %s: %s", path, script.error)
        return script

    entrypoint = getattr(module, ENTRYPOINT, None)
    if not callable(entrypoint):
        script.error = f"no {ENTRYPOINT}() function found — expected def {ENTRYPOINT}(ctx)"
        return script

    doc = (module.__doc__ or "").strip()
    script._module = module
    script.description = doc.splitlines()[0] if doc else path.stem
    script.writes = bool(getattr(module, FLAG_WRITES, False))
    script.background = bool(getattr(module, FLAG_BACKGROUND, False))
    script.uses_microscope = bool(getattr(module, FLAG_USES_MICROSCOPE, False))
    script.on_workflow_completed = bool(getattr(module, FLAG_ON_WORKFLOW_COMPLETED, False))
    return script


def discover_scripts(directory: Path) -> List[DiscoveredScript]:
    """Load every ``.py`` in ``directory``, sorted by filename.

    Returns failures alongside successes. A missing directory is not an error —
    it just means no scripts yet. Files starting with ``_`` are hidden from the
    listing; note that they are *not* importable from a script, since the folder
    is never on ``sys.path`` (FIB-386).

    Not cached: scripts are loaded fresh on every call, so editing a file and
    running it again picks up the change. A data script runs in milliseconds, so
    the import cost is irrelevant beside it, and "reload" stops being a feature.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    scripts: List[DiscoveredScript] = []
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        scripts.append(load_script(path))
    return scripts


@dataclass
class ScriptResult:
    """Outcome of one run, for the host to render and record."""

    script: DiscoveredScript
    value: Any = None
    error: Optional[BaseException] = None
    saved: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


def run_script(script: DiscoveredScript, ctx: Any) -> ScriptResult:
    """Run a script and return its outcome. Never raises.

    Exceptions are captured rather than propagated: hosts call this from a Qt slot,
    and PyQt5 aborts the whole process on any exception that escapes one (FIB-329).
    The host decides how to surface it.

    If the script declared ``writes`` and ``ctx`` provides a ``save()``, it is
    called after a successful run. A script that did not declare it cannot
    persist anything by accident.
    """
    if not script.is_runnable:
        return ScriptResult(script=script, error=RuntimeError(script.error or "not runnable"))

    logging.info(
        "Running script %s (%s) writes=%s microscope=%s",
        script.name, script.content_hash, script.writes, script.uses_microscope,
    )

    result = ScriptResult(script=script)
    # Tag anything the script logs with its name, so lines in a shared logfile are
    # attributable. Restored afterwards so a reused context is not left modified.
    original_log = getattr(ctx, "log", None)
    if callable(original_log):
        ctx.log = lambda message, _name=script.name: logging.info("[%s] %s", _name, message)
    try:
        result.value = script._module.run(ctx)  # type: ignore[union-attr]
    except BaseException as e:  # noqa: BLE001 - deliberately broad, see docstring
        result.error = e
        logging.exception("Script %s failed", script.name)
        return result
    finally:
        if callable(original_log):
            ctx.log = original_log

    save = getattr(ctx, "save", None)
    if script.writes and callable(save):
        save()
        result.saved = True

    return result
