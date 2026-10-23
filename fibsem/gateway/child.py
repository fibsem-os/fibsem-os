"""How the gateway runs an app with no window: ``python -m fibsem.gateway.child
<stop-file> <module:function> [args...]`` calls the app's serve function with
a stop event that is set when the stop file appears. A file works on every
platform, where a signal to a process with no console doesn't on Windows."""

import importlib
import sys
import threading
from pathlib import Path
from typing import List, Optional

POLL_S = 0.5


def watch(stop_file: Path, stop: threading.Event) -> None:
    """Set *stop* once *stop_file* exists."""
    while not stop.is_set():
        if stop_file.exists():
            stop.set()
            return
        stop.wait(POLL_S)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    stop_file, target, args = Path(argv[0]), argv[1], argv[2:]
    module, function = target.split(":")
    stop = threading.Event()
    threading.Thread(target=watch, args=(stop_file, stop), daemon=True).start()
    serve = getattr(importlib.import_module(module), function)
    return serve(args, stop=stop) or 0


if __name__ == "__main__":
    sys.exit(main())
