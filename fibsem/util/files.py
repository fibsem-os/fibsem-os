"""Writing a file whole, or not at all."""

import os
import threading
import time
from pathlib import Path
from typing import Union

# Windows can't replace a file another process has open for reading. Readers
# are quick, so the swap is tried again for up to two seconds.
REPLACE_TRIES = 40
REPLACE_WAIT_S = 0.05


def write_atomic(path: Union[str, Path], text: str, encoding: str = "utf-8") -> None:
    """Write *text* to *path* so a reader, or a crash part way, never leaves
    half a file: it is written beside *path*, flushed to disk and swapped in.
    A failure raises and leaves the file that was there.

    A plain ``open``, not ``mkstemp``: that makes the file 0600 and the swap
    keeps the mode, which on a PC shared between operator accounts would lock
    the next operator out.
    """
    path = Path(path)
    partial = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.partial"
    )
    try:
        with open(partial, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(REPLACE_TRIES):
            try:
                os.replace(partial, path)
                return
            except PermissionError:
                if attempt == REPLACE_TRIES - 1:
                    raise
                time.sleep(REPLACE_WAIT_S)
    except BaseException:
        try:
            os.remove(partial)
        except OSError:
            pass
        raise
