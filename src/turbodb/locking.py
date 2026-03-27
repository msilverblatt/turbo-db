# src/turbodb/locking.py
"""Cross-platform file locking."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

__all__ = ["FileLock"]

_lock_counts: dict[str, int] = {}
_lock_fds: dict[str, int] = {}
_registry_lock = threading.Lock()
_thread_local = threading.local()


class FileLock:
    """Context manager for exclusive file locking.

    Uses fcntl.flock on Unix and msvcrt.locking on Windows.
    Reentrant from the same thread.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()
        self._key = str(self._path)
        self._fd: int | None = None

    def __enter__(self) -> FileLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if not hasattr(_thread_local, "counts"):
            _thread_local.counts = {}

        key = self._key
        depth = _thread_local.counts.get(key, 0)

        if depth == 0:
            fd = os.open(str(self._path), os.O_CREAT | os.O_RDWR)
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
            self._fd = fd
            with _registry_lock:
                _lock_fds[key] = fd
        else:
            self._fd = None

        _thread_local.counts[key] = depth + 1
        return self

    def __exit__(self, *exc) -> None:
        key = self._key
        depth = _thread_local.counts.get(key, 1)
        _thread_local.counts[key] = depth - 1

        if depth == 1:
            with _registry_lock:
                fd = _lock_fds.pop(key, None)
            if fd is not None:
                if sys.platform == "win32":
                    import msvcrt
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
