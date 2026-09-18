"""Small dependency-free cross-platform process lock for shared runtime state."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows
    fcntl = None


class RuntimeLockError(RuntimeError):
    """Raised when a runtime lock cannot be acquired within the configured timeout."""


class RuntimeFileLock:
    """OS-level exclusive lock that works on POSIX and Windows without third-party packages."""

    def __init__(self, path: str | Path, timeout: float = 180.0, poll_interval: float = 0.25) -> None:
        self.path = Path(path)
        self.timeout = max(1.0, float(timeout))
        self.poll_interval = max(0.05, float(poll_interval))
        self._handle: Any = None

    def __enter__(self) -> "RuntimeFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        # msvcrt.locking locks an existing byte; ensure the file has one.
        if self.path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        self._handle = handle
        deadline = time.monotonic() + self.timeout

        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:  # pragma: no cover - exercised on Windows deployments
                    import msvcrt  # type: ignore
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return self
            except (BlockingIOError, OSError) as exc:
                if time.monotonic() >= deadline:
                    handle.close()
                    self._handle = None
                    raise RuntimeLockError(f"Timed out waiting for runtime lock: {self.path}") from exc
                time.sleep(self.poll_interval)

    def __exit__(self, exc_type, exc, tb) -> bool:
        handle = self._handle
        self._handle = None
        if handle is None:
            return False
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:  # pragma: no cover - exercised on Windows deployments
                import msvcrt  # type: ignore
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()
        return False
