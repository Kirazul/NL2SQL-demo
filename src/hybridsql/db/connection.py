"""Access to `eicu.db`, read-only and under constraint.

Every read of the database goes through here. Three guarantees, enforced at open
time rather than left to the caller:

1. **Truly read-only** — the `mode=ro` URI is refused by SQLite itself on write.
   That is an engine constraint, not a convention.
2. **Bounded time** — a progress handler aborts any query that exceeds the
   deadline. Without it, a malformed JOIN blocks the process.
3. **Bounded volume** — the number of returned rows is capped.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hybridsql.config import settings


class QueryTimeout(RuntimeError):
    """The query exceeded the allowed time."""


class DatabaseNotFound(FileNotFoundError):
    """`eicu.db` is missing — run `python scripts/build_database.py`."""


def _install_time_limit(cx: sqlite3.Connection, timeout_s: float) -> None:
    """Abort the query past the deadline.

    SQLite calls this handler every N virtual-machine instructions; returning a
    non-zero value aborts the running statement.
    """
    deadline = time.monotonic() + timeout_s

    def _check() -> int:
        return 1 if time.monotonic() > deadline else 0

    cx.set_progress_handler(_check, 10_000)


@contextmanager
def connect(path: Path | None = None, timeout_s: float | None = None) -> Iterator[sqlite3.Connection]:
    """Open the database read-only. Always use inside a `with` block."""
    cfg = settings()
    path = Path(path or cfg.db_path)
    if not path.exists():
        raise DatabaseNotFound(f"{path} is missing. Run: python scripts/build_database.py")

    uri = f"file:{path.as_posix()}?mode=ro" if cfg.db_readonly else f"file:{path.as_posix()}"
    cx = sqlite3.connect(uri, uri=True, check_same_thread=False)
    cx.row_factory = sqlite3.Row
    try:
        _install_time_limit(cx, timeout_s if timeout_s is not None else cfg.sql_timeout_s)
        yield cx
    finally:
        cx.set_progress_handler(None, 0)
        cx.close()


def execute(
    sql: str,
    params: dict[str, Any] | Sequence[Any] | None = None,
    max_rows: int | None = None,
) -> tuple[list[str], list[tuple]]:
    """Run a SELECT and return (columns, rows).

    Values travel through `params`, never through concatenation. That is what
    makes SQL injection impossible and keeps real values out of any string sent
    to the cloud.
    """
    cfg = settings()
    cap = max_rows if max_rows is not None else cfg.sql_max_rows

    with connect() as cx:
        try:
            cur = cx.execute(sql, params or {})
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e).lower():
                raise QueryTimeout(f"Query aborted after {cfg.sql_timeout_s} s.") from e
            raise
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(cap)
    return columns, [tuple(r) for r in rows]
