"""The only way the database is opened: read-only, time-bounded, row-capped."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from nl2sql.config import settings


class QueryTimeout(RuntimeError):
    """The query ran past its deadline and was aborted."""


class DatabaseNotFound(FileNotFoundError):
    """`data/eicu.db` is missing — run `python -m nl2sql.cli database`."""


@contextmanager
def connect(path: Path | None = None, timeout_s: float | None = None) -> Iterator[sqlite3.Connection]:
    """Open the database read-only. SQLite itself refuses any write."""
    cfg = settings()
    path = Path(path or cfg.db_path)
    if not path.exists():
        raise DatabaseNotFound(f"{path} is missing. Run: python -m nl2sql.cli database")

    cx = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    cx.row_factory = sqlite3.Row
    deadline = time.monotonic() + (cfg.sql_timeout_s if timeout_s is None else timeout_s)
    cx.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        yield cx
    finally:
        cx.set_progress_handler(None, 0)
        cx.close()


def execute(
    sql: str,
    params: dict[str, Any] | Sequence[Any] | None = None,
    max_rows: int | None = None,
) -> tuple[list[str], list[tuple]]:
    """Run one SELECT. Values travel as bound parameters, never as text."""
    cfg = settings()
    with connect() as cx:
        try:
            cur = cx.execute(sql, params or {})
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(cfg.sql_max_rows if max_rows is None else max_rows)
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e).lower():
                raise QueryTimeout(f"Query aborted after {cfg.sql_timeout_s}s.") from e
            raise
    return columns, [tuple(r) for r in rows]
