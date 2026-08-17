"""The full pipeline, from question to written answer.

The three stages and their boundary
-----------------------------------
    STAGE 1  understand   LOCAL   question -> entities -> real values
    STAGE 2  anonymize    LOCAL   values -> :v1, :v2
             gate         LOCAL   check every outgoing byte
             generate     CLOUD   schema + masked question -> SQL
    STAGE 3  validate     LOCAL   one safe SELECT
             execute      LOCAL   read-only SQLite, bound parameters
             write        LOCAL   local model -> answer sentence

The only network call sits in stage 2, and it goes through `providers/cloud.py`,
which enforces the egress gate.

Why values are never concatenated
---------------------------------
The SQL returned by the cloud contains `:v1`. We hand it to SQLite **together
with** the parameter dictionary. The real value and the query text never meet in
a single string: not on the way out (the cloud does not have it), nor on the way
back (SQLite binds the parameter internally). SQL injection is impossible by
construction, not by filtering.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hybridsql.db.connection import QueryTimeout, execute
from hybridsql.pipeline import generate as gen
from hybridsql.pipeline.anonymize import Anonymization, UnmaskableQuestion, anonymize
from hybridsql.pipeline.understand import Understanding, understand
from hybridsql.security.egress_gate import LeakBlocked

_log = logging.getLogger(__name__)


@dataclass
class Answer:
    """The complete result. `rows` holds real data: do not serialise outward."""

    question: str
    success: bool
    text: str = ""
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    failed_stage: str = ""
    failure_reason: str = ""

    # Traceability
    ms_total: float = 0.0
    ms_understand: float = 0.0
    ms_generate: float = 0.0
    ms_execute: float = 0.0
    ms_write: float = 0.0
    cloud_target: str = ""
    cloud_tokens: int = 0
    cloud_calls: int = 0
    repairs: int = 0
    symbol_count: int = 0
    row_count: int = 0

    def summary(self) -> dict[str, Any]:
        """View without real data, for logs and measurements."""
        return {
            "success": self.success,
            "failed_stage": self.failed_stage,
            "failure_reason": self.failure_reason,
            "sql": self.sql,
            "symbol_count": self.symbol_count,
            "row_count": self.row_count,
            "cloud_target": self.cloud_target,
            "cloud_tokens": self.cloud_tokens,
            "cloud_calls": self.cloud_calls,
            "repairs": self.repairs,
            "ms": {
                "total": self.ms_total, "understand": self.ms_understand,
                "generate": self.ms_generate, "execute": self.ms_execute,
                "write": self.ms_write,
            },
        }


def _elapsed(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def answer(question: str, write_answer: bool = True) -> Answer:
    """Run the full pipeline. Never raises: returns a failed `Answer`."""
    start = time.perf_counter()
    a = Answer(question=question, success=False)

    # --- STAGE 1: understand (local) ----------------------------------------
    t = time.perf_counter()
    try:
        u: Understanding = understand(question)
    except Exception as e:  # noqa: BLE001
        a.failed_stage, a.failure_reason = "understand", f"{type(e).__name__}: {e}"
        a.ms_total = _elapsed(start)
        return a
    a.ms_understand = _elapsed(t)

    # --- STAGE 2a: anonymize (local) ----------------------------------------
    try:
        anon: Anonymization = anonymize(u)
    except UnmaskableQuestion as e:
        # Deliberate refusal: the database is de-identified, the question is moot.
        a.failed_stage, a.failure_reason = "anonymize", str(e)
        a.ms_total = _elapsed(start)
        return a
    a.symbol_count = anon.symbol_count

    # --- STAGE 2b: generate (cloud, behind the gate) ------------------------
    t = time.perf_counter()
    try:
        g = gen.generate(u, anon)
    except LeakBlocked as e:
        # The gate did its job. That is a security success, not a bug.
        a.failed_stage, a.failure_reason = "egress_gate", str(e)
        a.ms_total = _elapsed(start)
        return a
    except Exception as e:  # noqa: BLE001
        a.failed_stage, a.failure_reason = "cloud", f"{type(e).__name__}: {e}"
        a.ms_total = _elapsed(start)
        return a

    a.ms_generate = _elapsed(t)
    a.sql, a.cloud_target = g.sql, g.target
    a.cloud_tokens, a.cloud_calls, a.repairs = g.tokens, g.calls, g.repairs

    if not g.valid:
        a.failed_stage, a.failure_reason = "sql_validation", g.failure_reason
        a.ms_total = _elapsed(start)
        return a

    # --- STAGE 3a: execute (local, read-only, bound parameters) -------------
    t = time.perf_counter()
    try:
        a.columns, a.rows = execute(g.sql, anon.parameters())
    except QueryTimeout as e:
        a.failed_stage, a.failure_reason = "execution", str(e)
        a.ms_total = _elapsed(start)
        return a
    except Exception as e:  # noqa: BLE001
        a.failed_stage, a.failure_reason = "execution", f"{type(e).__name__}: {e}"
        a.ms_total = _elapsed(start)
        return a
    a.ms_execute = _elapsed(t)
    a.row_count = len(a.rows)

    # --- STAGE 3b: write (local) --------------------------------------------
    if write_answer:
        t = time.perf_counter()
        try:
            from hybridsql.providers import local_model

            written = local_model.write_answer(question, a.columns, a.rows)
            a.text = written.text
        except Exception as e:  # noqa: BLE001
            # The writer is a convenience, not a success condition: the data is
            # already there. We degrade to the raw table.
            _log.warning("answer writing unavailable: %s", e)
            a.text = _plain_text(a.columns, a.rows)
        a.ms_write = _elapsed(t)
    else:
        a.text = _plain_text(a.columns, a.rows)

    a.success = True
    a.ms_total = _elapsed(start)
    return a


def _plain_text(columns: list[str], rows: list[tuple], max_rows: int = 5) -> str:
    """Render without a model: enough to check the pipeline without loading 1 GB."""
    if not rows:
        return "No matching record."
    if len(rows) == 1 and len(rows[0]) == 1:
        return f"{columns[0]} = {rows[0][0]}"
    header = " | ".join(columns)
    body = "\n".join(
        " | ".join("" if v is None else str(v) for v in row) for row in rows[:max_rows]
    )
    more = f"\n… {len(rows)} rows total" if len(rows) > max_rows else ""
    return f"{header}\n{body}{more}"
