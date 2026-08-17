"""Egress gate journal — the record of what left, and what was blocked.

What this file is for
---------------------
A guarantee that cannot be verified after the fact is not a guarantee. The report
will claim that no business value crossed the boundary; this journal is the
evidence that lets anyone check it line by line, without re-reading the code.

It also provides the **leak rate**, the project's headline metric: number of
blocked sends over number of attempted sends.

What is written, and what is not
--------------------------------
The journal stays **inside the trust boundary**: it is a local file, never
transmitted. It therefore records the text actually sent — which is already
masked — and the refused tokens, without which a block could not be diagnosed.

It never records the symbol-to-value mapping. That is the system's only secret:
writing it to disk next to the queries would undo the anonymization for anyone
reading both files.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from hybridsql.config import settings

if TYPE_CHECKING:
    from hybridsql.security.egress_gate import Verdict

_lock = threading.Lock()


def _path() -> Path:
    path = Path(settings().egress_audit_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def record(verdict: Verdict, text: str) -> None:
    """Append a line to the journal. Never raises.

    A journal write failure must not fail an otherwise legitimate request — but it
    must not go unnoticed either, hence the trace on stderr.
    """
    line = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "context": verdict.context,
        "allowed": verdict.allowed,
        "token_count": verdict.token_count,
        "fingerprint": verdict.fingerprint,
        "refused_tokens": list(verdict.refused_tokens),
        "length": len(text or ""),
    }
    try:
        with _lock, _path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError as e:  # noqa: BLE001
        import sys

        print(f"[audit] cannot write: {e}", file=sys.stderr)


def record_bypass(context: str, characters: int) -> None:
    """Record a send that deliberately skipped the gate.

    Only the Full Cloud arm of the benchmark does this, and only because leaving
    unmasked is that arm's definition. It is journalled with `allowed: true` and
    `bypassed: true` so the report can state how many characters left unprotected
    instead of quietly excluding them from the totals — an omitted baseline is
    worse than a bad one.
    """
    line = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "context": context,
        "allowed": True,
        "bypassed": True,
        "token_count": 0,
        "fingerprint": "",
        "refused_tokens": [],
        "length": characters,
    }
    try:
        with _lock, _path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError as e:  # noqa: BLE001
        import sys

        print(f"[audit] cannot write: {e}", file=sys.stderr)


def read(source: Path | None = None) -> list[dict]:
    """Re-read the journal. Unreadable lines are skipped, not fatal."""
    path = Path(source or settings().egress_audit_log)
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            lines.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return lines


def leak_rate(source: Path | None = None) -> dict[str, float | int]:
    """The project's headline metric.

    A leak would be an *allowed* send that contained a real value. It is zero by
    construction while the gate works: a text holding an unmasked value is not in
    the allowlist, so it is blocked. Measuring it anyway is the only way to
    demonstrate rather than assert — that is what `tests/test_canary.py` does.
    """
    lines = read(source)
    total = len(lines)
    blocked = sum(1 for line in lines if not line.get("allowed"))
    return {
        "sends": total,
        "blocked": blocked,
        "allowed": total - blocked,
        "block_rate": round(blocked / total, 4) if total else 0.0,
    }


def clear(source: Path | None = None) -> None:
    """Empty the journal. For tests and at the start of a measurement campaign."""
    path = Path(source or settings().egress_audit_log)
    if path.exists():
        path.unlink()
