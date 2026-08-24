"""Egress journal — the evidence behind the claim that nothing leaked.

A local file, never transmitted. It records the masked text that was sent and the
tokens that were refused; it never records the symbol-to-value mapping, which is
the system's only secret.
"""

from __future__ import annotations

import json
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from nl2sql.config import settings

if TYPE_CHECKING:
    from nl2sql.privacy.gate import Verdict

_lock = threading.Lock()


def _append(line: dict) -> None:
    """Never raises: a journal failure must not fail a legitimate request."""
    try:
        path = Path(settings().audit_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[audit] cannot write: {e}", file=sys.stderr)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def record(verdict: Verdict, text: str) -> None:
    _append({
        "timestamp": _now(),
        "context": verdict.context,
        "allowed": verdict.allowed,
        "token_count": verdict.token_count,
        "fingerprint": verdict.fingerprint,
        "refused_tokens": list(verdict.refused_tokens),
        "length": len(text or ""),
    })


def record_bypass(context: str, characters: int) -> None:
    """A send that deliberately skipped the gate — only the Full Cloud baseline.

    Journalled rather than excluded, so the report can state how many characters
    left unprotected instead of quietly leaving the arm out of the totals.
    """
    _append({
        "timestamp": _now(), "context": context, "allowed": True, "bypassed": True,
        "token_count": 0, "fingerprint": "", "refused_tokens": [], "length": characters,
    })


def read(source: Path | None = None) -> list[dict]:
    path = Path(source or settings().audit_log)
    if not path.exists():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return out


def report(source: Path | None = None) -> dict[str, float | int]:
    """Sends, blocks and characters — the headline numbers."""
    lines = read(source)
    blocked = sum(1 for line in lines if not line.get("allowed"))
    return {
        "sends": len(lines),
        "blocked": blocked,
        "allowed": len(lines) - blocked,
        "bypassed": sum(1 for line in lines if line.get("bypassed")),
        "block_rate": round(blocked / len(lines), 4) if lines else 0.0,
        "characters_sent": sum(line.get("length", 0) for line in lines if line.get("allowed")),
    }


def clear(source: Path | None = None) -> None:
    Path(source or settings().audit_log).unlink(missing_ok=True)
