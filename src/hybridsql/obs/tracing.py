"""Tracing, with each stage declaring which side of the boundary it ran on.

LangSmith is a hosted dashboard: every span it records — prompt, output, latency,
token count — is uploaded to a third party. Rejecting it outright was the earlier
position, and it was the wrong one: the comparison this project has to deliver,
Full Cloud vs Hybrid vs Hybrid Opaque vs Full Local with per-stage latency and
token cost, is exactly what a tracing backend is good at.

**Everything is traced, values included.** That is defensible here and only here:
eICU-CRD is public and de-identified, and no proprietary data exists anywhere in
this project. On real data this file is where the boundary would have to be
reimposed — every stage already declares its `zone`, so the hook exists — but a
half-used redaction path is worse than none, so there is not one.

The zone is declared at the call site, once per node, and cannot be forgotten
silently: `node()` requires the argument. It is what a reader of a trace uses to
tell, at a glance, which spans crossed the line.

The local sink is not optional
------------------------------
Every run is also appended to `traces/runs.jsonl` on the local disk, in full.
That file never leaves the machine, and it is what you read when a demo goes
wrong and the dashboard is not reachable.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from hybridsql.config import settings

Zone = Literal["local", "cloud"]

_local_lock = threading.Lock()
_configured = False


# --------------------------------------------------------------------------------
#  Optional dependency
# --------------------------------------------------------------------------------
# The notebook installs langsmith, but the pipeline must run without it: a demo
# that dies because a dashboard SDK is missing is a bad demo. When the import
# fails, `node()` degrades to a decorator that only feeds the local sink.
try:  # pragma: no cover - exercised by the absence of the package
    from langsmith import traceable as _ls_traceable

    LANGSMITH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ls_traceable = None
    LANGSMITH_AVAILABLE = False


def configure() -> dict[str, Any]:
    """Wire LangSmith from settings. Idempotent, and safe to call at import time.

    LangSmith reads its configuration from environment variables at the moment a
    traced function first runs, not when the decorator is applied. Setting them
    here — rather than expecting the user to export them — is what makes the
    notebook reproducible from a clean kernel.
    """
    global _configured
    s = settings()

    enabled = bool(s.langsmith_tracing and s.langsmith_api_key and LANGSMITH_AVAILABLE)
    if enabled and not _key_accepted(s):
        # A rejected key does not fail quietly. The SDK retries in a background
        # thread and prints a stack trace per span, which in a notebook buries the
        # actual output of the demo under ingest errors. Better to find out once,
        # here, and run without the dashboard.
        enabled = False
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    # The v1 name is still what parts of the SDK look at. Set both, stay boring.
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enabled else "false"
    if enabled:
        os.environ["LANGSMITH_API_KEY"] = s.langsmith_api_key
        os.environ["LANGSMITH_ENDPOINT"] = s.langsmith_endpoint
        os.environ["LANGSMITH_PROJECT"] = s.langsmith_project

    s.trace_dir.mkdir(parents=True, exist_ok=True)
    _configured = True

    return {
        "langsmith": enabled,
        "project": s.langsmith_project if enabled else None,
        "local_sink": str(local_sink()),
        "reason": _why_disabled(s) if not enabled else "",
    }


_key_check: dict[str, Any] = {"done": False, "ok": False, "detail": ""}


def _key_accepted(s: Any) -> bool:
    """One cheap call to find out whether the key works. Cached for the process."""
    if _key_check["done"]:
        return bool(_key_check["ok"])
    _key_check["done"] = True
    try:
        import httpx

        r = httpx.get(
            f"{s.langsmith_endpoint.rstrip('/')}/api/v1/sessions",
            params={"limit": 1},
            headers={"x-api-key": s.langsmith_api_key},
            timeout=8.0,
        )
        _key_check["ok"] = r.status_code < 400
        _key_check["detail"] = f"HTTP {r.status_code}"
    except Exception as e:  # noqa: BLE001 — no dashboard is never a reason to stop
        _key_check["ok"] = False
        _key_check["detail"] = f"{type(e).__name__}: {e}"
    return bool(_key_check["ok"])


def _why_disabled(s: Any) -> str:
    if not LANGSMITH_AVAILABLE:
        return "the langsmith package is not installed"
    if not s.langsmith_tracing:
        return "LANGSMITH_TRACING is off"
    if not s.langsmith_api_key:
        return "LANGSMITH_API_KEY is empty"
    if _key_check["done"] and not _key_check["ok"]:
        return f"LangSmith rejected the key ({_key_check['detail']}) — regenerate it"
    return "unknown"


def local_sink() -> Path:
    return settings().trace_dir / "runs.jsonl"


# --------------------------------------------------------------------------------
#  The decorator
# --------------------------------------------------------------------------------
def node(name: str, zone: Zone) -> Callable:
    """Mark a pipeline stage as traceable, declaring which side of the line it sits on.

    `zone` has no default on purpose. A new stage must state whether it handles
    real data, and the answer belongs next to the code, not in a config file
    somebody forgets to update.
    """

    def decorate(fn: Callable) -> Callable:
        if not LANGSMITH_AVAILABLE:
            return fn

        return _ls_traceable(
            name=name,
            run_type="chain" if zone == "local" else "llm",
            metadata={"zone": zone},
        )(fn)

    return decorate


# --------------------------------------------------------------------------------
#  The local sink
# --------------------------------------------------------------------------------
@dataclass
class Run:
    """One question, end to end, as recorded on local disk."""

    question: str
    arm: str = "hybrid"
    started_at: str = ""
    ms_total: float = 0.0
    stages: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    def stage(self, name: str, zone: Zone, ms: float, **detail: Any) -> None:
        self.stages.append({"stage": name, "zone": zone, "ms": round(ms, 1), **detail})


@contextmanager
def record(question: str, arm: str = "hybrid"):
    """Open a run, append it to `traces/runs.jsonl` when the block exits.

    Full fidelity, always. This file never leaves the machine, so no policy
    has no business trimming it — and when a demo misbehaves, this is the only
    place the actual values are still visible.
    """
    if not _configured:
        configure()
    run = Run(question=question, arm=arm, started_at=datetime.now(timezone.utc).isoformat())
    start = time.perf_counter()
    try:
        yield run
    finally:
        run.ms_total = round((time.perf_counter() - start) * 1000, 1)
        line = json.dumps(
            {
                "started_at": run.started_at,
                "arm": run.arm,
                "question": run.question,
                "ms_total": run.ms_total,
                "stages": run.stages,
                "result": run.result,
            },
            ensure_ascii=False,
            default=str,
        )
        with _local_lock:
            with local_sink().open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def read_runs(path: Path | None = None) -> list[dict[str, Any]]:
    """Read back the local sink. Used by the notebook to build its tables."""
    target = path or local_sink()
    if not target.exists():
        return []
    out = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return out
