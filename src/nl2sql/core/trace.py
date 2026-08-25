"""Tracing engine — LangSmith upstream, a JSON journal on local disk."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from nl2sql.config import settings

Zone = Literal["local", "cloud"]

_lock = threading.Lock()
_configured = False
_key_check: dict[str, Any] = {"done": False, "ok": False, "detail": ""}

try:  # the pipeline must run without the SDK: a demo that dies because a
    from langsmith import traceable as _traceable  # dashboard library is missing
    from langsmith.run_helpers import trace as _ls_span  # is a bad demo

    AVAILABLE = True
except ImportError:  # pragma: no cover
    _traceable = _ls_span = None
    AVAILABLE = False


# ---------------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------------
def _key_accepted(cfg: Any) -> bool:
    """One cheap call to find out whether the key works. Cached for the process."""
    if _key_check["done"]:
        return bool(_key_check["ok"])
    _key_check["done"] = True
    try:
        import httpx

        r = httpx.get(
            f"{cfg.langsmith_endpoint.rstrip('/')}/api/v1/sessions",
            params={"limit": 1},
            headers={"x-api-key": cfg.langsmith_api_key},
            timeout=8.0,
        )
        _key_check.update(ok=r.status_code < 400, detail=f"HTTP {r.status_code}")
    except Exception as e:  # noqa: BLE001 — no dashboard is never a reason to stop
        _key_check.update(ok=False, detail=f"{type(e).__name__}: {e}")
    return bool(_key_check["ok"])


def _why_off(cfg: Any) -> str:
    if not AVAILABLE:
        return "the langsmith package is not installed"
    if not cfg.langsmith_tracing:
        return "LANGSMITH_TRACING is off"
    if not cfg.langsmith_api_key:
        return "LANGSMITH_API_KEY is empty"
    if _key_check["done"] and not _key_check["ok"]:
        return f"LangSmith rejected the key ({_key_check['detail']}) — regenerate it"
    return "unknown"


def configure() -> dict[str, Any]:
    """Wire LangSmith from settings. Idempotent, safe at import time."""
    global _configured
    cfg = settings()
    enabled = bool(cfg.langsmith_tracing and cfg.langsmith_api_key and AVAILABLE)
    if enabled and not _key_accepted(cfg):
        enabled = False

    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enabled else "false"
    if enabled:
        os.environ["LANGSMITH_API_KEY"] = cfg.langsmith_api_key
        os.environ["LANGSMITH_ENDPOINT"] = cfg.langsmith_endpoint
        os.environ["LANGSMITH_PROJECT"] = cfg.langsmith_project

    cfg.trace_dir.mkdir(parents=True, exist_ok=True)
    _configured = True
    return {
        "langsmith": enabled,
        "project": cfg.langsmith_project if enabled else None,
        "local_sink": str(sink()),
        "reason": "" if enabled else _why_off(cfg),
    }


def sink() -> Path:
    return settings().trace_dir / "runs.jsonl"


def enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING") == "true"


# ---------------------------------------------------------------------------------
#  The journal — the same steps, kept locally and handed to the interface
# ---------------------------------------------------------------------------------
@dataclass
class Step:
    """One traced step, as the interface receives it."""

    id: str
    label: str
    zone: Zone
    ms: float = 0.0
    summary: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "zone": self.zone,
            "ms": round(self.ms, 1),
            "summary": self.summary,
            "detail": self.detail,
        }


@dataclass
class Run:
    """One question, end to end, as recorded on local disk."""

    question: str
    arm: str = "hybrid"
    variant: str = "baseline"
    started_at: str = ""
    ms_total: float = 0.0
    steps: list[Step] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "arm": self.arm,
            "variant": self.variant,
            "question": self.question,
            "ms_total": self.ms_total,
            "steps": [s.as_dict() for s in self.steps],
            "result": self.result,
        }


_current: ContextVar[Run | None] = ContextVar("nl2sql_run", default=None)


def current() -> Run | None:
    return _current.get()


def steps_so_far() -> list[dict[str, Any]]:
    """Every step of the run in progress. What the streaming endpoint sends out."""
    run = _current.get()
    return [s.as_dict() for s in run.steps] if run else []


@contextmanager
def record(question: str, arm: str = "hybrid", variant: str = "baseline") -> Iterator[Run]:
    """Open a run; append it to the local sink when the block exits, whatever happened."""
    if not _configured:
        configure()
    run = Run(question, arm, variant, datetime.now(UTC).isoformat())
    token = _current.set(run)
    start = time.perf_counter()
    try:
        yield run
    finally:
        run.ms_total = round((time.perf_counter() - start) * 1000, 1)
        _current.reset(token)
        try:
            with _lock, sink().open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(run.as_dict(), ensure_ascii=False, default=str) + "\n")
        except OSError:  # a full disk must not fail an answered question
            pass


# ---------------------------------------------------------------------------------
#  Spans
# ---------------------------------------------------------------------------------
class Recorder:
    """Handed to the body of a `span()` so it can describe what it did."""

    def __init__(self, step: Step) -> None:
        self.step = step
        self.outputs: dict[str, Any] = {}

    def say(self, summary: str, **detail: Any) -> None:
        """One plain sentence for the interface, plus whatever the trace should keep."""
        self.step.summary = summary
        self.step.detail.update(detail)
        self.outputs.update(detail)


@contextmanager
def span(
    step_id: str,
    label: str,
    zone: Zone,
    kind: str = "chain",
    **inputs: Any,
) -> Iterator[Recorder]:
    """Open one child span in LangSmith and one entry in the journal, together."""
    step = Step(id=step_id, label=label, zone=zone)
    run = _current.get()
    if run is not None:
        run.steps.append(step)

    recorder = Recorder(step)
    start = time.perf_counter()

    if not AVAILABLE or not enabled():
        try:
            yield recorder
        except Exception as e:  # noqa: BLE001
            step.detail["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            step.ms = (time.perf_counter() - start) * 1000
        return

    with _ls_span(
        name=label,
        run_type=kind,
        inputs={k: v for k, v in inputs.items() if v is not None},
        metadata={"zone": zone, "step": step_id},
    ) as tracer:
        try:
            yield recorder
        except Exception as e:  # noqa: BLE001
            step.detail["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            step.ms = (time.perf_counter() - start) * 1000
            tracer.end(outputs={"summary": step.summary, **recorder.outputs})


def node(name: str, zone: Zone) -> Callable:
    """Decorate a graph node. `zone` has no default: a new stage must state it."""

    def decorate(fn: Callable) -> Callable:
        if not AVAILABLE:
            return fn
        return _traceable(name=name, run_type="chain", metadata={"zone": zone})(fn)

    return decorate


def read_runs(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the local sink back. Unreadable lines are skipped, never fatal."""
    target = path or sink()
    if not target.exists():
        return []
    out = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return out
