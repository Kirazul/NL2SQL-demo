"""GLiNER2 — spot the business mentions in a question. Local, in-process.

The first brick of the trust boundary: it separates a value ("aspirin") from
structure ("how many"). An LLM would do it better but would have to *receive the
question*, which is the thing being avoided. Zero-shot, so changing domain means
changing `ENTITY_TYPES` and nothing else.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nl2sql.config import ROOT, settings

_log = logging.getLogger(__name__)

ENTITY_TYPES: tuple[str, ...] = (
    # First because it is the gravest case: eICU is de-identified, so a name can
    # never resolve, but an analyst will still type one and it must not leave.
    "person name",
    "drug or medication name",
    "medical condition or diagnosis",
    "laboratory test name",
    "microorganism or bacteria",
    "medical procedure or treatment",
    "hospital or unit name",
    "patient demographic value",
    "geographic region",
)

LABELS = {
    "person name": "person",
    "drug or medication name": "drug",
    "medical condition or diagnosis": "diagnosis",
    "laboratory test name": "lab_test",
    "microorganism or bacteria": "organism",
    "medical procedure or treatment": "procedure",
    "hospital or unit name": "facility",
    "patient demographic value": "demographic",
    "geographic region": "region",
}

_model: Any | None = None
_lock = threading.Lock()
_state: dict[str, Any] = {"loaded": False, "model": None, "load_ms": None, "error": None}


@dataclass(frozen=True)
class Entity:
    text: str
    type: str
    score: float
    start: int = -1
    end: int = -1
    source: str = "gliner2"

    def __str__(self) -> str:
        return f"{self.text!r} ({self.type}, {self.score:.2f})"


def _source(name: str) -> str:
    """A repo-relative folder loads with no network access at all; prefer it."""
    candidate = Path(name)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return str(candidate) if candidate.is_dir() else name


def _utf8_console() -> None:
    """gliner2 prints an emoji on load; a cp1252 console turns that into a crash."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # Already detached, or not a real console: nothing to fix either way.
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def load(force: bool = False) -> Any | None:
    """Load the weights once per process. The lock stops two concurrent downloads."""
    global _model
    if _model is not None and not force:
        return _model
    with _lock:
        if _model is not None and not force:
            return _model
        cfg = settings()
        _utf8_console()
        from gliner2 import GLiNER2

        for name in (_source(cfg.gliner_model), cfg.gliner_fallback):
            if not name:
                continue
            start = time.perf_counter()
            try:
                _model = GLiNER2.from_pretrained(name)
                _state.update(
                    loaded=True,
                    model=name,
                    load_ms=round((time.perf_counter() - start) * 1000),
                    error=None,
                )
                return _model
            except Exception as e:  # noqa: BLE001 — fall back rather than fail
                _log.warning("GLiNER2 unavailable (%s): %s", name, e)
                _state.update(loaded=False, model=None, error=f"{type(e).__name__}: {e}")
    return None


def available() -> bool:
    return bool(_state["loaded"])


def state() -> dict[str, Any]:
    return dict(_state)


def warm_up() -> dict[str, Any]:
    """Call at API startup, or the first user pays for loading the weights."""
    load()
    if available():
        extract("warm-up query about aspirin")
    return state()


def _fallback(question: str) -> list[Entity]:
    """Degraded extraction from the glossary. A safety net, never a solution."""
    from nl2sql.nlp.glossary import recognize

    return [
        Entity(m.trigger, m.term.canonical, 0.5, source="glossary") for m in recognize(question)
    ]


def _unwrap(raw: Any) -> dict[str, Any]:
    """Pull the {type: values} table out of the response envelope."""
    if not isinstance(raw, dict):
        return {}
    content = raw.get("entities", raw)
    if isinstance(content, list):
        content = content[0] if content else {}
    return content if isinstance(content, dict) else {}


def _flatten(raw: dict[str, Any], question: str) -> list[Entity]:
    """Values come back as strings or as dicts depending on the call options."""
    entities: list[Entity] = []
    for raw_type, values in _unwrap(raw).items():
        label = LABELS.get(raw_type, raw_type)
        for v in values or ():
            if isinstance(v, dict):
                text = str(v.get("text") or v.get("value") or "").strip()
                score = float(v.get("confidence") or v.get("score") or 0.0)
                start, end = int(v.get("start", -1)), int(v.get("end", -1))
            else:
                text, score, start, end = str(v).strip(), 0.0, -1, -1
            if not text:
                continue
            if start < 0:
                start = question.lower().find(text.lower())
                end = start + len(text) if start >= 0 else -1
            entities.append(Entity(text, label, round(score, 4), start, end))
    return entities


def _deduplicate(entities: list[Entity]) -> list[Entity]:
    """One span per stretch of text, the most informative one."""
    def overlap(a: Entity, b: Entity) -> bool:
        return a.start >= 0 and b.start >= 0 and a.start < b.end and b.start < a.end

    kept: list[Entity] = []
    for e in sorted(entities, key=lambda x: (-(x.end - x.start), -x.score)):
        if not any(k.text.lower() == e.text.lower() or overlap(k, e) for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: e.start if e.start >= 0 else 9999)


def extract(question: str, threshold: float | None = None) -> list[Entity]:
    """Business mentions in the question. No network call. Empty is a normal answer."""
    question = (question or "").strip()
    if not question:
        return []
    model = load()
    if model is None:
        return _fallback(question)

    try:
        raw = model.extract_entities(
            question,
            list(ENTITY_TYPES),
            threshold=settings().gliner_threshold if threshold is None else threshold,
            include_confidence=True,
            include_spans=True,
        )
    except Exception as e:  # noqa: BLE001
        _log.warning("GLiNER2 extraction failed: %s", e)
        return _fallback(question)
    return _deduplicate(_flatten(raw, question))
