"""Entity extraction from the question — GLiNER2, local and in-memory.

Role in the architecture
------------------------
This is the **first brick of the trust boundary**. It spots, inside the question,
what is a business value ("aspirin", "sepsis") as opposed to what is structure
("how many", "per hospital"). Without it we would not know what to mask before
calling the cloud.

Why GLiNER2 and not an LLM
--------------------------
An LLM would do the job, but it would have to receive the question — hence the
business values — which is exactly what we are avoiding. GLiNER2 is a 208 M
parameter encoder that runs on CPU in a few hundred milliseconds, **with no
network call at inference**. The only network access happens at first load, to
download the weights.

It is also *zero-shot*: entity types are given in natural language at call time,
with no retraining. That is what makes the pipeline portable to another domain —
change `ENTITY_TYPES`, nothing else.

Fallback
--------
If the model cannot be loaded (no network on first start, full disk), we do not
crash: we fall back to glossary-based extraction. Worse, but the pipeline stays
demonstrable. `available()` tells which of the two is active, and the evaluation
metric records it.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hybridsql.config import settings

_log = logging.getLogger(__name__)

# Types are described in natural language: GLiNER2 is zero-shot. They mirror the
# eICU domain. Changing this list is enough to change domain.
ENTITY_TYPES: tuple[str, ...] = (
    # Person names come first because they are the gravest case. eICU is
    # de-identified: no name exists in it, so none will ever resolve. But an
    # analyst will write "did Mr. Bensalah get his insulin?" and that name must
    # not reach the cloud provider under any circumstance. Without this type,
    # GLiNER2 does not spot it, and nothing masks it.
    "person name",
    "drug or medication name",
    "medical condition or diagnosis",
    "laboratory test name",
    "microorganism or bacteria",
    "medical procedure or treatment",
    "hospital or unit name",
    "patient demographic value",
    # Added after measurement: without it, "Midwest" and "South" were not spotted
    # at all, although they are real values of `hospital.region`. What is not
    # spotted cannot be masked.
    "geographic region",
    # A "test result or susceptibility status" type was tried to catch "Resistant"
    # and "Sensitive" (values of `microlab.sensitivitylevel`). It did not find
    # them, and its presence lost "potassium" and "glucose": each extra type
    # dilutes the model's attention and costs latency. Removed — the trade was
    # not favourable.
)

# GLiNER2 returns verbose labels; we shorten them to the tags that appear in
# traces and in the report.
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
    """A mention spotted in the question, before any resolution."""

    text: str
    type: str
    score: float
    start: int = -1
    end: int = -1
    source: str = "gliner2"

    def __str__(self) -> str:
        return f"{self.text!r} ({self.type}, {self.score:.2f})"


# ---------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------
def _resolve_source(name: str) -> str:
    """Return an absolute path if `name` designates a folder of the repo, else `name`.

    `GLINER_MODEL` accepts both forms: a Hugging Face repo id
    (`fastino/gliner2-base-v1`) or a local path (`models/gliner2-base-v1`). The
    path is preferred because it loads the model **with no network access at
    all**, including on first start — which is the property we want to be able to
    claim in the report.

    Resolution is done from the repo root rather than the current directory:
    otherwise the model only loads when the command is run from the right folder.
    """
    from hybridsql.config import ROOT

    candidate = Path(name)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return str(candidate) if candidate.is_dir() else name


def _make_console_unicode_safe() -> None:
    """Stop a Windows console from taking the entity extractor down with it.

    Measured, not theoretical: on Windows the default stdout encoding is cp1252,
    and `gliner2` prints a brain emoji on load. The resulting `UnicodeEncodeError`
    propagates out of `from_pretrained`, both model sources are marked
    unavailable, and stage 1 silently degrades to the glossary — which then fails
    to spot `aspirin` as a value, so the egress gate blocks a question that should
    have gone through. A console encoding turned into what looked like a security
    refusal.

    Linux and Kaggle default to UTF-8 and never hit this. The three lines stay
    because the project has to run on the machine it is developed on, and because
    the failure it prevents is impossible to diagnose from its symptom.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # already detached, or not a real console
            continue


def load(force: bool = False) -> Any | None:
    """Load the model once for the whole process.

    The lock is not decorative: under uvicorn, two concurrent requests at startup
    would trigger two simultaneous downloads of the same weights.
    """
    global _model
    if _model is not None and not force:
        return _model

    with _lock:
        if _model is not None and not force:
            return _model

        cfg = settings()
        _make_console_unicode_safe()
        from gliner2 import GLiNER2

        for name in (_resolve_source(cfg.gliner_model), cfg.gliner_fallback_model):
            if not name:
                continue
            start = time.perf_counter()
            try:
                _model = GLiNER2.from_pretrained(name)
                ms = (time.perf_counter() - start) * 1000
                _state.update(loaded=True, model=name, load_ms=round(ms), error=None)
                _log.info("GLiNER2 loaded: %s (%.0f ms)", name, ms)
                return _model
            except Exception as e:  # noqa: BLE001 — we really want to catch everything
                _log.warning("GLiNER2 unavailable (%s): %s", name, e)
                _state.update(loaded=False, model=None, error=f"{type(e).__name__}: {e}")
    return None


def available() -> bool:
    return _state["loaded"]


def state() -> dict[str, Any]:
    """Returned by `/health`: know which extractor is actually running."""
    return dict(_state)


def warm_up() -> dict[str, Any]:
    """Call at API startup.

    Otherwise the first user pays for loading the weights — tens of seconds on a
    freshly woken host.
    """
    load()
    if available():
        extract("warm-up query about aspirin")
    return state()


# ---------------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------------
def _glossary_fallback(question: str) -> list[Entity]:
    """Degraded extraction: the glossary terms present in the question.

    Finds only what is already written in `config/glossary.yaml`, so never an
    unknown drug name. A safety net, not a solution.
    """
    from hybridsql.resources.glossary import recognize

    return [
        Entity(text=m.trigger, type=m.term.canonical, score=0.5, source="glossary")
        for m in recognize(question)
    ]


def _unwrap(raw: Any) -> dict[str, Any]:
    """Pull the {type: values} table out of GLiNER2's response.

    The response is wrapped in an `entities` key, and its shape depends on the
    call options: a dict with `format_results=True`, a single-element list
    otherwise. Not unwrapping gave a misleading result — we iterated the envelope
    and mistook the *type names* for entities.
    """
    if not isinstance(raw, dict):
        return {}
    content = raw.get("entities", raw)
    if isinstance(content, list):
        content = content[0] if content else {}
    return content if isinstance(content, dict) else {}


def _flatten(raw: dict[str, Any], question: str) -> list[Entity]:
    """Normalise GLiNER2's output.

    Values are sometimes strings, sometimes dicts, depending on
    `include_confidence` / `include_spans`. We accept both rather than depend on a
    precise library version.
    """
    entities: list[Entity] = []
    for raw_type, values in _unwrap(raw).items():
        label = LABELS.get(raw_type, raw_type)
        for v in values or ():
            if isinstance(v, dict):
                text = str(v.get("text") or v.get("value") or "").strip()
                score = float(v.get("confidence") or v.get("score") or 0.0)
                start = int(v.get("start", -1))
                end = int(v.get("end", -1))
            else:
                text, score, start, end = str(v).strip(), 0.0, -1, -1
            if not text:
                continue
            if start < 0:
                start = question.lower().find(text.lower())
                end = start + len(text) if start >= 0 else -1
            entities.append(Entity(text, label, round(score, 4), start, end))
    return entities


def _overlap(a: Entity, b: Entity) -> bool:
    """Do two entities occupy a common span of the question?"""
    if a.start < 0 or b.start < 0:
        return False
    return a.start < b.end and b.start < a.end


def _deduplicate(entities: list[Entity]) -> list[Entity]:
    """One span per stretch of text, the most informative one.

    Two distinct problems are handled here, both of which would corrupt masking:

    1. **Exact duplicates** — GLiNER2 offers "sepsis" both as a diagnosis and as a
       condition. Two identical entities would produce `:v1` and `:v2` for the
       same value, hence inconsistent SQL.

    2. **Nested spans** — measured on the evaluation set, the model returns
       "Staphylococcus aureus" *and* "Staphylococcus aureus culture", or
       "creatinine" *and* "creatinine results". Replacing both would substitute
       inside already-substituted text: the masked question becomes unreadable and
       the offsets no longer line up.

    We keep the longest span at comparable score — it carries more context, so it
    resolves better — and the best scored one at equal length.
    """
    kept: list[Entity] = []
    for e in sorted(entities, key=lambda x: (-(x.end - x.start), -x.score)):
        key = e.text.lower()
        if any(k.text.lower() == key or _overlap(k, e) for k in kept):
            continue
        kept.append(e)
    return sorted(kept, key=lambda e: e.start if e.start >= 0 else 9999)


def extract(
    question: str,
    types: tuple[str, ...] = ENTITY_TYPES,
    threshold: float | None = None,
) -> list[Entity]:
    """Spot the business mentions in the question. No network call.

    Returns an empty list when the question holds no value — a normal case
    ("how many hospitals are there?"), not an error.
    """
    question = (question or "").strip()
    if not question:
        return []

    model = load()
    if model is None:
        return _glossary_fallback(question)

    threshold = threshold if threshold is not None else settings().gliner_threshold
    try:
        raw = model.extract_entities(
            question,
            list(types),
            threshold=threshold,
            include_confidence=True,
            include_spans=True,
        )
    except Exception as e:  # noqa: BLE001
        _log.warning("GLiNER2 extraction failed, falling back to glossary: %s", e)
        return _glossary_fallback(question)

    return _deduplicate(_flatten(raw, question))
