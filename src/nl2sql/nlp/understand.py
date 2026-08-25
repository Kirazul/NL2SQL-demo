"""Stage 1 — Understand. Entirely local, no network call."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal

from nl2sql.core.steps import track
from nl2sql.db import catalog
from nl2sql.db.values import FoundValue, search
from nl2sql.nlp import extract as ner
from nl2sql.nlp import glossary

# Below this we ask rather than guess: a badly resolved value produces SQL that
# runs without error and answers a different question.
CONFIDENCE_THRESHOLD = 0.75

# A value found outside the columns its type predicts is a hint, not an answer.
# Without the penalty, "hemoglobin" landed in `diagnosis.diagnosisstring` at 1.00.
OUT_OF_SCOPE_PENALTY = 0.7

MAX_SPAN_RETRIES = 6

TYPE_TO_TERMS: dict[str, tuple[str, ...]] = {
    "drug": ("drug",),
    "diagnosis": ("diagnosis", "medical_history"),
    "lab_test": ("lab_test",),
    "organism": ("microbiology",),
    "procedure": ("procedure", "ventilation"),
    "facility": ("hospital", "care_unit", "teaching_status"),
    "demographic": ("sex", "ethnicity", "age"),
    "region": ("region",),
}

Kind = Literal["value", "concept", "quantity", "person"]

TITLE_RE = re.compile(r"^(mr|mrs|ms|miss|dr|prof|sir|m|mme|mlle)\b\.?", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w.])")


@dataclass(frozen=True)
class Resolution:
    """A mention from the question, tied to a real value of the database."""

    mention: str
    type: str
    value: str | None          # the exact stored value — NEVER LEAVES
    column: str | None         # `table.column` — may leave
    score: float
    tier: str = ""
    alternatives: tuple[str, ...] = ()
    kind: Kind = "value"
    out_of_scope: bool = False

    @property
    def resolved(self) -> bool:
        return self.value is not None

    @property
    def confident(self) -> bool:
        return self.resolved and self.score >= CONFIDENCE_THRESHOLD

    @property
    def to_mask(self) -> bool:
        """Values and numbers become symbols; concepts name columns and do not."""
        if self.kind == "quantity":
            return self.type == "number"
        return self.kind == "value" and self.confident


@dataclass
class Understanding:
    """Stage 1 result. Contains secrets: never serialise as-is."""

    question: str
    entities: list[ner.Entity] = field(default_factory=list)
    resolutions: list[Resolution] = field(default_factory=list)
    tables: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    ms: float = 0.0
    extractor: str = ""

    @property
    def needs_confirmation(self) -> list[Resolution]:
        return [r for r in self.resolutions if r.resolved and not r.confident]

    @property
    def unresolved(self) -> list[Resolution]:
        return [r for r in self.resolutions if r.kind == "value" and not r.resolved]

    @property
    def values(self) -> list[Resolution]:
        return [r for r in self.resolutions if r.to_mask]

    @property
    def persons(self) -> list[Resolution]:
        """The database is de-identified: a named person means the question has no answer."""
        return [r for r in self.resolutions if r.kind == "person"]

    @property
    def columns(self) -> list[str]:
        return sorted({r.column for r in self.resolutions if r.column})

    def for_the_cloud(self) -> dict[str, object]:
        """The only view allowed to cross the boundary: no value, no mention."""
        return {
            "tables": sorted(self.tables),
            "target_columns": self.columns,
            "notes": list(self.notes),
            "masked_value_count": len(self.values),
        }


def _looks_like_a_name(mention: str) -> bool:
    """Does the mention have the shape of a proper noun?"""
    from nl2sql.privacy.gate import generic_vocabulary

    text = mention.strip()
    if not text:
        return False
    if TITLE_RE.match(text):
        return True
    generic = generic_vocabulary()
    return any(
        w[:1].isupper() and w.lower() not in generic for w in re.findall(r"[A-Za-zÀ-ÿ']+", text)
    )


def classify(mention: str, entity_type: str = "") -> Kind:
    """Tell a value to look up from a concept or a quantity."""
    from nl2sql.db.values import is_exact_value
    from nl2sql.privacy.gate import generic_vocabulary

    if entity_type == "person":
        return "person" if _looks_like_a_name(mention) else "concept"

    generic = generic_vocabulary()
    words = glossary.WORD_RE.findall(glossary.normalize(mention))
    significant = [w for w in words if len(w) >= 3 and w.isalpha() and w not in generic]
    if not significant:
        return "quantity"

    # An exact stored value beats the glossary, but only for multi-word mentions:
    # "Other Hospital" and "Neuro ICU" are stored verbatim, while "icu" alone
    # names the concept. Applied to single words too, accuracy fell from 88% to 29%.
    if len(words) >= 2 and is_exact_value(mention):
        return "value"

    covered = {w for m in glossary.recognize(mention) for w in m.trigger.split()}
    if all(w in covered for w in significant):
        return "concept"
    # A column-shaped head word names a column whatever the glossary knows.
    if catalog.names_a_column(mention):
        return "concept"
    return "value"


def _concept_column(mention: str) -> str | None:
    """The column a concept designates, for prompt scoping."""
    columns = glossary.columns_for(mention)
    if columns:
        scores = {m.ref: m.score for m in catalog.link(mention, limit=len(catalog.cards()))}
        return max(columns, key=lambda c: (scores.get(c, 0.0), -columns.index(c)))
    match = catalog.best(mention)
    return match.ref if match else None


def _scope(entity: ner.Entity, default: list[str]) -> list[str]:
    """Columns to search this mention in, most likely first."""
    terms = glossary.load()
    columns: list[str] = []
    for name in TYPE_TO_TERMS.get(entity.type, ()):
        term = terms.get(name)
        if term:
            columns.extend(c for c in term.columns if c not in columns)
    columns.extend(c for c in default if c not in columns)
    return columns


def _shorter_spans(mention: str) -> list[str]:
    """Sub-spans, longest first, for when the whole span resolves to nothing."""
    from nl2sql.privacy.gate import generic_vocabulary

    words = mention.split()
    if len(words) < 2:
        return []
    generic = generic_vocabulary()
    spans: list[str] = []
    for size in range(len(words) - 1, 0, -1):
        for start in range(len(words) - size + 1):
            span = " ".join(words[start : start + size])
            if span in spans or all(w.lower().strip(".,;:()") in generic for w in span.split()):
                continue
            spans.append(span)
            if len(spans) >= MAX_SPAN_RETRIES:
                return spans
    return spans


def _resolve(entity: ner.Entity, scope: list[str], step: object) -> Resolution:
    """Turn a mention into a real value, widening the search only if needed."""
    kind = classify(entity.text, entity.type)
    if kind != "value":
        column = _concept_column(entity.text) if kind == "concept" else None
        step.say(
            {
                "concept": f"'{entity.text}' names a column, not a value"
                + (f" — {column}" if column else ""),
                "quantity": f"'{entity.text}' is a number the analyst chose",
                "person": f"'{entity.text}' looks like a person's name — the database has none",
            }[kind],
            kind=kind,
            column=column,
        )
        return Resolution(
            mention=entity.text,
            type=entity.type,
            value=None,
            column=column,
            score=0.0,
            kind=kind,
        )

    found: list[FoundValue] = []
    out_of_scope = False
    mention = entity.text
    if scope:
        found = search(mention, columns=scope, limit=4)
    if not found:
        found = search(mention, limit=4)
        out_of_scope = bool(scope)

    # A *weak* match triggers the retry, not only an empty one: asked for the whole
    # span the index does not fail cleanly, it returns the closest thing it has —
    # `hemoglobin lab test` -> `testes` at 0.56, which looks like an answer.
    if not found or found[0].score < CONFIDENCE_THRESHOLD:
        best_score = found[0].score if found else 0.0
        for shorter in _shorter_spans(entity.text):
            candidates = search(shorter, columns=scope, limit=4) if scope else []
            widened = False
            if not candidates:
                candidates = search(shorter, limit=4)
                widened = bool(scope)
            if candidates and candidates[0].score > best_score:
                found, mention, best_score = candidates, shorter, candidates[0].score
                out_of_scope = widened
                if best_score >= 0.99:
                    break

    if not found:
        step.say(f"nothing in the database looks like '{entity.text}'", kind="value", found=0)
        return Resolution(
            entity.text,
            entity.type,
            None,
            _concept_column(entity.text),
            0.0,
            kind="concept" if catalog.best(entity.text) else "value",
        )

    best, *others = found
    # An exact string match in an unexpected column means the type guess was off,
    # not that the value is wrong — penalising it refused perfect matches. "Exact"
    # means the value equals the mention, not that the scorer returned 1.00.
    exact = str(best.value).strip().lower() == mention.strip().lower()
    penalised = out_of_scope and not exact
    score = best.score * (OUT_OF_SCOPE_PENALTY if penalised else 1.0)

    # The arbitration: does this mention name a column better than it names a
    # value? Without it the index always wins, because a fuzzy search never comes
    # back empty. An exact stored value is never overruled.
    column = catalog.best(mention)
    if column and not exact and column.score >= score:
        with track("arbitrate", mention=mention) as arbitration:
            arbitration.say(
                f"'{mention}' matches the column {column.ref} ({column.score:.2f}) better than "
                f"it matches the value '{best.value}' ({score:.2f}) — treated as a column",
                column=column.ref,
                column_score=column.score,
                rejected_value=best.value,
                value_score=round(score, 4),
            )
        step.say(f"'{mention}' names the column {column.ref}", kind="concept")
        return Resolution(
            mention=mention,
            type=entity.type,
            value=None,
            column=column.ref,
            score=0.0,
            kind="concept",
            alternatives=(f"{best.ref} = {best.value} ({score:.2f}, not used)",),
        )

    step.say(
        f"'{mention}' is stored as '{best.value}' in {best.ref} ({score:.0%} sure)",
        kind="value",
        value=best.value,
        column=best.ref,
        score=round(score, 4),
        tier=best.tier,
        searched=len(scope) or "every column",
        also_found=[f"{o.ref} = {o.value}" for o in others[:3]],
    )
    return Resolution(
        mention=mention,
        type=entity.type,
        value=best.value,
        column=best.ref,
        score=round(score, 4),
        tier=best.tier,
        alternatives=tuple(f"{o.ref} = {o.value}" for o in others),
        out_of_scope=penalised,
    )


def _numbers(question: str, masked_spans: list[tuple[int, int]]) -> list[Resolution]:
    """Every number in the question, so none leaves in clear text."""
    out: list[Resolution] = []
    for match in NUMBER_RE.finditer(question):
        if any(s <= match.start() and match.end() <= e for s, e in masked_spans):
            continue
        out.append(
            Resolution(match.group(), "number", match.group(), None, 1.0, kind="quantity")
        )
    return out


def _missed_values(
    question: str, existing: list[Resolution], scope: list[str]
) -> list[Resolution]:
    """Stored values the extractor missed — masked here rather than refused later."""
    from nl2sql.privacy.gate import find_known_values

    already = " ".join(r.mention.lower() for r in existing if r.to_mask)
    out: list[Resolution] = []
    for candidate in find_known_values(question):
        if candidate in already or any(candidate == r.mention.lower() for r in existing):
            continue
        found = search(candidate, columns=scope, limit=2) if scope else []
        if not found or found[0].score < 1.0:
            found = search(candidate, limit=2)
        if not found or found[0].score < 1.0:
            continue
        best = found[0]
        out.append(
            Resolution(candidate, "stored-value", best.value, best.ref, best.score, best.tier)
        )
    return out


def understand(question: str) -> Understanding:
    """Run stage 1 end to end. Every sub-step is traced — see `core/steps.py`."""
    start = time.perf_counter()

    with track("extract", question=question) as step:
        entities = ner.extract(question)
        step.say(
            f"found {len(entities)} thing(s) worth checking: "
            + (", ".join(f"'{e.text}'" for e in entities) or "none"),
            extractor="gliner2" if ner.available() else "glossary",
            entities=[{"text": e.text, "type": e.type, "score": e.score} for e in entities],
        )

    with track("scope", question=question) as step:
        scope = glossary.columns_for(question)
        tables = glossary.tables_for(question)
        notes = glossary.notes_for(question)
        step.say(
            f"the words used point at {len(tables) or 'no'} table(s): "
            + (", ".join(sorted(tables)) or "none yet"),
            columns=scope,
            tables=sorted(tables),
            notes=notes,
        )

    resolutions = []
    for entity in entities:
        with track("lookup", mention=entity.text, type=entity.type) as step:
            resolutions.append(_resolve(entity, _scope(entity, scope), step))

    masked = [(e.start, e.end) for e, r in zip(entities, resolutions, strict=True) if r.to_mask]
    with track("classify") as step:
        resolutions += _numbers(question, masked)
        resolutions += _missed_values(question, resolutions, scope)
        kinds = {k: sum(1 for r in resolutions if r.kind == k) for k in ("value", "concept", "quantity", "person")}
        step.say(
            f"{kinds['value']} value(s) to hide, {kinds['concept']} column name(s), "
            f"{kinds['quantity']} number(s)",
            **kinds,
        )

    # A question saying only "aspirin" triggers no business term, but the
    # resolution does designate `medication`.
    for r in resolutions:
        if r.column:
            tables.add(r.column.split(".", 1)[0])

    return Understanding(
        question=question,
        entities=entities,
        resolutions=resolutions,
        tables=tables,
        notes=notes,
        ms=round((time.perf_counter() - start) * 1000, 1),
        extractor="gliner2" if ner.available() else "glossary",
    )
