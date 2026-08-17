"""The business glossary: from the analyst's vocabulary to database columns.

The value index (`db/value_index.py`) resolves **content**: "aspirin" becomes
`ASPIRIN EC 81 MG PO TBEC`. It can do nothing for "mortality", which is not a
value but a column name.

This module does the other half: it reads `config/glossary.yaml` and returns, for
a given question, the columns and tables involved. Two uses:

- **schema linking** — send the cloud only the useful tables, which shortens the
  prompt and reduces join mistakes;
- **resolution scope** — restrict the value search to plausible columns, making
  it faster and more accurate.

No data passes through here: only vocabulary and column names. The glossary is
designed to cross the trust boundary.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from hybridsql.config import settings

WORD_RE = re.compile(r"[a-z0-9&]+")


class InvalidGlossary(ValueError):
    """The glossary references a column that does not exist in the database."""


@dataclass(frozen=True)
class Term:
    canonical: str
    synonyms: tuple[str, ...]
    columns: tuple[str, ...]
    note: str = ""

    @property
    def tables(self) -> tuple[str, ...]:
        seen: list[str] = []
        for c in self.columns:
            t = c.split(".", 1)[0]
            if t not in seen:
                seen.append(t)
        return tuple(seen)


@dataclass
class Match:
    """A glossary term recognised in the question, and what triggered it."""

    term: Term
    trigger: str
    position: int = field(default=0, compare=False)


def _normalize(text: str) -> str:
    """Lowercase, accent-free. "Mortalité" and "mortalite" must coincide."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


@lru_cache(maxsize=1)
def load(path: Path | None = None) -> dict[str, Term]:
    """Read the glossary. Cached: the file does not change at runtime."""
    import yaml

    path = Path(path or settings().glossary_path)
    if not path.exists():
        return {}

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    terms: dict[str, Term] = {}
    for canonical, body in raw.items():
        body = body or {}
        terms[canonical] = Term(
            canonical=canonical,
            synonyms=tuple(body.get("synonyms") or ()),
            columns=tuple(body.get("columns") or ()),
            note=(body.get("note") or "").strip(),
        )
    return terms


@lru_cache(maxsize=1)
def _directory() -> list[tuple[tuple[str, ...], Term]]:
    """Inverted index: each trigger, split into words, points to its term.

    Sorted longest first so that "length of stay" wins over "stay", which is a
    subset of it.
    """
    entries: list[tuple[tuple[str, ...], Term]] = []
    for term in load().values():
        for raw in (term.canonical.replace("_", " "), *term.synonyms):
            words = tuple(WORD_RE.findall(_normalize(raw)))
            if words:
                entries.append((words, term))
    entries.sort(key=lambda e: -len(e[0]))
    return entries


def recognize(question: str) -> list[Match]:
    """Spot the business terms present in the question.

    Search runs on word n-grams, not substrings: otherwise "stay" would fire
    inside "understayed", and "los" inside "close".
    """
    words = WORD_RE.findall(_normalize(question))
    if not words:
        return []

    found: dict[str, Match] = {}
    covered: set[int] = set()

    for trigger, term in _directory():
        n = len(trigger)
        for i in range(len(words) - n + 1):
            if tuple(words[i : i + n]) != trigger:
                continue
            # A word already taken by a longer term does not count twice.
            if any(j in covered for j in range(i, i + n)):
                continue
            covered.update(range(i, i + n))
            found.setdefault(
                term.canonical, Match(term=term, trigger=" ".join(trigger), position=i)
            )
    return sorted(found.values(), key=lambda m: m.position)


def columns_for(question: str) -> list[str]:
    """Plausible columns for this question, deduplicated, in order."""
    seen: list[str] = []
    for m in recognize(question):
        for column in m.term.columns:
            if column not in seen:
                seen.append(column)
    return seen


def tables_for(question: str) -> set[str]:
    """Tables to keep for schema linking."""
    return {c.split(".", 1)[0] for c in columns_for(question)}


def notes_for(question: str) -> list[str]:
    """Business warnings to attach to the prompt.

    This is how schema traps travel — `*offset` columns in minutes, the two
    coexisting APACHE versions — which the model cannot infer from the DDL alone.
    """
    return [m.term.note for m in recognize(question) if m.term.note]


def validate() -> list[str]:
    """Check the glossary against the real schema. Returns the list of problems.

    A glossary pointing at a vanished column is worse than an empty one: it steers
    the model toward SQL that will never run.
    """
    from hybridsql.db.schema import read_schema

    schema = read_schema()
    known = {f"{t.name}.{c.name}" for t in schema.values() for c in t.columns}

    problems: list[str] = []
    for term in load().values():
        if not term.columns:
            problems.append(f"{term.canonical}: no column declared")
        for column in term.columns:
            if column not in known:
                problems.append(f"{term.canonical} -> {column}: unknown column")
    return problems
