"""Business vocabulary to columns.

The value index resolves *content* ("aspirin"). It can do nothing for
"mortality", which is not a value but a column. This reads `resources/glossary.yaml`
and answers the second question: which columns and tables a question is about.

No data passes through here — only vocabulary and column names — so the glossary
is designed to cross the trust boundary.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from nl2sql.config import settings

WORD_RE = re.compile(r"[a-z0-9&]+")


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
            table = c.split(".", 1)[0]
            if table not in seen:
                seen.append(table)
        return tuple(seen)


@dataclass
class Match:
    term: Term
    trigger: str
    position: int = field(default=0, compare=False)


def normalize(text: str) -> str:
    """Lowercase and accent-free, so "mortalité" and "mortalite" coincide."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


@lru_cache(maxsize=1)
def load(path: Path | None = None) -> dict[str, Term]:
    import yaml

    path = Path(path or settings().glossary_path)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        name: Term(
            canonical=name,
            synonyms=tuple((body or {}).get("synonyms") or ()),
            columns=tuple((body or {}).get("columns") or ()),
            note=((body or {}).get("note") or "").strip(),
        )
        for name, body in raw.items()
    }


@lru_cache(maxsize=1)
def _directory() -> list[tuple[tuple[str, ...], Term]]:
    """Every trigger split into words, longest first so "length of stay" beats "stay"."""
    entries = [
        (tuple(WORD_RE.findall(normalize(raw))), term)
        for term in load().values()
        for raw in (term.canonical.replace("_", " "), *term.synonyms)
    ]
    return sorted((e for e in entries if e[0]), key=lambda e: -len(e[0]))


def recognize(question: str) -> list[Match]:
    """Business terms present in the question, matched on word n-grams.

    Not on substrings: "stay" would otherwise fire inside "understayed".
    """
    words = WORD_RE.findall(normalize(question))
    if not words:
        return []

    found: dict[str, Match] = {}
    covered: set[int] = set()
    for trigger, term in _directory():
        n = len(trigger)
        for i in range(len(words) - n + 1):
            if tuple(words[i : i + n]) != trigger or any(j in covered for j in range(i, i + n)):
                continue
            covered.update(range(i, i + n))
            found.setdefault(term.canonical, Match(term, " ".join(trigger), i))
    return sorted(found.values(), key=lambda m: m.position)


def columns_for(question: str) -> list[str]:
    seen: list[str] = []
    for match in recognize(question):
        for column in match.term.columns:
            if column not in seen:
                seen.append(column)
    return seen


def tables_for(question: str) -> set[str]:
    return {c.split(".", 1)[0] for c in columns_for(question)}


def notes_for(question: str) -> list[str]:
    """Schema traps the model cannot infer from the DDL — offsets in minutes, and such."""
    return [m.term.note for m in recognize(question) if m.term.note]


def validate() -> list[str]:
    """Problems in the glossary. A term pointing at a vanished column is worse than none."""
    from nl2sql.db.schema import read_schema

    known = {f"{t.name}.{c.name}" for t in read_schema().values() for c in t.columns}
    problems: list[str] = []
    for term in load().values():
        if not term.columns:
            problems.append(f"{term.canonical}: no column declared")
        problems += [f"{term.canonical} -> {c}: unknown column" for c in term.columns if c not in known]
    return problems
