"""The column catalogue — decide which **column** a mention names.

The value index answers "which stored value is this?" and always finds
something, because a fuzzy search over thirty thousand values never comes back
empty. Asked about "the 10 most common diagnosis names" it answered
`pasthistory.pasthistoryvalue = 'clinical diagnosis'`, and the question came back
filtered on a value nobody asked for.

This module answers the other question, so that `understand` can make the two
compete. Matching is on character trigrams over `table column`: real column names
are concatenated words with no separator (`labname`, `routeadmin`), which no
word-level comparison can see inside without a dictionary of the very domain we
are trying to stay independent of. A real value scores low against every column,
which is what makes the arbitration possible.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

WORD_RE = re.compile(r"[a-z0-9]+")

MINIMUM_SCORE = 0.25        # below this, the mention is not naming a column
HEAD_WORD_OCCURRENCES = 3   # how many columns must share a suffix for it to count


@dataclass(frozen=True)
class ColumnCard:
    ref: str
    table: str
    column: str
    synonyms: tuple[str, ...] = ()

    @property
    def surface(self) -> str:
        return f"{self.table} {self.column}"


@dataclass(frozen=True)
class ColumnMatch:
    ref: str
    score: float
    why: str

    @property
    def table(self) -> str:
        return self.ref.split(".", 1)[0]


@lru_cache(maxsize=1)
def cards() -> tuple[ColumnCard, ...]:
    """One card per linkable column.

    Identifiers and `*offset` columns are dropped: they repeat their table's name,
    so they win every table-level match while never being what the analyst meant.
    Text columns the index rejected (free text, near-unique, constant) are dropped
    too. A column the glossary explicitly declares is always kept.
    """
    from nl2sql.db.schema import compact_type, read_schema
    from nl2sql.db.values import is_identifier, query_index

    schema = read_schema()
    table_names = frozenset(schema)
    indexed = {ref for (ref,) in query_index("SELECT ref FROM columns_meta")}

    synonyms: dict[str, list[str]] = {}
    try:
        from nl2sql.nlp.glossary import load

        for term in load().values():
            for column in term.columns:
                bucket = synonyms.setdefault(column, [])
                for phrase in (term.canonical.replace("_", " "), *term.synonyms):
                    if phrase not in bucket:
                        bucket.append(phrase)
    except Exception:  # noqa: BLE001 — a missing glossary degrades, it does not break
        synonyms = {}

    out: list[ColumnCard] = []
    for table in schema.values():
        for column in table.columns:
            ref = f"{table.name}.{column.name}"
            textual = compact_type(column.sql_type) == "TEXT"
            structural = (
                is_identifier(column.name, table_names)
                or column.name.lower().endswith(("id", "offset"))
                or (textual and ref not in indexed)
            )
            if structural and ref not in synonyms:
                continue
            out.append(ColumnCard(ref, table.name, column.name, tuple(synonyms.get(ref, ()))))
    return tuple(out)


@lru_cache(maxsize=1)
def head_words() -> frozenset[str]:
    """Words that name a *column* by construction, derived from the schema itself.

    "diagnosis names" is not a diagnosis: it is the *name* column of the diagnosis
    table. What separates "name" from "aspirin" is distribution, not meaning —
    designers put the structural word last, so `name`, `value`, `text`, `type`
    recur as the ending of many column names while no value's word does. Only
    maximal suffixes are kept, otherwise `name`, `ame` and `me` would all qualify.
    """
    from nl2sql.db.schema import read_schema

    schema = read_schema()
    tables = {t.lower() for t in schema}
    suffixes: Counter[str] = Counter()
    for table in schema.values():
        for column in table.columns:
            name = column.name.lower()
            for size in range(3, min(len(name), 10) + 1):
                suffixes[name[-size:]] += 1

    frequent = {s: n for s, n in suffixes.items() if n >= HEAD_WORD_OCCURRENCES and s not in tables}
    return frozenset(
        s
        for s, n in frequent.items()
        if not any(o != s and o.endswith(s) and frequent[o] == n for o in frequent)
    )


def _singular(word: str) -> str:
    """`names` -> `name`. Enough for column vocabulary; no library needed."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4 and word[-3] in "sxzh":
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def names_a_column(mention: str) -> str | None:
    """The column-shaped head word of a mention, if it has one.

    Only the last word is examined: "name" in the middle of a phrase does not make
    the phrase a column reference.
    """
    words = WORD_RE.findall(mention.lower())
    if len(words) < 2:
        return None
    head = _singular(words[-1])
    return head if head in head_words() else None


def _grams(text: str, n: int = 3) -> Counter[str]:
    padded = "  " + " ".join(WORD_RE.findall(text.lower())) + "  "
    return Counter(padded[i : i + n] for i in range(len(padded) - n + 1))


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    common = sum(a[g] * b[g] for g in a if g in b)
    if not common:
        return 0.0
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return common / norm if norm else 0.0


@lru_cache(maxsize=1)
def _vectors() -> tuple[tuple[ColumnCard, Counter[str], Counter[str], tuple[Counter[str], ...]], ...]:
    return tuple(
        (c, _grams(c.surface), _grams(c.column), tuple(_grams(s) for s in c.synonyms))
        for c in cards()
    )


@lru_cache(maxsize=512)
def link(mention: str, limit: int = 4) -> tuple[ColumnMatch, ...]:
    """Columns this mention could be naming, best first."""
    text = " ".join(WORD_RE.findall((mention or "").lower()))
    if not text:
        return ()

    query = _grams(text)
    head = names_a_column(text)
    # The head word proves nothing about relevance — it is what we match *with* —
    # so it is excluded from the context, or "diagnosis names" reaches
    # `lab.labmeasurenamesystem` on the strength of the word "names" alone.
    words = set(text.split()[:-1]) if head else set(text.split())

    scored: list[ColumnMatch] = []
    for card, surface, own, synonym_grams in _vectors():
        if any(s.lower() == text for s in card.synonyms):
            scored.append(ColumnMatch(card.ref, 1.0, "glossary synonym"))
            continue

        base, why = _cosine(query, surface), "trigram similarity"
        # A *partial* synonym match is evidence about the table, not the column:
        # every column of `diagnosis` carries the synonym "diagnosis". Halved, it
        # still lifts the right table without choosing the wrong column.
        for grams in synonym_grams:
            partial = 0.6 * _cosine(query, grams)
            if partial > base:
                base, why = partial, "glossary synonym, partial"

        score = 0.75 * base + 0.25 * _cosine(query, own)

        if head and head in card.column and any(
            w in card.table or w in card.column for w in words if len(w) > 3
        ):
            score, why = max(score, 0.90), f"names the '{head}' column of {card.table}"

        if score >= MINIMUM_SCORE:
            scored.append(ColumnMatch(card.ref, round(score, 4), why))

    scored.sort(key=lambda m: (-m.score, len(m.ref)))
    return tuple(scored[:limit])


def best(mention: str) -> ColumnMatch | None:
    matches = link(mention)
    return matches[0] if matches else None


def stats() -> dict[str, int]:
    return {
        "columns_catalogued": len(cards()),
        "with_synonyms": sum(1 for c in cards() if c.synonyms),
        "column_shaped_words": len(head_words()),
    }
