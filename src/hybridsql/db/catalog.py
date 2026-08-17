"""The column catalogue — turning a mention into a **column**, not into a value.

The half that was missing
-------------------------
`db/value_index.py` answers *"which stored value is this?"*. `resources/glossary.py`
answers *"which column does this business term name?"* — but only for the thirty
terms somebody wrote down by hand. Between the two there was a hole, and every
question that named a column we had not written down fell into it:

    "the 10 most common diagnosis names"  ->  pasthistory.pasthistoryvalue = 'clinical diagnosis'
    "how many laboratory records"         ->  note.notevalue               = 'Medical Records'
    "medication administration routes"    ->  (nothing)

None of those mentions is a value. They name columns — `diagnosis.diagnosisstring`,
`lab.labname`, `medication.routeadmin` — and the index, asked for a value, always
finds *something*, because a fuzzy search over 30 000 values never comes back
empty. The result is SQL that runs and answers a different question, which is this
project's worst failure mode.

This module supplies the missing answer, and `pipeline/understand.py` then makes
the two compete: a mention becomes a value only when it matches a value **better
than it matches a column**.

How a column is matched, and why this way
-----------------------------------------
Column names in a real database are concatenated words with no separator —
`labname`, `routeadmin`, `nursingchartcelltypecat`. Word-level comparison cannot
see inside them, and splitting them requires a dictionary of the very domain we
are trying to stay independent of.

So matching is done on **character trigrams**, which need no dictionary and no
model: `"administration routes"` and `medication.routeadmin` share `rou`, `out`,
`ute`, `adm`, `dmi`, `min`. Measured on the failing questions:

    administration routes   ->  medication.routeadmin      0.52
    medication frequencies  ->  medication.frequency       0.83
    hospital region         ->  hospital.region            1.00
    aspirin                 ->  best column                0.21   <- a value, not a column

That last line is the point: a real value scores *low* against every column, which
is exactly what makes the arbitration possible.

Three sources feed each card, in decreasing order of authority:

1. the **glossary**, when it declares the column — an exact business synonym wins
   outright;
2. the **column-shaped head word** rule (see `head_words`) — "diagnosis *names*",
   "drug *names*" name a column by construction;
3. **trigram similarity** against `table column`.

What is excluded, and why it matters
------------------------------------
Tier-C columns — identifiers, `*offset`, timestamps — are dropped. Without that,
`diagnosis names` matched `diagnosis.diagnosisid` (0.70) ahead of
`diagnosis.diagnosisstring` (0.67): every identifier repeats its table name, so
identifiers win every table-level match. The tier already computed by
`db/value_index.py` is reused rather than re-derived.

Scale
-----
The catalogue holds one card per column: 391 here, and it does not grow with the
number of rows. Linking a mention is 391 trigram comparisons, measured under 2 ms.
On a schema of several thousand columns the same structure holds; if the linear
scan ever became the bottleneck the cards are exactly what an embedding or FTS
index would be built from — the interface would not change.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from hybridsql.config import settings

WORD_RE = re.compile(r"[a-z0-9]+")

# A mention scoring below this against every column is not naming a column.
MINIMUM_SCORE = 0.25

# A word must appear in this many distinct column names before it counts as
# "column-shaped". Two would catch coincidences; measured on eICU, three isolates
# name, value, text, type, path, label, rate, status, category — and nothing else.
HEAD_WORD_OCCURRENCES = 3


@dataclass(frozen=True)
class ColumnCard:
    """One column, with everything that can be matched against a mention."""

    ref: str
    table: str
    column: str
    tier: str = ""
    distinct: int = 0
    synonyms: tuple[str, ...] = ()   # business terms the glossary attaches here

    @property
    def surface(self) -> str:
        """The text a mention is compared against."""
        return f"{self.table} {self.column}"


@dataclass(frozen=True)
class ColumnMatch:
    ref: str
    score: float
    why: str

    @property
    def table(self) -> str:
        return self.ref.split(".", 1)[0]


# ---------------------------------------------------------------------------------
# Building the catalogue
# ---------------------------------------------------------------------------------
def _tiers() -> dict[str, str]:
    """Column tiers, read from the value index. Empty when it has not been built."""
    path = settings().value_index_path
    if not path.exists():
        return {}
    cx = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    try:
        return {ref: tier for ref, tier in cx.execute("SELECT ref, tier FROM columns_meta")}
    except sqlite3.Error:
        return {}
    finally:
        cx.close()


def _distinct_counts() -> dict[str, int]:
    path = settings().value_index_path
    if not path.exists():
        return {}
    cx = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    try:
        return {ref: n for ref, n in cx.execute("SELECT ref, distinct_count FROM columns_meta")}
    except sqlite3.Error:
        return {}
    finally:
        cx.close()


@lru_cache(maxsize=1)
def cards() -> tuple[ColumnCard, ...]:
    """One card per linkable column. Cached: the schema does not change at runtime."""
    from hybridsql.db.schema import read_schema
    from hybridsql.db.value_index import _is_identifier

    schema = read_schema()
    table_names = frozenset(schema)
    tiers, counts = _tiers(), _distinct_counts()

    synonyms: dict[str, list[str]] = {}
    try:
        from hybridsql.resources.glossary import load

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
            # Identifiers, keys and offsets carry their table's name, so they win
            # every table-level match while never being what the analyst meant:
            # "nursing charting records" matched `nursecharting.nursingchartid`
            # ahead of `nursingchartvalue`. Dropped unless the glossary explicitly
            # declares one, which is how `hospital.hospitalid` stays reachable.
            structural = (
                tiers.get(ref) == "C"
                or column.is_pk
                or _is_identifier(column.name, table_names)
                or column.name.lower().endswith(("id", "offset"))
            )
            if structural and ref not in synonyms:
                continue
            out.append(
                ColumnCard(
                    ref=ref,
                    table=table.name,
                    column=column.name,
                    tier=tiers.get(ref, ""),
                    distinct=counts.get(ref, 0),
                    synonyms=tuple(synonyms.get(ref, ())),
                )
            )
    return tuple(out)


@lru_cache(maxsize=1)
def head_words() -> frozenset[str]:
    """Words that name a *column* by construction, derived from the schema itself.

    "diagnosis names" is not a diagnosis: it is the *name* column of the diagnosis
    table. What makes "name" different from "aspirin" is not meaning, it is
    **distribution**: database designers put the structural word last, so `name`,
    `value`, `text`, `type`, `path` and `label` recur as the ending of many column
    names, while no value's word does.

    So the rule is derived and not written: a **suffix** shared by at least
    `HEAD_WORD_OCCURRENCES` column names, that is not itself a table name. Only
    maximal suffixes are kept, otherwise `name`, `ame` and `me` would all qualify.

    The derivation also yields fragments that are not words — `ion`, `ath`, `sis`.
    They are harmless and deliberately left in rather than filtered with an English
    dictionary we do not have: this set is only ever consulted with the last word
    of a question typed by a human, and nobody ends a question with "sis".
    """
    from hybridsql.db.schema import read_schema

    schema = read_schema()
    tables = {t.lower() for t in schema}
    columns = [c.name.lower() for t in schema.values() for c in t.columns]

    suffixes: Counter[str] = Counter()
    for name in columns:
        for size in range(3, min(len(name), 10) + 1):
            suffixes[name[-size:]] += 1

    frequent = {
        s: n for s, n in suffixes.items() if n >= HEAD_WORD_OCCURRENCES and s not in tables
    }
    # `name` and `drugname` both occur 9 times: only the longer one is informative.
    return frozenset(
        s
        for s, n in frequent.items()
        if not any(other != s and other.endswith(s) and frequent[other] == n for other in frequent)
    )


def _singular(word: str) -> str:
    """`names` -> `name`. Enough for English column vocabulary, no library needed."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 4 and word[-3] in "sxzh":
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def names_a_column(mention: str) -> str | None:
    """The column-shaped head word of a mention, if it has one.

    "diagnosis names" -> "name". "aspirin" -> None. Only the **last** word is
    examined: "name" in the middle of a phrase does not make it a column reference.
    """
    words = WORD_RE.findall(mention.lower())
    if len(words) < 2:
        return None
    head = _singular(words[-1])
    return head if head in head_words() else None


# ---------------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------------
def _grams(text: str, n: int = 3) -> Counter[str]:
    padded = "  " + " ".join(WORD_RE.findall(text.lower())) + "  "
    return Counter(padded[i : i + n] for i in range(len(padded) - n + 1))


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = sum(a[g] * b[g] for g in a if g in b)
    if not common:
        return 0.0
    return common / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


@lru_cache(maxsize=1)
def _card_grams() -> tuple[tuple[ColumnCard, Counter[str], Counter[str], tuple[Counter[str], ...]], ...]:
    """Trigram vectors, computed once for the process."""
    return tuple(
        (card, _grams(card.surface), _grams(card.column), tuple(_grams(s) for s in card.synonyms))
        for card in cards()
    )


@lru_cache(maxsize=512)
def link(mention: str, limit: int = 4) -> tuple[ColumnMatch, ...]:
    """Columns this mention could be naming, best first.

    Cached per mention: the same words come back on every question, and the scan is
    pure computation over immutable data.
    """
    text = " ".join(WORD_RE.findall((mention or "").lower()))
    if not text:
        return ()

    query = _grams(text)
    head = names_a_column(text)
    # The head word is excluded from the context: it is what we are matching *with*,
    # so letting it also prove relevance made "diagnosis names" reach
    # `lab.labmeasurenamesystem`, whose column name happens to contain "names".
    words = set(text.split()[:-1]) if head else set(text.split())

    scored: list[ColumnMatch] = []
    for card, surface, column_only, synonym_grams in _card_grams():
        # 1. The glossary is an authority, not a hint: an exact business synonym
        #    settles the question without any similarity computation.
        if any(s.lower() == text for s in card.synonyms):
            scored.append(ColumnMatch(card.ref, 1.0, "glossary synonym"))
            continue

        base = _cosine(query, surface)
        why = "trigram similarity"
        # A *partial* synonym match is evidence about the table, not about the
        # column: "diagnosis names" partially matches the synonym "diagnosis",
        # which every column of `diagnosis` carries. Left at full weight it made
        # `diagnosis.icd9code` outrank `diagnosis.diagnosisstring`. Halved, it
        # still lifts the right table without choosing the column.
        for grams in synonym_grams:
            partial = 0.6 * _cosine(query, grams)
            if partial > base:
                base, why = partial, "glossary synonym, partial"

        # The column's own name breaks the ties the table-level evidence creates.
        score = 0.75 * base + 0.25 * _cosine(query, column_only)

        # 2. A column-shaped head word is strong evidence, but only for the columns
        #    whose table the rest of the mention actually names: "diagnosis names"
        #    must reach `diagnosis.diagnosisstring`, not every `*name` in the schema.
        if head and head in card.column and any(
            w in card.table or w in card.column for w in words if len(w) > 3
        ):
            score = max(score, 0.90)
            why = f"names the '{head}' column of {card.table}"

        if score >= MINIMUM_SCORE:
            scored.append(ColumnMatch(card.ref, round(score, 4), why))

    scored.sort(key=lambda m: (-m.score, len(m.ref)))
    return tuple(scored[:limit])


def best(mention: str) -> ColumnMatch | None:
    matches = link(mention)
    return matches[0] if matches else None


def stats() -> dict[str, object]:
    return {
        "columns_catalogued": len(cards()),
        "with_glossary_synonyms": sum(1 for c in cards() if c.synonyms),
        "column_shaped_words": len(head_words()),
    }
