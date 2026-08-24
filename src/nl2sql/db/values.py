"""Value index — turn "aspirin" into the exact stored `ASPIRIN EC 81 MG PO TBEC`.

Two tiers, and the cost of both is bounded by the number of columns rather than
by the number of rows:

    A  vocabulary   few distinct values: pre-indexed into FTS5.
    B  on demand    too many to store: resolved at question time with a bounded
                    LIKE against the database itself.

Columns that are neither (identifiers, free text, constants) are not registered.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from nl2sql.config import settings
from nl2sql.db.sqlite import connect

VOCABULARY_LIMIT = 5_000      # above this a column is tier B
GIVE_UP_LIMIT = 200_000       # above this it is not a vocabulary at all
MIN_MEAN_LENGTH = 2
MAX_MEAN_LENGTH = 120
UNIQUENESS_LIMIT = 0.9        # near-unique column: an identifier or free text
UNIQUENESS_LIMIT_NAMED = 0.3  # stricter when the name already ends in "id"
SAMPLE_SIZE = 200

TOKEN_RE = re.compile(r"[a-z0-9]+")
IDENTIFIER_RE = re.compile(r"^(id|.*_id|.*offset)$")
SEPARATORS_RE = re.compile(r"\s*[|/;:]\s*")

Tier = Literal["A", "B"]


@dataclass(frozen=True)
class ColumnTier:
    table: str
    column: str
    tier: Tier | None
    distinct: int
    reason: str

    @property
    def ref(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass(frozen=True)
class FoundValue:
    value: str
    table: str
    column: str
    score: float
    tier: Tier = "A"

    @property
    def ref(self) -> str:
        return f"{self.table}.{self.column}"


# --------------------------------------------------------------------------------
#  Classification
# --------------------------------------------------------------------------------
def is_identifier(column: str, table_names: frozenset[str]) -> bool:
    """Recognised by shape, not by the last two letters: `volumeoffluid` is not a key.

    Lower-cased first: six tables of the published export are written entirely in
    upper case, and `PATIENTUNITSTAYID` is as much a key as `patientunitstayid`.
    """
    column = column.lower()
    if IDENTIFIER_RE.match(column):
        return True
    return column.endswith("id") and column[:-2] in table_names


def _classify(
    cx: sqlite3.Connection,
    table: str,
    column: str,
    table_names: frozenset[str],
    row_count: int,
    limit: int,
) -> ColumnTier:
    def out(tier: Tier | None, distinct: int, reason: str) -> ColumnTier:
        return ColumnTier(table, column, tier, distinct, reason)

    if is_identifier(column, table_names):
        return out(None, 0, "identifier or time offset")
    try:
        (distinct,) = cx.execute(f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"').fetchone()
    except sqlite3.Error as e:
        return out(None, 0, f"unreadable ({type(e).__name__})")

    distinct = distinct or 0
    if distinct < 2:
        return out(None, distinct, "constant or empty")
    if distinct > GIVE_UP_LIMIT:
        return out(None, distinct, f"> {GIVE_UP_LIMIT} distinct: not a vocabulary")

    # An identifier also betrays itself by behaviour: nearly one value per row.
    uniqueness = distinct / row_count if row_count else 0.0
    if column.endswith("id") and uniqueness > UNIQUENESS_LIMIT_NAMED:
        return out(None, distinct, f"de-facto identifier ({uniqueness:.0%} unique)")
    if uniqueness > UNIQUENESS_LIMIT and distinct > 100:
        return out(None, distinct, f"near-unique ({uniqueness:.0%})")

    sample = [
        str(r[0]).strip()
        for r in cx.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL LIMIT {SAMPLE_SIZE}'
        )
        if r[0] is not None and str(r[0]).strip()
    ]
    if not sample:
        return out(None, distinct, "no usable value")

    mean = sum(len(v) for v in sample) / len(sample)
    if mean < MIN_MEAN_LENGTH:
        return out(None, distinct, f"values too short ({mean:.1f} chars)")
    if mean > MAX_MEAN_LENGTH:
        return out(None, distinct, f"free text ({mean:.0f} chars)")

    if distinct <= limit:
        return out("A", distinct, "bounded vocabulary")
    return out("B", distinct, f"{distinct} distinct: resolved on demand")


def classify(limit: int = VOCABULARY_LIMIT) -> list[ColumnTier]:
    """Apply the policy to every textual column.

    `limit` is a parameter so the classification can be replayed as if the
    database were much larger, which is how tier B is exercised on eICU.
    """
    from nl2sql.db.schema import read_schema, text_columns

    schema = read_schema()
    names = frozenset(schema)
    with connect(timeout_s=900) as cx:
        return [_classify(cx, t, c, names, schema[t].row_count, limit) for t, c in text_columns()]


# --------------------------------------------------------------------------------
#  Build
# --------------------------------------------------------------------------------
SCHEMA_SQL = """
PRAGMA journal_mode=OFF;
CREATE VIRTUAL TABLE values_fts USING fts5(
    value, ref UNINDEXED, tokenize = 'unicode61 remove_diacritics 2');
CREATE TABLE columns_meta (
    ref TEXT PRIMARY KEY, "table" TEXT, column_name TEXT,
    tier TEXT, distinct_count INTEGER, indexed_count INTEGER, reason TEXT);
CREATE TABLE tier_b_words (word TEXT PRIMARY KEY);
"""


def build(destination: Path | None = None, limit: int = VOCABULARY_LIMIT, verbose: bool = True) -> dict:
    """Write the index. Tier B contributes its word vocabulary, never its values."""
    cfg = settings()
    destination = Path(destination or cfg.index_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)

    tiers = classify(limit)
    idx = sqlite3.connect(destination)
    idx.executescript(SCHEMA_SQL)

    total = 0
    words: set[str] = set()
    with connect(timeout_s=900) as cx:
        for t in tiers:
            if t.tier is None:
                continue
            indexed = 0
            rows = cx.execute(
                f'SELECT DISTINCT "{t.column}" FROM "{t.table}" WHERE "{t.column}" IS NOT NULL'
            )
            if t.tier == "B":
                for (v,) in rows:
                    words.update(re.findall(r"[a-zà-ÿ0-9]{3,}", str(v).lower()))
            else:
                values = [str(v).strip() for (v,) in rows if v is not None and str(v).strip()]
                if values:
                    idx.executemany(
                        "INSERT INTO values_fts (value, ref) VALUES (?, ?)",
                        ((v, t.ref) for v in values),
                    )
                    indexed = len(values)
                    total += indexed
            idx.execute(
                "INSERT INTO columns_meta VALUES (?,?,?,?,?,?,?)",
                (t.ref, t.table, t.column, t.tier, t.distinct, indexed, t.reason),
            )
            if verbose:
                print(f"  {t.tier}  {t.ref:<48} {indexed:>7}", flush=True)

    idx.executemany("INSERT INTO tier_b_words VALUES (?)", ((w,) for w in words))
    idx.execute("INSERT INTO values_fts(values_fts) VALUES('optimize')")
    idx.commit()
    idx.close()

    report = {
        "vocabulary_limit": limit,
        "columns_examined": len(tiers),
        "tier_A": sum(1 for t in tiers if t.tier == "A"),
        "tier_B": sum(1 for t in tiers if t.tier == "B"),
        "not_indexable": sum(1 for t in tiers if t.tier is None),
        "values_indexed": total,
        "tier_b_words": len(words),
        "size_mb": round(destination.stat().st_size / 1048576, 2),
    }
    destination.with_suffix(".json").write_text(
        json.dumps({"stats": report, "columns": [t.__dict__ for t in tiers]}, indent=1),
        encoding="utf-8",
    )
    if verbose:
        print(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------------
#  Search
# --------------------------------------------------------------------------------
def _open(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def _parts(value: str) -> list[str]:
    """eICU packs hierarchies into one cell; compare against each segment too.

    Whole-string comparison against `hematology|coagulation disorders|DIC syndrome`
    rewards any question that happens to share a word with it.
    """
    if len(value) <= 32:
        return [value]
    parts = [p for p in SEPARATORS_RE.split(value) if len(p.strip()) > 3]
    return [value, *parts[:6]] if len(parts) > 1 else [value]


def similarity(mention: str, value: str) -> float:
    """How well a mention matches a stored value. Deliberately not symmetric.

    Partial matching is legitimate in one direction only — the analyst names part
    of a longer value ("aspirin" for `ASPIRIN EC 81 MG PO TBEC`). The reverse is
    not a match: a short value found inside a long mention only says the database
    holds a common short string, and treating it as one scored "10" against
    "10 most frequently recorded laboratory tests" at 1.00.
    """
    left, right = mention.lower().strip(), value.lower().strip()
    if not left or not right:
        return 0.0
    from rapidfuzz import fuzz

    best = 0.0
    for candidate in _parts(right):
        whole = fuzz.ratio(left, candidate) / 100.0
        if len(candidate) < len(left):
            best = max(best, whole)
        else:
            best = max(best, whole, fuzz.partial_ratio(left, candidate) / 100.0)
    return best


def _queries(mention: str) -> list[str]:
    """FTS5 queries, strictest first: AND before OR, prefixes always."""
    tokens = [t for t in TOKEN_RE.findall(mention.lower()) if len(t) > 1]
    if not tokens:
        return []
    prefixes = [f'"{t}"*' for t in tokens]
    return prefixes if len(prefixes) == 1 else [" AND ".join(prefixes), " OR ".join(prefixes)]


def _tier_a(path: Path, mention: str, columns: list[str] | None, threshold: float) -> list[FoundValue]:
    queries = _queries(mention)
    if not queries:
        return []
    sql = "SELECT value, ref FROM values_fts WHERE values_fts MATCH ?"
    params: list[object] = []
    if columns:
        sql += " AND ref IN ({})".format(",".join("?" * len(columns)))
        params = list(columns)
    sql += " ORDER BY bm25(values_fts) LIMIT 300"

    cx = _open(path)
    try:
        for query in queries:
            try:
                raw = cx.execute(sql, [query, *params]).fetchall()
            except sqlite3.OperationalError:
                continue
            out = []
            for value, ref in raw:
                table, _, column = ref.partition(".")
                score = similarity(mention, value)
                if score >= threshold:
                    out.append(FoundValue(value, table, column, round(score, 4), "A"))
            if out:
                return out
    finally:
        cx.close()
    return []


def _tier_b(
    path: Path, mention: str, threshold: float, columns: list[str] | None, cap: int = 5_000
) -> list[FoundValue]:
    """On-demand resolution: a bounded LIKE, nothing stored at rest."""
    sql = "SELECT \"table\", column_name FROM columns_meta WHERE tier='B'"
    params: list[object] = []
    if columns:
        sql += " AND ref IN ({})".format(",".join("?" * len(columns)))
        params = list(columns)
    cx = _open(path)
    try:
        targets = cx.execute(sql, params).fetchall()
    finally:
        cx.close()
    if not targets:
        return []

    pattern = f"%{mention.strip()}%"
    out: list[FoundValue] = []
    with connect() as db:
        for table, column in targets:
            try:
                rows = db.execute(
                    f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" LIKE ? LIMIT {cap}',
                    (pattern,),
                ).fetchall()
            except sqlite3.Error:
                continue
            for (v,) in rows:
                if v is None:
                    continue
                score = similarity(mention, str(v))
                if score >= threshold:
                    out.append(FoundValue(str(v), table, column, round(score, 4), "B"))
    return out


@lru_cache(maxsize=2)
def _vocabulary(path_str: str) -> tuple[tuple[str, str], ...]:
    cx = _open(Path(path_str))
    try:
        return tuple(cx.execute("SELECT value, ref FROM values_fts"))
    finally:
        cx.close()


def _fuzzy(path: Path, mention: str, columns: list[str] | None, threshold: float) -> list[FoundValue]:
    """Last resort for a misspelling: full scan of the indexed vocabulary.

    Full-text retrieval matches whole tokens, so `asspirin` retrieves nothing and
    never gets scored at all. The bar is higher here because everything becomes a
    candidate, and the scorer is whole-string `ratio` — `WRatio` blends in partial
    matching and returned `tan` at 0.90 for a word that is not a drug.
    """
    from rapidfuzz import fuzz, process

    vocabulary = _vocabulary(str(path))
    if columns:
        allowed = set(columns)
        vocabulary = tuple((v, r) for v, r in vocabulary if r in allowed)
    if not vocabulary:
        return []

    hits = process.extract(
        mention.lower(),
        [v.lower() for v, _ in vocabulary],
        scorer=fuzz.ratio,
        limit=20,
        score_cutoff=threshold * 100,
    )
    out = []
    for _text, score, position in hits:
        value, ref = vocabulary[position]
        table, _, column = ref.partition(".")
        out.append(FoundValue(value, table, column, round(score / 100.0, 4), "A"))
    return out


def search(
    mention: str,
    columns: list[str] | None = None,
    limit: int = 5,
    threshold: float = 0.55,
    fuzzy_threshold: float = 0.80,
    source: Path | None = None,
) -> list[FoundValue]:
    """A first, then B, then a fuzzy scan — each only if the previous found nothing."""
    path = Path(source or settings().index_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run: python -m nl2sql.cli index")

    results = _tier_a(path, mention, columns, threshold)
    if not results:
        results = _tier_b(path, mention, threshold, columns)
    if not results:
        results = _fuzzy(path, mention, columns, fuzzy_threshold)

    # `aspirin` is stored verbatim in four columns, all scoring 1.00. The glossary
    # declares which is canonical by listing it first, so ties are settled by
    # position in the requested scope rather than by FTS ordering.
    rank = {ref: i for i, ref in enumerate(columns or [])}
    results.sort(key=lambda v: (-v.score, rank.get(v.ref, len(rank) + 1), len(v.value)))

    seen: set[str] = set()
    unique = []
    for r in results:
        if r.ref not in seen:
            seen.add(r.ref)
            unique.append(r)
    return unique[:limit]


@lru_cache(maxsize=1)
def exact_values(source: Path | None = None) -> frozenset[str]:
    """Every indexed value, normalised, for an O(1) membership test."""
    path = Path(source or settings().index_path)
    if not path.exists():
        return frozenset()
    cx = _open(path)
    try:
        return frozenset(
            " ".join(str(v).lower().split()) for (v,) in cx.execute("SELECT value FROM values_fts")
        )
    except sqlite3.Error:
        return frozenset()
    finally:
        cx.close()


def is_exact_value(mention: str) -> bool:
    return " ".join(str(mention).lower().split()) in exact_values()


def query_index(sql: str) -> list[tuple]:
    """Read from the index file. Used by the gate and by the notebooks."""
    path = settings().index_path
    if not path.exists():
        return []
    cx = _open(path)
    try:
        return cx.execute(sql).fetchall()
    except sqlite3.Error:
        return []
    finally:
        cx.close()


def stats() -> dict:
    path = settings().index_path
    values = query_index("SELECT COUNT(*) FROM values_fts")
    return {
        "tiers": dict(query_index("SELECT tier, COUNT(*) FROM columns_meta GROUP BY tier")),
        "values_indexed": values[0][0] if values else 0,
        "size_mb": round(path.stat().st_size / 1048576, 2) if path.exists() else 0,
        "top": query_index(
            "SELECT ref, indexed_count FROM columns_meta WHERE tier='A' "
            "ORDER BY indexed_count DESC LIMIT 12"
        ),
    }
