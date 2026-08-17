"""Value index: recover the exact stored value from a user's mention.

The problem it solves
---------------------
The analyst writes "aspirin". The database holds `ASPIRIN EC 81 MG PO TBEC`. A
`WHERE drugname = 'aspirin'` returns nothing. So the mention must be turned into
a real value **locally, before any cloud call**.

This is also what makes anonymization possible: the cloud receives `:v1` while we
keep `ASPIRIN EC 81 MG PO TBEC` on our side.

Why we do not index everything
------------------------------
On eICU (demo) indexing everything would be affordable: 53,065 distinct values.
But this project must demonstrate an architecture that **transfers to a real
database**, where a single column can carry millions of distinct values. An
exhaustive index would explode there in both size and build time.

Hence a three-tier policy, applied column by column:

    Tier A — VOCABULARY, indexed
        Few distinct values (<= VOCABULARY_LIMIT), textual, stable. These are the
        dimensions: drug names, lab names, diagnoses.
        Cost bounded by n_columns x VOCABULARY_LIMIT, independent of row count.

    Tier B — HIGH CARDINALITY, resolved on demand
        Too many values to pre-index, but resolvable at question time with a
        targeted, bounded query. Zero storage at rest; scales with any table size.

    Tier C — EXCLUDED
        Measurements, timestamps, free text, identifiers. Nobody names these in
        natural language; indexing them would drown real values in noise.

Scalability comes from the classification, not from the index size.
See `docs/03-INDEXING.md`.

Not to be confused with the *SQL indexes* in `build_database.py`, which only
serve execution speed. This one is a semantic index of content.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from hybridsql.config import settings
from hybridsql.db.connection import connect

# --- Policy thresholds ---------------------------------------------------------------
VOCABULARY_LIMIT = 5_000       # beyond this: tier B, resolved on demand
GIVE_UP_LIMIT = 200_000        # beyond this: not a vocabulary at all
MAX_NUMERIC_SHARE = 0.6        # a mostly numeric column is not a vocabulary
MIN_MEAN_LENGTH = 2            # "5", "K": nothing to resolve
MAX_MEAN_LENGTH = 120          # beyond this it is free text, not a value
UNIQUENESS_LIMIT = 0.9         # near-unique column: not a vocabulary
UNIQUENESS_LIMIT_NAMED = 0.3   # same, but the name already ends in "id": stricter
SAMPLE_SIZE = 200

NUMERIC_RE = re.compile(r"^\s*[-+]?\d+([.,]\d+)?([eE][-+]?\d+)?\s*$")
TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
TOKEN_RE = re.compile(r"[a-z0-9]+")

# An identifier, not a word that happens to end in "id". `volumeoffluid` was
# mistaken for a key by a plain suffix rule: here we require a separator, or a
# prefix that is a real table name.
IDENTIFIER_RE = re.compile(r"^(id|.*_id|.*offset)$")

Tier = Literal["A", "B", "C"]


@dataclass(frozen=True)
class Classification:
    table: str
    column: str
    tier: Tier
    distinct: int
    reason: str

    @property
    def ref(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass(frozen=True)
class FoundValue:
    value: str           # the exact value, as stored
    table: str
    column: str
    score: float         # 0..1
    tier: Tier = "A"

    @property
    def ref(self) -> str:
        return f"{self.table}.{self.column}"


# ---------------------------------------------------------------------------------
# 1. Column classification
# ---------------------------------------------------------------------------------
def _is_identifier(column: str, table_names: frozenset[str]) -> bool:
    """An identifier is recognised by its shape, not by its last three letters.

    `patientunitstayid` and `hospitalid` are identifiers; `volumeoffluid` is not.
    So we require either an explicit separator, or a prefix naming a table that
    actually exists in the schema.
    """
    if IDENTIFIER_RE.match(column):
        return True
    return column.endswith("id") and column[:-2] in table_names


def _classify_one(
    cx: sqlite3.Connection,
    table: str,
    column: str,
    table_names: frozenset[str],
    row_count: int,
    vocabulary_limit: int = VOCABULARY_LIMIT,
) -> Classification:
    def verdict(tier: Tier, distinct: int, reason: str) -> Classification:
        return Classification(table, column, tier, distinct, reason)

    if _is_identifier(column, table_names):
        return verdict("C", 0, "identifier or time offset")

    try:
        (distinct,) = cx.execute(f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"').fetchone()
    except sqlite3.Error as e:
        return verdict("C", 0, f"unreadable ({type(e).__name__})")
    distinct = distinct or 0

    if distinct < 2:
        return verdict("C", distinct, "constant or empty")
    if distinct > GIVE_UP_LIMIT:
        return verdict("C", distinct, f"> {GIVE_UP_LIMIT} distinct: no longer a vocabulary")

    # An identifier betrays itself by behaviour: almost as many distinct values as
    # rows. `patient.uniquepid` passed the name rule but is 73% unique — indexing
    # those values would publish a patient list into a file meant for business
    # vocabulary.
    uniqueness = distinct / row_count if row_count else 0.0
    if column.endswith("id") and uniqueness > UNIQUENESS_LIMIT_NAMED:
        return verdict("C", distinct, f"de-facto identifier ({uniqueness:.0%} unique values)")
    if uniqueness > UNIQUENESS_LIMIT and distinct > 100:
        return verdict("C", distinct, f"near-unique ({uniqueness:.0%}): identifier or free text")

    sample = [
        str(r[0]).strip()
        for r in cx.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL LIMIT {SAMPLE_SIZE}'
        )
        if r[0] is not None and str(r[0]).strip()
    ]
    if not sample:
        return verdict("C", distinct, "no usable value")

    # Purely numeric columns used to be excluded here, on the argument that a
    # number is a quantity rather than vocabulary. That rule is now redundant and
    # was quietly harmful.
    #
    # Redundant, because stage 1 masks **every** number in a question
    # deterministically (`pipeline/understand._numbers`), so a numeric value
    # cannot reach a provider whether it is indexed or not. Harmful, because the
    # test samples values and asks whether they *look* numeric, which threw away
    # real vocabulary living in numeric-looking columns — `patient.age`, 77
    # distinct values, 99% "numeric", and one of them is the string `> 89` that
    # every age question in this database turns on.
    #
    # Measured before removing it: the entire text vocabulary of eICU is 51,201
    # distinct values across all 391 columns, under 1 MB. There was never a cost
    # worth this rule.
    mean_length = sum(len(v) for v in sample) / len(sample)
    if mean_length < MIN_MEAN_LENGTH:
        return verdict("C", distinct, f"values too short ({mean_length:.1f} chars)")
    if mean_length > MAX_MEAN_LENGTH:
        return verdict("C", distinct, f"free text ({mean_length:.0f} chars on average)")

    if distinct <= vocabulary_limit:
        return verdict("A", distinct, "bounded vocabulary")
    return verdict("B", distinct, f"{distinct} distinct: resolved on demand")


def classify_columns(vocabulary_limit: int = VOCABULARY_LIMIT) -> list[Classification]:
    """Apply the policy to every textual column.

    `vocabulary_limit` is a parameter rather than a frozen constant: it is what
    lets us replay the classification as if the database were much larger, and so
    exercise tier B on eICU (see `scripts/demo_scalability.py`).
    """
    from hybridsql.db.schema import read_schema, text_columns

    schema = read_schema()
    names = frozenset(schema)
    with connect() as cx:
        return [
            _classify_one(cx, t, c, names, schema[t].row_count, vocabulary_limit)
            for t, c in text_columns()
        ]


# ---------------------------------------------------------------------------------
# 2. Build: only tier A is pre-indexed
# ---------------------------------------------------------------------------------
def build(
    destination: Path | None = None,
    verbose: bool = True,
    vocabulary_limit: int = VOCABULARY_LIMIT,
) -> dict[str, object]:
    cfg = settings()
    destination = Path(destination or cfg.value_index_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    classifications = classify_columns(vocabulary_limit)

    idx = sqlite3.connect(destination)
    idx.executescript(
        """
        PRAGMA journal_mode=OFF;
        CREATE VIRTUAL TABLE values_fts USING fts5(
            value,
            ref UNINDEXED,
            tokenize = 'unicode61 remove_diacritics 2'
        );
        CREATE TABLE columns_meta (
            ref TEXT PRIMARY KEY, "table" TEXT, column_name TEXT,
            tier TEXT, distinct_count INTEGER, indexed_count INTEGER, reason TEXT
        );
        -- Distinct WORDS appearing in tier-B values, without the values
        -- themselves. The egress gate needs to answer "is this word part of a
        -- stored value?" for every outgoing token; for tier-B columns we cannot
        -- store the values (that is the whole point of the tier), but the word
        -- vocabulary is orders of magnitude smaller and bounded. Five million
        -- distinct product names might hold two hundred thousand distinct words.
        CREATE TABLE tier_b_words (word TEXT PRIMARY KEY);
        """
    )

    total = 0
    tier_b_words: set[str] = set()
    with connect() as cx:
        for cl in classifications:
            indexed = 0
            if cl.tier == "B":
                # Harvest the vocabulary, not the values. This is what keeps the
                # egress gate's guarantee complete on columns we deliberately do
                # not index.
                for (v,) in cx.execute(
                    f'SELECT DISTINCT "{cl.column}" FROM "{cl.table}" '
                    f'WHERE "{cl.column}" IS NOT NULL'
                ):
                    tier_b_words.update(re.findall(r"[a-zà-ÿ0-9]{3,}", str(v).lower()))
            if cl.tier == "A":
                values = [
                    str(r[0]).strip()
                    for r in cx.execute(
                        f'SELECT DISTINCT "{cl.column}" FROM "{cl.table}" '
                        f'WHERE "{cl.column}" IS NOT NULL'
                    )
                    if r[0] is not None and str(r[0]).strip()
                ]
                if values:
                    idx.executemany(
                        "INSERT INTO values_fts (value, ref) VALUES (?, ?)",
                        ((v, cl.ref) for v in values),
                    )
                    indexed = len(values)
                    total += indexed
                    if verbose:
                        print(f"  A  {cl.ref:<50} {indexed:>7}", flush=True)
            idx.execute(
                'INSERT INTO columns_meta (ref,"table",column_name,tier,distinct_count,'
                "indexed_count,reason) VALUES (?,?,?,?,?,?,?)",
                (cl.ref, cl.table, cl.column, cl.tier, cl.distinct, indexed, cl.reason),
            )

    idx.executemany("INSERT INTO tier_b_words (word) VALUES (?)", ((w,) for w in tier_b_words))
    idx.execute("INSERT INTO values_fts(values_fts) VALUES('optimize')")
    idx.commit()
    idx.close()

    per_tier = {t: sum(1 for c in classifications if c.tier == t) for t in ("A", "B", "C")}
    stats_out: dict[str, object] = {
        "vocabulary_limit": vocabulary_limit,
        "columns_examined": len(classifications),
        "tier_A_indexed": per_tier["A"],
        "tier_B_on_demand": per_tier["B"],
        "tier_C_excluded": per_tier["C"],
        "values_indexed": total,
        "tier_b_words": len(tier_b_words),
        "size_mb": round(destination.stat().st_size / 1048576, 2),
    }

    report = destination.with_name("column_classification.json")
    report.write_text(
        json.dumps(
            {
                "thresholds": {"vocabulary": vocabulary_limit, "give_up": GIVE_UP_LIMIT},
                "stats": stats_out,
                "columns": [c.__dict__ | {"ref": c.ref} for c in classifications],
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if verbose:
        print(f"\nTier A (indexed)      : {per_tier['A']:>3} columns, {total} values")
        print(f"Tier B (on demand)    : {per_tier['B']:>3} columns")
        print(f"Tier C (excluded)     : {per_tier['C']:>3} columns")
        print(f"Index size            : {stats_out['size_mb']} MB")
        print(f"  -> {destination}\n  -> {report}")
    return stats_out


# ---------------------------------------------------------------------------------
# 3. Lookup
# ---------------------------------------------------------------------------------
def _similarity(a: str, b: str) -> float:
    """How close two strings are, ignoring case.

    Ignoring case is not a detail here. rapidfuzz compares case-sensitively by
    default, and eICU stores values in whatever case the source hospital used —
    `ASPIRIN`, `Female`, `Alive` — while questions are typed in lower case. The
    effect was measured: `alive` against the stored `Alive` scored 0.89 instead of
    1.00, which is below the confidence threshold, so a perfect match was reported
    as an uncertain one and the question was refused.
    """
    left, right = a.lower(), b.lower()
    try:
        from rapidfuzz import fuzz

        return max(fuzz.token_set_ratio(left, right), fuzz.partial_ratio(left, right)) / 100.0
    except ImportError:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, left, right).ratio()


def _fts_queries(mention: str) -> list[str]:
    """Turn a mention into FTS5 queries, strictest first.

    The prefix (`"aspirin"*`) is essential: the database stores
    `ASPIRIN EC 81 MG PO TBEC`, and an exact match would find nothing.

    AND before OR matters. On "acute renal failure", OR alone returns everything
    containing "acute" — hundreds of unrelated values that push the right one out
    of the LIMIT. So we try the conjunction first and only widen if it is empty.
    """
    tokens = [t for t in TOKEN_RE.findall(mention.lower()) if len(t) > 1]
    if not tokens:
        return []
    prefixes = [f'"{t}"*' for t in tokens]
    if len(prefixes) == 1:
        return prefixes
    return [" AND ".join(prefixes), " OR ".join(prefixes)]


def _search_tier_a(
    path: Path, mention: str, columns: list[str] | None, threshold: float
) -> list[FoundValue]:
    queries = _fts_queries(mention)
    if not queries:
        return []

    sql = "SELECT value, ref FROM values_fts WHERE values_fts MATCH ?"
    filters: list[object] = []
    if columns:
        sql += " AND ref IN (%s)" % ",".join("?" * len(columns))
        filters = list(columns)
    sql += " ORDER BY bm25(values_fts) LIMIT 300"

    cx = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        for query in queries:
            try:
                raw = cx.execute(sql, [query, *filters]).fetchall()
            except sqlite3.OperationalError:
                continue
            out = []
            for value, ref in raw:
                table, _, column = ref.partition(".")
                s = _similarity(mention, value)
                if s >= threshold:
                    out.append(FoundValue(value, table, column, round(s, 4), "A"))
            if out:
                return out
    finally:
        cx.close()
    return []


def _search_tier_b(
    path: Path,
    mention: str,
    threshold: float,
    columns: list[str] | None = None,
    scan_cap: int = 5_000,
) -> list[FoundValue]:
    """On-demand resolution, without any prior index.

    We query the database itself with a bounded `LIKE`. That is the mechanism
    that holds up on a column with millions of values: nothing is stored at rest,
    and the cost is capped by `LIMIT`.

    `columns` narrows the scan. It is the only lever that makes this path viable
    in production: without it, an unknown mention triggers a `LIKE` on *every*
    tier-B column.
    """
    sql = "SELECT \"table\", column_name FROM columns_meta WHERE tier='B'"
    params: list[object] = []
    if columns:
        sql += " AND ref IN (%s)" % ",".join("?" * len(columns))
        params = list(columns)

    cx = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
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
                    f'SELECT DISTINCT "{column}" FROM "{table}" '
                    f'WHERE "{column}" LIKE ? LIMIT {scan_cap}',
                    (pattern,),
                ).fetchall()
            except sqlite3.Error:
                continue
            for (v,) in rows:
                if v is None:
                    continue
                s = _similarity(mention, str(v))
                if s >= threshold:
                    out.append(FoundValue(str(v), table, column, round(s, 4), "B"))
    return out


@lru_cache(maxsize=2)
def _vocabulary(path_str: str) -> tuple[tuple[str, str], ...]:
    """Every indexed value with its column. Read once, kept for the process.

    On eICU this is 14,774 pairs — small enough that holding it costs nothing and
    scanning it costs milliseconds. That is what makes the fallback below
    affordable as a general rule rather than a special case.
    """
    cx = sqlite3.connect(f"file:{Path(path_str).as_posix()}?mode=ro", uri=True)
    try:
        return tuple(cx.execute("SELECT value, ref FROM values_fts").fetchall())
    finally:
        cx.close()


def _search_fuzzy(
    path: Path, mention: str, columns: list[str] | None, threshold: float
) -> list[FoundValue]:
    """Last resort: compare the mention against the whole indexed vocabulary.

    Why this exists
    ---------------
    Tiers A and B both *retrieve* candidates with full-text search, which matches
    whole tokens. The similarity scorer then ranks whatever came back. A
    misspelling therefore never gets scored at all: `asspirin` matches no token,
    so nothing is retrieved, so the fuzzy comparison it needed never runs. The
    system answered "no such value" for a word one character away from a real one,
    and the pipeline went on to build a query returning zero rows.

    The fix is not a spelling dictionary and not a list of known drugs — those
    would be hardcoding, and would only ever cover the words somebody thought of.
    It is a full scan of the values the database actually contains, which is the
    only complete list that exists and stays correct when the data changes.

    The bar is deliberately higher than for retrieved candidates. Scanning
    everything means everything is a candidate, so a loose threshold would return
    a confident-looking match for any input at all.

    The scorer is `ratio` — whole string against whole string — and not `WRatio`,
    which was tried first and is wrong for this job. `WRatio` blends in partial
    matching, so a short value contained in the mention scores as if it were the
    same word: `asparatan`, which is not a drug at all, came back as `tan` with
    0.90 confidence, and `asspirin` matched a lab column literally named `s`.
    Edit distance over the full string is the question actually being asked here —
    *is this a misspelling of that* — and it answers 0.53 and 0.27 respectively.
    """
    try:
        from rapidfuzz import fuzz, process
    except ImportError:  # pragma: no cover — rapidfuzz is a declared dependency
        return []

    vocabulary = _vocabulary(str(path))
    if columns:
        allowed = set(columns)
        vocabulary = tuple((v, r) for v, r in vocabulary if r in allowed)
    if not vocabulary:
        return []

    values = [v for v, _ in vocabulary]
    hits = process.extract(
        mention.lower(),
        [v.lower() for v in values],
        scorer=fuzz.ratio,
        limit=20,
        score_cutoff=threshold * 100,
    )

    out: list[FoundValue] = []
    for _value, score, position in hits:
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
    """Find the real values matching `mention`.

    Tier A (pre-indexed) answers first. Tier B is only reached when A returns
    nothing: it is the expensive path, not taken without reason. The fuzzy scan
    is reached only when both found nothing at all, which is the case a
    misspelling produces.
    """
    path = Path(source or settings().value_index_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run: python scripts/build_value_index.py")

    results = _search_tier_a(path, mention, columns, threshold)
    if not results:
        results = _search_tier_b(path, mention, threshold, columns)
    if not results:
        results = _search_fuzzy(path, mention, columns, fuzzy_threshold)

    # Ties are common and the tie-break decides the answer. `aspirin` is stored
    # verbatim in four columns at once, all scoring 1.0, so whichever the FTS
    # engine happened to return first became the resolution — `admissiondrug` for
    # a question about medication, which is wrong in a way nothing downstream can
    # notice.
    #
    # The glossary already declares which column is canonical for a concept, by
    # listing it first: `drug` names `medication.drugname` ahead of
    # `admissiondrug.drugname`. That declared order is the scope passed in here,
    # so ranking by position in it settles ties with the business's own answer
    # rather than with a coincidence. Shortest value last, to keep a whole value
    # ahead of a long one that merely contains the mention.
    rank = {ref: i for i, ref in enumerate(columns or [])}
    results.sort(key=lambda v: (-v.score, rank.get(v.ref, len(rank) + 1), len(v.value)))

    # One suggestion per column: ten variants of aspirin help nobody.
    seen: set[str] = set()
    unique = []
    for r in results:
        if r.ref in seen:
            continue
        seen.add(r.ref)
        unique.append(r)
    return unique[:limit]


@lru_cache(maxsize=1)
def exact_values(source: Path | None = None) -> frozenset[str]:
    """All indexed values, normalised, for a membership test.

    Settles one precise conflict: "Other Hospital" and "Neuro ICU" are covered by
    the glossary ("hospital", "icu" are synonyms there) and would be taken for
    concepts. Yet they are values stored verbatim in `patient.unitadmitsource`
    and `patient.unittype`.
    """
    path = Path(source or settings().value_index_path)
    if not path.exists():
        return frozenset()
    cx = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
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


def stats(source: Path | None = None) -> dict[str, object]:
    path = Path(source or settings().value_index_path)
    if not path.exists():
        return {}
    cx = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        tiers = dict(cx.execute("SELECT tier, COUNT(*) FROM columns_meta GROUP BY tier").fetchall())
        (values,) = cx.execute("SELECT COUNT(*) FROM values_fts").fetchone()
        top = cx.execute(
            "SELECT ref, indexed_count FROM columns_meta WHERE tier='A' "
            "ORDER BY indexed_count DESC LIMIT 12"
        ).fetchall()
        tier_b = cx.execute(
            "SELECT ref, distinct_count FROM columns_meta WHERE tier='B' ORDER BY distinct_count DESC"
        ).fetchall()
    finally:
        cx.close()
    return {
        "tiers": tiers,
        "values_indexed": values,
        "size_mb": round(path.stat().st_size / 1048576, 2),
        "top_A": top,
        "tier_B": tier_b,
    }
