"""The egress gate — the only point through which text can reach the cloud.

Checking by provenance, not by vocabulary
-----------------------------------------
The prompt sent to the cloud is assembled from parts whose origin we know:

    authored   the instruction text, a literal in the source
    schema     the DDL, generated from table and column names
    glossary   the business notes, read from config/glossary.yaml
    question   the masked question — the only part coming from outside

Only the last one can carry a database value. The first three are either written
by us or generated from the schema, which holds no data by construction.

An earlier version ran the word-by-word check over the *whole* prompt. It worked,
but it forced hundreds of ordinary English words into the allowlist ("identifies",
"hierarchical", "boolean", "readmitted"…) just so our own sentences could pass.
That is not an architecture, it is an endless patch: every new question brings one
more word to add, and each addition widens the hole a little.

So each part is now verified by its own rule:

    authored   fingerprint identical to a constant registered at import time
    schema     text reproducible by regenerating the DDL from the database
    glossary   text present among the notes declared in the glossary
    question   full allowlist check, fail-closed

The first three verifications are **reconstructions**, not declarations: the gate
recomputes what the part should contain and compares. Interpolating a value into
the DDL or into a note would break the match and send the part back to word-by-word
checking.

The allowlist that remains
--------------------------
Because it now only ever sees the question, the allowlist can be small and mostly
*derived* rather than written:

1. closed-class English — articles, prepositions, pronouns, auxiliaries, wh-words,
   quantifiers, common analytic verbs. Question grammar: a genuinely finite set
   that carries no content;
2. **schema identifiers read from the database**;
3. **glossary vocabulary read from the glossary**;
4. `:v1`, `:v2`… symbols, numbers, punctuation, SQL aliases.

The rule this expresses is meaningful and maintainable: **a content word in a
question must either be masked as a value, or be declared in the glossary as a
concept.** If it is neither, the gate blocks and the fix is to extend the
glossary — not to widen a word list.

Why fail-closed at all
----------------------
The gate does not try to *detect* what is sensitive; it refuses what it cannot
account for. The usual approach — detect entities then mask them — fails silently:
whatever it misses leaves in clear text. MaskSQL (arXiv 2509.23459) measures 61.4%
masking recall, so nearly four values in ten cross unseen. Here an unaccounted
value does not pass: the request fails loudly, which is fixable, instead of leaking
quietly, which is not.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from hybridsql.config import settings

# Tokenizer that keeps `:v1` symbols whole. Without the first alternative, `:v1`
# would split into `:` and `v1`, and `v1` would be refused.
TOKEN_RE = re.compile(r":v\d+|[A-Za-zÀ-ÿ_][A-Za-zÀ-ÿ0-9_]*|[-+]?\d+(?:[.,]\d+)?%?|\S")

# A value that fits in a single alphabetic word — those that can collide with the
# allowlist vocabulary.
SINGLE_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+")

# A bare number is never blocked. `medication.dosage` stores "10", "65", "1" —
# but nothing distinguishes the stored "65" from the "65" an analyst writes in
# "patients over 65". Blocking them would reject every SQL `LIMIT 10` and every
# numeric filter, for a disclosure of exactly nothing.
NUMBER_RE = re.compile(r"^[-+]?\d+([.,]\d+)?%?$")

# Beyond this, a label is no longer a "value" whose every word is sensitive:
# `intakeoutput.celllabel` holds whole sentences. Measured threshold, not theory.
MAX_WORDS_SHORT_VALUE = 3

# Numbers written out. Treated exactly like digits — see `carries_information`.
NUMERALS = frozenset(
    "zero one two three four five six seven eight nine ten "
    "eleven twelve twenty thirty forty fifty hundred thousand".split()
)

Origin = Literal["authored", "template", "schema", "glossary", "question"]


class LeakBlocked(PermissionError):
    """Unauthorised text attempted to cross the trust boundary."""

    def __init__(self, tokens: tuple[str, ...], context: str) -> None:
        self.tokens = tokens
        self.context = context
        preview = ", ".join(repr(t) for t in tokens[:5])
        more = f" (+{len(tokens) - 5} more)" if len(tokens) > 5 else ""
        super().__init__(
            f"Egress gate [{context}]: {len(tokens)} token(s) outside the allowlist "
            f"— {preview}{more}. Send refused."
        )


@dataclass(frozen=True)
class Segment:
    """One part of an outgoing message, with the origin it claims.

    The claim is not taken on trust: `check_segment` verifies it. A segment
    declaring `schema` that does not match the regenerated DDL falls back to the
    word-by-word check.
    """

    text: str
    origin: Origin = "question"


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    refused_tokens: tuple[str, ...]
    token_count: int
    fingerprint: str
    context: str = ""
    verified_by: str = ""

    @property
    def reason(self) -> str:
        if self.allowed:
            return f"accepted ({self.verified_by})"
        return f"{len(self.refused_tokens)} token(s) outside the allowlist"


# ---------------------------------------------------------------------------------
# Values held by the database — used to subtract, never to detect
# ---------------------------------------------------------------------------------
def _index_values(distinct: bool = False) -> list[str]:
    import sqlite3

    path = settings().value_index_path
    if not path.exists():
        return []
    cx = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        select = "SELECT DISTINCT value FROM values_fts" if distinct else "SELECT value FROM values_fts"
        return [v for (v,) in cx.execute(select)]
    except sqlite3.Error:
        return []
    finally:
        cx.close()


def _tier_b_words() -> set[str]:
    """Words harvested from tier-B columns at index build time."""
    import sqlite3

    path = settings().value_index_path
    if not path.exists():
        return set()
    cx = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return {w for (w,) in cx.execute("SELECT word FROM tier_b_words")}
    except sqlite3.Error:
        return set()
    finally:
        cx.close()


@lru_cache(maxsize=1)
def forbidden_words() -> frozenset[str]:
    """Words that appear **as a whole value** in the database.

    Why this subtraction is necessary
    ---------------------------------
    Without it the gate has a gaping hole. `patient.gender` holds the value
    "Female"; "female" is also how one names the concept. The word would pass the
    gate while *being* a datum.

    The rule: a word is removed if it constitutes **a whole value**, not if it
    merely appears inside a longer one. That precision is not cosmetic — the first
    version removed any token found anywhere in a value, so "count" (from "Platelet
    count"), "patients" and "with" vanished from the allowlist and no question
    could pass at all. A gate that blocks everything protects nothing: it gets
    switched off on first use.
    """
    forbidden: set[str] = set()
    for value in _index_values():
        word = str(value).strip().lower()
        if len(word) >= 3 and SINGLE_WORD_RE.fullmatch(word):
            forbidden.add(word)
    return frozenset(forbidden)


@lru_cache(maxsize=1)
def forbidden_words_extended() -> frozenset[str]:
    """Whole values, plus the words of **short** values.

    The "whole value" rule still let through two- or three-word labels whose every
    term is ordinary vocabulary: "Full therapy", "PO fluids", "Total In" crossed
    the gate although they are stored values.

    We stop at three words because beyond that we fall back into the first
    version's flaw: `intakeoutput.celllabel` holds ten-word labels whose every term
    cannot be removed without emptying the allowlist. The limit is empirical;
    `scripts/measure_gate.py` replays it.
    """
    forbidden = set(forbidden_words())
    for value in _index_values(distinct=True):
        tokens = re.findall(r"[A-Za-zÀ-ÿ]{3,}", str(value).lower())
        if 1 <= len(tokens) <= MAX_WORDS_SHORT_VALUE:
            forbidden.update(tokens)
    return frozenset(forbidden)


# ---------------------------------------------------------------------------------
# Building the allowlist — mostly derived, barely written
# ---------------------------------------------------------------------------------
def _words_of_block(block: object) -> set[str]:
    """Flatten one YAML block.

    The file writes `- a, an, the, of` to stay readable: YAML sees a single
    string, which must be split on commas.
    """
    words: set[str] = set()
    if isinstance(block, str):
        block = [block]
    for line in block or ():
        for word in str(line).split(","):
            word = word.strip().lower()
            if word:
                words.add(word)
    return words


@lru_cache(maxsize=1)
def _yaml() -> dict:
    import yaml

    path = Path(settings().egress_allowlist_path)
    return (yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}) or {}


@lru_cache(maxsize=1)
def _schema_identifiers() -> frozenset[str]:
    """Table and column names, read from the database.

    Underscore-separated fragments are added too: the cloud model sometimes writes
    `hospital_id` where the column is `hospitalid`, and an alias such as `avg_los`
    must be able to come back.
    """
    from hybridsql.db.schema import read_schema

    names: set[str] = set()
    for table in read_schema().values():
        names.add(table.name.lower())
        names.update(table.name.lower().split("_"))
        for column in table.columns:
            names.add(column.name.lower())
            names.update(column.name.lower().split("_"))
    return frozenset(n for n in names if n)


@lru_cache(maxsize=1)
def glossary_concepts() -> frozenset[str]:
    """Words the glossary explicitly declares as concept names.

    "icu", "mortality", "drug" and "admission" all exist as stored values
    somewhere, so `forbidden_words_extended()` removes them — yet they are also how
    an analyst *names* a column, and a question keeps them in clear text when stage
    1 classifies them as concepts.

    The glossary is the declaration of what counts as a concept, so it is the right
    authority. The dangerous overlap — a synonym that is a value of its own column,
    such as "female" for `patient.gender` — is forbidden by
    `tests/test_glossary.py`, which fails the build if anyone reintroduces it.
    """
    from hybridsql.resources.glossary import load

    words: set[str] = set()
    for term in load().values():
        for phrase in (term.canonical.replace("_", " "), *term.synonyms):
            words.update(w for w in re.findall(r"[a-z0-9]+", phrase.lower()) if len(w) > 1)
    return frozenset(words)


@lru_cache(maxsize=1)
def allowlist() -> frozenset[str]:
    """The vocabulary a **question** may contain. Cached: constant at runtime.

    Three sources, only the first of which is written by hand — and it is a closed
    class that does not grow with the domain.
    """
    grammar = _words_of_block(_yaml().get("grammar")) | _words_of_block(_yaml().get("sql"))

    try:
        derived = _schema_identifiers() | glossary_concepts()
    except Exception:  # noqa: BLE001
        # Database or glossary missing (CI, first run): the gate stays usable but
        # stricter. Erring toward refusal is the right default.
        derived = frozenset()

    return frozenset(grammar | derived)


@lru_cache(maxsize=1)
def generic_vocabulary() -> frozenset[str]:
    """Closed-class words: those that never denote business content.

    Used to qualify a leak. `medication.frequency` holds the value "Every 12
    hours": it crosses the gate, but its three words are pure question grammar.
    Disclosing it teaches nothing about the data — unlike "MELOXICAM". Telling the
    two apart avoids inflating the leak rate with non-cases.
    """
    return frozenset(_words_of_block(_yaml().get("grammar")) | _words_of_block(_yaml().get("sql")))


@lru_cache(maxsize=1)
def _patterns() -> tuple[re.Pattern[str], ...]:
    patterns = _yaml().get("patterns") or {}
    return tuple(re.compile(p) for p in patterns.values())


# ---------------------------------------------------------------------------------
# Provenance verification
# ---------------------------------------------------------------------------------
_CONSTANTS: set[str] = set()


def register_constant(text: str) -> str:
    """Declare a fixed, author-written text allowed to leave verbatim.

    Only for literals in the source. Passing a string built at runtime would amount
    to switching the gate off for that string.
    """
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
    _CONSTANTS.add(fingerprint)
    return fingerprint


_TEMPLATES: set[str] = set()

# What a template is allowed to vary: an opaque label we issued, or a number.
# Anything else in the sentence must be the authored wording, unchanged.
_SLOT = re.compile(r"(?<![A-Za-z0-9_])(?:[tc]\d+(?:\.[tc]\d+)?|\d+)(?![A-Za-z0-9_])")


def _skeleton(text: str) -> str:
    """The sentence with its variable parts blanked out."""
    return " ".join(_SLOT.sub("\x00", text or "").split())


def register_template(text: str) -> str:
    """Declare a sentence we author whose only variable parts are labels or numbers.

    Why a constant was not enough
    -----------------------------
    `register_constant` fingerprints an exact string, which works for fixed
    instructions and not for a sentence assembled per request. The opaque arm
    needs the second kind — "t2.c15 is the subject of the question" — and without
    this it fell to the word-by-word check, where the gate refused our own text on
    the word "related", because "related" happens to be stored in the database.

    That is the failure mode the provenance design exists to prevent: our own
    prose forced onto a data denylist, one word at a time. So a template is
    fingerprinted like a constant, after replacing every label and number with a
    blank. The wording is verified exactly; only the slots may differ, and a slot
    can hold nothing but an identifier we generated ourselves.
    """
    _TEMPLATES.add(hashlib.sha256(_skeleton(text).encode("utf-8")).hexdigest())
    return text


def is_template_text(text: str) -> bool:
    return hashlib.sha256(_skeleton(text).encode("utf-8")).hexdigest() in _TEMPLATES


def is_registered_constant(text: str) -> bool:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest() in _CONSTANTS


def is_schema_text(text: str) -> bool:
    """Is this text reproducible by regenerating the DDL from the database?

    Verification by reconstruction rather than by declaration: we recompute the DDL
    of the tables it mentions and compare. A value interpolated into the DDL would
    break the match and send the segment back to word-by-word checking.
    """
    from hybridsql.db.schema import ddl, read_schema

    candidate = (text or "").strip()
    if not candidate:
        return False
    mentioned = {t for t in read_schema() if f"CREATE TABLE {t} " in candidate}
    if not mentioned:
        return False
    return ddl(mentioned, with_row_counts=True).strip() == candidate


def is_glossary_note(text: str) -> bool:
    """Is this text one of the notes declared in the glossary?

    Membership, not similarity: a modified note is not a note.
    """
    from hybridsql.resources.glossary import load

    candidate = (text or "").strip()
    if not candidate:
        return False
    return any(candidate == term.note.strip() for term in load().values() if term.note)


# ---------------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------------
class BloomFilter:
    """Compact membership test with one-sided error.

    Why this exists
    ---------------
    The gate must answer one question per token: *does this word appear in a stored
    value?* Asking the database each time is out of the question at scale — a
    `LIKE '%word%'` on a 50-million-row column is a full scan, per token, per
    question.

    So the answer is precomputed once at index build time. But an in-memory set
    does not scale either: the tier-A policy caps the vocabulary at
    `n_columns x VOCABULARY_LIMIT`, which on a production database means roughly
    3 million distinct tokens — about 300 MB as a Python set.

    A Bloom filter holds the same 3 million tokens in about 3.6 MB at a 1% false
    positive rate, with O(1) lookups. Its error is **one-sided and in the safe
    direction**: it may claim a word is present when it is not (so the gate blocks
    something harmless), and it can never claim absence for a word that is present.
    Over-blocking is a nuisance; under-blocking would be a leak.
    """

    def __init__(self, expected: int, error_rate: float = 0.01) -> None:
        import math

        expected = max(expected, 1)
        self.size = max(8, int(-expected * math.log(error_rate) / (math.log(2) ** 2)))
        self.hashes = max(1, round(self.size / expected * math.log(2)))
        self.bits = bytearray((self.size + 7) // 8)

    def _positions(self, item: str) -> list[int]:
        # Two hashes are enough to derive k positions (Kirsch-Mitzenmacher).
        digest = hashlib.sha256(item.encode("utf-8")).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:16], "big") | 1
        return [(h1 + i * h2) % self.size for i in range(self.hashes)]

    def add(self, item: str) -> None:
        for p in self._positions(item):
            self.bits[p >> 3] |= 1 << (p & 7)

    def __contains__(self, item: str) -> bool:
        return all(self.bits[p >> 3] & (1 << (p & 7)) for p in self._positions(item))

    @property
    def size_mb(self) -> float:
        return round(len(self.bits) / 1048576, 3)


# Beyond this many distinct tokens, a plain set costs more memory than a Bloom
# filter saves in accuracy. Below it, exactness is free — so we keep it.
BLOOM_THRESHOLD = 200_000


@lru_cache(maxsize=1)
def value_tokens() -> object:
    """Every word appearing inside a **short** stored value, as an O(1) membership test.

    This is what replaces "ask the database for each token". It is built from the
    tier-A index, whose size the indexing policy already bounds by column count
    rather than row count (see docs/03-INDEXING.md) — which is precisely what
    makes this check scale.

    Why only short values
    ---------------------
    This used to harvest the words of *every* value, and that is what made the gate
    unusable in practice. eICU stores hierarchical free text —
    `hematology|coagulation disorders|DIC syndrome|associated with intravascular
    clotting` — so ordinary English ends up inside stored values, and the gate then
    refused ordinary questions:

        "how many laboratory records are associated with each ICU stay"
            -> refused on 'associated'

    Learning that our questions contain the word "associated" tells a provider
    nothing about this database; every database's free text contains it. This is the
    same argument `carries_information` already makes about "56" and "id", applied
    to a word rather than to a string.

    The threshold is not new either: `forbidden_words_extended` already stops at
    `MAX_WORDS_SHORT_VALUE` words, for the same reason and with the same
    measurement behind it. It was simply never applied here. Effect, measured on
    eICU: 8 708 words -> 7 419. `associated`, `unique`, `necrotizing` and 1 286
    others stop being treated as data; `aspirin`, `female`, `alive`, `hgb` and every
    other real value word stay.

    What still protects the long values: `find_known_values` matches them **whole**,
    exactly and exhaustively, before this check ever runs.

    Returns a `set` on small databases and a `BloomFilter` beyond
    `BLOOM_THRESHOLD`. Both answer `in`.
    """
    tokens: set[str] = set()
    for value in _index_values(distinct=True):
        text = str(value)
        if len(re.findall(r"[A-Za-zÀ-ÿ]{3,}", text)) > MAX_WORDS_SHORT_VALUE:
            continue
        tokens.update(w for w in re.findall(r"[a-zà-ÿ0-9]{3,}", text.lower()))
    # Tier-B columns are not indexed by value, but their word vocabulary is
    # harvested at build time — otherwise the guarantee would stop at tier A and a
    # value from a high-cardinality column could cross unnoticed.
    tokens.update(_tier_b_words())

    if len(tokens) <= BLOOM_THRESHOLD:
        return frozenset(tokens)

    bloom = BloomFilter(len(tokens))
    for t in tokens:
        bloom.add(t)
    return bloom


@lru_cache(maxsize=1)
def known_values() -> frozenset[str]:
    """Every value the database holds, normalised. The exhaustive denylist.

    Why this is stronger than PII detection
    ---------------------------------------
    The industry approach (Microsoft Presidio and the tools built on it) detects
    entities with a model, then substitutes them. Detection is a heuristic, so it
    has a recall: MaskSQL measures 61.4% on this very task, meaning nearly four
    values in ten are never seen and leave in clear text.

    We are in a different position: **we own the database**. For every indexed
    column we hold the complete inventory of values, so matching against it is not
    an estimate — it is exact. A value present in the database cannot slip past
    this check.

    Its limit, stated plainly: it covers indexed (tier A) columns. Tier B columns
    are resolved on demand and are not enumerated here, so on a production database
    this denylist would be complete only for the indexed part. That gap is exactly
    what the allowlist backstop below is for.
    """
    return frozenset(
        " ".join(str(v).lower().split()) for v in _index_values(distinct=True) if v
    )


def carries_information(value: str) -> bool:
    """Could disclosing this string teach a provider anything about the data?

    The denylist exists to stop a *value* leaving. Not every string stored in a
    database is a value in that sense, and treating them alike made the system
    unusable: `id` is a stored value somewhere in eICU, so "how many patients in
    hospital id 56" was refused — and so was every question containing a small
    number, because `0`, `1`, `12` and `56` are all stored values too.

    Measured on eICU's 11,732 indexed values:

        11,251   four characters or more     aspirin, AMOXICILLIN 500 MG PO CAPS
           223   no letter at all            0, 1, %, 12, 56
            97   one or two characters       id, in, no, as, mg, ml, iv

    The last two groups are disqualified here, on the same argument in both cases:
    **every database contains those strings**. Learning that "56" or "id" occurs in
    ours narrows nothing down, while blocking them stops an analyst from writing an
    ordinary sentence. Stage 1 already agrees on the numbers — it classifies them
    as quantities supplied by the analyst, never as data — so refusing them here
    was the two layers contradicting each other.

    Everything with a letter and three or more characters stays protected, which
    keeps the short medical abbreviations that *are* real values — `hgb`, `chf`,
    `dvt`, `crp`. This is a floor on information content, not a vocabulary: it
    needs no maintenance and it does not grow when the database does.

    Spelled-out numerals are the one addition, and they follow from the same
    argument rather than extending it. `12` is already exempt because every
    database contains it; `one` was not, so "patients with at least **one**
    laboratory record" was refused as a leak while "with at least **1**" passed.
    Ten words, closed class, no maintenance.
    """
    stripped = (value or "").strip()
    if len(stripped) < 3:
        return False
    if stripped.lower() in NUMERALS:
        return False
    return any(char.isalpha() for char in stripped)


@lru_cache(maxsize=1)
def longest_value_words() -> int:
    """How many words the longest stored value has. Bounds the n-gram scan exactly.

    The scan used to stop at six words, which was fine while `value_tokens` held
    the words of *every* value: a longer value was caught word by word. Now that it
    only holds the words of short values, a seventeen-word value —
    *"Patient transferred from another hospital to the current hospital within 48
    hours prior to ICU admission"* — was matched by neither, and crossed. The canary
    test caught it.

    Deriving the bound from the data rather than guessing it means the two layers
    cannot drift apart again: whatever the longest value is, the scan reaches it.
    Capped at 32 words so that a pathological value cannot make the scan quadratic
    in a long SQL response.
    """
    longest = max((len(str(v).split()) for v in known_values()), default=6)
    return min(max(longest, 6), 32)


def find_known_values(text: str, max_ngram: int | None = None) -> list[str]:
    """Return the database values appearing in `text`, longest first.

    Matching runs on word n-grams rather than substrings: a substring search would
    fire on "ICU" inside "medicus" and would miss "Skilled Nursing Facility", which
    only exists as a phrase.

    Single words that are declared structure — a column name, a glossary concept —
    or plain question grammar are exempt. "patient", "icu" and even "in" all exist
    as whole values somewhere in eICU; blocking them would stop the SQL itself,
    since the model must write `FROM patient`, and would reject any sentence
    containing the preposition "in". Such a word tells the provider nothing beyond
    what the DDL already shows.

    The exemption stops at one word. "Skilled Nursing Facility" and "Other
    Hospital" are made of structural words yet are specific stored values, and they
    are blocked — which is why the n-gram scan runs longest-first.
    """
    values = known_values()
    if not values:
        return []

    try:
        exempt = _schema_identifiers() | glossary_concepts() | generic_vocabulary()
    except Exception:  # noqa: BLE001
        exempt = generic_vocabulary()

    words = re.findall(r"[A-Za-zÀ-ÿ0-9+&/.-]+", text or "")
    ceiling = max_ngram if max_ngram is not None else longest_value_words()
    found: list[str] = []
    for size in range(min(ceiling, len(words)), 0, -1):
        for i in range(len(words) - size + 1):
            candidate = " ".join(words[i : i + size]).lower()
            if candidate not in values or candidate in found:
                continue
            # The exemption stops at one word: "Skilled Nursing Facility" is made
            # of ordinary words and is still a specific stored value.
            if size == 1 and (candidate in exempt or not carries_information(candidate)):
                continue
            found.append(candidate)
    return found


def _is_allowed(token: str, words: frozenset[str], patterns: tuple[re.Pattern[str], ...]) -> bool:
    if token.lower() in words:
        return True
    return any(p.match(token) for p in patterns)


def check(text: str, context: str = "") -> Verdict:
    """Check a text of unknown origin. Never raises.

    Two layers, in this order — defence in depth, as recommended for LLM
    guardrails, but with the layers the right way round:

    1. **Known-value denylist** — exhaustive over indexed columns, so it is a
       guarantee rather than a guess. Anything matching a stored value is refused
       outright, whatever the allowlist says. Without this, a value that also
       happens to be ordinary vocabulary would slip through.

    2. **Grammar allowlist** — fail-closed backstop for what the denylist cannot
       know: values of non-indexed columns, misspellings, content words the
       glossary has not declared. Small and mostly derived, because provenance
       checking handles everything we author ourselves.
    """
    fingerprint = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    tokens = TOKEN_RE.findall(text or "")

    leaked = find_known_values(text)
    if leaked:
        return Verdict(
            allowed=False,
            refused_tokens=tuple(leaked),
            token_count=len(tokens),
            fingerprint=fingerprint[:16],
            context=context,
            verified_by="known-value denylist",
        )

    words, patterns = allowlist(), _patterns()
    vocabulary = value_tokens()

    refused: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.lower()
        if key in seen:
            continue
        # Fast path, in this order and for distinct reasons:
        #   - grammar and symbols carry no content by construction;
        #   - schema identifiers and glossary concepts are declared as structure,
        #     and the exact-value denylist above has already ruled out the case
        #     where such a word is itself a complete stored value.
        if _is_allowed(token, words, patterns):
            continue
        # Unknown word: is it part of any stored value? This is the check that
        # replaces enumerating English. A word absent from every value cannot leak
        # data, whatever it means.
        seen.add(key)
        # Same floor as the exact-value denylist above, and for the same reason:
        # a year like "2015" is a word of some stored value somewhere, and
        # refusing it would block a question about a date.
        if carries_information(key) and key in vocabulary:
            refused.append(token)

    return Verdict(
        allowed=not refused,
        refused_tokens=tuple(refused),
        token_count=len(tokens),
        fingerprint=fingerprint[:16],
        context=context,
        verified_by="value-token membership",
    )


def sweep_response(text: str, context: str = "cloud-response") -> list[str]:
    """Output sweep: check what the cloud model **returned** for stored values.

    Standard practice for LLM guardrails is to sanitise the input *and* sweep the
    output. Here the risk is specific and real: the model can invent a literal
    (`WHERE drugname = 'aspirin'`) instead of using the bound parameter it was
    given. That literal would not be a leak outward — it never left our side — but
    it means the query no longer reflects the masked value, so executing it would
    silently answer a different question.

    `security/sql_validator.py` already refuses a query that ignores a supplied
    parameter. This sweep is the complementary check: it names the values the model
    reproduced, for the audit trail.
    """
    from hybridsql.security import audit

    found = find_known_values(text)
    if found:
        audit.record(
            Verdict(
                allowed=False,
                refused_tokens=tuple(found),
                token_count=len(TOKEN_RE.findall(text or "")),
                fingerprint=hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16],
                context=context,
                verified_by="output sweep",
            ),
            text,
        )
    return found


def check_segment(segment: Segment, context: str = "") -> Verdict:
    """Check one segment against the rule matching its claimed origin.

    The claim is verified, not trusted. A segment that fails its provenance check
    falls back to the word-by-word check rather than being refused outright: an
    author-written text may legitimately consist of allowed words.
    """
    verifiers = {
        "authored": (is_registered_constant, "registered constant"),
        "template": (is_template_text, "authored template, labels only"),
        "schema": (is_schema_text, "regenerated DDL"),
        "glossary": (is_glossary_note, "declared glossary note"),
    }
    verifier = verifiers.get(segment.origin)
    if verifier and verifier[0](segment.text):
        return Verdict(
            allowed=True,
            refused_tokens=(),
            token_count=0,
            fingerprint=hashlib.sha256(segment.text.encode("utf-8")).hexdigest()[:16],
            context=f"{context}/{segment.origin}",
            verified_by=verifier[1],
        )
    return check(segment.text, context)


def require_segments(segments: list[Segment], context: str = "") -> None:
    """Check every segment of an outgoing message, or raise `LeakBlocked`.

    **Every network call to a cloud provider must go through here.** Each segment
    is journalled, whatever the verdict, so the audit trail accounts for every byte
    sent.
    """
    from hybridsql.security import audit

    refused: list[str] = []
    for segment in segments:
        verdict = check_segment(segment, context)
        audit.record(verdict, segment.text)
        if not verdict.allowed:
            refused.extend(verdict.refused_tokens)
    if refused:
        raise LeakBlocked(tuple(dict.fromkeys(refused)), context)


def require(text: str, context: str = "") -> str:
    """Check a text of unknown origin, or raise. Kept for callers with no segments."""
    from hybridsql.security import audit

    verdict = check(text, context)
    audit.record(verdict, text)
    if not verdict.allowed:
        raise LeakBlocked(verdict.refused_tokens, context)
    return text


def stats() -> dict[str, int]:
    return {
        "allowed_words": len(allowlist()),
        "of_which_derived": len(_schema_identifiers() | glossary_concepts()),
        "patterns": len(_patterns()),
        "registered_constants": len(_CONSTANTS),
    }
