"""The egress gate — the only point through which text can reach the cloud."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from nl2sql.config import settings

# Keeps `:v1` whole; without the first alternative it splits and `v1` is refused.
TOKEN_RE = re.compile(r":v\d+|[A-Za-zÀ-ÿ_][A-Za-zÀ-ÿ0-9_]*|[-+]?\d+(?:[.,]\d+)?%?|\S")
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+")
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Beyond three words a label is a sentence, not a value: `intakeoutput.celllabel`
# holds ten-word strings whose every term is ordinary English.
MAX_WORDS_SHORT_VALUE = 3

# Written-out numbers, treated exactly like digits — see `carries_information`.
NUMERALS = frozenset(
    "zero one two three four five six seven eight nine ten "
    "eleven twelve twenty thirty forty fifty hundred thousand".split()
)

DDL_WORDS = frozenset(
    "create table int real text blob numeric rows primary key foreign references".split()
)

# Three declaration forms, and nothing else. The LIKE form carries the same
# information as the value form - a symbol and a real column - so it is verified
# the same way, by provenance. Letting it fall through to the word check instead
# would have put "pattern" in the allowlist, which is the hole that file warns
# against: authored scaffolding must be verified by regeneration, not vocabulary.
PARAM_LINE_RE = re.compile(
    r"^\s*:v\d+ = (?:a value of ([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)"
    r"|a LIKE pattern for ([A-Za-z0-9_]+)\.([A-Za-z0-9_]+) - match it with "
    r"LIKE :v\d+, never with ="
    r"|a value of ([tc]\d+)\.([tc]\d+)|a number given by the analyst)\s*$"
)
LABEL_RE = re.compile(r"^[tc]\d+$")

Origin = Literal["authored", "template", "schema", "params", "opaque", "glossary", "question"]


class LeakBlocked(PermissionError):
    """Unauthorised text tried to cross the trust boundary."""

    def __init__(self, tokens: tuple[str, ...], context: str) -> None:
        self.tokens, self.context = tokens, context
        preview = ", ".join(repr(t) for t in tokens[:5])
        more = f" (+{len(tokens) - 5} more)" if len(tokens) > 5 else ""
        super().__init__(
            f"Egress gate [{context}]: {len(tokens)} token(s) outside the allowlist "
            f"— {preview}{more}. Send refused."
        )


@dataclass(frozen=True)
class Segment:
    """One part of an outgoing message. The claimed origin is verified, not trusted."""

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


def _fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------
#  What the database holds
# ---------------------------------------------------------------------------------
def _values(distinct: bool = False) -> list[str]:
    from nl2sql.db.values import query_index

    select = "SELECT DISTINCT value FROM values_fts" if distinct else "SELECT value FROM values_fts"
    return [v for (v,) in query_index(select)]


@lru_cache(maxsize=1)
def known_values() -> frozenset[str]:
    """Every indexed value, normalised: an exact denylist, not a detector's guess."""
    return frozenset(" ".join(str(v).lower().split()) for v in _values(distinct=True) if v)


def carries_information(value: str) -> bool:
    """Could disclosing this string teach a provider anything?"""
    text = (value or "").strip()
    if len(text) < 3 or text.lower() in NUMERALS:
        return False
    return any(c.isalpha() for c in text)


class BloomFilter:
    """Membership test whose error is one-sided: it over-blocks, never under-blocks."""

    def __init__(self, expected: int, error_rate: float = 0.01) -> None:
        expected = max(expected, 1)
        self.size = max(8, int(-expected * math.log(error_rate) / (math.log(2) ** 2)))
        self.hashes = max(1, round(self.size / expected * math.log(2)))
        self.bits = bytearray((self.size + 7) // 8)

    def _positions(self, item: str) -> list[int]:
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


BLOOM_THRESHOLD = 200_000


@lru_cache(maxsize=1)
def value_tokens() -> object:
    """Words inside **short** stored values, as an O(1) test."""
    from nl2sql.db.values import query_index

    tokens: set[str] = set()
    for value in _values(distinct=True):
        text = str(value)
        if len(WORD_RE.findall(text)) > MAX_WORDS_SHORT_VALUE:
            continue
        tokens.update(re.findall(r"[a-zà-ÿ0-9]{3,}", text.lower()))
    tokens.update(w for (w,) in query_index("SELECT word FROM tier_b_words"))

    if len(tokens) <= BLOOM_THRESHOLD:
        return frozenset(tokens)
    bloom = BloomFilter(len(tokens))
    for token in tokens:
        bloom.add(token)
    return bloom


@lru_cache(maxsize=1)
def longest_value_words() -> int:
    """Bound the n-gram scan by the data, so the two layers cannot drift apart."""
    longest = max((len(str(v).split()) for v in known_values()), default=6)
    return min(max(longest, 6), 32)


# ---------------------------------------------------------------------------------
#  The allowlist — mostly derived, barely written
# ---------------------------------------------------------------------------------
def _flatten(block: object) -> set[str]:
    """The YAML writes `- a, an, the` to stay readable; split it back on commas."""
    words: set[str] = set()
    for line in ([block] if isinstance(block, str) else block or ()):
        words.update(w.strip().lower() for w in str(line).split(",") if w.strip())
    return words


@lru_cache(maxsize=1)
def _yaml() -> dict:
    import yaml

    path = Path(settings().allowlist_path)
    return (yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}) or {}


@lru_cache(maxsize=1)
def schema_identifiers() -> frozenset[str]:
    """Table and column names, read from the database, plus their `_` fragments."""
    from nl2sql.db.schema import read_schema

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
    """Words the glossary declares as concept names."""
    from nl2sql.nlp.glossary import load

    words: set[str] = set()
    for term in load().values():
        for phrase in (term.canonical.replace("_", " "), *term.synonyms):
            words.update(w for w in re.findall(r"[a-z0-9]+", phrase.lower()) if len(w) > 1)
    return frozenset(words)


@lru_cache(maxsize=1)
def generic_vocabulary() -> frozenset[str]:
    """Closed-class words: they never denote business content."""
    return frozenset(_flatten(_yaml().get("grammar")) | _flatten(_yaml().get("sql")))


@lru_cache(maxsize=1)
def allowlist() -> frozenset[str]:
    """The vocabulary a question may contain. Only the grammar block is hand-written."""
    try:
        derived = schema_identifiers() | glossary_concepts()
    except Exception:  # noqa: BLE001 — no database yet: stay stricter, never looser
        derived = frozenset()
    return frozenset(generic_vocabulary() | derived)


@lru_cache(maxsize=1)
def _patterns() -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p) for p in (_yaml().get("patterns") or {}).values())


# ---------------------------------------------------------------------------------
#  Provenance rules
# ---------------------------------------------------------------------------------
_CONSTANTS: set[str] = set()
_TEMPLATES: set[str] = set()
_SLOT = re.compile(r"(?<![A-Za-z0-9_])(?:[tc]\d+(?:\.[tc]\d+)?|\d+)(?![A-Za-z0-9_])")


def register_constant(text: str) -> str:
    """Declare a fixed, author-written literal allowed to leave verbatim."""
    _CONSTANTS.add(_fingerprint(text))
    return text


def register_template(text: str) -> str:
    """Declare an authored sentence whose only variable parts are labels or numbers."""
    _TEMPLATES.add(_fingerprint(" ".join(_SLOT.sub("\x00", text or "").split())))
    return text


def is_constant(text: str) -> bool:
    return _fingerprint(text) in _CONSTANTS


def is_template(text: str) -> bool:
    return _fingerprint(" ".join(_SLOT.sub("\x00", text or "").split())) in _TEMPLATES


def is_schema_text(text: str) -> bool:
    """DDL made only of real identifiers, types and counts — it cannot carry data."""
    identifiers = schema_identifiers()
    if "CREATE TABLE" not in (text or ""):
        return False
    return all(
        token.lower() in DDL_WORDS or token.lower() in identifiers
        for token in IDENTIFIER_RE.findall(text)
    )


def is_opaque_text(text: str) -> bool:
    """Pseudonymised DDL or question: nothing but labels we issued, and grammar."""
    tokens = IDENTIFIER_RE.findall(text or "")
    if not tokens:
        return False
    return all(LABEL_RE.match(t) or t.lower() in DDL_WORDS for t in tokens)


def is_parameter_block(text: str) -> bool:
    """Every line declares one bound parameter against a real column or a label."""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return False
    if lines == ["  (none)"]:
        return True
    known = schema_identifiers()
    for line in lines:
        match = PARAM_LINE_RE.match(line)
        if not match:
            return False
        # Groups 1-2 and 3-4 are the two forms that name a real column, and both
        # must be checked against the schema. Groups 5-6 are the opaque arm's
        # pseudonyms, `t1.c7`, which are labels rather than identifiers and are
        # deliberately not looked up here.
        table = match.group(1) or match.group(3)
        column = match.group(2) or match.group(4)
        if table and (table not in known or column not in known):
            return False
    return True


def is_glossary_note(text: str) -> bool:
    """Membership, not similarity: a modified note is not a note."""
    from nl2sql.nlp.glossary import load

    candidate = (text or "").strip()
    return bool(candidate) and any(
        candidate == term.note.strip() for term in load().values() if term.note
    )


VERIFIERS: dict[str, tuple] = {
    "authored": (is_constant, "registered constant"),
    "template": (is_template, "authored template, labels only"),
    "schema": (is_schema_text, "schema identifiers only"),
    "params": (is_parameter_block, "bound-parameter declarations"),
    "opaque": (is_opaque_text, "opaque labels only"),
    "glossary": (is_glossary_note, "declared glossary note"),
}


# ---------------------------------------------------------------------------------
#  The word check
# ---------------------------------------------------------------------------------
def find_known_values(text: str, max_ngram: int | None = None) -> list[str]:
    """Database values appearing in `text`, longest first."""
    values = known_values()
    if not values:
        return []
    try:
        exempt = schema_identifiers() | glossary_concepts() | generic_vocabulary()
    except Exception:  # noqa: BLE001
        exempt = generic_vocabulary()

    words = re.findall(r"[A-Za-zÀ-ÿ0-9+&/.-]+", text or "")
    ceiling = longest_value_words() if max_ngram is None else max_ngram
    found: list[str] = []
    for size in range(min(ceiling, len(words)), 0, -1):
        for i in range(len(words) - size + 1):
            candidate = " ".join(words[i : i + size]).lower()
            if candidate not in values or candidate in found:
                continue
            if size == 1 and (candidate in exempt or not carries_information(candidate)):
                continue
            found.append(candidate)
    return found


def check(text: str, context: str = "") -> Verdict:
    """Check text of unknown origin. Never raises."""
    tokens = TOKEN_RE.findall(text or "")
    leaked = find_known_values(text)
    if leaked:
        return Verdict(
            False, tuple(leaked), len(tokens), _fingerprint(text)[:16], context,
            "known-value denylist",
        )

    words, patterns, vocabulary = allowlist(), _patterns(), value_tokens()
    refused: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.lower()
        if key in seen or key in words or any(p.match(token) for p in patterns):
            continue
        seen.add(key)
        if carries_information(key) and key in vocabulary:
            refused.append(token)

    return Verdict(
        not refused, tuple(refused), len(tokens), _fingerprint(text)[:16], context,
        "value-token membership",
    )


def check_segment(segment: Segment, context: str = "") -> Verdict:
    """Check one segment against the rule its origin claims."""
    verifier = VERIFIERS.get(segment.origin)
    if verifier and verifier[0](segment.text):
        return Verdict(
            True, (), 0, _fingerprint(segment.text)[:16],
            f"{context}/{segment.origin}", verifier[1],
        )
    return check(segment.text, context)


def verdicts(segments: list[Segment], context: str = "") -> list[dict]:
    """One verdict per segment, in a shape the interface can render."""
    out = []
    for segment in segments:
        verdict = check_segment(segment, context)
        out.append({
            "origin": segment.origin,
            "allowed": verdict.allowed,
            "checked_by": verdict.verified_by,
            "tokens": verdict.token_count,
            "refused": list(verdict.refused_tokens),
            "preview": " ".join(segment.text.split())[:90],
        })
    return out


def require_segments(segments: list[Segment], context: str = "") -> None:
    """Check every segment of an outgoing message, or raise. Every send is journalled."""
    from nl2sql.core.steps import track
    from nl2sql.privacy import audit

    with track("gate", parts=len(segments)) as step:
        refused: list[str] = []
        checked: list[dict] = []
        for segment in segments:
            verdict = check_segment(segment, context)
            audit.record(verdict, segment.text)
            checked.append({
                "origin": segment.origin,
                "allowed": verdict.allowed,
                "checked_by": verdict.verified_by,
                "tokens": verdict.token_count,
                "refused": list(verdict.refused_tokens),
                "preview": " ".join(segment.text.split())[:90],
            })
            if not verdict.allowed:
                refused.extend(verdict.refused_tokens)

        if refused:
            step.say(
                f"blocked: {len(refused)} word(s) that belong to the database "
                f"were about to leave",
                parts=checked,
                refused=sorted(set(refused)),
            )
            raise LeakBlocked(tuple(dict.fromkeys(refused)), context)

        step.say(
            f"all {len(segments)} parts of the message checked and cleared",
            parts=checked,
        )


def require(text: str, context: str = "") -> str:
    """Check text of unknown origin, or raise."""
    from nl2sql.privacy import audit

    verdict = check(text, context)
    audit.record(verdict, text)
    if not verdict.allowed:
        raise LeakBlocked(verdict.refused_tokens, context)
    return text


def sweep_response(text: str, context: str = "cloud-response") -> list[str]:
    """Check what the model returned for stored values."""
    from nl2sql.privacy import audit

    found = find_known_values(text)
    if found:
        audit.record(
            Verdict(False, tuple(found), len(TOKEN_RE.findall(text or "")),
                    _fingerprint(text)[:16], context, "output sweep"),
            text,
        )
    return found


def stats() -> dict[str, int]:
    return {
        "allowed_words": len(allowlist()),
        "of_which_derived": len(schema_identifiers() | glossary_concepts()),
        "value_tokens": len(value_tokens()) if isinstance(value_tokens(), frozenset) else -1,
        "known_values": len(known_values()),
        "patterns": len(_patterns()),
        "registered_constants": len(_CONSTANTS),
    }
