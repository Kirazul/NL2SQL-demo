"""Stage 2a — replace real values with symbols before any network call.

The mechanism
-------------
"how many patients received aspirin?" becomes "how many patients received :v1?",
and the mapping `:v1 -> 'aspirin'` stays on our side. The cloud model writes SQL
containing `:v1`; we bind it to the real value at execution time, as a bound
parameter.

Why symbols and not a hash or a pseudonym
-----------------------------------------
A stable pseudonym (`DRUG_A17` for aspirin, always) would let the provider track
a value across requests and infer its frequency, hence its identity. Symbols are
**numbered in order of appearance and restart from 1 on every question**: two
requests about the same drug do not look alike.

Why `:v1` and not `<VALUE_1>`
-----------------------------
`:v1` is SQLite's bound-parameter syntax. The model therefore produces directly
executable SQL, and the value is never concatenated into the string — which makes
injection impossible and guarantees no value can end up in the text sent to the
cloud by accident.

What is NOT masked, and why
---------------------------
- **concepts** ("mortality rate") name a column: masking them would deprive the
  model of what it needs, and the column name goes into the DDL anyway;
- **quantities** ("over 65") come from the analyst, not from the database;
- **person names** are not masked but **block the request**: the database is
  de-identified, the question cannot be satisfied, and there is no reason to send
  anything to the cloud.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from hybridsql.pipeline.understand import Resolution, Understanding


class UnresolvableValue(RuntimeError):
    """A value in the question matches nothing in the database, or matches it too weakly.

    Why this stops the request instead of continuing
    ------------------------------------------------
    It used to continue. The mention stayed in the question unmasked, the cloud
    model received a word it could not place, and it did what a language model
    always does with an impossible constraint: it produced *something*. Observed
    output, twice: `WHERE apacheadmissiondx = 'asparatan'` for a drug that does
    not exist, and `WHERE 1=0` for a misspelling. Both ran without error and both
    returned `0`, which the answer writer then reported as a fact.

    A zero that means "no such patient" and a zero that means "I did not
    understand you" are indistinguishable to the person reading the answer, and
    the second one is a lie the system told confidently. So it is refused, with
    whatever near matches the index found attached — the analyst decides, we do
    not guess on their behalf.
    """

    def __init__(self, unknown: list[tuple[str, tuple[str, ...]]]) -> None:
        self.unknown = unknown
        parts = []
        for mention, suggestions in unknown:
            if suggestions:
                shown = ", ".join(repr(s) for s in suggestions[:3])
                parts.append(f"{mention!r} — did you mean {shown}?")
            else:
                parts.append(f"{mention!r} — nothing like it is stored in this database")
        super().__init__(
            "The question uses a value this database does not recognise: "
            + "; ".join(parts)
            + ". Nothing was sent."
        )

    @property
    def suggestions(self) -> list[str]:
        """Flat list of proposals, for an interface to offer as one click each."""
        return [s for _, group in self.unknown for s in group]


class UnmaskableQuestion(RuntimeError):
    """The question names a person: it cannot be sent."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        super().__init__(
            "The question names one or more people ("
            + ", ".join(repr(n) for n in names)
            + "). The database is de-identified and holds no names. "
            "Rephrase without a proper noun."
        )


@dataclass
class Anonymization:
    """Stage 2a result.

    `masked_question` is meant for the cloud. `mapping` must never leave the
    process: it is the system's only secret.
    """

    masked_question: str
    mapping: dict[str, str] = field(default_factory=dict)    # ':v1' -> real value
    columns: dict[str, str] = field(default_factory=dict)    # ':v1' -> 'table.column'
    unresolved: list[str] = field(default_factory=list)

    @property
    def symbol_count(self) -> int:
        return len(self.mapping)

    def parameters(self) -> dict[str, str | int | float]:
        """Bound parameters for SQLite: `{'v1': 'aspirin', 'v2': 56}`.

        A masked number is bound as a number. SQLite's type affinity would usually
        rescue `hospitalid = '56'` on an INTEGER column, but not everywhere — an
        `IN` list, a `CAST`, or a comparison against a TEXT column all behave
        differently — and the analyst wrote a number, so a number is what should
        reach the database.
        """
        out: dict[str, str | int | float] = {}
        for symbol, value in self.mapping.items():
            name = symbol.lstrip(":")
            # Symbols with no column are the masked numbers: a value resolved from
            # the database always carries the column it came from.
            out[name] = _as_number(value) if symbol not in self.columns else value
        return out

    def for_the_prompt(self) -> str:
        """Describe the symbols to the model **without revealing the values**.

        The model needs to know which column each symbol applies to, otherwise it
        cannot place the `WHERE`. It does not need the value.

        A masked number is described as a number and nothing else. That is enough
        to write `> :v2` or `= :v2`, which is all the model was ever doing with it.
        """
        if not self.mapping:
            return "  (none)"
        lines = []
        for symbol in sorted(self.mapping, key=lambda s: int(s.lstrip(":v") or 0)):
            column = self.columns.get(symbol)
            lines.append(
                f"  {symbol} = a value from {column}" if column
                else f"  {symbol} = a number given by the analyst"
            )
        return "\n".join(lines)


def _replace(question: str, resolution: Resolution, symbol: str) -> str:
    """Replace the mention with its symbol, on whole words.

    Substring substitution would break on a mention appearing inside another word,
    so we bound with `\\b` where possible.
    """
    pattern = re.escape(resolution.mention)
    if resolution.mention[:1].isalnum() and resolution.mention[-1:].isalnum():
        pattern = rf"\b{pattern}\b"
    new, n = re.subn(pattern, symbol, question, count=1, flags=re.IGNORECASE)
    if n:
        return new
    # Fallback: the mention returned by the model does not match the text exactly
    # (case, inner punctuation). Try again without word boundaries.
    return re.sub(re.escape(resolution.mention), symbol, question, count=1, flags=re.IGNORECASE)


def _as_number(value: str) -> str | int | float:
    """`'56'` -> `56`, `'12.5'` -> `12.5`, anything else unchanged."""
    text = str(value).strip().replace(",", ".")
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return value


def _proposals(resolution: Resolution) -> tuple[str, ...]:
    """The stored values worth offering instead of what the analyst wrote.

    The best match comes first when there is one — for a near miss it *is* the
    answer, it simply scored below the bar at which we are willing to substitute
    it silently. `alternatives` carries `table.column = value`, of which only the
    value is useful in a suggestion.
    """
    # A weak match is not a suggestion, it is noise. Proposing "testes" to somebody
    # who asked about haemoglobin is worse than admitting we found nothing.
    MIN_USEFUL = 0.65

    out: list[str] = []
    if resolution.value and resolution.score >= MIN_USEFUL:
        out.append(resolution.value)
    for alternative in resolution.alternatives:
        _, _, value = alternative.partition(" = ")
        if value and value not in out:
            out.append(value)
    return tuple(out[:4])


def anonymize(understanding: Understanding) -> Anonymization:
    """Turn the understanding into a masked question ready for the egress gate.

    Raises `UnmaskableQuestion` when a person name is present: a deliberate
    refusal, not a technical error.
    """
    if understanding.persons:
        raise UnmaskableQuestion([r.mention for r in understanding.persons])

    # A value we could not pin down, or pinned down too weakly to act on. Both are
    # refused here rather than passed through: see `UnresolvableValue`. Concepts
    # and quantities are untouched — they are not looked up in the first place.
    doubtful = understanding.unresolved + understanding.needs_confirmation
    if doubtful:
        raise UnresolvableValue([
            (r.mention, _proposals(r)) for r in doubtful
        ])

    question = understanding.question
    mapping: dict[str, str] = {}
    columns: dict[str, str] = {}

    # Longest mentions first: otherwise replacing "aspirin" before "aspirin 81 mg"
    # would leave ":v1 81 mg" in the question.
    to_mask = sorted(understanding.values, key=lambda r: -len(r.mention))

    for i, resolution in enumerate(to_mask, start=1):
        symbol = f":v{i}"
        question = _replace(question, resolution, symbol)
        mapping[symbol] = resolution.value or ""
        if resolution.column:
            columns[symbol] = resolution.column

    return Anonymization(
        masked_question=question,
        mapping=mapping,
        columns=columns,
        unresolved=[r.mention for r in understanding.unresolved],
    )
