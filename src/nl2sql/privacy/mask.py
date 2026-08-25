"""Stage 2a — replace real values with symbols before any network call.

"received aspirin?" becomes "received :v1?", and the mapping stays here. Symbols
restart at 1 per question, so two questions about the same drug do not look alike.

`:v1` is SQLite's own bound-parameter syntax, so the model returns directly
executable SQL and the value never meets the query text in one string — injection
is impossible by construction rather than by filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nl2sql.nlp.understand import Resolution, Understanding

MIN_USEFUL_SUGGESTION = 0.65


class UnresolvableValue(RuntimeError):
    """A value in the question matches nothing in the database, or matches too weakly.

    It used to be left unmasked and passed on, and the model did what a language model
    always does with an impossible constraint: it produced something — `WHERE
    """

    def __init__(self, unknown: list[tuple[str, tuple[str, ...]]]) -> None:
        self.unknown = unknown
        parts = [
            f"{mention!r} — did you mean {', '.join(repr(s) for s in suggestions[:3])}?"
            if suggestions
            else f"{mention!r} — nothing like it is stored in this database"
            for mention, suggestions in unknown
        ]
        super().__init__(
            "The question uses a value this database does not recognise: "
            + "; ".join(parts)
            + ". Nothing was sent."
        )

    @property
    def suggestions(self) -> list[str]:
        return [s for _, group in self.unknown for s in group]


class UnmaskableQuestion(RuntimeError):
    """The question names a person, so it cannot be answered or sent."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        super().__init__(
            "The question names one or more people (" + ", ".join(repr(n) for n in names) + "). "
            "The database is de-identified and holds no names. Rephrase without a proper noun."
        )


@dataclass
class Masked:
    """`masked_question` is for the cloud; `mapping` must never leave the process."""

    question: str
    mapping: dict[str, str] = field(default_factory=dict)   # ':v1' -> real value
    columns: dict[str, str] = field(default_factory=dict)   # ':v1' -> 'table.column'
    unresolved: list[str] = field(default_factory=list)

    @property
    def symbol_count(self) -> int:
        return len(self.mapping)

    def parameters(self) -> dict[str, str | int | float]:
        """Bound parameters for SQLite. A masked number is bound as a number."""
        return {
            symbol.lstrip(":"): (value if symbol in self.columns else _as_number(value))
            for symbol, value in self.mapping.items()
        }

    def declare(self, rename: dict[str, str] | None = None) -> str:
        """Describe the symbols to the model without revealing the values.

        The model needs the column to place the `WHERE`; it never needs the value.
        `rename` maps a real column to its pseudonym for the opaque arm.
        """
        if not self.mapping:
            return "  (none)"
        lines = []
        for symbol in sorted(self.mapping, key=lambda s: int(s.lstrip(":v") or 0)):
            column = self.columns.get(symbol)
            if column and rename:
                column = rename.get(column, column)
            lines.append(
                f"  {symbol} = a value of {column}" if column
                else f"  {symbol} = a number given by the analyst"
            )
        return "\n".join(lines)


def _as_number(value: str) -> str | int | float:
    text = str(value).strip().replace(",", ".")
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return value


def _replace(question: str, mention: str, symbol: str) -> str:
    """Substitute on whole words where possible, so a mention inside a word is safe."""
    pattern = re.escape(mention)
    if mention[:1].isalnum() and mention[-1:].isalnum():
        pattern = rf"\b{pattern}\b"
    new, count = re.subn(pattern, symbol, question, count=1, flags=re.IGNORECASE)
    if count:
        return new
    return re.sub(re.escape(mention), symbol, question, count=1, flags=re.IGNORECASE)


def _proposals(resolution: Resolution) -> tuple[str, ...]:
    """Values worth offering back. A weak match is noise, not a suggestion."""
    out: list[str] = []
    if resolution.value and resolution.score >= MIN_USEFUL_SUGGESTION:
        out.append(resolution.value)
    for alternative in resolution.alternatives:
        _, _, value = alternative.partition(" = ")
        if value and value not in out:
            out.append(value)
    return tuple(out[:4])


def mask(understanding: Understanding) -> Masked:
    """Turn an understanding into a masked question ready for the gate."""
    from nl2sql.core.steps import track

    with track("mask", question=understanding.question) as step:
        if understanding.persons:
            names = [r.mention for r in understanding.persons]
            step.say(f"stopped: the question names {', '.join(names)}", refused="person named")
            raise UnmaskableQuestion(names)

        doubtful = understanding.unresolved + understanding.needs_confirmation
        if doubtful:
            step.say(
                "stopped: " + ", ".join(f"'{r.mention}' is not in this database" for r in doubtful),
                refused="value not recognised",
                unknown=[r.mention for r in doubtful],
            )
            raise UnresolvableValue([(r.mention, _proposals(r)) for r in doubtful])

        question = understanding.question
        mapping: dict[str, str] = {}
        columns: dict[str, str] = {}

        # Longest first, or replacing "aspirin" before "aspirin 81 mg" leaves ":v1 81 mg".
        for i, r in enumerate(sorted(understanding.values, key=lambda r: -len(r.mention)), 1):
            symbol = f":v{i}"
            question = _replace(question, r.mention, symbol)
            mapping[symbol] = r.value or ""
            if r.column:
                columns[symbol] = r.column

        step.say(
            f"{len(mapping)} value(s) replaced by a symbol; the real values stay here"
            if mapping
            else "nothing to hide: this question contains no stored value",
            before=understanding.question,
            after=question,
            symbols=list(mapping),
            columns=columns,
        )
        return Masked(question, mapping, columns, [r.mention for r in understanding.unresolved])
