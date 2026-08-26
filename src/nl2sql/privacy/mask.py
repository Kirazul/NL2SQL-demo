"""Stage 2a — replace real values with symbols before any network call."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nl2sql.nlp.understand import Resolution, Understanding

MIN_USEFUL_SUGGESTION = 0.65


class UnresolvableValue(RuntimeError):
    """A value in the question matches nothing in the database, or matches too weakly."""

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
    # ':v1' -> the word the analyst typed, where that word is a fragment of the
    # stored value rather than the whole of it. See `_pattern_for`.
    patterns: dict[str, str] = field(default_factory=dict)

    @property
    def symbol_count(self) -> int:
        return len(self.mapping)

    def parameters(self) -> dict[str, str | int | float]:
        """Bound parameters for SQLite. A masked number is bound as a number."""
        out: dict[str, str | int | float] = {}
        for symbol, value in self.mapping.items():
            name = symbol.lstrip(":")
            if symbol in self.patterns:
                out[name] = f"%{self.patterns[symbol]}%"
            elif symbol in self.columns:
                out[name] = value
            else:
                out[name] = _as_number(value)
        return out

    def declare(self, rename: dict[str, str] | None = None) -> str:
        """Describe the symbols to the model without revealing the values."""
        if not self.mapping:
            return "  (none)"
        lines = []
        for symbol in sorted(self.mapping, key=lambda s: int(s.lstrip(":v") or 0)):
            column = self.columns.get(symbol)
            if column and rename:
                column = rename.get(column, column)
            if column and symbol in self.patterns:
                # The analyst named part of what is stored - "aspirin" where the
                # column holds "ASPIRIN EC 81 MG PO TBEC" and twenty other
                # spellings. `= :v1` would find one of them, so say plainly that
                # this symbol is a pattern.
                lines.append(f"  {symbol} = a LIKE pattern for {column} - match it with "
                             f"LIKE :{symbol.lstrip(':')}, never with =")
            elif column:
                lines.append(f"  {symbol} = a value of {column}")
            else:
                lines.append(f"  {symbol} = a number given by the analyst")
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


def _spellings(column: str, mention: str) -> int:
    """How many distinct values of this column contain the analyst's word (max 2).

    Stops at two, because one is all the caller needs to distinguish "this word
    names the whole value" from "this word appears in several of them".
    """
    from nl2sql.db import schema as sch
    from nl2sql.db.sqlite import execute

    table, _, name = column.partition(".")
    tables = sch.read_schema()
    if table not in tables or name not in {c.name for c in tables[table].columns}:
        return 0
    # Identifiers cannot be bound, so they are checked against the schema above
    # and never taken from the question; the pattern itself is bound.
    sql = (f"SELECT COUNT(*) FROM (SELECT DISTINCT {name} FROM {table} "
           f"WHERE {name} LIKE :needle LIMIT 2)")
    try:
        _, rows = execute(sql, {"needle": f"%{mention}%"}, max_rows=1)
    except Exception:  # noqa: BLE001 - a slow or missing table must not stop masking
        return 0
    return int(rows[0][0]) if rows else 0


def _pattern_for(resolution: Resolution) -> str:
    """The analyst's word, when the column stores it in more than one spelling.

    `medication.drugname` holds twenty-one values containing ASPIRIN, and
    `diagnosis.diagnosisstring` thirteen paths containing pneumonia. Stage 1
    resolves the mention to exactly one of them, and `= :v1` then answers about
    that one spelling: 79 patients where 433 received the drug. Where several
    stored values contain the word, the honest query matches all of them, so the
    symbol becomes a pattern rather than a literal.

    Returns "" where the word names the whole value and nothing else - a gender,
    an ethnicity, an organism - and `=` is exactly right.
    """
    mention = " ".join(str(resolution.mention).lower().split())
    if not mention or not resolution.column:
        return ""
    return resolution.mention if _spellings(resolution.column, mention) > 1 else ""


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
        patterns: dict[str, str] = {}

        # Longest first, or replacing "aspirin" before "aspirin 81 mg" leaves ":v1 81 mg".
        for i, r in enumerate(sorted(understanding.values, key=lambda r: -len(r.mention)), 1):
            symbol = f":v{i}"
            question = _replace(question, r.mention, symbol)
            mapping[symbol] = r.value or ""
            if r.column:
                columns[symbol] = r.column
                fragment = _pattern_for(r)
                if fragment:
                    patterns[symbol] = fragment

        step.say(
            f"{len(mapping)} value(s) replaced by a symbol; the real values stay here"
            if mapping
            else "nothing to hide: this question contains no stored value",
            before=understanding.question,
            after=question,
            symbols=list(mapping),
            columns=columns,
        )
        return Masked(question, mapping, columns,
                      [r.mention for r in understanding.unresolved], patterns)
