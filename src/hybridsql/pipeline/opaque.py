"""Schema pseudonymisation — hiding the structure as well as the data.

What this adds to the hybrid architecture
-----------------------------------------
The hybrid pipeline masks values and sends the schema in clear, because the model
cannot write SQL without knowing that a table called `medication` exists. That is
a real disclosure and the report says so: 31 table names and 391 column names
describe the business even when no row does.

This module removes it. Every identifier in the outgoing DDL is replaced by an
opaque one, and the question's structural words go with it:

    CREATE TABLE medication (            CREATE TABLE t3 (
      medicationid INT PRIMARY KEY,        c1 INT PRIMARY KEY,
      patientunitstayid INT,        ->     c2 INT,
      drugname TEXT                        c7 TEXT
    );                                   );

    "how many patients received :v1"  ->  "how many c2 received :v1, a value of c7"

The provider assembles a join path over a shape. It never learns that the shape
is a hospital.

Why this can work here and not in general
-----------------------------------------
Stripping schema names is known to hurt text-to-SQL badly, because those names
carry most of the semantic signal — a model shown `c7` cannot guess it holds drug
names. This pipeline is in an unusual position: **the semantic work already
happened locally.** GLiNER2 and the value index have already established that
`aspirin` lives in `medication.drugname`, so the prompt can state
`:v1 is a value of c7` outright. What remains for the provider is mechanical —
follow the foreign keys, place the aggregate — and that survives anonymisation
far better than "work out which column means what".

It is still expected to cost accuracy. Measuring that cost is the point: it is
the price of removing the last disclosure, and a number is worth more than an
opinion.

Why the mapping is redrawn every request
----------------------------------------
Stable pseudonyms would let a provider accumulate a dictionary across questions —
`t3` appearing beside drug-like values often enough stops being opaque. The
mapping is therefore shuffled per request, exactly as `:v1` is renumbered per
request (decision 5, docs/02-DECISIONS.md). Two questions about the same table do not look alike.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from hybridsql.db import schema as sch

# Identifiers are matched as whole words. `c1` must not match inside `c10`, and a
# column named `age` must not be rewritten inside `average`.
_WORD = r"(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])"


@dataclass
class Pseudonyms:
    """The dictionary for one request. Never leaves the process."""

    tables: dict[str, str] = field(default_factory=dict)     # medication -> t3
    columns: dict[str, str] = field(default_factory=dict)    # drugname   -> c7
    ddl: str = ""

    @property
    def reverse_tables(self) -> dict[str, str]:
        return {alias: real for real, alias in self.tables.items()}

    @property
    def reverse_columns(self) -> dict[str, str]:
        return {alias: real for real, alias in self.columns.items()}

    def column(self, reference: str) -> str:
        """`medication.drugname` -> `t3.c7`, for describing a bound parameter."""
        table, _, column = reference.partition(".")
        return f"{self.tables.get(table, table)}.{self.columns.get(column, column)}"


def build(tables: set[str], seed: int | None = None) -> Pseudonyms:
    """Draw a fresh dictionary and render the opaque DDL.

    Column names are shared across tables on purpose: eICU's join key
    `patientunitstayid` appears in 28 tables, and giving it one alias is what lets
    the model see that those tables join at all. Renaming it differently per table
    would hide the relationships along with the names, and the foreign keys would
    become the only remaining hint.
    """
    schema = sch.read_schema()
    chosen = sorted(t for t in tables if t in schema)

    rng = random.Random(seed)

    table_ids = list(range(1, len(chosen) + 1))
    rng.shuffle(table_ids)
    table_map = {name: f"t{table_ids[i]}" for i, name in enumerate(chosen)}

    every_column: list[str] = []
    for name in chosen:
        for column in schema[name].columns:
            if column.name not in every_column:
                every_column.append(column.name)

    column_ids = list(range(1, len(every_column) + 1))
    rng.shuffle(column_ids)
    column_map = {name: f"c{column_ids[i]}" for i, name in enumerate(every_column)}

    chunks: list[str] = []
    for name in chosen:
        table = schema[name]
        lines = []
        for column in table.columns:
            line = f"  {column_map[column.name]} {sch._compact_type(column.sql_type)}"
            if column.is_pk:
                line += " PRIMARY KEY"
            lines.append(line)
        for fk in table.foreign_keys:
            # A foreign key pointing outside the selected tables would name a
            # table with no alias, which would leak it. Dropped instead.
            if fk.target_table not in table_map:
                continue
            lines.append(
                f"  FOREIGN KEY ({column_map[fk.column]}) "
                f"REFERENCES {table_map[fk.target_table]}({column_map[fk.target_column]})"
            )
        chunks.append(
            f"CREATE TABLE {table_map[name]} (  -- {table.row_count} rows\n"
            + ",\n".join(lines)
            + "\n);"
        )

    return Pseudonyms(tables=table_map, columns=column_map, ddl="\n\n".join(chunks))


def opaque_question(masked_question: str, pseudonyms: Pseudonyms) -> tuple[str, list[str]]:
    """Replace the question's business words with their pseudonyms.

    Masking values is not enough to hide the subject. "How many patients received
    :v1" still says *patients*, and combined with the rest of the sentence it says
    the database is clinical. The glossary already maps such words to the columns
    they denote — `patients` to `patient.patientunitstayid`, `mortality` to
    `patient.hospitaldischargestatus` — so the same mapping, run through the
    request's dictionary, turns them into aliases.

    Returns the rewritten question and the notes describing what each alias is,
    because after this the model has a sentence of pure grammar and needs to be
    told which alias to aggregate and which to filter.

    What deliberately stays
    -----------------------
    Words the glossary does not recognise. They are English, not schema, and they
    carry the *question's* meaning rather than the database's: "how many", "each",
    "average". Removing them would leave nothing to write SQL from at all.
    """
    from hybridsql.resources import glossary
    from hybridsql.security import egress_gate

    text = masked_question
    notes: list[str] = []
    seen: set[str] = set()

    # Longest trigger first: "mortality rate" must be consumed before "mortality".
    matches = sorted(glossary.recognize(masked_question), key=lambda m: -len(m.trigger))
    for match in matches:
        columns = [c for c in match.term.columns if c.split(".", 1)[0] in pseudonyms.tables]
        if not columns:
            continue
        alias = pseudonyms.column(columns[0])
        pattern = _WORD.format(re.escape(match.trigger))
        rewritten, count = re.subn(pattern, alias, text, flags=re.IGNORECASE)
        if not count:
            continue
        text = rewritten
        if alias not in seen:
            seen.add(alias)
            others = ", ".join(pseudonyms.column(c) for c in columns[1:4])
            notes.append(
                egress_gate.register_template(
                    f"{alias} is the subject of the question"
                    + (f"; related columns: {others}" if others else "")
                )
            )
    return text, notes


def restore(sql: str, pseudonyms: Pseudonyms) -> str:
    """Put the real identifiers back into the SQL the provider returned.

    Longest alias first. `c1` would otherwise match the leading characters of
    `c12` under a naive replacement, and the word-boundary pattern is what makes
    that safe — both belts, because a mis-substitution here produces a query that
    runs against the wrong column and returns a plausible wrong answer.
    """
    out = sql
    for alias, real in sorted(
        {**pseudonyms.reverse_tables, **pseudonyms.reverse_columns}.items(),
        key=lambda pair: -len(pair[0]),
    ):
        out = re.sub(_WORD.format(re.escape(alias)), real, out)
    return out


def leftovers(sql: str, pseudonyms: Pseudonyms) -> list[str]:
    """Aliases the model invented that we never issued.

    A model that writes `FROM t9` when only `t1..t4` were given has hallucinated a
    table. Left alone, `restore` would pass `t9` through untouched and the query
    would fail at execution with an unhelpful error; caught here, it is a clean
    rejection with a reason the repair loop can act on.
    """
    issued = set(pseudonyms.tables.values()) | set(pseudonyms.columns.values())
    used = set(re.findall(r"(?<![A-Za-z0-9_])([tc]\d+)(?![A-Za-z0-9_])", sql))
    return sorted(used - issued)
