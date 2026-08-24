"""Schema pseudonymisation — hiding the structure as well as the data.

The hybrid arm masks values and sends the schema in clear, because a model cannot
write SQL without knowing a table called `medication` exists. That is a real
disclosure: 31 table names and 391 column names describe the business even when
no row does. Here every identifier becomes `t3` / `c7`, redrawn at random per
request, and the question's business words go with them.

This can work only because the semantic work already happened locally: the prompt
can state `:v1 is a value of c7` outright, leaving the provider a mechanical
join-assembly task rather than "work out which column means what".
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from nl2sql.db import schema as sch

# `c1` must not match inside `c10`, and `age` must not be rewritten in `average`.
WORD = r"(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])"
LABEL_RE = re.compile(r"(?<![A-Za-z0-9_])([tc]\d+)(?![A-Za-z0-9_])")


@dataclass
class Pseudonyms:
    """The dictionary for one request. Never leaves the process."""

    tables: dict[str, str] = field(default_factory=dict)
    columns: dict[str, str] = field(default_factory=dict)
    ddl: str = ""

    @property
    def reverse(self) -> dict[str, str]:
        return {alias: real for real, alias in {**self.tables, **self.columns}.items()}

    def column(self, reference: str) -> str:
        """`medication.drugname` -> `t3.c7`."""
        table, _, column = reference.partition(".")
        return f"{self.tables.get(table, table)}.{self.columns.get(column, column)}"


def build(tables: set[str], seed: int | None = None) -> Pseudonyms:
    """Draw a fresh dictionary and render the opaque DDL.

    A column name keeps one alias across tables on purpose: `patientunitstayid`
    appears in 28 tables, and that repetition is the only thing telling the model
    those tables join at all.
    """
    schema = sch.read_schema()
    chosen = sorted(t for t in tables if t in schema)
    rng = random.Random(seed)

    ids = list(range(1, len(chosen) + 1))
    rng.shuffle(ids)
    table_map = {name: f"t{ids[i]}" for i, name in enumerate(chosen)}

    every: list[str] = []
    for name in chosen:
        for column in schema[name].columns:
            if column.name not in every:
                every.append(column.name)
    ids = list(range(1, len(every) + 1))
    rng.shuffle(ids)
    column_map = {name: f"c{ids[i]}" for i, name in enumerate(every)}

    chunks = []
    for name in chosen:
        table = schema[name]
        lines = [
            f"  {column_map[c.name]} {sch.compact_type(c.sql_type)}"
            + (" PRIMARY KEY" if c.is_pk else "")
            for c in table.columns
        ]
        # A key pointing outside the selected tables would name a table with no
        # alias, which would leak it. Dropped instead.
        lines += [
            f"  FOREIGN KEY ({column_map[k.column]}) "
            f"REFERENCES {table_map[k.target_table]}({column_map[k.target_column]})"
            for k in table.foreign_keys
            if k.target_table in table_map and k.target_column in column_map
        ]
        chunks.append(
            f"CREATE TABLE {table_map[name]} (  -- {table.row_count} rows\n"
            + ",\n".join(lines)
            + "\n);"
        )

    return Pseudonyms(table_map, column_map, "\n\n".join(chunks))


def rewrite_question(question: str, pseudonyms: Pseudonyms) -> tuple[str, list[str]]:
    """Replace the question's business words with their pseudonyms.

    Masking values is not enough to hide the subject: "how many patients received
    :v1" still says *patients*. The glossary already maps such words to columns,
    so the same mapping run through the request's dictionary turns them into
    labels. Words the glossary does not know stay — they are English, not schema,
    and removing them would leave nothing to write SQL from.
    """
    from nl2sql.nlp import glossary
    from nl2sql.privacy import gate

    text, notes, seen = question, [], set()
    # Longest trigger first: "mortality rate" before "mortality".
    for match in sorted(glossary.recognize(question), key=lambda m: -len(m.trigger)):
        columns = [c for c in match.term.columns if c.split(".", 1)[0] in pseudonyms.tables]
        if not columns:
            continue
        alias = pseudonyms.column(columns[0])
        rewritten, count = re.subn(
            WORD.format(re.escape(match.trigger)), alias, text, flags=re.IGNORECASE
        )
        if not count:
            continue
        text = rewritten
        if alias not in seen:
            seen.add(alias)
            others = ", ".join(pseudonyms.column(c) for c in columns[1:4])
            notes.append(
                gate.register_template(
                    f"{alias} is the subject of the question"
                    + (f"; related columns: {others}" if others else "")
                )
            )
    return text, notes


def restore(sql: str, pseudonyms: Pseudonyms) -> str:
    """Put the real identifiers back. Longest alias first, on word boundaries."""
    out = sql
    for alias, real in sorted(pseudonyms.reverse.items(), key=lambda p: -len(p[0])):
        out = re.sub(WORD.format(re.escape(alias)), real, out)
    return out


def invented(sql: str, pseudonyms: Pseudonyms) -> list[str]:
    """Labels the model made up. `FROM t9` when only t1..t4 exist is a hallucination.

    Caught here it becomes a clean rejection the repair loop can act on; passed
    through, `restore` would leave `t9` untouched and SQLite would report
    something unhelpful.
    """
    issued = set(pseudonyms.tables.values()) | set(pseudonyms.columns.values())
    return sorted(set(LABEL_RE.findall(sql)) - issued)
