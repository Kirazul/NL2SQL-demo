"""Prompt assembly — one place, four arms, five variants.

Everything a model is ever shown is built here, split into segments whose origin
the egress gate can verify. Keeping it in one module is what makes the benchmark
compare *architectures* rather than prompt-writing: every arm receives the same
instruction text, character for character, except the one rule that has to differ.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from nl2sql.db import schema as sch
from nl2sql.nlp.understand import Understanding
from nl2sql.privacy import opaque as opq
from nl2sql.privacy.gate import Segment, register_constant
from nl2sql.privacy.mask import Masked

MAX_TABLES = 8   # past this the prompt grows without improving joins
LABEL_RE = re.compile(r"(?<![A-Za-z0-9_])([tc]\d+)(?![A-Za-z0-9_])")

_HEAD = """You write SQLite queries. Follow these rules exactly.

1. Output ONE SQLite SELECT statement. No markdown, no explanation, no semicolon.
"""

# The only rule that differs between the hybrid arms and the two baselines.
_RULE_PARAMETERS = """2. Use the bound parameters (:v1, :v2, ...) EXACTLY as given. Never write a
   literal value where a parameter is provided — the real values are secret and
   you have not been shown them.
"""

# The baselines were shown the real values, so there is nothing to bind. Telling
# them to use parameters anyway produced `WHERE drugname = :v1` with no :v1 in
# existence: a query that passes validation and then fails at execution.
_RULE_LITERALS = """2. Write the values from the question directly into the query as SQL string or
   numeric literals. There are no bound parameters in this task.
"""

_TAIL = """3. Use only the tables and columns of the schema below.
4. Join tables by following the declared FOREIGN KEY relationships.
5. Prefer COUNT(DISTINCT uniquepid) when counting patients: one patient may have
   several stays.
6. If the question asks for a rate or a proportion, compute it with
   CAST(... AS REAL) so integer division does not return 0.
7. Give every output column a short readable alias (AS patient_count, AS
   mortality_rate). The answer is written from the column names."""

INSTRUCTIONS = _HEAD + _RULE_PARAMETERS + _TAIL
INSTRUCTIONS_LITERAL = _HEAD + _RULE_LITERALS + _TAIL

INSTRUCTIONS_OPAQUE = """You write SQLite queries over an anonymised schema.

The table and column names have been replaced by opaque labels (t1, c7, ...).
That is deliberate: you are not expected to understand what the data means. You
are given everything needed to assemble the query mechanically.

1. Output ONE SQLite SELECT statement. No markdown, no explanation, no semicolon.
2. Use ONLY the labels that appear in the schema below. Never invent a t- or c-
   label, and never write a real-world column name — there are none here.
3. Use the bound parameters (:v1, :v2, ...) EXACTLY as given. Each one is
   described with the column it applies to.
4. Join tables by following the declared FOREIGN KEY relationships.
5. When counting subjects, prefer COUNT(DISTINCT ...) on the label the foreign
   keys point at: one subject may have several rows.
6. If the question asks for a rate, compute it with CAST(... AS REAL).
7. Give every output column a short alias (AS n, AS rate). Do not try to name it
   after what it means — you have not been told."""

# The headings that hold the user message together. Declared so that every
# character actually sent is covered by some segment — otherwise scaffolding text
# would travel unchecked simply because nobody thought to list it.
SCAFFOLD = "Schema: Bound parameters: Domain notes: Question: SQL:"

REPAIR_TAIL = (
    "Rewrite it as ONE valid SQLite SELECT that respects the rules. Output only the SQL."
)

# Declared constants: the gate lets these through on their fingerprint, not their
# vocabulary. Interpolating anything into them would change the fingerprint and
# send them back to word-by-word checking.
for _literal in (INSTRUCTIONS, INSTRUCTIONS_LITERAL, INSTRUCTIONS_OPAQUE, SCAFFOLD, REPAIR_TAIL):
    register_constant(_literal)


@dataclass
class Prompt:
    """One assembled prompt, with everything needed to display and audit it."""

    messages: list[dict[str, str]]
    segments: list[Segment]
    pseudonyms: Any = None
    view: dict[str, Any] = field(default_factory=dict)

    @property
    def characters(self) -> int:
        return sum(len(m["content"]) for m in self.messages)


def relevant_tables(understanding: Understanding) -> set[str]:
    """Tables to describe. All 31 would cost ~6,000 tokens for no improvement."""
    schema = sch.read_schema()
    tables = {t for t in understanding.tables if t in schema} or {"patient"}
    tables.add("patient")  # eICU's hub: almost every join goes through it
    if len(tables) > MAX_TABLES:
        # Keep the smallest: bulky fact tables add columns without helping.
        tables = set(sorted(tables, key=lambda t: schema[t].row_count)[:MAX_TABLES])
    return tables


def lean_columns(understanding: Understanding, tables: set[str]) -> dict[str, list[str]]:
    """Only the columns the question actually reaches, plus the join keys."""
    wanted: dict[str, list[str]] = {t: [] for t in tables}
    for ref in understanding.columns:
        table, _, column = ref.partition(".")
        if table in wanted:
            wanted[table].append(column)
    return wanted


def _user_message(ddl: str, parameters: str, notes: list[str], question: str) -> str:
    note_block = "\n\nDomain notes:\n" + "\n".join(f"- {n}" for n in notes) if notes else ""
    return (
        f"Schema:\n{ddl}\n\n"
        f"Bound parameters:\n{parameters}{note_block}\n\n"
        f"Question: {question}\n\nSQL:"
    )


def hybrid(
    understanding: Understanding, masked: Masked, lean: bool = False, notes: bool = True
) -> Prompt:
    """Clear schema, masked question. The architecture this project defends."""
    tables = relevant_tables(understanding)
    ddl = sch.ddl(tables, lean_columns(understanding, tables) if lean else None)
    parameters = masked.declare()
    used_notes = list(understanding.notes) if notes else []

    segments = [
        Segment(INSTRUCTIONS, "authored"),
        Segment(SCAFFOLD, "authored"),
        Segment(ddl, "schema"),
        Segment(parameters, "params"),
        *[Segment(n, "glossary") for n in used_notes],
        Segment(masked.question, "question"),
    ]
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": _user_message(ddl, parameters, used_notes, masked.question)},
    ]
    return Prompt(messages, segments, view={"tables": sorted(tables), "lean": lean})


def opaque(understanding: Understanding, masked: Masked, seed: int | None = None) -> Prompt:
    """Pseudonymised schema *and* question: no business word of any kind leaves.

    The dictionary is drawn per request and travels with the prompt — rebuilding it
    later would describe labels that were never sent.
    """
    tables = relevant_tables(understanding)
    pseudonyms = opq.build(tables, seed=seed)
    question, notes = opq.rewrite_question(masked.question, pseudonyms)
    parameters = masked.declare(rename={c: pseudonyms.column(c) for c in understanding.columns})

    segments = [
        Segment(INSTRUCTIONS_OPAQUE, "authored"),
        Segment(SCAFFOLD, "authored"),
        Segment(pseudonyms.ddl, "opaque"),
        Segment(parameters, "params"),
        *[Segment(n, "template") for n in notes],
        Segment(question, "opaque"),
    ]
    messages = [
        {"role": "system", "content": INSTRUCTIONS_OPAQUE},
        {"role": "user", "content": _user_message(pseudonyms.ddl, parameters, notes, question)},
    ]

    # Only the sentence and its parameters are decoded for display: decoding the
    # DDL too would list several hundred columns nobody reads.
    reverse = pseudonyms.reverse
    used = set(LABEL_RE.findall(f"{question}\n{parameters}"))
    view = {
        "question": question,
        "parameters": parameters,
        "ddl": pseudonyms.ddl,
        "labels": {a: reverse[a] for a in sorted(used) if a in reverse},
        "tables": len(pseudonyms.tables),
        "columns": len(pseudonyms.columns),
    }
    return Prompt(messages, segments, pseudonyms, view)


def clear(understanding: Understanding, question: str) -> Prompt:
    """The unprotected baseline: the question leaves exactly as it was typed."""
    ddl = sch.ddl(relevant_tables(understanding))
    note_block = (
        "\n\nDomain notes:\n" + "\n".join(f"- {n}" for n in understanding.notes)
        if understanding.notes
        else ""
    )
    messages = [
        {"role": "system", "content": INSTRUCTIONS_LITERAL},
        {"role": "user", "content": f"Schema:\n{ddl}{note_block}\n\nQuestion: {question}\n\nSQL:"},
    ]
    return Prompt(messages, [], view={"tables": sorted(relevant_tables(understanding))})


def repair(prompt: Prompt, sql: str, reason: str, opaque_arm: bool = False) -> Prompt:
    """Hand the model its own query back with the reason it was rejected."""
    complaint = f"That query was rejected: {reason}."
    instruction = f"{complaint}\n{REPAIR_TAIL}"
    origin = "opaque" if opaque_arm else "question"
    return Prompt(
        messages=[
            *prompt.messages,
            {"role": "assistant", "content": sql},
            {"role": "user", "content": instruction},
        ],
        segments=[
            *prompt.segments,
            Segment(sql, origin),
            Segment(complaint, "question"),
            Segment(REPAIR_TAIL, "authored"),
        ],
        pseudonyms=prompt.pseudonyms,
        view=prompt.view,
    )
