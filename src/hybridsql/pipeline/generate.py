"""Stage 2b — build the prompt, call the cloud, repair if needed.

Repair loop rather than multiple candidates
-------------------------------------------
The architecture diagram called for 5 candidate queries generated in parallel,
then a selector. We do otherwise, for a measurable reason: free Groq caps at **30
requests per minute**. Five calls per question means six questions per minute and
an exhausted quota partway through evaluating 107 questions.

Instead: **one call, then a second only on failure**, with the exact error handed
back to the model. The average cost stays close to 1 call instead of 5, and a
targeted repair corrects better than an extra sample — the model is told *why* it
was wrong.

What crosses the boundary
-------------------------
The prompt contains: the DDL of the selected tables, the symbol descriptions
(`:v1 = a value from medication.drugname`), the masked question, and the glossary
notes. It contains **no value**. Everything goes through the egress gate, in
`providers/cloud.py`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from hybridsql.db import schema as sch
from hybridsql.pipeline.anonymize import Anonymization
from hybridsql.pipeline.understand import Understanding
from hybridsql.providers import cloud
from hybridsql.security import sql_validator
from hybridsql.security import egress_gate
from hybridsql.security.egress_gate import Segment

_log = logging.getLogger(__name__)

MAX_TABLES = 8   # beyond this the prompt grows without improving joins

_RULES_HEAD = """You write SQLite queries. Follow these rules exactly.

1. Output ONE SQLite SELECT statement. No markdown, no explanation, no semicolon.
"""

# The one rule that differs between the hybrid architecture and the two baselines.
# It is isolated rather than duplicated so the benchmark compares architectures
# and not prompt-writing: every other instruction the three arms receive is the
# same text, character for character.
_RULE_PARAMETERS = """2. Use the bound parameters (:v1, :v2, ...) EXACTLY as given. Never write a
   literal value where a parameter is provided — the real values are secret and
   you have not been shown them.
"""

# The baselines were shown the real values, so there is nothing to bind. Telling
# them to use parameters anyway produced `WHERE drugname = :v1` with no :v1 in
# existence — a query that passes validation and then fails at execution.
_RULE_LITERALS = """2. Write the values from the question directly into the query as SQL string or
   numeric literals. There are no bound parameters in this task.
"""

_RULES_TAIL = """3. Use only the tables and columns of the schema below.
4. Prefer COUNT(DISTINCT patientunitstayid) when counting patients: a patient
   may have several unit stays.
5. If the question asks for a rate or a proportion, compute it with
   CAST(... AS REAL) so integer division does not return 0.
6. Give every output column a short readable alias (AS patient_count, AS
   mortality_rate). The answer is written from the column names, so
   COUNT(DISTINCT m.patientunitstayid) as a header makes it unreadable."""

# What the Hybrid arm sends. Unchanged, character for character, from the version
# the published measurements were taken against.
INSTRUCTIONS = _RULES_HEAD + _RULE_PARAMETERS + _RULES_TAIL

# What the Full Cloud and Full Local baselines send. Neither passes through the
# egress gate — Full Cloud bypasses it by definition, Full Local never opens a
# socket — so this text is not registered as a constant.
INSTRUCTIONS_LITERAL = _RULES_HEAD + _RULE_LITERALS + _RULES_TAIL

# Declared constant: this text is a source literal, depending neither on the
# database nor on the question. The gate lets it through on its fingerprint, not
# its vocabulary — if a value were ever interpolated into it, the fingerprint
# would change and word-by-word checking would resume.
egress_gate.register_constant(INSTRUCTIONS)


INSTRUCTIONS_OPAQUE = """You write SQLite queries over an anonymised schema.

The table and column names have been replaced by opaque labels (t1, c7, ...).
That is deliberate: you are not expected to understand what the data means. You
are given everything you need to assemble the query mechanically.

1. Output ONE SQLite SELECT statement. No markdown, no explanation, no semicolon.
2. Use ONLY the labels that appear in the schema below. Never invent a t- or c-
   label, and never write a real-world column name — there are none here.
3. Use the bound parameters (:v1, :v2, ...) EXACTLY as given. Each one is
   described with the column it applies to.
4. Join tables by following the declared FOREIGN KEY relationships.
5. When counting subjects, prefer COUNT(DISTINCT ...) on the key that the foreign
   keys point at: one subject may have several rows.
6. If the question asks for a rate or a proportion, compute it with
   CAST(... AS REAL) so integer division does not return 0.
7. Give every output column a short alias (AS n, AS rate). Do not try to name it
   after what it means — you have not been told."""

egress_gate.register_constant(INSTRUCTIONS_OPAQUE)


@dataclass
class Generation:
    sql: str
    valid: bool
    target: str = ""
    ms: float = 0.0
    tokens: int = 0
    calls: int = 0
    repairs: int = 0
    failure_reason: str = ""
    history: list[str] = field(default_factory=list)

    # The prompt as it was actually sent. Carried out rather than rebuilt by the
    # caller: the opaque arm draws its dictionary at random per request, so a
    # second `build_opaque` would describe a prompt that was never sent.
    segments: list[Segment] = field(default_factory=list)
    opaque: dict[str, Any] = field(default_factory=dict)


def relevant_tables(understanding: Understanding) -> set[str]:
    """Tables to describe in the prompt.

    We start from what stage 1 identified, add what joins need, and cap the count.
    Sending all 31 tables would cost roughly 6,000 tokens per question for no
    improvement: the model drowns as much as it learns.
    """
    tables = set(understanding.tables)
    if not tables:
        tables = {"patient"}

    # `patient` is eICU's hub: almost every join goes through it.
    tables.add("patient")

    schema = sch.read_schema()
    tables = {t for t in tables if t in schema}
    if len(tables) > MAX_TABLES:
        # Keep the smallest: bulky fact tables (vitalperiodic, nursecharting) add
        # columns without helping.
        tables = set(sorted(tables, key=lambda t: schema[t].row_count)[:MAX_TABLES])
    return tables


def build_segments(
    understanding: Understanding, anonymization: Anonymization
) -> list[Segment]:
    """The outgoing prompt, split by origin so the gate can verify each part.

    This decomposition is what lets the gate check the DDL and the notes by
    **regenerating** them instead of scanning them word by word. Only the masked
    question goes through the allowlist — it is the only part that comes from
    outside.
    """
    tables = relevant_tables(understanding)
    segments = [
        Segment(INSTRUCTIONS, "authored"),
        Segment(sch.ddl(tables, with_row_counts=True), "schema"),
        Segment(anonymization.for_the_prompt(), "schema"),
    ]
    segments += [Segment(n, "glossary") for n in understanding.notes]
    segments.append(Segment(anonymization.masked_question, "question"))
    return segments


def build_messages(
    understanding: Understanding, anonymization: Anonymization
) -> list[dict[str, str]]:
    """Assemble the segments into the chat messages actually sent."""
    tables = relevant_tables(understanding)
    ddl = sch.ddl(tables, with_row_counts=True)

    notes = ""
    if understanding.notes:
        notes = "\n\nDomain notes:\n" + "\n".join(f"- {n}" for n in understanding.notes)

    user = (
        f"Schema:\n{ddl}\n\n"
        f"Bound parameters:\n{anonymization.for_the_prompt()}"
        f"{notes}\n\n"
        f"Question: {anonymization.masked_question}\n\n"
        f"SQL:"
    )
    return [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": user},
    ]


def generate(
    understanding: Understanding,
    anonymization: Anonymization,
    max_repairs: int = 1,
) -> Generation:
    """One call, then a targeted repair if the query is rejected."""
    messages = build_messages(understanding, anonymization)
    segments = build_segments(understanding, anonymization)
    expected = set(anonymization.mapping)

    total_ms = 0.0
    total_tokens = 0
    calls = 0
    history: list[str] = []
    last_reason = ""

    for attempt in range(max_repairs + 1):
        response = cloud.call(messages, segments=segments)
        calls += 1
        total_ms += response.ms
        total_tokens += response.tokens

        sql = cloud.extract_sql(response.text)
        history.append(sql)

        # Output sweep: standard practice is to sanitise the input *and* check the
        # response. Here the specific risk is the model writing a literal instead
        # of using the bound parameter it was given.
        egress_gate.sweep_response(sql)

        verdict = sql_validator.validate(sql, expected)

        if verdict.valid:
            return Generation(
                sql=sql, valid=True, target=response.target, ms=round(total_ms, 1),
                tokens=total_tokens, calls=calls, repairs=attempt, history=history,
            )

        last_reason = verdict.reason
        if attempt >= max_repairs:
            break

        # The repair tells the model what was wrong. That is what makes it more
        # effective than simply sampling again.
        _log.info("repairing: %s", verdict.reason)
        # The repair turn sends the model's own SQL back to it. That text did not
        # come from us, so it is added as a `question`-origin segment and gets the
        # full word check — a literal the model invented must not slip out on the
        # return trip.
        segments = segments + [
            Segment(sql, "question"),
            Segment(f"That query was rejected: {verdict.reason}.", "question"),
        ]
        messages = messages + [
            {"role": "assistant", "content": sql},
            {
                "role": "user",
                "content": (
                    f"That query was rejected: {verdict.reason}.\n"
                    "Rewrite it as ONE valid SQLite SELECT that respects the rules. "
                    "Output only the SQL."
                ),
            },
        ]

    return Generation(
        sql=history[-1] if history else "", valid=False, target="",
        ms=round(total_ms, 1), tokens=total_tokens, calls=calls,
        repairs=max_repairs, failure_reason=last_reason, history=history,
    )


# =================================================================================
#  Opaque arm — the schema is anonymised too
# =================================================================================
@dataclass
class OpaquePrompt:
    """One request's opaque prompt, and the dictionary that produced it.

    Everything travels together because the dictionary is drawn at random per
    request: rebuilding it later would produce a *different* set of labels, and
    anything shown from that rebuild would describe a prompt that was never sent.
    """

    pseudonyms: Any                      # pipeline.opaque.Pseudonyms
    messages: list[dict[str, str]]
    segments: list[Segment]
    question: str                        # the question, in labels
    parameters: str                      # the bound-parameter block, in labels

    def view(self) -> dict[str, Any]:
        """What the interface shows: the labels sent, and what they stood for.

        Only the labels of the question and its bound parameters are decoded.
        Decoding the DDL as well would list every column of every selected
        table — several hundred rows nobody reads — when the point to be made is
        about the sentence: this is what the question became.
        """
        sent = f"{self.question}\n{self.parameters}"
        used = set(re.findall(r"(?<![A-Za-z0-9_])([tc]\d+)(?![A-Za-z0-9_])", sent))
        reverse = {
            **self.pseudonyms.reverse_tables,
            **self.pseudonyms.reverse_columns,
        }
        return {
            "question": self.question,
            "parameters": self.parameters,
            "ddl": self.pseudonyms.ddl,
            "labels": {alias: reverse[alias] for alias in sorted(used) if alias in reverse},
            "tables": len(self.pseudonyms.tables),
            "columns": len(self.pseudonyms.columns),
        }


def build_opaque(
    understanding: Understanding, anonymization: Anonymization, seed: int | None = None
) -> OpaquePrompt:
    """Assemble the prompt in which nothing business-related appears."""
    from hybridsql.pipeline import opaque

    tables = relevant_tables(understanding)
    pseudonyms = opaque.build(tables, seed=seed)
    question, subject_notes = opaque.opaque_question(anonymization.masked_question, pseudonyms)

    # Bound parameters are described against the *opaque* column, which is what
    # carries the local semantic work across: the provider is told where each
    # value belongs without being told what either of them is.
    if anonymization.mapping:
        described = []
        for symbol in sorted(anonymization.mapping, key=lambda s: int(s.lstrip(":v") or 0)):
            column = anonymization.columns.get(symbol)
            described.append(
                f"  {symbol} = a value of {pseudonyms.column(column)}" if column
                else f"  {symbol} = a number given by the analyst"
            )
        parameters = "\n".join(described)
    else:
        parameters = "  (none)"

    notes = "\n".join(f"- {n}" for n in subject_notes)
    user = (
        f"Schema:\n{pseudonyms.ddl}\n\n"
        f"Bound parameters:\n{parameters}\n\n"
        + (f"Notes:\n{notes}\n\n" if notes else "")
        + f"Question: {question}\n\nSQL:"
    )

    messages = [
        {"role": "system", "content": INSTRUCTIONS_OPAQUE},
        {"role": "user", "content": user},
    ]
    segments = [
        Segment(INSTRUCTIONS_OPAQUE, "authored"),
        # The opaque DDL and the opaque question are checked word by word like any
        # untrusted text. They should contain nothing but labels and grammar, and
        # if a real identifier or value ever survived the rewriting, the gate is
        # what notices.
        Segment(pseudonyms.ddl, "question"),
        Segment(parameters, "question"),
        Segment(question, "question"),
    ]
    segments += [Segment(n, "template") for n in subject_notes]
    return OpaquePrompt(
        pseudonyms=pseudonyms,
        messages=messages,
        segments=segments,
        question=question,
        parameters=parameters,
    )


def generate_opaque(
    understanding: Understanding,
    anonymization: Anonymization,
    max_repairs: int = 1,
    prompt: OpaquePrompt | None = None,
) -> Generation:
    """Generate SQL over an anonymised schema, then translate it back.

    The repair loop matters more here than in the clear-schema arm. A model given
    opaque labels invents them more often — `FROM t9` when only t1..t4 exist — and
    that is a failure the loop can actually fix, because the error message names
    the labels it was allowed to use.

    `prompt` may be supplied by a caller that already built it — the graph node
    does, so that it can display the gate's verdict on the very prompt that goes
    out, even when the gate refuses it and no call happens at all.
    """
    from hybridsql.pipeline import opaque

    prompt = prompt or build_opaque(understanding, anonymization)
    pseudonyms, messages, segments = prompt.pseudonyms, prompt.messages, prompt.segments
    view = prompt.view()
    expected = set(anonymization.mapping)

    total_ms = 0.0
    total_tokens = 0
    calls = 0
    history: list[str] = []
    last_reason = ""

    for attempt in range(max_repairs + 1):
        response = cloud.call(messages, segments=segments, context="sql-generation-opaque")
        calls += 1
        total_ms += response.ms
        total_tokens += response.tokens

        raw = cloud.extract_sql(response.text)
        invented = opaque.leftovers(raw, pseudonyms)
        sql = opaque.restore(raw, pseudonyms)
        history.append(sql)

        egress_gate.sweep_response(sql)

        if invented:
            verdict = sql_validator.Verdict(
                False, f"unknown label: {', '.join(invented)} — not in the schema you were given"
            )
        else:
            verdict = sql_validator.validate(sql, expected)

        if verdict.valid:
            return Generation(
                sql=sql, valid=True, target=response.target, ms=round(total_ms, 1),
                tokens=total_tokens, calls=calls, repairs=attempt, history=history,
                segments=segments, opaque=view,
            )

        last_reason = verdict.reason
        if attempt >= max_repairs:
            break

        _log.info("repairing (opaque): %s", verdict.reason)
        # The model is corrected in its own vocabulary. Handing back the restored
        # SQL would show it the real identifiers, which would defeat the arm.
        segments = segments + [Segment(raw, "question")]
        messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    f"That query was rejected: {verdict.reason}.\n"
                    "Rewrite it as ONE valid SQLite SELECT using only the labels in "
                    "the schema above. Output only the SQL."
                ),
            },
        ]

    return Generation(
        sql=history[-1] if history else "", valid=False, target="",
        ms=round(total_ms, 1), tokens=total_tokens, calls=calls,
        repairs=max_repairs, failure_reason=last_reason, history=history,
        segments=segments, opaque=view,
    )
