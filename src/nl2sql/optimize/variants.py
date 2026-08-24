"""Five ways of running the hybrid arm, so the benchmark can say which is best.

Each variant changes exactly one thing, and they share one code path — a variant
is a handful of flags, not a fork of the pipeline. That is what makes the
comparison mean something: if two variants differed in code as well as in
configuration, the benchmark would not know which difference produced the result.

    baseline   the pipeline as it stands. The reference.
    lean       prompt size    — only the columns the question reaches are described.
    fewshot    prompt content — three similar solved questions are shown as examples.
    cascade    model choice   — climbs a ladder of models, starting at the rung the
                                question looks like it needs and stopping as soon
                                as one answers confidently (see `perplexity`).
    consensus  sampling       — three cheap answers, executed; the majority result
                                wins, and perplexity breaks a tie.

**Perplexity** is how surprised a model was by its own output: low means it was
confident, high means it was guessing. It is the only quality signal available
without knowing the right answer in advance, which is why `cascade` uses it to
decide whether to climb and `consensus` uses it to break ties.

It is not free everywhere, and the report should say so: Groq refuses `logprobs`
on every model it serves, OpenRouter returns them. So the ladder is arranged with
an OpenRouter model on the bottom rung — the rung whose confidence decides whether
to climb is the one that has to be able to report it. Where a rung answers without
a perplexity, the decision falls back to what the validator says, which is a
weaker signal but never a wrong one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from nl2sql.config import DATA
from nl2sql.core import prompt as pr
from nl2sql.core.state import State
from nl2sql.core.steps import track
from nl2sql.llm import cloud
from nl2sql.nlp.understand import Understanding
from nl2sql.privacy import gate, sqlcheck
from nl2sql.privacy.mask import Masked

_log = logging.getLogger(__name__)

# A question the small model answered with a perplexity above this was guessing,
# and is asked again of the large one. Calibrated in `benchmark.calibrate`.
ESCALATE_ABOVE = 1.8

Strategy = Literal["single", "cascade", "consensus"]


@dataclass(frozen=True)
class Variant:
    """One configuration of the hybrid arm. Flags only — no behaviour of its own."""

    name: str
    changes: str                     # the single variable this one moves
    what: str                        # one plain sentence, shown in the interface
    size: cloud.Size = "large"       # the rung a single-shot variant uses
    strategy: Strategy = "single"
    lean: bool = False
    notes: bool = True
    fewshot: int = 0
    samples: int = 1
    escalate_above: float = ESCALATE_ABOVE
    max_repairs: int = 1

    @property
    def logprobs(self) -> bool:
        """Only ask for them where they are used: they add tokens to the response."""
        return self.strategy in ("cascade", "consensus")

    def build_prompt(self, understanding: Understanding, masked: Masked) -> pr.Prompt:
        built = pr.hybrid(understanding, masked, lean=self.lean, notes=self.notes)
        if self.fewshot:
            built = _with_examples(built, masked.question, self.fewshot)
        built.view["variant"] = self.name
        return built


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in (
        Variant("baseline", "nothing", "The pipeline as it stands: full tables, the large model."),
        Variant("lean", "prompt size",
                "Describes only the columns the question needs, to see if a shorter prompt costs accuracy.",
                lean=True),
        Variant("fewshot", "prompt content",
                "Shows the model three similar questions that were answered correctly before.",
                lean=True, fewshot=3),
        Variant("cascade", "model choice",
                "Starts at the cheapest model that looks able to answer and climbs only when it must.",
                size="small", strategy="cascade"),
        Variant("consensus", "sampling",
                "Asks a cheap model three times, runs all three queries, and keeps the answer they agree on.",
                size="small", strategy="consensus", samples=3),
    )
}

DEFAULT = "baseline"


# ---------------------------------------------------------------------------------
#  The one code path
# ---------------------------------------------------------------------------------
def run(state: State, built: pr.Prompt, opaque_arm: bool = False) -> dict[str, Any]:
    """Generate the SQL for this state, by whichever strategy the variant asks for."""
    variant = VARIANTS.get(state.get("variant", DEFAULT), VARIANTS[DEFAULT])
    if opaque_arm:
        # The opaque arm is an architecture, not a variant: it always runs single.
        variant = VARIANTS[DEFAULT]

    if variant.strategy == "consensus":
        return _consensus(state, built, variant)
    if variant.strategy == "cascade":
        return _cascade(state, built, variant)
    return _single(state, built, variant, opaque_arm)


@dataclass
class Attempt:
    """One call and what came back, before anything is chosen."""

    sql: str
    valid: bool
    reason: str
    response: Any
    perplexity: float | None = None


def _ask(built: pr.Prompt, variant: Variant, size: cloud.Size, expected: set[str],
         opaque_arm: bool, temperature: float = 0.0) -> Attempt:
    """One call, then validation. The only place a provider is reached."""
    response = cloud.call(
        built.messages,
        segments=built.segments,
        size=size,
        temperature=temperature,
        logprobs=variant.logprobs,
        context="sql-generation-opaque" if opaque_arm else "sql-generation",
    )
    raw = cloud.extract_sql(response.text)
    sql, invented = _decode(raw, built.pseudonyms) if opaque_arm else (raw, [])

    with track("validate", sql=sql) as step:
        gate.sweep_response(sql)
        if invented:
            verdict = sqlcheck.Verdict(False, f"unknown label: {', '.join(invented)}")
        else:
            verdict = sqlcheck.validate(sql, expected)
        step.say(
            "the query is safe to run" if verdict.valid else f"rejected — {verdict.reason}",
            valid=verdict.valid,
        )
    return Attempt(sql, verdict.valid, verdict.reason, response, response.perplexity)


def _decode(raw: str, pseudonyms: Any) -> tuple[str, list[str]]:
    from nl2sql.privacy import opaque as opq

    if pseudonyms is None:
        return raw, []
    return opq.restore(raw, pseudonyms), opq.invented(raw, pseudonyms)


def _single(state: State, built: pr.Prompt, variant: Variant, opaque_arm: bool) -> dict[str, Any]:
    """One call, then one targeted repair if the query was rejected.

    Not five candidates: free Groq caps at 30 requests a minute, and a model told
    *why* it was wrong corrects better than an extra sample. Measured across 107
    questions: 0 repairs were needed.
    """
    expected = set(state["masked"].mapping)
    totals = _Totals()
    history: list[str] = []
    attempt = None
    reason = ""

    for round_number in range(variant.max_repairs + 1):
        attempt = _ask(built, variant, variant.size, expected, opaque_arm)
        totals.add(attempt.response)
        history.append(attempt.sql)
        if attempt.valid:
            return _result(built, attempt, history, totals, round_number, opaque_arm)
        reason = attempt.reason
        if round_number >= variant.max_repairs:
            break
        _log.info("repairing: %s", reason)
        built = pr.repair(built, attempt.sql, reason, opaque_arm)

    out = _result(built, attempt, history, totals, variant.max_repairs, opaque_arm)
    out.update(success=False, failed_stage="sql_validation", failure_reason=reason)
    return out


def _cascade(state: State, built: pr.Prompt, variant: Variant) -> dict[str, Any]:
    """Climb the ladder: start where the question looks like it belongs, stop when
    a rung answers confidently.

    Two things keep this from burning calls. The starting rung comes from the
    difficulty score, so a question with four joins and a ratio does not waste a
    call on the smallest model first. And a rung that answers is only accepted if
    the validator passed *and* its perplexity says it was not guessing — those are
    the two ways an answer can be wrong, and both are checked before stopping.
    """
    expected = set(state["masked"].mapping)
    totals = _Totals()
    history: list[str] = []

    with track("route", question=state["question"]) as step:
        score = difficulty(state["understanding"])
        first = starting_rung(score)
        step.say(
            f"this question scores {score:.2f} out of 1 for difficulty — starting at "
            f"the {first} model",
            difficulty=score,
            starting_rung=first,
        )

    rungs = cloud.LADDER[cloud.LADDER.index(first):]
    attempt = None
    for rung in rungs:
        attempt = _ask(built, variant, rung, expected, False)
        totals.add(attempt.response)
        history.append(attempt.sql)

        unsure = (attempt.perplexity or 0.0) > variant.escalate_above
        if attempt.valid and not unsure:
            out = _result(built, attempt, history, totals, 0, False)
            out.update(difficulty=score, escalated=rung != first, rung=rung)
            return out

        if rung != rungs[-1]:
            with track("rank") as step:
                step.say(
                    f"the {rung} model was rejected ({attempt.reason}) — climbing"
                    if not attempt.valid
                    else f"the {rung} model was unsure (perplexity {attempt.perplexity}) — climbing",
                    rung=rung,
                    perplexity=attempt.perplexity,
                )

    out = _result(built, attempt, history, totals, 0, False)
    out.update(difficulty=score, escalated=True, rung=rungs[-1])
    if not attempt.valid:
        out.update(success=False, failed_stage="sql_validation", failure_reason=attempt.reason)
    return out


def starting_rung(score: float) -> cloud.Size:
    """Which model a question of this difficulty is worth asking first.

    The thresholds are deliberately generous towards the small model: a wasted
    cheap call costs a fraction of a cent, while starting too high costs the whole
    point of the exercise.
    """
    if score < 0.35:
        return "small"
    return "medium" if score < 0.70 else "large"


def _consensus(state: State, built: pr.Prompt, variant: Variant) -> dict[str, Any]:
    """Several cheap answers; keep the one whose *result* the others agree with.

    Voting on the executed result rather than on the SQL text is what makes this
    work: two differently written queries returning the same number agree, and a
    text comparison would call them different. Perplexity breaks a tie.
    """
    from nl2sql.optimize.benchmark import fingerprint_result

    expected = set(state["masked"].mapping)
    parameters = state["masked"].parameters()
    totals = _Totals()
    attempts: list[Attempt] = []

    for i in range(variant.samples):
        attempt = _ask(built, variant, variant.size, expected, False,
                       temperature=0.0 if i == 0 else 0.6)
        totals.add(attempt.response)
        attempts.append(attempt)

    valid = [a for a in attempts if a.valid]
    if not valid:
        out = _result(built, attempts[0], [a.sql for a in attempts], totals, 0, False)
        out.update(success=False, failed_stage="sql_validation", failure_reason=attempts[0].reason)
        return out

    with track("rank", candidates=len(valid)) as step:
        votes: dict[str, list[Attempt]] = {}
        for attempt in valid:
            votes.setdefault(fingerprint_result(attempt.sql, parameters), []).append(attempt)

        # Most agreement first; among equals, the one the model was surest of.
        winner_key = max(
            votes,
            key=lambda k: (len(votes[k]), -min((a.perplexity or 9.9) for a in votes[k])),
        )
        group = votes[winner_key]
        chosen = min(group, key=lambda a: a.perplexity or 9.9)
        step.say(
            f"{len(group)} of {len(valid)} queries returned the same result — keeping that one"
            if len(group) > 1
            else "the queries disagreed; keeping the one the model was surest of",
            agreement=f"{len(group)}/{len(valid)}",
            perplexities=[a.perplexity for a in valid],
        )

    out = _result(built, chosen, [a.sql for a in attempts], totals, 0, False)
    out["candidates"] = [
        {"sql": a.sql, "perplexity": a.perplexity, "chosen": a is chosen} for a in valid
    ]
    return out


class _Totals:
    """Token and call counters, accumulated across however many calls a variant makes."""

    def __init__(self) -> None:
        self.tokens = self.prompt = self.completion = self.calls = 0

    def add(self, response: Any) -> None:
        self.calls += 1
        self.tokens += response.tokens
        self.prompt += response.prompt_tokens
        self.completion += response.completion_tokens


def _result(built: pr.Prompt, attempt: Attempt, history: list[str], totals: _Totals,
            repairs: int, opaque_arm: bool) -> dict[str, Any]:
    return {
        "sql": attempt.sql,
        "sql_history": history,
        "sql_author": attempt.response.target,
        "cloud_target": attempt.response.target,
        "cloud_tokens": totals.tokens,
        "prompt_tokens": totals.prompt,
        "completion_tokens": totals.completion,
        "cloud_calls": totals.calls,
        "repairs": repairs,
        "perplexity": attempt.perplexity,
        "egress_chars": built.characters,
        # Zero by construction: the gate verified every segment before the socket
        # opened. It is the gate that says so, not this line.
        "egress_values": 0,
        "egress_segments": gate.verdicts(built.segments),
        "opaque": built.view if opaque_arm else {},
    }


# ---------------------------------------------------------------------------------
#  Worked examples — the cheapest accuracy there is
# ---------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def example_bank() -> tuple[tuple[str, str], ...]:
    """Solved (masked question, SQL) pairs from `data/examples.yaml`.

    Every pair is checked against the value denylist when it is loaded, and one
    carrying a stored value is dropped rather than shown to a provider — an
    example is prompt text like any other, and it goes out on every request that
    uses it. Each surviving pair is registered as a constant, so the gate lets it
    through on its fingerprint instead of word by word.
    """
    import yaml

    path = DATA / "examples.yaml"
    if not path.exists():
        return ()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []

    kept: list[tuple[str, str]] = []
    for item in raw:
        question, sql = str(item.get("question", "")).strip(), str(item.get("sql", "")).strip()
        if not question or not sql:
            continue
        if gate.find_known_values(f"{question} {sql}"):
            _log.warning("example dropped, it contains a stored value: %s", question)
            continue
        gate.register_constant(_render_example(question, sql))
        kept.append((question, sql))
    return tuple(kept)


def _render_example(question: str, sql: str) -> str:
    return "Question: " + question + "\nSQL: " + sql


def _with_examples(built: pr.Prompt, question: str, k: int) -> pr.Prompt:
    """Show the model the k most similar questions it has already seen answered."""
    from rapidfuzz import fuzz

    bank = example_bank()
    if not bank:
        return built

    ranked = sorted(bank, key=lambda pair: -fuzz.token_set_ratio(question, pair[0]))[:k]
    rendered = [_render_example(q, sql) for q, sql in ranked]
    block = "Examples of questions already answered correctly:\n\n" + "\n\n".join(rendered)
    gate.register_constant(block)

    user = built.messages[-1]
    merged = block + "\n\n" + user["content"]
    return pr.Prompt(
        messages=[*built.messages[:-1], {**user, "content": merged}],
        segments=[*built.segments, gate.Segment(block, "authored")],
        pseudonyms=built.pseudonyms,
        view={**built.view, "examples": [q for q, _ in ranked]},
    )


# ---------------------------------------------------------------------------------
#  How hard is this question?
# ---------------------------------------------------------------------------------
def difficulty(understanding: Understanding) -> float:
    """A 0-to-1 score, from what stage 1 already worked out. No model, no guessing.

    Everything here is a real cost driver for the query that has to be written:
    more tables means more joins, more values means more filters, and a question
    asking for a rate or a ranking needs an aggregate the model must get right.
    """
    question = understanding.question.lower()
    tables = len(understanding.tables)
    values = len(understanding.values)
    hard_words = sum(
        word in question
        for word in ("per ", "each ", "average", "rate", "ratio", "percentage", "proportion",
                     "most", "least", "top", "compare", "between", "trend", "median")
    )
    score = 0.25 * min(tables / 4, 1) + 0.25 * min(values / 3, 1) + 0.5 * min(hard_words / 3, 1)
    return round(min(score, 1.0), 3)


def catalogue() -> list[dict[str, str]]:
    """The variant table, for `/meta` and the notebooks."""
    return [
        {"name": v.name, "changes": v.changes, "what": v.what,
         "model": v.size, "strategy": v.strategy}
        for v in VARIANTS.values()
    ]
