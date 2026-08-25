"""Run every variant over the same questions and say which one is best.

Accuracy is measured against a hand-written reference query where one exists, and
otherwise against the result at least two variants agreed on. Executable rate is a
floor, not a measure of correctness; perplexity is the only signal needing no
answer key.

Cost and latency sit beside them because the point is not the most accurate
variant but the best trade.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nl2sql.config import DATA
from nl2sql.core import graph
from nl2sql.core.state import State, is_refusal
from nl2sql.db.sqlite import execute
from nl2sql.optimize.variants import VARIANTS

# US dollars per million tokens, as published in August 2026. Used only to turn
# token counts into a number a reader can compare; change these, not the code.
PRICE_PER_MTOK = {
    "openrouter/openai/gpt-4.1-nano": (0.10, 0.40),
    "groq/openai/gpt-oss-20b": (0.10, 0.50),
    "groq/openai/gpt-oss-120b": (0.15, 0.75),
    "groq/qwen/qwen3.6-27b": (0.29, 0.59),
}


def fingerprint_result(sql: str, parameters: dict[str, Any] | None = None) -> str:
    """A hash of what a query returns, so two queries can be compared by result."""
    try:
        columns, rows = execute(sql, parameters, max_rows=500)
    except Exception as e:  # noqa: BLE001
        return f"error:{type(e).__name__}"
    if not rows:
        return "empty"
    body = sorted("\x1f".join("" if v is None else str(v) for v in row) for row in rows)
    return hashlib.sha256(("\x1e".join(body)).encode("utf-8")).hexdigest()[:16]


def cost(state: State) -> float:
    """Dollars this run would have cost at list price."""
    price = PRICE_PER_MTOK.get(state.get("cloud_target") or "", (0.0, 0.0))
    return round(
        (state.get("prompt_tokens", 0) * price[0] + state.get("completion_tokens", 0) * price[1])
        / 1_000_000,
        8,
    )


@dataclass
class Result:
    """One variant answering one question."""

    question: str
    variant: str
    success: bool
    refusal: bool
    sql: str = ""
    result_hash: str = ""
    correct: bool | None = None      # None when the question has no reference query
    rows: int = 0
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    escalated: bool = False
    perplexity: float | None = None
    difficulty: float = 0.0
    ms: float = 0.0
    dollars: float = 0.0
    model: str = ""
    failed_stage: str = ""
    failure_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__


def load_questions(path: Path | None = None) -> list[dict[str, Any]]:
    """The benchmark set: a question, and optionally the reference query for it."""
    import yaml

    source = Path(path or DATA / "questions.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or []
    return [q for q in raw if q.get("question")]


def ask(question: str, variant: str, arm: str = "hybrid") -> Result:
    """Run one question through one variant and record everything measurable."""
    started = time.perf_counter()
    state = graph.run(question, arm=arm, write=False, variant=variant)
    elapsed = round((time.perf_counter() - started) * 1000, 1)

    result = Result(
        question=question,
        variant=variant,
        success=bool(state.get("success")),
        refusal=is_refusal(state),
        sql=state.get("sql", ""),
        rows=state.get("row_count", 0),
        tokens=state.get("cloud_tokens", 0),
        prompt_tokens=state.get("prompt_tokens", 0),
        completion_tokens=state.get("completion_tokens", 0),
        calls=state.get("cloud_calls", 0),
        escalated=bool(state.get("escalated")),
        perplexity=state.get("perplexity"),
        difficulty=state.get("difficulty", 0.0),
        ms=elapsed,
        dollars=cost(state),
        model=state.get("cloud_target", ""),
        failed_stage=state.get("failed_stage", ""),
        failure_reason=state.get("failure_reason", "")[:200],
    )
    if result.success:
        masked = state.get("masked")
        result.result_hash = fingerprint_result(
            result.sql, masked.parameters() if masked is not None else None
        )
    return result


def score_against_reference(results: list[Result], reference: dict[str, str]) -> None:
    """Mark each result correct or not, where a reference answer exists."""
    for result in results:
        expected = reference.get(result.question)
        if expected:
            result.correct = result.success and result.result_hash == expected


def consensus_reference(results: list[Result]) -> dict[str, str]:
    """For questions with no hand-written query: what most variants agreed on.

    A single variant's answer is never taken as the reference — that would score every
    other variant against one arbitrary opinion.
    """
    by_question: dict[str, list[str]] = {}
    for result in results:
        if result.success and result.result_hash not in ("", "empty"):
            by_question.setdefault(result.question, []).append(result.result_hash)

    reference: dict[str, str] = {}
    for question, hashes in by_question.items():
        counts = {h: hashes.count(h) for h in set(hashes)}
        best, votes = max(counts.items(), key=lambda kv: kv[1])
        if votes >= 2:
            reference[question] = best
    return reference


@dataclass
class Summary:
    """One variant, across the whole question set."""

    variant: str
    asked: int = 0
    executed: int = 0
    correct: int = 0
    scored: int = 0
    refusals: int = 0
    escalations: int = 0
    tokens: int = 0
    dollars: float = 0.0
    ms_median: float = 0.0
    perplexity_median: float | None = None
    failures: dict[str, int] = field(default_factory=dict)

    @property
    def executable_rate(self) -> float:
        return round(self.executed / self.asked, 4) if self.asked else 0.0

    @property
    def accuracy(self) -> float:
        return round(self.correct / self.scored, 4) if self.scored else 0.0

    @property
    def tokens_per_question(self) -> int:
        return round(self.tokens / self.asked) if self.asked else 0

    def as_row(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "accuracy": self.accuracy,
            "executable": self.executable_rate,
            "tokens/question": self.tokens_per_question,
            "dollars/100q": round(self.dollars / self.asked * 100, 4) if self.asked else 0.0,
            "median ms": self.ms_median,
            "median perplexity": self.perplexity_median,
            "escalations": self.escalations,
            "refusals": self.refusals,
            "scored on": self.scored,
        }


def summarise(variant: str, results: list[Result]) -> Summary:
    mine = [r for r in results if r.variant == variant]
    scored = [r for r in mine if r.correct is not None]
    latencies = [r.ms for r in mine] or [0.0]
    perplexities = [r.perplexity for r in mine if r.perplexity is not None]

    failures: dict[str, int] = {}
    for r in mine:
        if not r.success and r.failed_stage:
            failures[r.failed_stage] = failures.get(r.failed_stage, 0) + 1

    return Summary(
        variant=variant,
        asked=len(mine),
        executed=sum(1 for r in mine if r.success),
        correct=sum(1 for r in scored if r.correct),
        scored=len(scored),
        refusals=sum(1 for r in mine if r.refusal),
        escalations=sum(1 for r in mine if r.escalated),
        tokens=sum(r.tokens for r in mine),
        dollars=round(sum(r.dollars for r in mine), 6),
        ms_median=round(statistics.median(latencies), 1),
        perplexity_median=round(statistics.median(perplexities), 3) if perplexities else None,
        failures=failures,
    )


def rank(summaries: list[Summary]) -> list[tuple[str, float]]:
    """Order by accuracy first, then by what it cost to get it.

    Cost is a tie-breaker, not a term traded against accuracy: how much accuracy a euro
    is worth is the reader's call, which is why the full table is printed too.
    """
    return [
        (s.variant, s.accuracy)
        for s in sorted(summaries, key=lambda s: (-s.accuracy, s.tokens_per_question, s.ms_median))
    ]


def compare(
    variants: list[str] | None = None,
    questions: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    destination: Path | None = None,
) -> dict[str, Any]:
    """Run every variant over every question, score, and write the report."""
    names = variants or list(VARIANTS)
    items = questions if questions is not None else load_questions()
    if limit:
        items = items[:limit]

    results: list[Result] = []
    for item in items:
        for name in names:
            result = ask(item["question"], name)
            results.append(result)
            flag = "ok " if result.success else "-- "
            print(f"  {flag}{name:<10} {result.tokens:>5}t {result.ms:>7.0f}ms  {item['question'][:60]}",
                  flush=True)

    reference = {q["question"]: "" for q in items if q.get("sql")}
    for question in list(reference):
        gold = next(q["sql"] for q in items if q["question"] == question)
        reference[question] = fingerprint_result(gold)
    reference = {q: h for q, h in reference.items() if h and not h.startswith("error:")}
    reference.update({q: h for q, h in consensus_reference(results).items() if q not in reference})
    score_against_reference(results, reference)

    summaries = [summarise(name, results) for name in names]
    report = {
        "questions": len(items),
        "with_reference_query": sum(1 for q in items if q.get("sql")),
        "scored_by_consensus": len(reference) - sum(1 for q in items if q.get("sql")),
        "table": [s.as_row() for s in summaries],
        "ranking": rank(summaries),
        "failures": {s.variant: s.failures for s in summaries},
        "results": [r.as_dict() for r in results],
    }

    out = Path(destination or DATA / "benchmark.json")
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report


def calibrate(results: list[Result] | None = None) -> dict[str, Any]:
    """Where to put the escalation threshold, from measured perplexities."""
    source = results or [Result(**r) for r in json.loads(
        (DATA / "benchmark.json").read_text(encoding="utf-8")
    )["results"]]
    scored = [r for r in source if r.perplexity is not None and r.correct is not None]
    if not scored:
        return {"threshold": None, "reason": "no scored run carries a perplexity"}

    right = [r.perplexity for r in scored if r.correct]
    wrong = [r.perplexity for r in scored if not r.correct]
    if not right or not wrong:
        return {"threshold": None, "reason": "every scored run landed on the same side"}

    candidates = sorted({round(p, 2) for p in right + wrong})
    best, best_score = candidates[0], -1.0
    for threshold in candidates:
        kept = sum(1 for p in right if p <= threshold)
        caught = sum(1 for p in wrong if p > threshold)
        score = kept / len(right) + caught / len(wrong)
        if score > best_score:
            best, best_score = threshold, score

    return {
        "threshold": best,
        "median_when_right": round(statistics.median(right), 3),
        "median_when_wrong": round(statistics.median(wrong), 3),
        "separates": round(best_score / 2, 3),
        "samples": len(scored),
    }
