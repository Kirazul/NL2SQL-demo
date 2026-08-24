"""The state that travels through the graph.

One shape for every arm and every variant. That is a constraint rather than a
convenience: if each arm carried its own fields, the benchmark would be comparing
different measurements. Here an empty column is empty for a reason you can point
at — Full Local has no `cloud_target` because it never called one.

Two things in here must never be serialised outward: `masked.mapping`, which is
the system's only secret, and `rows`, which is the data itself. The API returns
`public()`, never the state.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

#   full_cloud     question and schema leave unmasked, and the provider writes the
#                  answer from the rows too. The accuracy ceiling, the privacy floor.
#   hybrid         this project: the provider sees the schema and a masked question.
#   hybrid_opaque  as hybrid, plus a pseudonymised schema — no business word leaves.
#   full_local     nothing leaves. A 1.7B model writes the SQL itself.
Arm = Literal["full_cloud", "hybrid", "hybrid_opaque", "full_local"]

ARMS: tuple[Arm, ...] = ("full_cloud", "hybrid", "hybrid_opaque", "full_local")

# A failure here means the architecture worked. Counting them as errors would
# score the system down for protecting the data it exists to protect.
REFUSAL_STAGES = frozenset({"mask", "gate"})


class State(TypedDict, total=False):
    question: str
    arm: Arm
    variant: str
    write: bool

    # stage 1 — understand (local)
    understanding: Any
    tables: list[str]
    notes: list[str]
    extractor: str

    # stage 2a — mask (local)
    masked: Any                 # privacy.mask.Masked — HOLDS THE SECRET
    masked_question: str
    symbol_count: int

    # stage 2b — generate
    sql: str
    sql_history: list[str]
    sql_author: str
    cloud_target: str
    cloud_tokens: int
    prompt_tokens: int
    completion_tokens: int
    cloud_calls: int
    repairs: int
    difficulty: float           # router score, 0..1
    rung: str                   # which model of the ladder actually answered
    perplexity: float | None    # confidence of the chosen candidate
    candidates: list[dict[str, Any]]
    escalated: bool             # the small model was not trusted, the large one ran
    opaque: dict[str, Any]

    # stage 3 — execute and answer (local, except in Full Cloud)
    columns: list[str]
    rows: list[tuple]           # REAL DATA
    row_count: int
    answer: str
    answer_author: str

    # control
    success: bool
    failed_stage: str
    failure_reason: str
    refusal: bool
    suggestions: list[str]

    # measurement
    ms: dict[str, float]
    trace: list[dict[str, Any]]   # every traced step, for the interface
    egress_chars: int
    egress_values: int          # real values among them — the number the report turns on
    egress_segments: list[dict[str, Any]]


def blank(question: str, arm: Arm = "hybrid", write: bool = True, variant: str = "baseline") -> State:
    return State(
        question=question, arm=arm, variant=variant, write=write,
        tables=[], notes=[], sql="", sql_history=[], columns=[], rows=[], row_count=0,
        answer="", success=False, failed_stage="", failure_reason="", refusal=False,
        candidates=[], escalated=False, difficulty=0.0, rung="", perplexity=None, trace=[],
        ms={}, egress_chars=0, egress_values=0, symbol_count=0,
        cloud_tokens=0, prompt_tokens=0, completion_tokens=0, cloud_calls=0, repairs=0,
    )


def is_refusal(state: State) -> bool:
    return state.get("failed_stage", "") in REFUSAL_STAGES


def public(state: State) -> dict[str, Any]:
    """The view allowed to cross a network boundary — the API's response body.

    Built by listing what goes *in*, never by removing what stays out: a denylist
    over a dataclass grows a hole the first time somebody adds a field.
    """
    keys = (
        "question", "arm", "variant", "answer", "sql", "masked_question", "columns",
        "row_count", "failed_stage", "failure_reason", "suggestions", "symbol_count",
        "sql_author", "answer_author", "cloud_target", "cloud_tokens", "prompt_tokens",
        "completion_tokens", "cloud_calls", "repairs", "egress_chars", "egress_values",
        "difficulty", "rung", "perplexity", "escalated",
    )
    out: dict[str, Any] = {k: state.get(k) for k in keys}
    out["success"] = state.get("success", False)
    out["refusal"] = is_refusal(state)
    # Shown to the analyst who asked and already knows the values; it is the
    # provider that must not see them, and the provider is not on this connection.
    out["opaque"] = dict(state.get("opaque", {}))
    out["ms"] = dict(state.get("ms", {}))
    return out
