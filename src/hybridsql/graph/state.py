"""The state that travels through the graph.

One dictionary, three arms
--------------------------
The same state object serves Full Cloud, Hybrid and Full Local. That is a
deliberate constraint rather than a convenience: if each arm carried its own
shape, the benchmark would end up comparing three different measurements. Here
every arm fills the same fields, so a column that is empty is empty for a reason
you can point at — Full Local has no `cloud_target` because it never called one.

What is in here that must never be serialised outward
-----------------------------------------------------
`anonymization.mapping` (`:v1 -> 'AMOXICILLIN 500 MG PO CAPS'`) and `rows`. The
first is the system's only secret; the second is the data itself. The API layer
returns `public()`, never the state.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

# The three architectures the internship has to compare.
#
#   full_cloud   the question, unmasked, plus the schema go to the provider, and
#                the provider also writes the final answer from the returned
#                rows. The accuracy ceiling, and the privacy floor.
#   hybrid       this project. The provider sees the schema and a masked
#                question, nothing else. Execution and answer writing are local.
#   hybrid_opaque  as hybrid, but the schema is pseudonymised too: the provider
#                sees t1/c7 labels and no business word at all.
#   full_local   nothing leaves. A 1.7B model writes the SQL itself.
Arm = Literal["full_cloud", "hybrid", "hybrid_opaque", "full_local"]

ARMS: tuple[Arm, ...] = ("full_cloud", "hybrid", "hybrid_opaque", "full_local")


class PipelineState(TypedDict, total=False):
    """Everything a run accumulates, in the order the nodes fill it."""

    # --- input -------------------------------------------------------------------
    question: str
    arm: Arm
    write: bool                 # run the answer writer, or stop at the rows

    # --- stage 1: understand (local) ---------------------------------------------
    understanding: Any          # pipeline.understand.Understanding
    tables: list[str]
    notes: list[str]
    extractor: str

    # --- stage 2a: anonymize (local) ---------------------------------------------
    anonymization: Any          # pipeline.anonymize.Anonymization — HOLDS THE SECRET
    masked_question: str
    symbol_count: int

    # --- stage 2b: generate ------------------------------------------------------
    sql: str
    sql_history: list[str]
    cloud_target: str
    cloud_tokens: int
    cloud_calls: int
    repairs: int
    sql_author: str             # which model wrote the SQL, for the benchmark table
    opaque: dict[str, Any]      # Hybrid Opaque only: the t/c labels as transmitted

    # --- stage 3: execute and answer (local, except in Full Cloud) ---------------
    columns: list[str]
    rows: list[tuple]           # REAL DATA
    row_count: int
    answer: str
    answer_author: str

    # --- control -----------------------------------------------------------------
    success: bool
    failed_stage: str
    failure_reason: str
    refusal: bool               # a refusal by design, not a defect — see `is_refusal`
    suggestions: list[str]      # stored values to offer when a mention did not resolve

    # --- measurement -------------------------------------------------------------
    ms: dict[str, float]
    egress_chars: int           # characters that actually left the machine
    egress_values: int          # real values among them. The number the report turns on.
    egress_segments: list[dict[str, Any]]   # the gate's verdict per prompt segment


# A failure at one of these stages means the architecture worked. Counting them as
# errors would score the system down for protecting the data it was built to
# protect, which is how a privacy benchmark ends up rewarding the leakiest arm.
REFUSAL_STAGES = frozenset({"anonymize", "egress_gate"})


def blank(question: str, arm: Arm = "hybrid", write: bool = True) -> PipelineState:
    return PipelineState(
        question=question,
        arm=arm,
        write=write,
        tables=[],
        notes=[],
        sql="",
        sql_history=[],
        columns=[],
        rows=[],
        row_count=0,
        answer="",
        success=False,
        failed_stage="",
        failure_reason="",
        refusal=False,
        ms={},
        egress_chars=0,
        egress_values=0,
        symbol_count=0,
        cloud_tokens=0,
        cloud_calls=0,
        repairs=0,
    )


def is_refusal(state: PipelineState) -> bool:
    return state.get("failed_stage", "") in REFUSAL_STAGES


def public(state: PipelineState) -> dict[str, Any]:
    """The view that may cross a network boundary — the API's response body.

    Deliberately built by listing what goes *in*, not by removing what stays out.
    A denylist over a dataclass grows a hole the first time somebody adds a field;
    this one stays closed by default.
    """
    return {
        "question": state.get("question", ""),
        "arm": state.get("arm", ""),
        "success": state.get("success", False),
        "refusal": is_refusal(state),
        "answer": state.get("answer", ""),
        "sql": state.get("sql", ""),
        "masked_question": state.get("masked_question", ""),
        # Hybrid Opaque only. Like the `:v1` mapping, this is shown to the analyst
        # who asked — it is the provider that must not see it, and the provider is
        # not on this connection.
        "opaque": dict(state.get("opaque", {})),
        "columns": list(state.get("columns", [])),
        "row_count": state.get("row_count", 0),
        "failed_stage": state.get("failed_stage", ""),
        "failure_reason": state.get("failure_reason", ""),
        "suggestions": list(state.get("suggestions", [])),
        "symbol_count": state.get("symbol_count", 0),
        "sql_author": state.get("sql_author", ""),
        "answer_author": state.get("answer_author", ""),
        "cloud_target": state.get("cloud_target", ""),
        "cloud_tokens": state.get("cloud_tokens", 0),
        "cloud_calls": state.get("cloud_calls", 0),
        "repairs": state.get("repairs", 0),
        "egress_chars": state.get("egress_chars", 0),
        "egress_values": state.get("egress_values", 0),
        "ms": dict(state.get("ms", {})),
    }
