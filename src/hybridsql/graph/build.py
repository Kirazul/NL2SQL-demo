"""Assembling the three architectures as LangGraph state machines.

Why an orchestrator at all
--------------------------
`pipeline/answer.py` already runs the hybrid chain, correctly, in eighty lines of
straight-line Python with try/except around each stage. Replacing it with a graph
buys three things it could not give:

1. **Three arms from one description.** Full Cloud, Hybrid and Full Local differ
   by which nodes they contain, not by their code. Written as functions they
   would be three near-copies drifting apart; written as graphs they share every
   node they have in common, so a fix to `execute` is a fix in all three.
2. **A drawing that cannot lie.** `mermaid()` renders the graph that actually
   ran. An architecture diagram maintained by hand is out of date the day after
   it is drawn; this one is generated from the object being executed.
3. **A trace with structure.** Each node is a span, so the tracing backend shows
   where the time went per stage instead of one flat duration.

What it deliberately does not buy
---------------------------------
No LangChain LLM wrapper, no retriever, no agent, no tool-calling loop. The
providers stay in `providers/`, which is what keeps the claim "exactly one module
opens a socket" checkable by reading one file. LangGraph here is a scheduler and
nothing more.

Routing
-------
Every node may fail, and a failure ends the run. Rather than an error edge per
node, one predicate — `_completed` — is reused on every conditional edge: if
`failed_stage` is set, go to END. A refusal (a person named, or the gate firing)
travels the same path as a technical failure, and is distinguished at the end by
`state.is_refusal`, because the *control flow* is identical even though the two
mean opposite things about the system's health.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from hybridsql.graph import nodes
from hybridsql.graph.state import ARMS, Arm, PipelineState, blank, public
from hybridsql.obs import tracing


def _completed(state: PipelineState) -> str:
    """Continue, or stop because a stage failed or refused."""
    return "stop" if state.get("failed_stage") else "continue"


def _chain(graph: StateGraph, sequence: list[str]) -> None:
    """Wire a linear sequence where every step may end the run."""
    graph.add_edge(START, sequence[0])
    for current, following in zip(sequence, sequence[1:]):
        graph.add_conditional_edges(
            current, _completed, {"continue": following, "stop": END}
        )
    graph.add_edge(sequence[-1], END)


def build(arm: Arm = "hybrid") -> Any:
    """Compile the graph for one architecture."""
    graph = StateGraph(PipelineState)

    if arm == "hybrid":
        graph.add_node("understand", nodes.understand_node)
        graph.add_node("anonymize", nodes.anonymize_node)
        graph.add_node("generate", nodes.generate_hybrid_node)
        graph.add_node("execute", nodes.execute_node)
        graph.add_node("write", nodes.write_local_node)
        _chain(graph, ["understand", "anonymize", "generate", "execute", "write"])

    elif arm == "hybrid_opaque":
        # Same shape as hybrid; only the generation node differs, which is the
        # point of expressing the arms as graphs rather than as functions.
        graph.add_node("understand", nodes.understand_node)
        graph.add_node("anonymize", nodes.anonymize_node)
        graph.add_node("generate", nodes.generate_opaque_node)
        graph.add_node("execute", nodes.execute_node)
        graph.add_node("write", nodes.write_local_node)
        _chain(graph, ["understand", "anonymize", "generate", "execute", "write"])

    elif arm == "full_cloud":
        # No `anonymize`: its absence *is* this arm. The answer is written by the
        # provider too, so the rows leave as well as the question.
        graph.add_node("understand", nodes.understand_node)
        graph.add_node("generate", nodes.generate_full_cloud_node)
        graph.add_node("execute", nodes.execute_node)
        graph.add_node("write", nodes.write_cloud_node)
        _chain(graph, ["understand", "generate", "execute", "write"])

    elif arm == "full_local":
        graph.add_node("understand", nodes.understand_node)
        graph.add_node("generate", nodes.generate_full_local_node)
        graph.add_node("execute", nodes.execute_node)
        graph.add_node("write", nodes.write_local_node)
        _chain(graph, ["understand", "generate", "execute", "write"])

    else:
        raise ValueError(f"unknown arm: {arm}")

    return graph.compile()


@lru_cache(maxsize=len(ARMS))
def compiled(arm: Arm = "hybrid") -> Any:
    """Compiled graphs are immutable and cheap to keep — build each one once."""
    return build(arm)


def run(question: str, arm: Arm = "hybrid", write: bool = True) -> PipelineState:
    """Execute one question end to end. Never raises: failures come back in the state.

    The local sink records the run whatever happens, including the failures — a
    benchmark that only writes down its successes measures nothing.
    """
    tracing.configure()
    initial = blank(question, arm=arm, write=write)

    with tracing.record(question, arm=arm) as record:
        final: PipelineState = compiled(arm).invoke(
            initial,
            config={
                "run_name": f"hybridsql:{arm}",
                "metadata": {"arm": arm},
                "tags": [arm, "hybridsql"],
            },
        )
        for stage, ms in final.get("ms", {}).items():
            record.stage(stage, "local", ms)
        record.result = public(final)

    return final


def mermaid(arm: Arm = "hybrid") -> str:
    """The graph as a mermaid diagram, generated from the executed object.

    Paste into the report, or render inline in the notebook. It is worth
    preferring this over a hand-drawn figure for a reason that has nothing to do
    with effort: this one cannot describe an architecture that is not the one
    running.
    """
    return compiled(arm).get_graph().draw_mermaid()
