"""The four architectures, assembled as LangGraph state machines."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from nl2sql.core import nodes, trace
from nl2sql.core.state import ARMS, Arm, State, blank, public

# Which nodes each arm is made of. Full Cloud has no `mask` — that absence *is*
# the arm — and it lets the provider write the answer, so the rows leave too.
LAYOUT: dict[Arm, list[tuple[str, Any]]] = {
    "hybrid": [
        ("understand", nodes.understand_node),
        ("mask", nodes.mask_node),
        ("generate", nodes.generate_hybrid_node),
        ("execute", nodes.execute_node),
        ("write", nodes.write_local_node),
    ],
    "hybrid_opaque": [
        ("understand", nodes.understand_node),
        ("mask", nodes.mask_node),
        ("generate", nodes.generate_opaque_node),
        ("execute", nodes.execute_node),
        ("write", nodes.write_local_node),
    ],
    "full_cloud": [
        ("understand", nodes.understand_node),
        ("generate", nodes.generate_full_cloud_node),
        ("execute", nodes.execute_node),
        ("write", nodes.write_cloud_node),
    ],
    "full_local": [
        ("understand", nodes.understand_node),
        ("generate", nodes.generate_full_local_node),
        ("execute", nodes.execute_node),
        ("write", nodes.write_local_node),
    ],
}


def _completed(state: State) -> str:
    """Continue, or stop because a stage failed or refused."""
    return "stop" if state.get("failed_stage") else "continue"


def build(arm: Arm = "hybrid") -> Any:
    """Compile one architecture."""
    if arm not in LAYOUT:
        raise ValueError(f"unknown arm: {arm}")

    graph = StateGraph(State)
    sequence = LAYOUT[arm]
    for name, fn in sequence:
        graph.add_node(name, fn)

    graph.add_edge(START, sequence[0][0])
    for (current, _), (following, _) in zip(sequence, sequence[1:], strict=False):
        graph.add_conditional_edges(current, _completed, {"continue": following, "stop": END})
    graph.add_edge(sequence[-1][0], END)
    return graph.compile()


@lru_cache(maxsize=len(ARMS))
def compiled(arm: Arm = "hybrid") -> Any:
    """Compiled graphs are immutable and cheap to keep — build each one once."""
    return build(arm)


def run(question: str, arm: Arm = "hybrid", write: bool = True, variant: str = "baseline") -> State:
    """Execute one question end to end. Never raises: failures come back in the state."""
    trace.configure()
    with trace.record(question, arm=arm, variant=variant) as recorded:
        final: State = compiled(arm).invoke(
            blank(question, arm=arm, write=write, variant=variant),
            config={
                "run_name": f"nl2sql:{arm}:{variant}",
                "metadata": {"arm": arm, "variant": variant},
                "tags": [arm, variant, "nl2sql"],
            },
        )
        recorded.result = public(final)
    return final


def mermaid(arm: Arm = "hybrid") -> str:
    """The graph as a diagram, generated from the object that runs."""
    return compiled(arm).get_graph().draw_mermaid()


def diagram(arm: Arm = "hybrid") -> bytes | None:
    """The same diagram drawn, as PNG bytes.

    `mermaid` returns source, and a notebook prints source as thirty lines of
    text — which is not a diagram. Rendering needs mermaid.ink, so this returns
    None rather than raising when there is no network: the caller can fall back
    to the source it would have drawn.
    """
    try:
        png: bytes = compiled(arm).get_graph().draw_mermaid_png()
    except Exception:
        return None
    return png
