"""The steps every pipeline is made of — named once, in plain language.

Each arm and each variant is a different route through the *same* list of steps,
so naming them here lets two traces be compared line by line and gives the
interface its wording.

    with track("lookup", mention="aspirin") as t:
        t.say("found ASPIRIN EC 81 MG PO TBEC in medication.drugname", score=1.0)

`say()` writes one sentence for the interface and keeps the detail for LangSmith.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from nl2sql.core import trace
from nl2sql.core.trace import Recorder, Zone


@dataclass(frozen=True)
class StepKind:
    """One named step: what it is called on screen, and which side of the line it runs on."""

    id: str
    label: str
    zone: Zone
    explains: str
    kind: str = "chain"   # "llm" for a model call, so token counts land in the dashboard


# The order is the order they happen in. `zone` is the default; a step that can
# run on either side (a model call, writing the answer) is told which at runtime.
STEPS: tuple[StepKind, ...] = (
    StepKind("understand", "Reading the question", "local",
             "The whole local stage: nothing has left the machine yet."),
    StepKind("extract", "Finding the words that matter", "local",
             "A small model on this machine picks out drugs, diagnoses, tests and names."),
    StepKind("classify", "Sorting words into values and concepts", "local",
             "'aspirin' is content to look up; 'mortality rate' is the name of a column."),
    StepKind("lookup", "Matching a word to what is really stored", "local",
             "The analyst writes 'aspirin'; the database holds 'ASPIRIN EC 81 MG PO TBEC'."),
    StepKind("arbitrate", "Deciding between a value and a column", "local",
             "'diagnosis names' asks for a column, not for a diagnosis called 'names'."),
    StepKind("scope", "Choosing which tables to describe", "local",
             "Only the tables the question needs are sent, not all 31."),
    StepKind("mask", "Hiding the real values", "local",
             "Each value becomes :v1, :v2. The list of what they stand for stays here."),
    StepKind("gate", "Checking what is allowed to leave", "local",
             "Every part of the message is verified before any connection is opened."),
    StepKind("prompt", "Writing the request for the model", "local",
             "Schema, bound parameters and the masked question, assembled."),
    StepKind("route", "Judging how hard the question is", "local",
             "An easy question does not need the most expensive model."),
    StepKind("model", "Asking a model to write the SQL", "cloud",
             "The only step that can leave the machine — and it never sees a value."),
    StepKind("rank", "Comparing the candidate queries", "local",
             "Perplexity: how sure the model was. Lower is more confident."),
    StepKind("validate", "Checking the SQL is safe and complete", "local",
             "One SELECT, no invented columns, every parameter used."),
    StepKind("execute", "Running the query on the database", "local",
             "Read-only, time-limited, with the real values bound in here."),
    StepKind("write", "Turning the result into a sentence", "local",
             "The only step that sees the actual rows, so it runs on this machine."),
)

BY_ID: dict[str, StepKind] = {s.id: s for s in STEPS}


@contextmanager
def track(
    step_id: str, zone: Zone | None = None, label: str | None = None, **inputs: Any
) -> Iterator[Recorder]:
    """Open one named step. Unknown ids fail loudly rather than tracing a typo.

    `label` overrides the wording where one step kind does two jobs — a model call
    writing SQL and one writing the answer are both `model`.
    """
    step = BY_ID[step_id]
    with trace.span(step.id, label or step.label, zone or step.zone, step.kind, **inputs) as recorder:
        yield recorder


def catalogue() -> list[dict[str, str]]:
    """The step table, for `/meta` — so the interface and the traces agree on wording."""
    return [
        {"id": s.id, "label": s.label, "zone": s.zone, "explains": s.explains} for s in STEPS
    ]
