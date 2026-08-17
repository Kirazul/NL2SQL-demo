"""The local answer writer: guard rails and prompt shaping.

These tests load no model. They cover what must be true before a single weight is
read: the backend refused in strict mode, the prompt that bounds what the model
can invent, the cleanup of its output.

Writing quality itself belongs to the evaluation (week 3), not to a unit test —
one does not test a language model by assertion.
"""

from __future__ import annotations

import pytest

from hybridsql.config import Settings
from hybridsql.providers import local_model as lm


# --- The trust boundary --------------------------------------------------------
def test_a_network_backend_is_refused_in_strict_mode(monkeypatch):
    """The module's most important test.

    The writer is the only component that sees the real data. If it can leave the
    network, the whole architecture is pointless: we would have protected the
    question in order to export the answer.
    """
    monkeypatch.setattr(lm, "settings", lambda: Settings(privacy_mode="strict"))
    with pytest.raises(lm.ForbiddenBackend, match="strict"):
        lm._check_boundary("hf-inference")


def test_a_local_backend_passes_in_strict_mode(monkeypatch):
    monkeypatch.setattr(lm, "settings", lambda: Settings(privacy_mode="strict"))
    lm._check_boundary("llamacpp")   # must not raise
    lm._check_boundary("ollama")


def test_demo_mode_tolerates_but_warns(monkeypatch, caplog):
    """In demo we let it through, but the trace must stay in the logs:
    a leak-rate measurement made under those conditions is worthless."""
    monkeypatch.setattr(lm, "settings", lambda: Settings(privacy_mode="demo"))
    with caplog.at_level("WARNING"):
        lm._check_boundary("hf-inference")
    assert "boundary" in caplog.text


# --- The prompt ----------------------------------------------------------------
def test_the_prompt_forbids_computing_and_inventing():
    """These two instructions are what make a 1.7-billion-parameter model acceptable."""
    p = lm.build_prompt("how many patients?", ["n"], [(42,)])
    assert "Never compute, estimate or invent" in p
    assert "ONLY the query results" in p


def test_the_prompt_carries_the_question_and_results():
    p = lm.build_prompt("how many patients?", ["n"], [(42,)])
    assert "how many patients?" in p
    assert "42" in p


def test_an_empty_result_is_announced_as_such():
    """The model must not embroider over an absence of results."""
    p = lm.build_prompt("how many unicorns?", ["n"], [])
    assert "no row" in p
    assert "empty" in p


def test_a_single_number_is_stated_as_a_fact():
    """The commonest case, and the one a 1.7B model most often mishandles.

    Shown as a one-cell table with `COUNT(DISTINCT m.patientunitstayid)` as its
    header, the model read the header as the answer and replied "no matching
    record" for a result of 4.
    """
    table = lm._table(["patient_count"], [(4,)])
    assert table == "patient_count = 4"


def test_long_results_are_truncated_with_the_exact_count():
    """Past a few dozen rows a small model copies them wrong.
    We give it a sample *and* the total, so it does not extrapolate."""
    rows = [(i, f"drug-{i}") for i in range(200)]
    table = lm._table(["id", "name"], rows)

    assert "drug-0" in table
    assert "drug-199" not in table
    assert "200 rows total" in table
    assert table.count("\n") < lm.MAX_PROMPT_ROWS + 6


def test_null_values_do_not_break_formatting():
    table = lm._table(["a", "b"], [(None, "x"), (1, None)])
    assert "None" not in table


# --- Output cleanup ------------------------------------------------------------
def test_the_thinking_block_is_removed():
    """Qwen3 emits `<think>…</think>`: showing it would expose the scratchpad."""
    raw = "<think>Let me count the rows...\nOK.</think>\nThere are 42 patients."
    assert lm._strip_thinking(raw) == "There are 42 patients."


def test_cleanup_handles_output_without_thinking():
    assert lm._strip_thinking("  There are 42 patients.  ") == "There are 42 patients."


def test_cleanup_handles_empty_output():
    assert lm._strip_thinking("") == ""
    assert lm._strip_thinking(None) == ""
