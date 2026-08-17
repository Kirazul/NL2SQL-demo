"""Canaries: proof by execution that no value crosses the gate.

Why these tests exist
---------------------
The report will claim that no business data reaches the cloud provider. That claim
only holds if a machine checks it on every change. A reviewer cannot guarantee a
value does not pass; a test that takes **real database values** and observes them
blocked, can.

The canary principle: we deliberately send toward the gate what must never leave.
If one of these tests ever turns green when it should fail, the gate has been
weakened.

These must run in CI. A failure here is not a functional regression: it is a leak.
"""

from __future__ import annotations

import random
import re
import sqlite3

import pytest

from hybridsql.config import settings
from hybridsql.security import audit, egress_gate

pytestmark = pytest.mark.skipif(
    not settings().value_index_path.exists(),
    reason="index missing — run scripts/build_value_index.py",
)

# Non-regression threshold, from a measurement rather than a wish: on 15 August
# 2026, `scripts/measure_gate.py` records 1.74% of information-bearing values
# crossing the gate. We leave headroom, but any drift beyond it must fail CI —
# that is the only way to notice a weakened gate.
MAX_RESIDUAL_RATE = 0.03

# These pass and that is acceptable: categorical fillers with no content.
FILLERS = {"other", "yes", "no", "none", "unknown", "normal", "all", "not"}


@pytest.fixture(scope="module")
def real_values() -> list[str]:
    """A sample of real values, drawn from the database index."""
    path = settings().value_index_path
    cx = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        every = [v for (v,) in cx.execute("SELECT value FROM values_fts") if v and len(v) > 3]
    finally:
        cx.close()
    random.seed(20260815)   # reproducible draw: a canary must not flicker
    return random.sample(every, min(400, len(every)))


def _informative(value: str, identifiers: frozenset[str]) -> bool:
    """A value whose disclosure would genuinely teach something.

    Three categories are set aside, each for a distinct reason:

    - **numbers and units** ("65", "mg") — nothing tells them apart from what the
      analyst types in the question itself;
    - **column names** (`albumin`, `urine`, `wbc` of `apacheapsvar`) — blocking them
      would forbid the model from writing `SELECT albumin FROM apacheapsvar`;
    - **generic vocabulary** ("Every 12 hours", "Current", "ratio") — made only of
      function words, it says nothing about the database content.

    What remains — "MELOXICAM", "melanoma" — is what must be blocked.
    """
    words = [w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ]{3,}", value)]
    if not words:
        return False
    generic = egress_gate.generic_vocabulary()
    return any(w not in identifiers and w not in FILLERS and w not in generic for w in words)


# --- The canaries --------------------------------------------------------------
def test_no_truly_informative_value_crosses_the_gate(real_values):
    """THE project test.

    Excluded are the values that cannot be blocked without breaking the system:
    bare numbers, categorical fillers, and words that are also column names.
    Everything else must be stopped.
    """
    identifiers = frozenset(egress_gate._schema_identifiers())
    passed = [
        v
        for v in real_values
        if _informative(v, identifiers) and egress_gate.check(v, "canary").allowed
    ]
    assert not passed, f"{len(passed)} informative value(s) crossed the gate: {passed[:8]}"


def test_the_residual_rate_does_not_drift():
    """Non-regression guard over the whole population.

    A test sampling only 400 values can miss a drift. This one measures across all
    information-bearing values in the database.
    """
    path = settings().value_index_path
    cx = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        every = [v for (v,) in cx.execute("SELECT DISTINCT value FROM values_fts") if v]
    finally:
        cx.close()

    bearing = [v for v in every if re.search(r"[A-Za-zÀ-ÿ]{3,}", v)]
    passed = [v for v in bearing if egress_gate.check(v).allowed]
    rate = len(passed) / len(bearing)

    assert rate <= MAX_RESIDUAL_RATE, (
        f"residual rate {rate:.2%} beyond the {MAX_RESIDUAL_RATE:.0%} threshold — "
        "the gate has weakened. Re-run scripts/measure_gate.py."
    )


def test_a_value_buried_in_a_sentence_is_still_blocked(real_values):
    """A value does not hide by surrounding itself with allowed words.

    This is the realistic case: a leak does not happen on an isolated value, but
    inside a question that looks innocuous.
    """
    identifiers = frozenset(egress_gate._schema_identifiers())
    for value in real_values[:80]:
        if not _informative(value, identifiers):
            continue
        sentence = f"how many patients received {value} in the last year?"
        verdict = egress_gate.check(sentence, "canary")
        assert not verdict.allowed, f"sentence passed with the value {value!r}"


def test_a_word_absent_from_the_database_is_allowed():
    """The boundary of the guarantee, stated rather than glossed over.

    "ZORBLAXIMAB" appears nowhere in the database. It is the analyst's own word,
    not UNIMED data, so letting it through discloses nothing that belongs to the
    organisation.

    This is a deliberate change of design. An earlier version blocked every word
    absent from a hand-written allowlist — which meant maintaining hundreds of
    ordinary English words ("readmitted", "hierarchical", "weather") and adding one
    more after every new question. That is not an architecture; it is an endless
    patch, and each addition widened the hole.

    What replaced it is a check on the data itself: the gate blocks anything that
    **is** in the database. That inventory is exhaustive — we own the database —
    and it costs no vocabulary maintenance.
    """
    verdict = egress_gate.check("count patients with ZORBLAXIMAB", "canary")
    assert verdict.allowed


def test_an_ordinary_word_never_needs_to_be_declared():
    """The point of the redesign: the gate does not have to know English."""
    for question in (
        "how many patients were readmitted last year?",
        "what is the weather today?",
        "which cases should I look at first?",
        "compare the outcomes between the two groups",
    ):
        verdict = egress_gate.check(question, "canary")
        assert verdict.allowed, f"wrongly blocked: {verdict.refused_tokens}"


def test_a_multi_word_value_made_of_ordinary_words_is_still_blocked():
    """"Skilled Nursing Facility" is three unremarkable words — and a stored value.

    Single structural words are exempt, phrases are not. That is why the n-gram
    scan runs longest-first.
    """
    verdict = egress_gate.check("patients discharged to a Skilled Nursing Facility", "canary")
    assert not verdict.allowed
    assert "skilled nursing facility" in verdict.refused_tokens


def test_female_is_blocked_because_it_is_a_database_value():
    """The hole that automatic subtraction closes.

    "female" was in the business vocabulary of the allowlist, while it is also the
    value stored in `patient.gender`. Without subtraction, the word crossed the
    gate while being data.
    """
    assert "female" in egress_gate.forbidden_words()
    assert not egress_gate.check("how many female patients", "canary").allowed


# --- What must pass ------------------------------------------------------------
def test_a_correctly_masked_question_passes():
    """The indispensable counterpart: a gate that blocks everything proves nothing."""
    verdict = egress_gate.check("how many patients received :v1 during their stay?", "canary")
    assert verdict.allowed, f"wrongly blocked on: {verdict.refused_tokens}"


def test_sql_over_the_schema_passes():
    """The cloud model must be able to receive and return SQL."""
    sql = (
        "SELECT COUNT(DISTINCT p.patientunitstayid) FROM patient p "
        "JOIN medication m ON m.patientunitstayid = p.patientunitstayid "
        "WHERE m.drugname = :v1 GROUP BY p.gender ORDER BY 1 DESC LIMIT 10"
    )
    verdict = egress_gate.check(sql, "canary")
    assert verdict.allowed, f"wrongly blocked on: {verdict.refused_tokens}"


def test_numbers_and_punctuation_pass():
    verdict = egress_gate.check("patients older than 65 (about 12.5%)", "canary")
    assert verdict.allowed, f"wrongly blocked on: {verdict.refused_tokens}"


def test_symbols_are_not_split():
    """`:v1` must stay one token. Split, `v1` would be refused and everything would break."""
    verdict = egress_gate.check(":v1 :v2 :v10", "canary")
    assert verdict.allowed
    assert verdict.token_count == 3


# --- Provenance verification ----------------------------------------------------
def test_an_authored_segment_is_verified_by_fingerprint():
    """Our own instruction text is checked for constancy, not for vocabulary."""
    text = "Fixed instruction text for the canary test."
    egress_gate.register_constant(text)

    verdict = egress_gate.check_segment(egress_gate.Segment(text, "authored"), "canary")
    assert verdict.allowed
    assert verdict.verified_by == "registered constant"


def test_a_modified_authored_segment_falls_back_to_the_word_check():
    """If a value is interpolated into it, the fingerprint changes and the gate catches it."""
    text = "Fixed instruction text for the canary test."
    egress_gate.register_constant(text)

    tampered = egress_gate.Segment(text + " MELOXICAM", "authored")
    verdict = egress_gate.check_segment(tampered, "canary")
    assert not verdict.allowed
    assert "meloxicam" in verdict.refused_tokens


def test_the_ddl_is_verified_by_regenerating_it():
    """Provenance by reconstruction: the gate recomputes the DDL and compares.

    This is what removed the need to allowlist every word of our own prompt
    scaffolding.
    """
    from hybridsql.db.schema import ddl

    real = ddl({"patient"}, with_row_counts=True)
    verdict = egress_gate.check_segment(egress_gate.Segment(real, "schema"), "canary")
    assert verdict.allowed
    assert verdict.verified_by == "regenerated DDL"


def test_a_ddl_with_an_injected_value_is_rejected():
    from hybridsql.db.schema import ddl

    tampered = ddl({"patient"}, with_row_counts=True) + "\n-- MELOXICAM"
    verdict = egress_gate.check_segment(egress_gate.Segment(tampered, "schema"), "canary")
    assert not verdict.allowed


def test_a_glossary_note_is_verified_by_membership():
    from hybridsql.resources.glossary import load

    note = next(t.note for t in load().values() if t.note)
    verdict = egress_gate.check_segment(egress_gate.Segment(note, "glossary"), "canary")
    assert verdict.allowed
    assert verdict.verified_by == "declared glossary note"


# --- Output sweep ---------------------------------------------------------------
def test_the_output_sweep_spots_a_literal_the_model_reproduced():
    """Sanitising the input is only half of it; the response is swept too.

    The real risk: the model writes `WHERE drugname = 'aspirin'` instead of using
    the bound parameter. Nothing left our side, but the query no longer reflects
    the masked value, so running it would silently answer a different question.
    """
    found = egress_gate.sweep_response("SELECT * FROM medication WHERE drugname = 'MELOXICAM'")
    assert "meloxicam" in found

    clean = egress_gate.sweep_response("SELECT * FROM medication WHERE drugname = :v1")
    assert clean == []


# --- The journal ---------------------------------------------------------------
def test_a_refused_send_is_recorded(tmp_path, monkeypatch):
    """A guarantee that is not traced cannot be verified after the fact."""
    journal = tmp_path / "egress.jsonl"
    monkeypatch.setattr(audit, "_path", lambda: journal)

    with pytest.raises(egress_gate.LeakBlocked):
        egress_gate.require("count MELOXICAM patients", "test")

    lines = audit.read(journal)
    assert len(lines) == 1
    assert lines[0]["allowed"] is False
    assert "meloxicam" in lines[0]["refused_tokens"]


def test_the_journal_never_holds_the_clear_text(tmp_path, monkeypatch):
    """The journal records a fingerprint and the refused tokens, not the message.

    Otherwise, protecting the request only to copy its content into a file beside
    it would make no sense.
    """
    journal = tmp_path / "egress.jsonl"
    monkeypatch.setattr(audit, "_path", lambda: journal)

    egress_gate.require("how many patients received :v1", "test")

    content = journal.read_text(encoding="utf-8")
    assert "how many patients" not in content
    assert len(audit.read(journal)) == 1


def test_the_block_rate_is_computed(tmp_path, monkeypatch):
    journal = tmp_path / "egress.jsonl"
    monkeypatch.setattr(audit, "_path", lambda: journal)

    egress_gate.require("count patients with :v1", "test")
    with pytest.raises(egress_gate.LeakBlocked):
        egress_gate.require("count patients with MELOXICAM", "test")

    stats = audit.leak_rate(journal)
    assert stats == {"sends": 2, "blocked": 1, "allowed": 1, "block_rate": 0.5}
