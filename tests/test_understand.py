"""Stage 1 end to end.

The extractor is replaced by a fake in most tests. Two reasons: GLiNER2's weights
weigh 800 MB and are not guaranteed present in CI, and above all we want to test
**the orchestration**, not the model's quality — that belongs to the week-3
evaluation, not to unit tests.
"""

from __future__ import annotations

import pytest

from hybridsql.config import settings
from hybridsql.pipeline import understand as stage1
from hybridsql.providers.extractor import Entity

pytestmark = pytest.mark.skipif(
    not settings().value_index_path.exists(),
    reason="index missing — run scripts/build_value_index.py",
)


@pytest.fixture
def fake_extractor(monkeypatch):
    """Inject chosen entities without loading the model."""

    def _install(entities: list[Entity]):
        monkeypatch.setattr(stage1.extractor, "extract", lambda q: entities)
        monkeypatch.setattr(stage1.extractor, "available", lambda: True)

    return _install


# --- The confidentiality contract ---------------------------------------------
def test_the_cloud_view_holds_no_real_value(fake_extractor):
    """The most important test in this file.

    `for_the_cloud()` is the only authorised path outward. If a real value shows up
    in it, the whole architecture is pointless.
    """
    fake_extractor([Entity("aspirin", "drug", 0.9)])
    u = stage1.understand("how many patients received aspirin?")

    view = u.for_the_cloud()
    serialised = repr(view).lower()

    assert u.resolutions[0].value is not None, "the test only means something if it resolved"
    for r in u.resolutions:
        assert r.value.lower() not in serialised
        assert r.mention.lower() not in serialised


def test_the_cloud_view_does_carry_the_scoping(fake_extractor):
    """The converse: what must leave has to leave."""
    fake_extractor([Entity("aspirin", "drug", 0.9)])
    view = stage1.understand("how many patients received aspirin?").for_the_cloud()

    assert "medication" in view["tables"]
    assert view["masked_value_count"] >= 1
    assert any(c.startswith("medication.") for c in view["target_columns"])


# --- Scoping -------------------------------------------------------------------
def test_entity_type_steers_the_search(fake_extractor):
    """Without scoping, "female" returns an oncology diagnosis before `patient.gender`."""
    fake_extractor([Entity("female", "demographic", 0.9)])
    u = stage1.understand("how many female patients?")

    assert u.resolutions[0].column == "patient.gender"
    assert u.resolutions[0].value == "Female"


def test_a_drug_is_searched_in_drug_columns(fake_extractor):
    fake_extractor([Entity("aspirin", "drug", 0.9)])
    u = stage1.understand("patients on aspirin")

    assert u.resolutions[0].column in {
        "medication.drugname",
        "admissiondrug.drugname",
        "infusiondrug.drugname",
    }


def test_glossary_and_resolution_tables_are_merged(fake_extractor):
    fake_extractor([Entity("aspirin", "drug", 0.9)])
    u = stage1.understand("mortality of patients on aspirin")

    assert "medication" in u.tables       # from the resolution
    assert "patient" in u.tables          # from the glossary ("mortality")


# --- Kind classification -------------------------------------------------------
def test_a_concept_is_not_looked_up():
    """"mortality rate" names a column; resolving it produced a wrong answer at 0.79."""
    assert stage1._classify("mortality rate") == "concept"
    assert stage1._classify("length of stay") == "concept"


def test_a_quantity_is_not_looked_up():
    """"over 65" is a numeric filter written by the analyst, not a stored value."""
    assert stage1._classify("over 65") == "quantity"
    assert stage1._classify("12.5 %") == "quantity"


def test_a_person_name_is_never_looked_up():
    assert stage1._classify("Sarah Johnson", "person") == "person"


def test_a_compound_stored_value_beats_the_glossary():
    """"Neuro ICU" is covered by the synonym "icu" but stored verbatim in `unittype`."""
    assert stage1._classify("Neuro ICU") == "value"
    # A single glossary word stays a concept: applying the rule to every word made
    # classification accuracy fall from 88% to 29%.
    assert stage1._classify("icu") == "concept"


# --- Cases where we must not guess ---------------------------------------------
def test_an_unfindable_mention_is_reported_not_invented(fake_extractor):
    fake_extractor([Entity("zzzqqxyzunknown", "drug", 0.9)])
    u = stage1.understand("patients on zzzqqxyzunknown")

    assert len(u.unresolved) == 1
    assert u.resolutions[0].value is None
    assert not u.resolutions[0].confident


def test_a_question_without_value_yields_no_resolution(fake_extractor):
    """"how many hospitals" holds no value to mask. Normal case."""
    fake_extractor([])
    u = stage1.understand("how many hospitals are there?")

    assert u.resolutions == []
    assert u.for_the_cloud()["masked_value_count"] == 0
    assert "hospital" in u.tables


def test_an_uncertain_resolution_asks_for_confirmation():
    r = stage1.Resolution("aspirn", "drug", "aspirin", "medication.drugname", 0.60)
    assert r.resolved
    assert not r.confident
    assert r.score < stage1.CONFIDENCE_THRESHOLD


def test_an_out_of_scope_resolution_is_penalised():
    """"hemoglobin", absent from `lab.labname`, landed in `diagnosisstring` at 1.00."""
    assert stage1.OUT_OF_SCOPE_PENALTY < 1.0
    penalised = 1.0 * stage1.OUT_OF_SCOPE_PENALTY
    assert penalised < stage1.CONFIDENCE_THRESHOLD


# --- Fallback ------------------------------------------------------------------
def test_the_pipeline_survives_without_gliner2(monkeypatch):
    """If the model does not load, we degrade — we do not crash."""
    from hybridsql.providers import extractor

    monkeypatch.setattr(extractor, "load", lambda force=False: None)
    monkeypatch.setattr(extractor, "available", lambda: False)

    u = stage1.understand("mortality by hospital region")
    assert u.active_extractor == "glossary"
    assert u.entities, "the glossary fallback must still recognise terms"
