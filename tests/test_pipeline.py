"""Stages 2 and 3: anonymization, SQL validation, orchestration.

No cloud call here. The network is exercised by
`scripts/evaluate_pipeline.py`, which measures the real thing; these tests cover
what must hold regardless of the model's answer.
"""

from __future__ import annotations

import pytest

from hybridsql.config import settings
from hybridsql.pipeline.anonymize import UnmaskableQuestion, anonymize
from hybridsql.pipeline.understand import Resolution, Understanding
from hybridsql.security import sql_validator as sv

pytestmark = pytest.mark.skipif(
    not settings().value_index_path.exists(),
    reason="index missing — run scripts/build_value_index.py",
)


def _understanding(question: str, resolutions: list[Resolution]) -> Understanding:
    return Understanding(question=question, resolutions=resolutions, tables={"medication"})


# --- Anonymization -------------------------------------------------------------
def test_a_value_becomes_a_symbol():
    u = _understanding(
        "how many patients received aspirin?",
        [Resolution("aspirin", "drug", "aspirin", "medication.drugname", 1.0)],
    )
    a = anonymize(u)

    assert ":v1" in a.masked_question
    assert "aspirin" not in a.masked_question.lower()
    assert a.mapping[":v1"] == "aspirin"
    assert a.parameters() == {"v1": "aspirin"}


def test_the_prompt_view_never_reveals_the_value():
    """The model needs the column, not the content."""
    u = _understanding(
        "patients on aspirin",
        [Resolution("aspirin", "drug", "ASPIRIN EC 81 MG PO TBEC", "medication.drugname", 1.0)],
    )
    view = anonymize(u).for_the_prompt()

    assert "medication.drugname" in view
    assert "ASPIRIN" not in view.upper()


def test_the_longest_mention_is_masked_first():
    """Otherwise replacing "aspirin" before "aspirin 81 mg" leaves ":v1 81 mg"."""
    u = _understanding(
        "patients on aspirin 81 mg",
        [
            Resolution("aspirin", "drug", "aspirin", "medication.drugname", 1.0),
            Resolution("aspirin 81 mg", "drug", "ASPIRIN 81 MG", "medication.drugname", 1.0),
        ],
    )
    a = anonymize(u)
    assert "81 mg" not in a.masked_question


def test_a_concept_is_not_masked():
    """"mortality rate" names a column: masking it would blind the model."""
    u = _understanding(
        "mortality rate for patients",
        [Resolution("mortality rate", "diagnosis", None, "patient.hospitaldischargestatus",
                    0.0, kind="concept")],
    )
    a = anonymize(u)
    assert a.symbol_count == 0
    assert "mortality rate" in a.masked_question


def test_a_person_name_blocks_the_request():
    """The database is de-identified: the question cannot be satisfied, and there is
    no reason to send anything to the cloud."""
    u = _understanding(
        "did Mr. Bensalah get his insulin?",
        [Resolution("Mr. Bensalah", "person", None, None, 0.0, kind="person")],
    )
    with pytest.raises(UnmaskableQuestion, match="Bensalah"):
        anonymize(u)


def test_symbols_restart_from_one_on_every_question():
    """A stable pseudonym would let the provider track a value across requests."""
    for _ in range(3):
        u = _understanding(
            "patients on aspirin",
            [Resolution("aspirin", "drug", "aspirin", "medication.drugname", 1.0)],
        )
        assert list(anonymize(u).mapping) == [":v1"]


# --- SQL validation ------------------------------------------------------------
def test_a_plain_select_is_accepted():
    assert sv.validate("SELECT COUNT(*) FROM patient").valid


def test_a_cte_is_accepted():
    assert sv.validate("WITH x AS (SELECT 1 AS n) SELECT n FROM x").valid


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE patient",
        "DELETE FROM patient",
        "UPDATE patient SET gender = 'X'",
        "INSERT INTO patient VALUES (1)",
        "PRAGMA table_info(patient)",
        "ATTACH DATABASE '/tmp/x.db' AS x",
    ],
)
def test_write_statements_are_refused(sql):
    verdict = sv.validate(sql)
    assert not verdict.valid


def test_two_statements_are_refused():
    """SQLite happily runs `SELECT 1; DROP TABLE patient` through executescript."""
    verdict = sv.validate("SELECT 1; DROP TABLE patient")
    assert not verdict.valid
    assert "multiple statements" in verdict.reason


def test_a_comment_cannot_hide_a_statement():
    """`SELECT 1 /* */ ; DROP` must not escape the first-keyword check."""
    assert not sv.validate("SELECT 1 /* comment */ ; DROP TABLE patient").valid


def test_a_literal_named_like_a_keyword_does_not_trip_the_check():
    """`WHERE name = 'update'` is legitimate."""
    assert sv.validate("SELECT * FROM patient WHERE gender = 'update'").valid


def test_load_extension_is_refused():
    assert not sv.validate("SELECT load_extension('evil.so')").valid


def test_an_unknown_parameter_is_refused():
    verdict = sv.validate("SELECT * FROM patient WHERE gender = :v9", {":v1"})
    assert not verdict.valid
    assert ":v9" in verdict.reason


def test_a_forgotten_parameter_is_refused():
    """The gravest case: the model wrote the value in clear text instead of binding it."""
    verdict = sv.validate("SELECT * FROM medication WHERE drugname = 'aspirin'", {":v1"})
    assert not verdict.valid
    assert "clear text" in verdict.reason


def test_an_empty_query_is_refused():
    assert not sv.validate("").valid
    assert not sv.validate("-- nothing but a comment").valid


def test_require_raises_on_a_bad_query():
    with pytest.raises(sv.SqlRejected):
        sv.require("DROP TABLE patient")
