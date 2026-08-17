"""The value index: the classification policy, then resolution.

These tests need `data/warehouse/value_index.db`. They are skipped when it is
missing rather than failing: on a fresh machine the index gets built.
"""

from __future__ import annotations

import pytest

from hybridsql.config import settings
from hybridsql.db import value_index as vi

pytestmark = pytest.mark.skipif(
    not settings().value_index_path.exists(),
    reason="index missing — run scripts/build_value_index.py",
)


# --- The classification policy ------------------------------------------------
def test_a_word_ending_in_id_is_not_an_identifier():
    """`volumeoffluid` was mistaken for a key by a suffix rule."""
    tables = frozenset({"patient", "hospital", "lab"})
    assert not vi._is_identifier("volumeoffluid", tables)
    assert not vi._is_identifier("valid", tables)


def test_real_identifiers_are_recognised():
    tables = frozenset({"patient", "hospital", "lab"})
    assert vi._is_identifier("id", tables)
    assert vi._is_identifier("patient_id", tables)
    assert vi._is_identifier("hospitalid", tables)      # prefix is a table name
    assert vi._is_identifier("labresultoffset", tables)


def test_no_patient_identifier_in_the_index():
    """The module's most important regression test.

    `patient.uniquepid` slipped into tier A during development: 1,841 patient
    identifiers would have landed in a file meant for business vocabulary.
    """
    stats = vi.stats()
    indexed = {ref for ref, _ in stats["top_A"]}
    assert "patient.uniquepid" not in indexed


# --- Query construction --------------------------------------------------------
def test_conjunction_is_tried_before_disjunction():
    """On several words, OR alone drowns the right value among homonyms."""
    queries = vi._fts_queries("acute renal failure")
    assert len(queries) == 2
    assert " AND " in queries[0]
    assert " OR " in queries[1]


def test_a_single_word_yields_one_query():
    assert vi._fts_queries("aspirin") == ['"aspirin"*']


def test_an_empty_mention_yields_no_query():
    assert vi._fts_queries("") == []
    assert vi._fts_queries("!!!") == []


# --- Resolution ----------------------------------------------------------------
def test_a_short_mention_reaches_the_long_value():
    """The case that justifies the whole module: `aspirin` -> `ASPIRIN EC 81 MG PO TBEC`."""
    results = vi.search("aspirin", limit=5)
    assert results
    assert any(r.ref == "medication.drugname" for r in results)


def test_a_multi_word_mention_reaches_the_exact_value():
    results = vi.search("acute renal failure", limit=3)
    assert results
    assert "acute renal failure" in results[0].value.lower()


def test_column_scoping_is_respected():
    results = vi.search("aspirin", columns=["medication.drugname"], limit=5)
    assert results
    assert {r.ref for r in results} == {"medication.drugname"}


def test_an_unknown_mention_returns_nothing():
    assert vi.search("zzzzqqqxyzunknown", limit=3) == []


def test_one_suggestion_per_column():
    """Ten aspirin variants from the same column help nobody."""
    results = vi.search("insulin", limit=5)
    refs = [r.ref for r in results]
    assert len(refs) == len(set(refs))


def test_exact_value_lookup():
    """Used to settle concept-versus-value ambiguity in stage 1."""
    assert vi.is_exact_value("Female")
    assert vi.is_exact_value("  female  ")     # normalised
    assert not vi.is_exact_value("zzzqqxyzunknown")
