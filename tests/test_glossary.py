"""The glossary is hand-written text: that is where mistakes settle."""

from __future__ import annotations

import pytest

from hybridsql.resources import glossary


def test_glossary_only_points_at_existing_columns():
    """The test that matters.

    A glossary referencing a vanished column is worse than an empty one: it steers
    the model toward SQL that will never run, and the mistake only shows up at
    execution time.
    """
    problems = glossary.validate()
    assert problems == [], "\n".join(problems)


def test_basic_recognition():
    found = {m.term.canonical for m in glossary.recognize("mortality by hospital region")}
    assert "mortality" in found
    assert "region" in found


def test_longest_term_wins():
    """"length of stay" must not be reduced to "stay"."""
    found = {m.term.canonical for m in glossary.recognize("average length of stay")}
    assert "length_of_stay" in found
    assert "patient" not in found


@pytest.mark.parametrize("question", ["understayed patients", "close the case"])
def test_no_trigger_on_a_word_fragment(question):
    """Search runs on whole words.

    With substring search, "stay" would fire inside "understayed" and "los" inside
    "close".
    """
    triggers = {m.trigger for m in glossary.recognize(question)}
    assert "stay" not in triggers
    assert "los" not in triggers


def test_accents_are_ignored():
    assert glossary.recognize("mortalité") == glossary.recognize("mortalite")


def test_no_synonym_is_a_database_value():
    """The rule that prevents a silent leak.

    A mention fully covered by the glossary is treated as a *concept*: it is
    therefore not replaced by a symbol before the cloud call. Listing a real value
    among the synonyms amounts to authorising it to leave in clear text.

    The case we hit: `sex` declared `male, female` as synonyms, while "Female" is
    the value stored in `patient.gender` — precisely the column that term points
    at.

    The check targets that collision, not a distant coincidence. "temperature" is
    a legitimate synonym of `vital_signs` even though it also exists as a value in
    `lab.labname`: the two senses coexist without contradiction. The conflict is
    real only when the word is a value **of the very column** the term points at.
    """
    import sqlite3

    from hybridsql.config import settings

    path = settings().value_index_path
    if not path.exists():
        pytest.skip("index missing — run scripts/build_value_index.py")

    cx = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        per_column: dict[str, set[str]] = {}
        for value, ref in cx.execute("SELECT value, ref FROM values_fts"):
            per_column.setdefault(ref, set()).add(str(value).strip().lower())
    finally:
        cx.close()

    offenders = [
        f"{term.canonical} -> {s!r} is a value of {column}"
        for term in glossary.load().values()
        for s in term.synonyms
        for column in term.columns
        if s.lower() in per_column.get(column, ())
    ]
    assert not offenders, (
        "These synonyms name a value of their own column: the pipeline would treat "
        "them as concepts and let them leave in clear text.\n  " + "\n  ".join(offenders)
    )


def test_business_notes_surface():
    """Schema traps must reach the prompt."""
    notes = glossary.notes_for("what is the average length of stay")
    assert any("minutes" in n for n in notes)
