"""The allowlist is hand-written text, and the gate is what stands between the
database and a third party. Both are tested here for the same reason
`test_glossary.py` exists: this is where a mistake settles quietly.

`test_canary.py` proves that values do not cross. These tests prove the two other
things that make a gate usable rather than merely strict:

- an ordinary question is **not** refused;
- and no word added to make that true has opened a hole.
"""

from __future__ import annotations

import pytest

from hybridsql.config import settings
from hybridsql.security import egress_gate as gate

pytestmark = pytest.mark.skipif(
    not settings().value_index_path.exists(),
    reason="index missing — run scripts/build_value_index.py",
)


# Grammar words that are also, in this database, a whole stored value. Each one is
# a small hole: the word passes the gate although the database contains it exactly.
#
# They are recorded rather than removed because every one of them has to pass for
# the system to work — eleven are SQL keywords the model must be able to write
# (`SELECT ... LEFT JOIN`, `CAST`, `PRIMARY KEY`), and the rest are question
# grammar whose absence refuses ordinary sentences ("more than once", "no", "yes").
# Disclosing them teaches a provider nothing: every database contains "left" and
# "normal".
#
# What this list is for: it must **not grow**. A word added to `allowlist.yaml`
# that is also a stored value fails this test, and the author then has to say why —
# which is exactly the review that was missing when the list was extended.
KNOWN_COLLISIONS = frozenset(
    """abnormal all cast current equal index int intubated left lower minute none
    normal note now once other primary ratio right seconds top unknown upper value
    ventilated yes""".split()
)


def test_no_new_grammar_word_hides_a_stored_value():
    collisions = {
        word
        for word in gate.generic_vocabulary()
        if word in gate.known_values() and gate.carries_information(word)
    }
    new = collisions - KNOWN_COLLISIONS
    assert not new, (
        f"{sorted(new)} were added to config/allowlist.yaml although the database "
        "stores each of them as a complete value. Either drop them, or record them "
        "in KNOWN_COLLISIONS with the reason they have to pass."
    )


@pytest.mark.parametrize(
    "question",
    [
        "How many :v1 records are associated with each patient ICU stay?",
        "What are the 10 most common diagnosis names?",
        "What are the most common medication administration routes?",
        "How many patients belong to each ethnicity?",
        "Which patients have the largest number of nursing charting records?",
        "How many patients received :v1 and have at least one laboratory record?",
        "For each hospital, how many unique patients received medication?",
    ],
)
def test_an_ordinary_analytical_question_is_not_refused(question):
    """The failure this project actually had in the field.

    Every one of these was refused, on a word no database owns: `records`,
    `associated`, `one`, `unique`. A gate that refuses these is not strict, it is
    broken — the analyst reformulates until something passes, which teaches them to
    work around the guarantee.
    """
    verdict = gate.check(question, "test")
    assert verdict.allowed, f"refused on {list(verdict.refused_tokens)}"


def test_a_spelled_out_numeral_is_treated_like_a_digit():
    """"at least 1" passed and "at least one" did not — the two layers disagreeing."""
    assert not gate.carries_information("one")
    assert gate.check("at least one record", "test").allowed
    assert gate.check("at least 1 record", "test").allowed


def test_value_tokens_holds_the_words_of_short_values_only():
    """The rule that made the gate usable, stated as a test.

    `associated` occurs only inside seven-word hierarchical diagnosis strings;
    `aspirin` is a value on its own. Only the second is data.
    """
    vocabulary = gate.value_tokens()
    assert "aspirin" in vocabulary
    assert "female" in vocabulary
    assert "alive" in vocabulary
    assert "associated" not in vocabulary
    assert "unique" not in vocabulary


def test_a_long_stored_value_is_still_matched_whole():
    """What restricting `value_tokens` gave up, and how it is paid back.

    A seventeen-word value used to be caught word by word. It is now caught as a
    whole by the n-gram scan, whose ceiling is derived from the longest value in the
    database instead of being frozen at six words.
    """
    # Only plain-word values: the n-gram scan tokenises on word characters, so a
    # value written `flowsheet|I&O|volume (ml)` cannot be reassembled from its
    # tokens. Those are caught by their words instead, and that is a known limit of
    # the exact layer rather than something this test can assert away.
    long_values = [
        v for v in gate.known_values() if len(v.split()) > 8 and v.replace(" ", "").isalnum()
    ]
    if not long_values:  # pragma: no cover — depends on the database
        pytest.skip("this database holds no long plain-word value")

    for value in long_values[:20]:
        assert value in gate.find_known_values(value), f"no longer matched whole: {value!r}"
        assert not gate.check(value, "test").allowed
