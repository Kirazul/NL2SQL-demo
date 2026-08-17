"""Stage 3a — run one SELECT and nothing else.

Why this module exists
----------------------
The SQL executed here was **written by a remote model**. Treating it as trusted
input would be absurd: it is text produced by a third party that can be wrong, be
pushed into a mistake by a twisted question, or return anything at all if the API
is compromised.

The database is already opened read-only (`db/connection.py`), which would block a
`DELETE` at the engine level. This module adds a second barrier, upstream, for
three reasons:

1. **Clear message** — "forbidden statement: DROP" beats an opaque SQLite error
   at execution time;
2. **Defence in depth** — if `DB_READONLY` were flipped to 0 by a configuration
   mistake, the check would remain;
3. **Refusing multiple statements** — SQLite happily runs
   `SELECT 1; DROP TABLE patient` through `executescript`.

What the validator does not do
------------------------------
It does not check that the query *answers the question*. A safe query can still be
wrong. That is the job of execution and evaluation, not of the security check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Statements that must never reach the engine, even read-only.
FORBIDDEN_STATEMENTS = {
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "truncate", "attach", "detach", "vacuum", "reindex", "pragma",
    "begin", "commit", "rollback", "savepoint", "release", "analyze",
}

# SQLite functions that touch the filesystem or load code.
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}

COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
PARAMETER_RE = re.compile(r":v\d+")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class SqlRejected(ValueError):
    """The query does not satisfy the execution conditions."""


@dataclass(frozen=True)
class Verdict:
    valid: bool
    reason: str = ""
    parameters_used: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.valid


def _without_comments(sql: str) -> str:
    """Strip comments before analysis.

    Without this, `SELECT 1 -- ; DROP TABLE x` and especially `/* */ DROP` escape
    the first-keyword check.
    """
    return COMMENT_RE.sub(" ", sql or "")


def _without_strings(sql: str) -> str:
    """Neutralise literals, so `WHERE name = 'update'` does not trip the check."""
    return re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", " '' ", sql)


def validate(sql: str, expected_parameters: set[str] | None = None) -> Verdict:
    """Return a verdict without raising. `require()` raises."""
    raw = (sql or "").strip()
    if not raw:
        return Verdict(False, "the model did not return SQL — the question is probably out of scope")

    bare = _without_strings(_without_comments(raw)).strip().rstrip(";").strip()
    if not bare:
        return Verdict(False, "empty query after comment removal")

    # A single statement. A trailing semicolon is tolerated, a second statement is not.
    if ";" in bare:
        return Verdict(False, "multiple statements: only one query is allowed")

    words = [w.lower() for w in WORD_RE.findall(bare)]
    if not words:
        return Verdict(False, "no usable keyword")

    if words[0] not in ("select", "with"):
        return Verdict(False, f"the query must start with SELECT or WITH, not '{words[0]}'")

    forbidden = sorted(set(words) & FORBIDDEN_STATEMENTS)
    if forbidden:
        return Verdict(False, f"forbidden statement: {', '.join(forbidden).upper()}")

    dangerous = sorted(set(words) & FORBIDDEN_FUNCTIONS)
    if dangerous:
        return Verdict(False, f"forbidden function: {', '.join(dangerous)}")

    # Cited parameters must be the ones we supplied. An invented `:v9` would fail
    # at execution with an unhelpful error; worse, a forgotten parameter would mean
    # the model wrote the value in clear text.
    used = tuple(sorted(set(PARAMETER_RE.findall(raw))))
    if expected_parameters is not None:
        unknown = sorted(set(used) - expected_parameters)
        if unknown:
            return Verdict(False, f"unknown parameter: {', '.join(unknown)}", used)
        missing = sorted(expected_parameters - set(used))
        if missing:
            return Verdict(
                False,
                f"unused parameter: {', '.join(missing)} — the value may have been "
                "written in clear text instead of bound",
                used,
            )

    unknown_columns = _unknown_columns(bare)
    if unknown_columns:
        return Verdict(
            False,
            f"unknown column: {', '.join(unknown_columns)} — not in the schema you were given",
            used,
        )

    return Verdict(True, "query accepted", used)


def _unknown_columns(sql: str) -> list[str]:
    """Column names the query invents.

    Why this belongs in the validator
    ---------------------------------
    A hallucinated column is caught by SQLite anyway — but only at execution, as
    `no such column: glucose`, at which point the request has already failed for
    the user. Catching it here turns a dead end into a repair: the generator sends
    the reason back to the model, which rewrites the query.

    Measured on the 107-question evaluation, this was one of the two remaining
    defects: the model wrote `SELECT AVG(glucose) FROM patient`, inventing a column
    that exists in `apacheapsvar` but not in `patient`.

    Detection is deliberately conservative. Only a `table.column` or `alias.column`
    form is checked, because a bare identifier cannot be told apart from an alias
    or a function name without parsing SQL properly. Under-reporting is the right
    error direction: a missed hallucination still fails at execution, whereas a
    false accusation would reject a correct query.
    """
    from hybridsql.db.schema import read_schema

    try:
        schema = read_schema()
    except Exception:  # noqa: BLE001 — database unavailable: skip the check
        return []

    anywhere: set[str] = set()
    for table in schema.values():
        anywhere.update(c.name.lower() for c in table.columns)

    # Columns of the tables this query actually reads. Checking against the whole
    # schema was not enough: `glucose` exists — in `apacheapsvar` — so
    # `SELECT AVG(glucose) FROM patient` passed while being wrong.
    cited = {
        t.lower()
        for t in re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", sql, re.IGNORECASE)
    }
    in_scope: set[str] = set()
    for name in cited:
        table = schema.get(name)
        if table:
            in_scope.update(c.name.lower() for c in table.columns)

    unknown: list[str] = []

    def flag(name: str, allowed: set[str]) -> None:
        low = name.lower()
        if low == "*" or low in allowed or name in unknown:
            return
        unknown.append(name)

    # `alias.column` — checked against the whole schema only, because resolving the
    # alias to its table would require parsing the query properly.
    for name in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]+)\b", sql):
        flag(name, anywhere)

    # Bare column inside an aggregate, e.g. `AVG(glucose)`. Here the scope is
    # unambiguous: it must belong to one of the tables named in FROM/JOIN.
    if in_scope:
        for name in re.findall(
            r"\b(?:AVG|SUM|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?([A-Za-z_][A-Za-z0-9_]+)\s*\)",
            sql,
            re.IGNORECASE,
        ):
            flag(name, in_scope)

    return unknown


def require(sql: str, expected_parameters: set[str] | None = None) -> str:
    """Validate and return the SQL, or raise `SqlRejected`."""
    verdict = validate(sql, expected_parameters)
    if not verdict.valid:
        raise SqlRejected(verdict.reason)
    return sql.strip().rstrip(";").strip()
