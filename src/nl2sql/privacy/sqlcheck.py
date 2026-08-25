"""Validate the SQL a remote model wrote, before it reaches the engine."""

from __future__ import annotations

import re
from dataclasses import dataclass

FORBIDDEN_STATEMENTS = frozenset(
    "insert update delete drop alter create replace truncate attach detach vacuum "
    "reindex pragma begin commit rollback savepoint release analyze".split()
)
FORBIDDEN_FUNCTIONS = frozenset({"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"})

COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
STRING_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
PARAMETER_RE = re.compile(r":v\d+")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
QUALIFIED_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]+)\b")
AGGREGATE_RE = re.compile(
    r"\b(?:AVG|SUM|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?([A-Za-z_][A-Za-z0-9_]+)\s*\)", re.IGNORECASE
)
SOURCE_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


class SqlRejected(ValueError):
    """The query does not satisfy the execution conditions."""


@dataclass(frozen=True)
class Verdict:
    valid: bool
    reason: str = ""
    parameters_used: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.valid


def unknown_columns(sql: str) -> list[str]:
    """Column names the query invents."""
    from nl2sql.db.schema import read_schema

    try:
        schema = read_schema()
    except Exception:  # noqa: BLE001 — no database: skip the check
        return []

    anywhere = {c.name.lower() for t in schema.values() for c in t.columns}
    # Scoped to the tables actually read: `glucose` exists in `apacheapsvar`, so
    # `SELECT AVG(glucose) FROM patient` passed a whole-schema check while wrong.
    in_scope: set[str] = set()
    for name in {t.lower() for t in SOURCE_RE.findall(sql)}:
        table = schema.get(name)
        if table:
            in_scope.update(c.name.lower() for c in table.columns)

    unknown: list[str] = []

    def flag(name: str, allowed: set[str]) -> None:
        if name.lower() != "*" and name.lower() not in allowed and name not in unknown:
            unknown.append(name)

    for name in QUALIFIED_RE.findall(sql):
        flag(name, anywhere)
    if in_scope:
        for name in AGGREGATE_RE.findall(sql):
            flag(name, in_scope)
    return unknown


def validate(sql: str, expected_parameters: set[str] | None = None) -> Verdict:
    """Return a verdict without raising."""
    raw = (sql or "").strip()
    if not raw:
        return Verdict(False, "the model did not return SQL — the question is probably out of scope")

    bare = STRING_RE.sub(" '' ", COMMENT_RE.sub(" ", raw)).strip().rstrip(";").strip()
    if not bare:
        return Verdict(False, "empty query after comment removal")
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

    invented = unknown_columns(bare)
    if invented:
        return Verdict(
            False, f"unknown column: {', '.join(invented)} — not in the schema you were given", used
        )
    return Verdict(True, "query accepted", used)


def require(sql: str, expected_parameters: set[str] | None = None) -> str:
    verdict = validate(sql, expected_parameters)
    if not verdict.valid:
        raise SqlRejected(verdict.reason)
    return sql.strip().rstrip(";").strip()
