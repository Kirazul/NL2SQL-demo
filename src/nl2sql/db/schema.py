"""The schema, as introspected and as shown to a model.

The only structural element that crosses the boundary. It carries no data, and
has to be complete enough for correct joins yet short enough not to inflate the
prompt. The keys are the part that matters: a declared FOREIGN KEY tells the
model how tables relate instead of leaving it to guess, at no extra token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from nl2sql.db.sqlite import connect


@dataclass(frozen=True)
class Column:
    name: str
    sql_type: str
    is_pk: bool = False


@dataclass(frozen=True)
class ForeignKey:
    column: str
    target_table: str
    target_column: str


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...] = field(default_factory=tuple)
    row_count: int = 0

    @property
    def primary_key(self) -> str | None:
        return next((c.name for c in self.columns if c.is_pk), None)


@lru_cache(maxsize=1)
def read_schema() -> dict[str, Table]:
    """Introspect the database once per process. It does not change at runtime."""
    tables: dict[str, Table] = {}
    with connect(timeout_s=120) as cx:
        names = [
            r[0]
            for r in cx.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for name in names:
            columns = tuple(
                Column(r[1], r[2] or "TEXT", bool(r[5]))
                for r in cx.execute(f'PRAGMA table_info("{name}")')
            )
            keys = tuple(
                ForeignKey(r[3], r[2], r[4])
                for r in cx.execute(f'PRAGMA foreign_key_list("{name}")')
            )
            (n,) = cx.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()
            tables[name] = Table(name, columns, keys, n)
    return tables


def compact_type(sql_type: str) -> str:
    """VARCHAR(220) is noise; INT / REAL / TEXT is all a model needs."""
    t = (sql_type or "TEXT").upper().split("(")[0]
    if t.startswith(("INT", "BIGINT", "SMALLINT", "BOOL")):
        return "INT"
    if t.startswith(("NUMERIC", "DOUBLE", "REAL", "FLOAT", "DECIMAL")):
        return "REAL"
    return "TEXT"


@lru_cache(maxsize=1)
def key_columns() -> frozenset[str]:
    """Columns that carry a relationship: a primary key, or one end of a foreign key."""
    names: set[str] = set()
    for table in read_schema().values():
        names.update(c.name for c in table.columns if c.is_pk)
        for key in table.foreign_keys:
            names.update((key.column, key.target_column))
    return frozenset(names)


def ddl(tables: set[str] | None = None, columns: dict[str, list[str]] | None = None) -> str:
    """The DDL sent to a model: names, types, row counts, keys.

    `tables` restricts which tables appear; `columns` restricts which columns of
    each, which is the lever the lean variant pulls. Keys are never dropped by
    that restriction — a table with its join columns removed cannot be joined.
    """
    schema = read_schema()
    names = sorted(tables) if tables else sorted(schema)
    keys = key_columns()

    chunks: list[str] = []
    for name in names:
        table = schema.get(name)
        if table is None:
            continue
        keep = columns.get(name) if columns else None
        kept = [c for c in table.columns if keep is None or c.name in keep or c.name in keys]

        lines = [
            f"  {c.name} {compact_type(c.sql_type)}" + (" PRIMARY KEY" if c.is_pk else "")
            for c in kept
        ]
        lines += [
            f"  FOREIGN KEY ({k.column}) REFERENCES {k.target_table}({k.target_column})"
            for k in table.foreign_keys
            if k.target_table in names
        ]
        body = ",\n".join(lines)
        chunks.append(f"CREATE TABLE {table.name} (  -- {table.row_count} rows\n{body}\n);")
    return "\n\n".join(chunks)


def summary() -> dict[str, int]:
    schema = read_schema()
    return {
        "tables": len(schema),
        "columns": sum(len(t.columns) for t in schema.values()),
        "foreign_keys": sum(len(t.foreign_keys) for t in schema.values()),
        "rows": sum(t.row_count for t in schema.values()),
    }


def text_columns() -> list[tuple[str, str]]:
    """(table, column) pairs worth considering for the value index."""
    return [
        (t.name, c.name)
        for t in read_schema().values()
        for c in t.columns
        if compact_type(c.sql_type) == "TEXT" and not c.is_pk
    ]
