"""The database schema, as shown to the cloud model.

This is the only structural element that crosses the trust boundary. It carries
no data: only table names, column names, types and keys. Two opposing
requirements to reconcile:

- **complete enough** for the model to write correct joins;
- **short enough** not to inflate the prompt (hence cost and latency).

Foreign keys are the critical part. eICU declares none; we rebuilt them (see
`scripts/build_database.py`), and that is precisely what tells the model how to
relate tables instead of making it guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from hybridsql.db.connection import connect


@dataclass(frozen=True)
class Column:
    name: str
    sql_type: str
    is_pk: bool = False
    not_null: bool = False


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
    """Introspect the database. Cached: the schema does not change at runtime."""
    tables: dict[str, Table] = {}
    with connect() as cx:
        names = [
            r[0]
            for r in cx.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for name in names:
            columns = tuple(
                Column(name=r[1], sql_type=r[2] or "TEXT", is_pk=bool(r[5]), not_null=bool(r[3]))
                for r in cx.execute(f'PRAGMA table_info("{name}")')
            )
            fks = tuple(
                ForeignKey(column=r[3], target_table=r[2], target_column=r[4])
                for r in cx.execute(f'PRAGMA foreign_key_list("{name}")')
            )
            (n,) = cx.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()
            tables[name] = Table(name=name, columns=columns, foreign_keys=fks, row_count=n)
    return tables


def _compact_type(t: str) -> str:
    """Reduce the type to what matters: the model does not need VARCHAR(220)."""
    t = (t or "TEXT").upper().split("(")[0]
    if t.startswith(("INT", "BIGINT", "SMALLINT")):
        return "INT"
    if t.startswith(("NUMERIC", "DOUBLE", "REAL", "FLOAT", "DECIMAL")):
        return "REAL"
    if t.startswith("BOOL"):
        return "INT"
    return "TEXT"


def ddl(selected_tables: set[str] | None = None, with_row_counts: bool = True) -> str:
    """Render the DDL sent to the model.

    `selected_tables` keeps only the tables relevant to the question (schema
    linking). Without it, the full schema is rendered.
    """
    schema = read_schema()
    names = sorted(selected_tables) if selected_tables else sorted(schema)

    chunks: list[str] = []
    for name in names:
        t = schema.get(name)
        if t is None:
            continue
        lines = []
        for c in t.columns:
            line = f"  {c.name} {_compact_type(c.sql_type)}"
            if c.is_pk:
                line += " PRIMARY KEY"
            lines.append(line)
        for fk in t.foreign_keys:
            lines.append(
                f"  FOREIGN KEY ({fk.column}) REFERENCES {fk.target_table}({fk.target_column})"
            )
        comment = f"  -- {t.row_count} rows" if with_row_counts else ""
        chunks.append(f"CREATE TABLE {t.name} ({comment}\n" + ",\n".join(lines) + "\n);")
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
    """(table, column) for textual columns — candidates for the value index."""
    return [
        (t.name, c.name)
        for t in read_schema().values()
        for c in t.columns
        if _compact_type(c.sql_type) == "TEXT" and not c.is_pk
    ]
