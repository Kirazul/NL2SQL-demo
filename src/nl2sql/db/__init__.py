"""Read-only access to the database and its schema."""

from nl2sql.db.schema import Column, Table, ddl, read_schema, summary, text_columns
from nl2sql.db.sqlite import DatabaseNotFound, QueryTimeout, connect, execute

__all__ = [
    "Column",
    "DatabaseNotFound",
    "QueryTimeout",
    "Table",
    "connect",
    "ddl",
    "execute",
    "read_schema",
    "summary",
    "text_columns",
]
