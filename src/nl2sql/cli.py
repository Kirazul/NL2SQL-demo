"""Command line: get the data in place, ask a question, serve, benchmark.

    python -m nl2sql.cli database          put eicu.db in data/, as published
    python -m nl2sql.cli index             build the value index
    python -m nl2sql.cli check             glossary and gate self-check
    python -m nl2sql.cli ask "..."         one question through one arm
    python -m nl2sql.cli serve             the REST API
    python -m nl2sql.cli bench             run every variant and rank them
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from nl2sql.config import settings

# The demo is published by PhysioNet and needs no credentials.
SOURCE_URL = (
    "https://physionet.org/static/published-projects/eicu-crd-demo/"
    "eicu-collaborative-research-database-demo-2.0.1.zip"
)
SQLITE_IN_ZIP = "eicu-collaborative-research-database-demo-2.0.1/sqlite/eicu_v2_0_1.sqlite3.gz"


def _fetch(destination: Path, source: str | None) -> None:
    """Get the published export onto disk, from a local copy or from PhysioNet."""
    import gzip
    import io
    import urllib.request
    import zipfile

    local = Path(source) if source else None
    if local and local.exists():
        print(f"copying {local}")
        shutil.copy2(local, destination)
        return

    print(f"downloading {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL) as response:  # noqa: S310 — a fixed https URL
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    with archive.open(SQLITE_IN_ZIP) as compressed, destination.open("wb") as out:
        shutil.copyfileobj(gzip.GzipFile(fileobj=compressed), out)


def _relationships(cx) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
    """Read each table's key and the columns that point at another table's key.

    A key is found by behaviour rather than by name: the first identifier column
    holding exactly as many distinct values as there are rows. A column repeating
    another table's key is a reference to it. Both are properties of the data, so
    they stay right when the extract is refreshed.
    """
    tables = [r[0] for r in cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )]

    keys: dict[str, str] = {}
    columns: dict[str, list[tuple[str, str]]] = {}
    for table in tables:
        info = [(r[1], r[2] or "TEXT") for r in cx.execute(f'PRAGMA table_info("{table}")')]
        columns[table] = [(name.lower(), sql_type) for name, sql_type in info]
        (rows,) = cx.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
        for name, _ in info:
            if not name.lower().endswith("id") or name.lower().endswith("offset"):
                continue
            (distinct,) = cx.execute(f'SELECT COUNT(DISTINCT "{name}") FROM "{table}"').fetchone()
            if rows and distinct == rows:
                keys[table] = name.lower()
                break

    owner = {key: table for table, key in keys.items()}
    references = {
        table: [(name, owner[name]) for name, _ in cols if name in owner and owner[name] != table]
        for table, cols in columns.items()
    }
    return keys, references


def database(args: argparse.Namespace) -> int:
    """Build `data/eicu.db` from the published eICU-CRD v2.0.1 export."""
    import sqlite3
    import time

    cfg = settings()
    destination = cfg.db_path
    if destination.exists() and not args.force:
        print(f"{destination} already present ({destination.stat().st_size / 1048576:.0f} MB). "
              "--force to rebuild.")
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    source = destination.with_suffix(".source.db")
    if not source.exists():
        _fetch(source, args.source)

    started = time.perf_counter()
    src = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    try:
        keys, references = _relationships(src)
    finally:
        src.close()
    print(f"{len(keys)} keys, {sum(len(r) for r in references.values())} references")

    destination.unlink(missing_ok=True)
    out = sqlite3.connect(destination)
    out.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
    out.execute("ATTACH DATABASE ? AS src", (str(source),))

    for table in sorted(references):
        columns = [
            (r[1].lower(), r[2] or "TEXT")
            for r in out.execute(f'PRAGMA src.table_info("{table}")')
        ]
        key = keys.get(table)
        lines = [
            f'  "{name}" {sql_type}' + (" PRIMARY KEY" if name == key else "")
            for name, sql_type in columns
        ]
        lines += [
            f'  FOREIGN KEY ("{column}") REFERENCES "{target}"("{keys[target]}")'
            for column, target in references[table]
        ]
        out.execute(f'CREATE TABLE "{table}" (\n' + ",\n".join(lines) + "\n)")
        listed = ", ".join(f'"{name}"' for name, _ in columns)
        out.execute(f'INSERT INTO "{table}" SELECT {listed} FROM src."{table}"')
        (copied,) = out.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
        print(f"  {table:<24} {copied:>9}", flush=True)

    # Indexes on the columns joins actually travel through, so a four-table query
    # finishes in milliseconds instead of scanning 1.5 million rows.
    for table, refs in references.items():
        for column, _ in refs:
            out.execute(f'CREATE INDEX IF NOT EXISTS "ix_{table}_{column}" ON "{table}"("{column}")')
    out.commit()
    out.execute("DETACH DATABASE src")
    out.execute("VACUUM")
    out.close()

    from nl2sql.db.schema import summary

    print(f"\nbuilt in {time.perf_counter() - started:.0f}s -> {destination}")
    print(json.dumps(summary(), indent=2))
    if not args.keep_source:
        source.unlink(missing_ok=True)
    return 0


def index(args: argparse.Namespace) -> int:
    """Classify every text column and index the bounded ones."""
    from nl2sql.db.values import build

    build(limit=args.limit)
    return 0


def check(_: argparse.Namespace) -> int:
    """Everything that can be verified without calling a model."""
    from nl2sql.db import catalog, schema
    from nl2sql.db.values import stats as index_stats
    from nl2sql.nlp import glossary
    from nl2sql.privacy import gate

    problems = glossary.validate()
    report = {
        "schema": schema.summary(),
        "index": index_stats(),
        "catalog": catalog.stats(),
        "gate": gate.stats(),
        "glossary_problems": problems,
    }
    print(json.dumps(report, indent=2, default=str))
    return 1 if problems else 0


def ask(args: argparse.Namespace) -> int:
    """One question, end to end, with every traced step printed."""
    from nl2sql.core import graph
    from nl2sql.core.state import public

    state = graph.run(args.question, arm=args.arm, write=not args.no_write, variant=args.variant)
    for step in state.get("trace", []):
        print(f"  [{step['zone']:>5}] {step['label']:<38} {step['ms']:>7.0f} ms  {step['summary']}")
    print()
    print(json.dumps(public(state), indent=2, default=str))
    return 0 if state.get("success") else 1


def serve(_: argparse.Namespace) -> int:
    import uvicorn

    cfg = settings()
    uvicorn.run("nl2sql.api:app", host=cfg.api_host, port=cfg.api_port, log_level=cfg.log_level.lower())
    return 0


def bench(args: argparse.Namespace) -> int:
    """Run the variants over the question set and print the ranking."""
    from nl2sql.optimize.benchmark import calibrate, compare

    report = compare(variants=args.variants, limit=args.limit)
    print()
    header = list(report["table"][0]) if report["table"] else []
    print(" | ".join(f"{h:>16}" for h in header))
    for row in report["table"]:
        print(" | ".join(f"{str(row[h]):>16}" for h in header))
    print("\nranking:", " > ".join(f"{n} ({a:.0%})" for n, a in report["ranking"]))
    print("escalation threshold:", json.dumps(calibrate(), indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nl2sql", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("database", help="put the published eICU export in data/")
    p.add_argument("--source", help="a local copy of eicu_v2_0_1.sqlite3")
    p.add_argument("--force", action="store_true", help="rebuild even if present")
    p.add_argument("--keep-source", action="store_true", help="keep the downloaded export")
    p.set_defaults(fn=database)

    p = sub.add_parser("index", help="build the value index")
    p.add_argument("--limit", type=int, default=5000, help="tier A cutoff, distinct values")
    p.set_defaults(fn=index)

    p = sub.add_parser("check", help="schema, index, glossary and gate self-check")
    p.set_defaults(fn=check)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.add_argument("--arm", default="hybrid")
    p.add_argument("--variant", default="baseline")
    p.add_argument("--no-write", action="store_true", help="stop at the rows")
    p.set_defaults(fn=ask)

    p = sub.add_parser("serve", help="run the REST API")
    p.set_defaults(fn=serve)

    p = sub.add_parser("bench", help="compare the variants")
    p.add_argument("--variants", nargs="*", help="defaults to all of them")
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.set_defaults(fn=bench)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
