"""Build `data/warehouse/eicu.db` from the published eICU-CRD v2.0.1 SQLite export.

The export PhysioNet publishes has no primary keys, no foreign keys and no
indexes: 31 tables, 4.6 M rows, no constraints at all. This script rebuilds the
schema.

Where the constraints come from
-------------------------------
- **Primary keys and indexes: from the official source.** The consortium's own
  repository (MIT-LCP/eicu-code, `build-db/postgres/postgres_add_indexes.sql`)
  defines 17 primary keys and 22 indexes. We take them as they are — it is the
  reference, and it is citable.
- **Foreign keys: added by us.** eICU declares none, even officially. The
  relationships are documented all the same: `patientunitstayid` links 28 tables
  to `patient`, `hospitalid` links `patient` to `hospital`. We declare them.

Why this is not cosmetic
------------------------
The DDL is the *only* thing the cloud model ever sees of the database. A DDL that
declares `FOREIGN KEY (patientunitstayid) REFERENCES patient` tells it explicitly
how to join; without that it guesses. Accuracy improves for not one extra token.
The indexes make those joins run in reasonable time on two cores.

Usage
-----
    python scripts/build_database.py                 # download if needed, then build
    python scripts/build_database.py --force         # rebuild even if present
    python scripts/build_database.py --source X.sqlite3
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "warehouse" / "eicu.db"
CACHE = ROOT / "data" / "raw"

URL = (
    "https://physionet.org/static/published-projects/eicu-crd-demo/"
    "eicu-collaborative-research-database-demo-2.0.1.zip"
)
SQLITE_PATH_IN_ZIP = (
    "eicu-collaborative-research-database-demo-2.0.1/sqlite/eicu_v2_0_1.sqlite3.gz"
)

# The eICU consortium's official reference: 17 primary keys + 22 indexes.
REFERENCE_URL = (
    "https://raw.githubusercontent.com/MIT-LCP/eicu-code/main/"
    "build-db/postgres/postgres_add_indexes.sql"
)

# --- eICU's foreign-key graph -----------------------------------------------------
# `patientunitstayid` appears in 28 of the 31 tables: it is the backbone.
PARENT_OF = {
    "patientunitstayid": ("patient", "patientunitstayid"),
    "hospitalid": ("hospital", "hospitalid"),
}

# Composite indexes of the form "join key + filtered column".
#
# This is the most profitable correction of the lot. Without them, a question like
# "which patients on aspirin had a high creatinine?" started from ix_lab_labname
# (thousands of rows) and then bounced back and forth: the rebuilt database was
# *slower* than the raw export (107 ms against 46 ms). With them, SQLite switches
# to a COVERING INDEX and drops to 42 ms.
#
# Deliberately absent: vitalperiodic and nursecharting (1.6 M and 1.5 M rows).
# Their composite cost 64 MB for no gain — the plain index on the join key is
# already enough there (1,634x faster).
COMPOSITE_INDEXES = {
    "lab": [("patientunitstayid", "labname")],
    "medication": [("patientunitstayid", "drugname")],
    "infusiondrug": [("patientunitstayid", "drugname")],
    "admissiondrug": [("patientunitstayid", "drugname")],
    "diagnosis": [("patientunitstayid", "diagnosisstring")],
    "treatment": [("patientunitstayid", "treatmentstring")],
    "allergy": [("patientunitstayid", "allergyname")],
}

# Columns questions filter on often → indexes that help beyond the joins.
BUSINESS_INDEXES = {
    "patient": ["gender", "age", "ethnicity", "hospitaldischargestatus", "unittype",
                "hospitaldischargeyear", "patienthealthsystemstayid"],
    "medication": ["drugname", "routeadmin"],
    "admissiondrug": ["drugname"],
    "infusiondrug": ["drugname"],
    "lab": ["labname"],
    "diagnosis": ["diagnosisstring", "icd9code"],
    "treatment": ["treatmentstring"],
    "allergy": ["allergyname"],
    "apachepatientresult": ["apacheversion", "actualiculos"],
    "pasthistory": ["pasthistorypath"],
    "hospital": ["region"],
}


def log(message: str) -> None:
    # The Windows console is cp1252: without this, a single arrow crashes the script.
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(message.encode(encoding, errors="replace").decode(encoding), flush=True)


# ---------------------------------------------------------------------------------
# 0. The official reference: PKs and indexes as the consortium defines them
# ---------------------------------------------------------------------------------
def load_reference() -> tuple[dict[str, str], list[tuple[str, list[str]]]]:
    """Parse `postgres_add_indexes.sql` and extract its primary keys and indexes."""
    import re

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "postgres_add_indexes.sql"
    if not path.exists():
        log("Fetching the official MIT-LCP reference...")
        urllib.request.urlretrieve(REFERENCE_URL, path)  # noqa: S310
    sql = path.read_text(encoding="utf-8", errors="replace")

    primary_keys = {
        m.group(1).lower(): m.group(2).strip().lower()
        for m in re.finditer(
            r"ALTER\s+TABLE\s+(\w+)\s+ADD\s+CONSTRAINT\s+\w+\s+primary\s+key\s*\(([^)]+)\)",
            sql, re.I,
        )
        if "," not in m.group(2)  # composite primary keys are ignored
    }
    indexes = [
        (m.group(2).lower(), [c.strip().lower() for c in m.group(3).split(",")])
        for m in re.finditer(
            r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(\w+)\s+ON\s+(\w+)\s*\(([^)]+)\)", sql, re.I
        )
    ]
    log(f"Official reference: {len(primary_keys)} primary keys, {len(indexes)} indexes.")
    return primary_keys, indexes


# ---------------------------------------------------------------------------------
# 1. Getting the source
# ---------------------------------------------------------------------------------
def get_source(source: Path | None) -> Path:
    if source and source.exists():
        log(f"Source supplied: {source}")
        return source

    CACHE.mkdir(parents=True, exist_ok=True)
    raw = CACHE / "eicu_v2_0_1.sqlite3"
    if raw.exists():
        log(f"Source already cached: {raw}")
        return raw

    archive = CACHE / "eicu-demo.zip"
    if not archive.exists():
        log(f"Downloading eICU-CRD (~130 MB)...\n  {URL}")
        urllib.request.urlretrieve(URL, archive)  # noqa: S310

    import zipfile

    log("Extracting the provided SQLite file...")
    with zipfile.ZipFile(archive) as z:
        with z.open(SQLITE_PATH_IN_ZIP) as compressed, \
             gzip.open(compressed) as f, open(raw, "wb") as out:
            shutil.copyfileobj(f, out)
    return raw


# ---------------------------------------------------------------------------------
# 2. Detecting primary keys — we verify, we do not assume
# ---------------------------------------------------------------------------------
def detect_primary_key(cx: sqlite3.Connection, table: str, columns: list[tuple]) -> str | None:
    """Return the column that can serve as a primary key, or None.

    The rule: a column whose name ends in 'id', which is not an obvious foreign
    key, and whose values really are unique and never null.
    """
    total = cx.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    if total == 0:
        return None

    candidates = [c[1] for c in columns if c[1].lower().endswith("id")]
    # The table's own column comes before any foreign key.
    candidates.sort(key=lambda c: c.lower() in PARENT_OF)

    for column in candidates:
        distinct, nulls = cx.execute(
            f'SELECT COUNT(DISTINCT "{column}"), '
            f'SUM(CASE WHEN "{column}" IS NULL THEN 1 ELSE 0 END) FROM "{table}"'
        ).fetchone()
        if distinct == total and not nulls:
            return column
    return None


# ---------------------------------------------------------------------------------
# 3. Rebuilding
# ---------------------------------------------------------------------------------
def build(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()

    official_keys, official_indexes = load_reference()

    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    tables = [r[0] for r in src.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )]
    log(f"\n{len(tables)} tables found in the source.")

    info: dict[str, dict] = {}
    from_reference = detected = 0
    for table in tables:
        columns = src.execute(f'PRAGMA table_info("{table}")').fetchall()
        names = {c[1].lower() for c in columns}

        key = official_keys.get(table.lower())
        if key and key in names:
            origin = "official"
            from_reference += 1
        else:
            key = detect_primary_key(src, table, columns)
            origin = "detected" if key else "none"
            detected += 1 if key else 0

        # Only declare a primary key if it really is unique in THIS dataset: a key
        # that holds on the full database may not hold on an extract.
        if key:
            total = src.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            distinct, nulls = src.execute(
                f'SELECT COUNT(DISTINCT "{key}"), '
                f'SUM(CASE WHEN "{key}" IS NULL THEN 1 ELSE 0 END) FROM "{table}"'
            ).fetchone()
            if total and (distinct != total or nulls):
                log(f"  ! {table}.{key} ({origin}) is not unique here -> key dropped")
                key = None

        info[table] = {"columns": columns, "key": key}

    without = [t for t, i in info.items() if not i["key"]]
    log(f"Primary keys kept: {len(tables) - len(without)}/{len(tables)} "
        f"({from_reference} official, {detected} detected)"
        + (f"\n  no key: {', '.join(without)}" if without else ""))

    # Parents have to exist before children.
    order = [t for t in ("hospital", "patient") if t in tables]
    order += [t for t in tables if t not in order]

    dst = sqlite3.connect(dest)
    dst.execute("PRAGMA journal_mode=OFF")
    dst.execute("PRAGMA synchronous=OFF")

    foreign_keys = 0
    for table in order:
        columns, key = info[table]["columns"], info[table]["key"]
        lines = []
        for column in columns:
            # Case normalised to lowercase. The source is inconsistent:
            # respiratorycare and respiratorycharting are entirely UPPERCASE,
            # nursecharting/nurseassessment/nursecare/careplangoal/note are half and
            # half. SQLite identifiers are case-insensitive so this is safe — and it
            # matters, because this DDL is exactly what the cloud model reads.
            # Note: the source's typos are preserved (cplcareprovderid). Fixing them
            # would break every query written against the real eICU.
            name, sql_type, not_null = column[1].lower(), column[2] or "TEXT", column[3]
            line = f'    "{name}" {sql_type}'
            if key and name == key.lower():
                line += " PRIMARY KEY"
            elif not_null:
                line += " NOT NULL"
            lines.append(line)

        for column in columns:
            name = column[1].lower()
            if name in PARENT_OF and name != (key or "").lower():
                parent_table, parent_column = PARENT_OF[name]
                parent_key = (info.get(parent_table, {}).get("key") or "").lower()
                if parent_table in tables and parent_table != table and parent_key == parent_column:
                    lines.append(
                        f'    FOREIGN KEY ("{name}") '
                        f'REFERENCES "{parent_table}"("{parent_column}")'
                    )
                    foreign_keys += 1

        dst.execute(f'CREATE TABLE "{table}" (\n' + ",\n".join(lines) + "\n)")

    log(f"Foreign keys declared: {foreign_keys}")

    log("\nCopying the data...")
    dst.execute("ATTACH DATABASE ? AS src", (str(source),))
    total_rows = 0
    for table in order:
        target = ", ".join(f'"{c[1].lower()}"' for c in info[table]["columns"])
        origin = ", ".join(f'"{c[1]}"' for c in info[table]["columns"])
        dst.execute(f'INSERT INTO main."{table}" ({target}) SELECT {origin} FROM src."{table}"')
        n = dst.execute(f'SELECT COUNT(*) FROM main."{table}"').fetchone()[0]
        total_rows += n
        log(f"  {table:<28} {n:>10,}")
    dst.commit()
    dst.execute("DETACH DATABASE src")
    log(f"  {'TOTAL':<28} {total_rows:>10,}")

    log("\nCreating indexes...")
    n_official = n_ours = 0

    for table, columns in official_indexes:
        if table not in info:
            continue
        available = {c[1].lower(): c[1].lower() for c in info[table]["columns"]}
        if not all(c in available for c in columns):
            continue
        real = [available[c] for c in columns]
        name = "ix_{}_{}".format(table, "_".join(real))
        listed = ", ".join('"{}"'.format(c) for c in real)
        dst.execute(f'CREATE INDEX IF NOT EXISTS "{name}" ON "{table}"({listed})')
        n_official += 1

    for table in order:
        available = {c[1].lower(): c[1].lower() for c in info[table]["columns"]}
        key = info[table]["key"]

        # Composites first: they are what pays off on filtered joins.
        for columns in COMPOSITE_INDEXES.get(table, []):
            if not all(c in available for c in columns):
                continue
            real = [available[c] for c in columns]
            name = "ix_{}_{}".format(table, "_".join(real))
            listed = ", ".join('"{}"'.format(c) for c in real)
            dst.execute(f'CREATE INDEX IF NOT EXISTS "{name}" ON "{table}"({listed})')
            n_ours += 1

        # Then the plain ones. An index duplicating the primary key is useless and
        # costs space, so it is skipped.
        targets = [v for k, v in available.items() if k in PARENT_OF and v != key]
        targets += [c for c in BUSINESS_INDEXES.get(table, []) if c.lower() in available]
        for column in dict.fromkeys(targets):
            if key and column.lower() == key.lower():
                continue
            name = f"ix_{table}_{column}"
            existed = dst.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?", (name,)
            ).fetchone()[0]
            dst.execute(f'CREATE INDEX IF NOT EXISTS "{name}" ON "{table}"("{column}")')
            if not existed:
                n_ours += 1

    dst.commit()
    log(f"{n_official} official + {n_ours} added = {n_official + n_ours} indexes.")

    log("\nANALYZE + VACUUM...")
    dst.execute("ANALYZE")
    dst.commit()
    dst.execute("VACUUM")
    dst.close()
    src.close()


# ---------------------------------------------------------------------------------
# 4. Verification
# ---------------------------------------------------------------------------------
BENCHMARK_QUERIES = {
    "wide join + LIKE": """
        SELECT p.gender, COUNT(DISTINCT p.patientunitstayid) FROM patient p
        JOIN medication m ON m.patientunitstayid = p.patientunitstayid
        JOIN lab l        ON l.patientunitstayid = p.patientunitstayid
        WHERE m.drugname LIKE 'ASPIRIN%' AND l.labname = 'creatinine'
        GROUP BY p.gender""",
    "single-patient lookup": "SELECT COUNT(*) FROM nursecharting WHERE patientunitstayid = 141765",
    "filter on drugname": "SELECT COUNT(*) FROM medication WHERE drugname = 'ASPIRIN EC 81 MG PO TBEC'",
    "hospital join": """
        SELECT h.region, COUNT(*) FROM patient p
        JOIN hospital h ON h.hospitalid = p.hospitalid GROUP BY h.region""",
    "one patient's labs": """
        SELECT labname, labresult FROM lab
        WHERE patientunitstayid = 141765 AND labname = 'creatinine'""",
    "one patient's vitals": "SELECT AVG(heartrate) FROM vitalperiodic WHERE patientunitstayid = 141765",
    "top drugs": """
        SELECT drugname, COUNT(*) c FROM medication
        GROUP BY drugname ORDER BY c DESC LIMIT 10""",
}


def time_query(path: Path, query: str, n: int = 5) -> float:
    """Best of n runs, after one warm-up pass."""
    cx = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cx.execute(query).fetchall()
    times = []
    for _ in range(n):
        started = time.perf_counter()
        cx.execute(query).fetchall()
        times.append(time.perf_counter() - started)
    cx.close()
    return min(times)


def verify(dest: Path, source: Path) -> None:
    cx = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
    tables = [r[0] for r in cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    foreign = sum(len(cx.execute(f'PRAGMA foreign_key_list("{t}")').fetchall()) for t in tables)
    primary = sum(1 for t in tables for c in cx.execute(f'PRAGMA table_info("{t}")') if c[5])
    indexes = len(list(cx.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")))
    violations = cx.execute("PRAGMA foreign_key_check").fetchall()
    cx.close()

    log("\n" + "=" * 62)
    log("  RESULT")
    log("=" * 62)
    log(f"  file             : {dest}")
    log(f"  size             : {dest.stat().st_size / 1048576:.0f} MB")
    log(f"  tables           : {len(tables)}")
    log(f"  primary keys     : {primary}")
    log(f"  foreign keys     : {foreign}")
    log(f"  indexes          : {indexes}")
    log(f"  key violations   : {len(violations)}")

    if source.exists():
        log("\n  Performance, 7 benchmark queries (best of 5):")
        log(f"  {'':<26}{'raw':>10}{'rebuilt':>13}{'gain':>10}")
        log("  " + "-" * 58)
        for name, query in BENCHMARK_QUERIES.items():
            before = time_query(source, query)
            after = time_query(dest, query)
            log(f"  {name:<26}{before * 1000:>8.1f}ms{after * 1000:>11.2f}ms{before / after:>9.1f}x")
    log("=" * 62)


def already_built(dest: Path) -> bool:
    """Is the database already there, and complete?

    Rebuilding costs several minutes and 429 MB of writing. On Kaggle, where the
    setup notebook is re-run on every saved version, that cost was being paid for
    nothing. So we check what distinguishes the rebuilt database from the raw
    export: the foreign keys, which eICU declares nowhere.
    """
    if not dest.exists():
        return False
    try:
        cx = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
        tables = [r[0] for r in cx.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        foreign = sum(len(cx.execute(f'PRAGMA foreign_key_list("{t}")').fetchall()) for t in tables)
        cx.close()
    except sqlite3.Error:
        return False
    return len(tables) >= 30 and foreign > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the database is already complete")
    arguments = parser.parse_args()

    if not arguments.force and already_built(arguments.dest):
        log(f"Already built: {arguments.dest} "
            f"({arguments.dest.stat().st_size / 1048576:.0f} MB). Use --force to redo it.")
        return 0

    source = get_source(arguments.source)
    build(source, arguments.dest)
    verify(arguments.dest, source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
