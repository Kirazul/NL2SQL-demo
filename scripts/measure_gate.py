"""Measure the egress gate against every value in the database.

What this script produces
-------------------------
The **residual leak rate**: the share of real values that would cross the gate if
sent unmasked. It is the project's headline metric, and it must be measured, not
asserted.

Two populations are separated, because they do not mean the same thing:

- **all values** — includes bare numbers ("65", "2") and two-letter units ("mg").
  Those necessarily pass: nothing distinguishes the stored "65" from the "65" the
  analyst writes in "patients over 65". Counting them as leaks would inflate the
  figure without saying anything.

- **information-bearing values** — at least one word of three letters. That is the
  population that matters: the one whose disclosure would teach the cloud provider
  something.

The expected residue is not zero, and it is explainable: some words are both a
column name and a value. `apacheapsvar` has columns `albumin`, `creatinine`,
`bun`, `wbc`, `urine` — which are also lab names stored in `lab.labname`. Blocking
them would forbid the model from writing SQL against those columns. So the script
separates leaks by cause.

    python scripts/measure_gate.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybridsql.config import settings  # noqa: E402
from hybridsql.security import egress_gate as gate  # noqa: E402

BEARING_RE = re.compile(r"[A-Za-zÀ-ÿ]{3,}")


def load_values() -> list[str]:
    path = settings().value_index_path
    if not path.exists():
        raise SystemExit(f"{path} missing — run scripts/build_value_index.py")
    cx = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return [v for (v,) in cx.execute("SELECT DISTINCT value FROM values_fts") if v]
    finally:
        cx.close()


def cause(value: str, identifiers: frozenset[str]) -> str:
    """Why did this value pass?"""
    bearing = [w.lower() for w in BEARING_RE.findall(value)]
    if not bearing:
        return "no bearing word (number or unit)"
    if all(w in identifiers for w in bearing):
        return "collides with a column name"
    if any(w in identifiers for w in bearing):
        return "partially collides with a column name"
    return "allowlist vocabulary"


def main() -> None:
    values = load_values()
    identifiers = frozenset(gate._schema_identifiers())
    words = gate.allowlist()

    bearing = [v for v in values if BEARING_RE.search(v)]
    passed_all = [v for v in values if gate.check(v).allowed]
    passed_bearing = [v for v in bearing if gate.check(v).allowed]

    causes: dict[str, list[str]] = {}
    for v in passed_bearing:
        causes.setdefault(cause(v, identifiers), []).append(v)

    width = 46
    print("=" * 72)
    print("EGRESS GATE — residual leak rate")
    print("=" * 72)
    print(f"{'Allowlist (permitted words)':<{width}} {len(words):>8}")
    print(f"{'Words removed as database values':<{width}} {len(gate.forbidden_words_extended()):>8}")
    print()
    print(f"{'Distinct values tested':<{width}} {len(values):>8}")
    print(f"{'  of which information-bearing':<{width}} {len(bearing):>8}")
    print()
    rate_all = 100 * len(passed_all) / len(values) if values else 0
    rate_bearing = 100 * len(passed_bearing) / len(bearing) if bearing else 0
    print(f"{'Cross the gate (all values)':<{width}} {len(passed_all):>8}  {rate_all:5.2f} %")
    print(f"{'Cross the gate (bearing)':<{width}} {len(passed_bearing):>8}  {rate_bearing:5.2f} %")
    print()
    print("-" * 72)
    print("Breakdown by cause (information-bearing values only)")
    print("-" * 72)
    for reason, items in sorted(causes.items(), key=lambda kv: -len(kv[1])):
        share = 100 * len(items) / len(bearing)
        print(f"  {reason:<44} {len(items):>5}  {share:5.2f} %")
        for example in items[:4]:
            print(f"      - {example[:56]}")

    report = Path("data/warehouse/gate_measurement.json")
    report.write_text(
        json.dumps(
            {
                "allowlist_words": len(words),
                "removed_words": len(gate.forbidden_words_extended()),
                "values_tested": len(values),
                "values_bearing": len(bearing),
                "passed_all": len(passed_all),
                "passed_bearing": len(passed_bearing),
                "rate_all_pct": round(rate_all, 3),
                "rate_bearing_pct": round(rate_bearing, 3),
                "by_cause": {k: {"count": len(v), "examples": v[:12]} for k, v in causes.items()},
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n  -> {report}")


if __name__ == "__main__":
    main()
