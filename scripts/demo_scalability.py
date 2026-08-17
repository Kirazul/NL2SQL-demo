"""Check that the indexing policy holds on a database larger than eICU.

The methodological problem
--------------------------
eICU-Demo is small: no textual column exceeds 5,000 distinct values, so **none
falls into tier B**. The "resolve on demand" path — the one carrying the whole
scalability argument — would therefore never run, and the demonstration would be
worthless.

The workaround
--------------
The vocabulary limit is a parameter, not a constant. Lowering it simulates a
database where the same columns would be far richer: at limit = 200,
`medication.drugname` (1,401 distinct) tips into tier B exactly as a column with
5 million values would in production.

What the script measures
------------------------
1. the A/B/C split and indexed volume for several limits;
2. resolution latency through tier A (pre-indexed) versus tier B (on demand), on
   the same mentions.

Output: `data/warehouse/scalability.json`, quoted in docs/03-INDEXING.md.

    python scripts/demo_scalability.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybridsql.db.value_index import (  # noqa: E402
    VOCABULARY_LIMIT,
    build,
    classify_columns,
    search,
)

LIMITS = [VOCABULARY_LIMIT, 1_000, 200]
MENTIONS = ["aspirin", "insulin", "potassium", "heparin", "warfarin", "sepsis"]
SCRATCH = Path("data/warehouse/_scalability")


def rule(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title), flush=True)


def split() -> list[dict[str, object]]:
    """Classification only, without building: we want the shape of the policy."""
    rule("1. Column split by vocabulary limit")
    print(f"{'limit':>8} | {'A':>4} {'B':>4} {'C':>4} | {'values in A':>12} | columns tipped into B")
    lines = []
    for limit in LIMITS:
        cls = classify_columns(limit)
        a = [c for c in cls if c.tier == "A"]
        b = [c for c in cls if c.tier == "B"]
        c_ = [c for c in cls if c.tier == "C"]
        values = sum(x.distinct for x in a)
        examples = ", ".join(x.ref for x in sorted(b, key=lambda x: -x.distinct)[:3])
        print(
            f"{limit:>8} | {len(a):>4} {len(b):>4} {len(c_):>4} | {values:>12} | {examples or '—'}",
            flush=True,
        )
        lines.append(
            {
                "limit": limit,
                "tier_A": len(a),
                "tier_B": len(b),
                "tier_C": len(c_),
                "values_indexed": values,
                "tipped_into_B": [{"ref": x.ref, "distinct": x.distinct} for x in b],
            }
        )
    return lines


def _timed(mention: str, source: Path) -> tuple[float, list]:
    start = time.perf_counter()
    results = search(mention, limit=3, source=source)
    return (time.perf_counter() - start) * 1000, results


def latencies(low_limit: int) -> dict[str, object]:
    """Compare the cost of both paths on the same mentions."""
    rule(f"2. Resolution latency — tier A vs tier B (limit lowered to {low_limit})")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    reference = SCRATCH / "index_reference.db"
    low = SCRATCH / "index_low_limit.db"

    build(destination=reference, verbose=False, vocabulary_limit=VOCABULARY_LIMIT)
    build(destination=low, verbose=False, vocabulary_limit=low_limit)

    print(f"{'mention':<12} | {'A (indexed)':>12} | {'B (on demand)':>15} | ratio")
    measures = []
    for m in MENTIONS:
        ms_a, res_a = _timed(m, reference)
        ms_b, res_b = _timed(m, low)
        ratio = ms_b / ms_a if ms_a else 0
        tiers_b = {r.tier for r in res_b} or {"—"}
        print(
            f"{m:<12} | {ms_a:>9.1f} ms | {ms_b:>12.1f} ms | x{ratio:.0f}  "
            f"(tier {'/'.join(sorted(tiers_b))})",
            flush=True,
        )
        measures.append(
            {
                "mention": m,
                "ms_tier_A": round(ms_a, 1),
                "ms_low_limit": round(ms_b, 1),
                "ratio": round(ratio, 1),
                "tiers_returned": sorted(tiers_b),
                "found_A": [r.ref for r in res_a],
                "found_low": [r.ref for r in res_b],
            }
        )

    size_reference = reference.stat().st_size / 1048576
    size_low = low.stat().st_size / 1048576
    print(f"\nIndex size — limit {VOCABULARY_LIMIT}: {size_reference:.2f} MB")
    print(f"Index size — limit {low_limit:>4}: {size_low:.2f} MB")
    return {
        "low_limit": low_limit,
        "measures": measures,
        "size_mb_reference": round(size_reference, 2),
        "size_mb_low_limit": round(size_low, 2),
    }


def main() -> None:
    report = {"split": split(), "latencies": latencies(200)}
    out = Path("data/warehouse/scalability.json")
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    rule("3. Reading")
    print(
        "Tier B answers more slowly — expected, it queries the database instead of\n"
        "an index. What it buys: zero size at rest, and a cost independent of the\n"
        "column's distinct-value count, bounded only by the scan cap. That is what\n"
        "lets the same pipeline attach to a column with millions of values without\n"
        "rebuilding anything."
    )
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
