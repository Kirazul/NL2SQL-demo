"""Measure the egress gate — in both directions.

Why two measurements and not one
--------------------------------
A gate has two ways of being wrong, and optimising either one alone produces a
useless system:

- it **lets a value through** — a leak, which is what the first measurement counts;
- it **refuses an ordinary question** — which is what the second one counts, and
  what was missing. A gate that blocks everything scores a perfect leak rate and
  gets switched off on first use. Measured before this was added: 11 of 28
  realistic analytical questions were refused, every one of them on ordinary
  English (`records`, `associated`, `one`, `unique`) that no database owns.

So both numbers are printed together, and no change to the gate is justified by
one of them alone.

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
from collections import Counter
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


QUESTION_SETS = (
    Path("data/evaluation/questions_standard.yaml"),
    Path("data/evaluation/questions_hard.yaml"),
    Path("data/evaluation/questions_analytics.yaml"),
)


def load_questions() -> list[str]:
    import yaml

    questions: list[str] = []
    for path in QUESTION_SETS:
        if not path.exists():
            continue
        for case in yaml.safe_load(path.read_text(encoding="utf-8")) or ():
            question = str(case.get("question") or "").strip()
            if question:
                questions.append(question)
    return questions


def refusal_rate() -> dict[str, object]:
    """How often does the gate refuse a question that carries nothing to hide?

    The whole pipeline is replayed, not just the gate: the question is understood,
    its values are masked, and the segments that would actually be sent are the ones
    checked. Checking the raw question instead would count every drug name as a
    refusal, when masking is precisely what stops it from being one.

    A question refused **because a person's name is in it** is not counted: that
    refusal is the system working.
    """
    from hybridsql.pipeline import generate
    from hybridsql.pipeline.anonymize import UnmaskableQuestion, anonymize
    from hybridsql.pipeline.understand import understand

    refused: list[tuple[str, tuple[str, ...]]] = []
    tokens: Counter[str] = Counter()
    checked = skipped = 0

    for question in load_questions():
        try:
            understanding = understand(question)
            masked = anonymize(understanding)
        except UnmaskableQuestion:
            skipped += 1
            continue
        except Exception:  # noqa: BLE001 — a broken question is not a gate verdict
            skipped += 1
            continue

        checked += 1
        bad: list[str] = []
        for segment in generate.build_segments(understanding, masked):
            verdict = gate.check_segment(segment, "measure")
            if not verdict.allowed:
                bad.extend(verdict.refused_tokens)
        if bad:
            refused.append((question, tuple(dict.fromkeys(bad))))
            tokens.update(bad)

    return {
        "questions_checked": checked,
        "questions_skipped": skipped,
        "questions_refused": len(refused),
        "refusal_rate_pct": round(100 * len(refused) / checked, 2) if checked else 0.0,
        "tokens": tokens.most_common(),
        "examples": [{"question": q, "refused": list(t)} for q, t in refused[:12]],
    }


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

    print()
    print("=" * 72)
    print("EGRESS GATE — refusal rate on questions that carry nothing to hide")
    print("=" * 72)
    refusals = refusal_rate()
    print(f"{'Questions replayed end to end':<{width}} {refusals['questions_checked']:>8}")
    print(f"{'  refused by the gate':<{width}} {refusals['questions_refused']:>8}"
          f"  {refusals['refusal_rate_pct']:5.2f} %")
    print(f"{'  not counted (a name, or unmaskable)':<{width}} {refusals['questions_skipped']:>8}")
    if refusals["tokens"]:
        print("\n  words the gate refused, most frequent first")
        for token, n in refusals["tokens"][:15]:
            print(f"      {n:>3}x  {token}")
        print("\n  examples")
        for case in refusals["examples"][:6]:
            print(f"      {case['question'][:60]}")
            print(f"          refused: {', '.join(case['refused'])}")
    else:
        print("\n  no ordinary question is refused.")

    report = Path("data/warehouse/gate_measurement.json")
    report.write_text(
        json.dumps(
            {
                "allowlist_words": len(words),
                "removed_words": len(gate.forbidden_words_extended()),
                "value_token_vocabulary": len(gate.value_tokens()),
                "values_tested": len(values),
                "values_bearing": len(bearing),
                "passed_all": len(passed_all),
                "passed_bearing": len(passed_bearing),
                "rate_all_pct": round(rate_all, 3),
                "rate_bearing_pct": round(rate_bearing, 3),
                "by_cause": {k: {"count": len(v), "examples": v[:12]} for k, v in causes.items()},
                "questions": refusals,
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n  -> {report}")


if __name__ == "__main__":
    main()
