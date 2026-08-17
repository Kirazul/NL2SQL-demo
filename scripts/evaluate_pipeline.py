"""Run the full pipeline over the question sets and measure what happens.

What is measured
----------------
Stage 1 was evaluated on its own (`evaluate_understanding.py`). This script
measures the whole chain, which adds three things the earlier one could not see:

1. **Executable rate** — questions that reach a SELECT which runs on the database.
   The end-to-end figure a user would experience.
2. **Failure breakdown by stage** — a failure at `egress_gate` is a security
   success, a failure at `cloud` is an infrastructure problem, a failure at
   `sql_validation` is a model problem. Lumping them into one number would hide
   which part to fix.
3. **Real leak check** — the audit journal is emptied before the run and read
   afterwards. Every byte sent is accounted for.

Rate limiting
-------------
Free Groq caps at 30 requests per minute. The repair loop averages slightly above
one call per question, so we pace deliberately rather than collect 429s.

    python scripts/evaluate_pipeline.py            # 25 questions, no answer writing
    python scripts/evaluate_pipeline.py --all      # every question
    python scripts/evaluate_pipeline.py --write    # with the local answer writer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from hybridsql.pipeline.answer import answer  # noqa: E402
from hybridsql.security import audit  # noqa: E402

SETS = {
    "standard": Path("data/evaluation/questions_standard.yaml"),
    "hard": Path("data/evaluation/questions_hard.yaml"),
}
OUTPUT = Path("data/evaluation/pipeline_results.json")

# Free Groq allows 30 requests per minute. Slightly over two seconds between
# questions keeps us under it even when a repair adds a second call.
PACE_S = 2.2

# Failures that are *desired behaviour*, not defects: the pipeline correctly
# refused to send. Counting them as errors would penalise the system for working.
INTENTIONAL_REFUSALS = {"anonymize", "egress_gate"}


def load_questions(only: str | None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name, path in SETS.items():
        if only and name != only:
            continue
        if not path.exists():
            continue
        for case in yaml.safe_load(path.read_text(encoding="utf-8")):
            out.append((name, case["question"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="every question, not a sample")
    parser.add_argument("--write", action="store_true", help="run the local answer writer")
    parser.add_argument("--set", dest="only", choices=list(SETS), help="a single set")
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    questions = load_questions(args.only)
    if not args.all:
        # A spread sample rather than the first N: the sets are grouped by theme,
        # so taking the head would only measure drugs.
        step = max(1, len(questions) // args.limit)
        questions = questions[::step][: args.limit]

    audit.clear()
    print(f"Running {len(questions)} questions "
          f"({'with' if args.write else 'without'} answer writing)\n")

    records: list[dict] = []
    start_all = time.perf_counter()

    for i, (set_name, question) in enumerate(questions, start=1):
        if i > 1:
            time.sleep(PACE_S)
        a = answer(question, write_answer=args.write)
        record = a.summary() | {"set": set_name, "question": question}
        records.append(record)

        mark = "OK  " if a.success else "FAIL"
        detail = (
            f"{a.row_count:>5} rows  {a.ms_total:>7.0f} ms  {a.cloud_tokens:>5} tk"
            if a.success
            else f"[{a.failed_stage}] {a.failure_reason[:70]}"
        )
        print(f"{i:>3}/{len(questions)} {mark} {question[:56]:<58} {detail}", flush=True)

    wall = time.perf_counter() - start_all

    # --- Aggregation -----------------------------------------------------------
    total = len(records)
    ok = [r for r in records if r["success"]]
    failed = [r for r in records if not r["success"]]
    refusals = [r for r in failed if r["failed_stage"] in INTENTIONAL_REFUSALS]
    defects = [r for r in failed if r["failed_stage"] not in INTENTIONAL_REFUSALS]

    by_stage = Counter(r["failed_stage"] for r in failed)
    by_target = Counter(r["cloud_target"] for r in ok if r["cloud_target"])

    def avg(key: str, source: list[dict]) -> float:
        vals = [r["ms"][key] for r in source]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    journal = audit.leak_rate()

    width = 44
    print("\n" + "=" * 72)
    print("FULL PIPELINE")
    print("=" * 72)
    print(f"{'Questions run':<{width}} {total:>8}")
    print(f"{'Executable answers':<{width}} {len(ok):>8}  {100*len(ok)/total:5.1f} %")
    print(f"{'Deliberate refusals (correct behaviour)':<{width}} {len(refusals):>8}  "
          f"{100*len(refusals)/total:5.1f} %")
    print(f"{'Actual defects':<{width}} {len(defects):>8}  {100*len(defects)/total:5.1f} %")
    print()
    print(f"{'Success excluding refusals':<{width}} "
          f"{100*len(ok)/max(1, total-len(refusals)):>7.1f} %")
    print()
    print("-" * 72)
    print("Latency breakdown (successful questions)")
    print("-" * 72)
    print(f"{'  stage 1 understand (local)':<{width}} {avg('understand', ok):>7} ms")
    print(f"{'  stage 2 generate (cloud)':<{width}} {avg('generate', ok):>7} ms")
    print(f"{'  stage 3 execute (local)':<{width}} {avg('execute', ok):>7} ms")
    if args.write:
        print(f"{'  stage 3 write (local)':<{width}} {avg('write', ok):>7} ms")
    print(f"{'  TOTAL':<{width}} {avg('total', ok):>7} ms")
    print()
    tokens = sum(r["cloud_tokens"] for r in records)
    calls = sum(r["cloud_calls"] for r in records)
    repairs = sum(r["repairs"] for r in records)
    print(f"{'Cloud calls':<{width}} {calls:>8}")
    print(f"{'  of which repairs':<{width}} {repairs:>8}")
    print(f"{'Cloud tokens':<{width}} {tokens:>8}")
    print(f"{'  average per question':<{width}} {round(tokens/max(1,total)):>8}")
    for target, n in by_target.most_common():
        print(f"{'  served by ' + target:<{width}} {n:>8}")
    print()
    print("-" * 72)
    print("EGRESS GATE — everything sent during this run")
    print("-" * 72)
    print(f"{'Sends attempted':<{width}} {journal['sends']:>8}")
    print(f"{'  allowed':<{width}} {journal['allowed']:>8}")
    print(f"{'  blocked':<{width}} {journal['blocked']:>8}")

    if by_stage:
        print("\n" + "-" * 72)
        print("Failures by stage")
        print("-" * 72)
        for stage, n in by_stage.most_common():
            tag = "refusal (expected)" if stage in INTENTIONAL_REFUSALS else "defect"
            print(f"  {stage:<28} {n:>4}   {tag}")

    if defects:
        print("\nDefects in detail:")
        for r in defects:
            print(f"\n  - {r['question']}")
            print(f"      [{r['failed_stage']}] {r['failure_reason'][:150]}")
            if r["sql"]:
                print(f"      SQL: {r['sql'][:120]}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "questions": total,
                "executable": len(ok),
                "executable_pct": round(100 * len(ok) / total, 1),
                "deliberate_refusals": len(refusals),
                "defects": len(defects),
                "by_stage": dict(by_stage),
                "by_target": dict(by_target),
                "cloud_calls": calls,
                "cloud_repairs": repairs,
                "cloud_tokens": tokens,
                "gate": journal,
                "wall_seconds": round(wall, 1),
                "answer_writing": args.write,
                "records": records,
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n  -> {OUTPUT}")


if __name__ == "__main__":
    main()
