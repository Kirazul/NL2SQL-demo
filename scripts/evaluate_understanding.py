"""Measure stage 1 performance against the annotated question sets.

What this measures, and why these metrics
-----------------------------------------
Saying "it works" is worthless. Four figures, each on a distinct failure mode:

1. **Extraction recall** — of the business values the question contains, how many
   does GLiNER2 spot? What is not spotted cannot be masked: this is the direct
   measure of leak risk.

2. **Resolution accuracy** — of the spotted values, how many are attached to the
   right column? A wrong column produces SQL that runs and returns a wrong
   answer — the costliest failure.

3. **Kind-classification accuracy** — concepts ("mortality rate") must not be
   looked up in the database, values must. Getting this wrong masks what should
   not be, or lets through what should be masked.

4. **Full-understanding rate** — questions where *everything* is right. The only
   figure that describes what the user will see.

The first three are lenient (each element is scored separately), the fourth is
strict. Publishing all four avoids picking the flattering one.

    python scripts/evaluate_understanding.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from hybridsql.pipeline.understand import understand  # noqa: E402

SETS = {
    "standard": Path("data/evaluation/questions_standard.yaml"),
    "hard": Path("data/evaluation/questions_hard.yaml"),
    # Questions that name a column and aggregate over it, which is what an analyst
    # actually types. They carry no value to mask, so the two older sets never
    # exercised the path where the index invents one.
    "analytics": Path("data/evaluation/questions_analytics.yaml"),
}
OUTPUT = Path("data/evaluation/understanding_results.json")


def normalize(t: str) -> str:
    return " ".join(str(t).lower().split())


def find_mention(expected: str, obtained: list[str]) -> str | None:
    """Match an expected mention to an extracted one.

    Matching is lenient: GLiNER2 returns "female patients" where the set annotates
    "female". Requiring strict equality would measure the model's span boundaries,
    not its ability to spot the value.
    """
    a = normalize(expected)
    for o in obtained:
        n = normalize(o)
        if a == n or a in n or n in a:
            return o
    return None


def evaluate(path: Path) -> dict:
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))

    total_values = found = well_resolved = 0
    total_concepts = concepts_ok = 0
    total_persons = persons_ok = 0
    complete = 0
    latencies: list[float] = []
    details: list[dict] = []

    for case in cases:
        question = case["question"]
        expected_values = case.get("values") or {}
        expected_concepts = case.get("concepts") or []

        start = time.perf_counter()
        u = understand(question)
        latencies.append((time.perf_counter() - start) * 1000)

        obtained = {normalize(r.mention): r for r in u.resolutions}
        names = list(obtained)

        perfect = True
        problems: list[str] = []
        # GLiNER2 often returns a wider span than the annotation: "female patients"
        # where the set notes "female". Lenient matching then also ties the concept
        # "patients" to that same mention, and we would wrongly count a
        # classification error. So we hold back mentions already claimed by a value
        # and do not re-evaluate them as concepts.
        claimed: set[str] = set()

        # --- 0: person names -------------------------------------------------
        # Top priority: eICU is de-identified, a name matches nothing and must
        # never reach the cloud. Two conditions: the name is spotted, and it is
        # classified as a person (so never looked up, so stopped by the gate for
        # want of being in the allowlist).
        for mention in case.get("persons") or []:
            total_persons += 1
            key = find_mention(mention, names)
            if key is None:
                perfect = False
                problems.append(f"NAME NOT SPOTTED: {mention!r} — it would leave in clear text")
                continue
            claimed.add(key)
            r = obtained[key]
            if r.kind == "person":
                persons_ok += 1
            else:
                perfect = False
                problems.append(
                    f"NAME MISCLASSIFIED: {mention!r} treated as '{r.kind}' -> {r.column}"
                )

        # --- 1 & 2: the values ------------------------------------------------
        for mention, ok_columns in expected_values.items():
            total_values += 1
            key = find_mention(mention, names)
            if key is not None:
                claimed.add(key)
            if key is None:
                perfect = False
                problems.append(f"not extracted: {mention!r}")
                continue
            found += 1
            r = obtained[key]
            acceptable = {c.strip() for c in str(ok_columns).split("|")}
            if r.kind != "value":
                perfect = False
                problems.append(f"{mention!r} classified '{r.kind}' instead of 'value'")
            elif r.column in acceptable:
                well_resolved += 1
            else:
                perfect = False
                problems.append(f"{mention!r} -> {r.column} (expected: {ok_columns})")

        # --- 2b: what must NOT be resolved confidently ------------------------
        # Some mentions have no matching value in the database ("hemoglobin" does
        # not exist in `lab.labname` on eICU-Demo). The right answer is then "I do
        # not know", not a forced resolution into an unrelated column.
        for mention in case.get("unresolvable") or []:
            key = find_mention(mention, names)
            if key is None:
                continue
            claimed.add(key)
            r = obtained[key]
            total_values += 1
            found += 1
            if r.confident:
                perfect = False
                problems.append(
                    f"{mention!r} confidently resolved to {r.column} ({r.score:.2f}) "
                    "although no value matches"
                )
            else:
                well_resolved += 1

        # --- 3: the concepts ---------------------------------------------------
        free = [n for n in names if n not in claimed]
        for mention in expected_concepts:
            key = find_mention(mention, free)
            if key is None:
                continue          # failing to extract a concept is not a fault
            total_concepts += 1
            if obtained[key].kind in ("concept", "quantity"):
                concepts_ok += 1
            else:
                perfect = False
                problems.append(
                    f"{mention!r} treated as a value -> {obtained[key].column}"
                )

        # --- 3b: where a concept must point -------------------------------------
        # Classifying "diagnosis names" as a concept is only half the job: it also
        # has to reach `diagnosis.diagnosisstring`, because that column is what
        # scopes the schema sent to the cloud. Checked only where the annotation
        # says so — many concepts have no single defensible column.
        for mention, ok_columns in (case.get("columns") or {}).items():
            key = find_mention(mention, names)
            if key is None:
                continue
            total_concepts += 1
            acceptable = {c.strip() for c in str(ok_columns).split("|")}
            if obtained[key].column in acceptable:
                concepts_ok += 1
            else:
                perfect = False
                problems.append(
                    f"{mention!r} -> {obtained[key].column} (expected column: {ok_columns})"
                )

        if perfect:
            complete += 1
        details.append(
            {
                "question": question,
                "correct": perfect,
                "ms": round(latencies[-1], 1),
                "problems": problems,
                "resolutions": [
                    {
                        "mention": r.mention, "kind": r.kind, "column": r.column,
                        "value": r.value, "score": r.score,
                    }
                    for r in u.resolutions
                ],
            }
        )

    def pct(n: int, d: int) -> float:
        return round(100 * n / d, 1) if d else 0.0

    latencies.sort()
    return {
        "questions": len(cases),
        "expected_values": total_values,
        "extraction_recall_pct": pct(found, total_values),
        "resolution_accuracy_pct": pct(well_resolved, found),
        "resolution_end_to_end_pct": pct(well_resolved, total_values),
        "concepts_evaluated": total_concepts,
        "kind_accuracy_pct": pct(concepts_ok, total_concepts),
        "expected_names": total_persons,
        "names_protected_pct": pct(persons_ok, total_persons),
        "full_understanding_pct": pct(complete, len(cases)),
        "correct_questions": complete,
        "median_ms": round(latencies[len(latencies) // 2], 1),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 1),
        "details": details,
    }


def show(name: str, r: dict) -> None:
    width = 44
    print("\n" + "=" * 70)
    print(f"SET '{name.upper()}'")
    print("=" * 70)
    print(f"{'Questions':<{width}} {r['questions']:>8}")
    print(f"{'Annotated business values':<{width}} {r['expected_values']:>8}")
    if r["expected_names"]:
        print(f"{'Annotated person names':<{width}} {r['expected_names']:>8}")
    print()
    print(f"{'Extraction recall':<{width}} {r['extraction_recall_pct']:>7} %")
    print(f"{'Resolution accuracy (if extracted)':<{width}} {r['resolution_accuracy_pct']:>7} %")
    print(f"{'Resolution end to end':<{width}} {r['resolution_end_to_end_pct']:>7} %")
    print(f"{'Kind-classification accuracy':<{width}} {r['kind_accuracy_pct']:>7} %")
    if r["expected_names"]:
        print(f"{'PERSON NAMES PROTECTED':<{width}} {r['names_protected_pct']:>7} %")
    print("-" * 70)
    print(
        f"{'FULL UNDERSTANDING':<{width}} {r['full_understanding_pct']:>7} %"
        f"  ({r['correct_questions']}/{r['questions']})"
    )
    print("-" * 70)
    print(f"{'Median / p95 latency':<{width}} {r['median_ms']:>7} / {r['p95_ms']} ms")

    failures = [d for d in r["details"] if not d["correct"]]
    if failures:
        print(f"\n{len(failures)} failing question(s):")
        for d in failures:
            print(f"\n  - {d['question']}")
            for p in d["problems"]:
                print(f"      * {p}")


def main() -> None:
    everything = {}
    for name, path in SETS.items():
        if not path.exists():
            continue
        everything[name] = evaluate(path)
        show(name, everything[name])

    if len(everything) > 1:
        q = sum(r["questions"] for r in everything.values())
        ok = sum(r["correct_questions"] for r in everything.values())
        print("\n" + "=" * 70)
        print(f"{'ALL SETS COMBINED':<44} {round(100 * ok / q, 1):>7} %  ({ok}/{q})")
        print("=" * 70)
        everything["overall"] = {
            "questions": q,
            "correct_questions": ok,
            "full_understanding_pct": round(100 * ok / q, 1),
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(everything, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {OUTPUT}")


if __name__ == "__main__":
    main()
