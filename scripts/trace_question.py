"""Show every stage of the pipeline for one question.

Why this exists
---------------
The measurement scripts give rates; they do not show the machine working. This one
prints what each stage produced, in order, so the transformation can be read
rather than trusted:

    the question as typed
    -> entities spotted, and how each was classified
    -> the real values resolved, and in which column
    -> THE MASKED SENTENCE — what actually leaves the building
    -> what the egress gate checked, segment by segment
    -> the SQL the cloud model wrote
    -> the rows it returned
    -> the written answer

The two lines that matter for the report sit next to each other: the question with
its values, and the same question with `:v1`. Everything to the right of the gate
only ever saw the second.

    python scripts/trace_question.py "how many patients received aspirin?"
    python scripts/trace_question.py --demo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybridsql.db.connection import execute  # noqa: E402
from hybridsql.pipeline import generate as gen  # noqa: E402
from hybridsql.pipeline.anonymize import UnmaskableQuestion, anonymize  # noqa: E402
from hybridsql.pipeline.understand import understand  # noqa: E402
from hybridsql.security import egress_gate as gate  # noqa: E402
from hybridsql.security import sql_validator  # noqa: E402

DEMO = [
    "How many patients received aspirin for sepsis?",
    "What is the mortality rate for patients treated with vancomycin?",
    "How many female patients were discharged to a Skilled Nursing Facility?",
    "Did Mr. Bensalah receive his insulin?",
]

W = 78


def rule(title: str, char: str = "=") -> None:
    print(f"\n{char * W}")
    print(f" {title}")
    print(char * W)


def field(label: str, value: str) -> None:
    print(f"  {label:<22} {value}")


def trace(question: str, write: bool = False) -> None:
    print("\n")
    print("#" * W)
    print(f"#  {question}")
    print("#" * W)

    # ---- STAGE 1 --------------------------------------------------------------
    t = time.perf_counter()
    u = understand(question)
    ms1 = (time.perf_counter() - t) * 1000

    rule("STAGE 1 — UNDERSTAND (local, no network)")
    field("extractor", f"{u.active_extractor}  ({ms1:.0f} ms)")
    field("glossary tables", ", ".join(sorted(u.tables)) or "—")
    print()
    if not u.resolutions:
        print("  no entity spotted")
    for r in u.resolutions:
        mark = {"value": "VALUE", "concept": "concept", "quantity": "quantity",
                "person": "PERSON"}[r.kind]
        print(f"  [{mark:<8}] {r.mention!r}")
        if r.kind == "value" and r.resolved:
            print(f"      resolved -> {r.column} = {r.value!r}")
            print(f"      score {r.score:.2f} (tier {r.tier})"
                  f"{'  OUT OF SCOPE' if r.out_of_scope else ''}"
                  f"{'  -> will be masked' if r.to_mask else '  -> too uncertain, ask analyst'}")
        elif r.kind == "value":
            print("      NOT FOUND in the database — the question may be out of scope")
        elif r.kind == "concept":
            print(f"      names a column -> {r.column}   (not masked: it is structure)")
        elif r.kind == "quantity":
            print("      a number from the analyst, not from the database")
        elif r.kind == "person":
            print("      PERSON NAME — never looked up, and the request will stop here")

    for n in u.notes:
        print(f"\n  note for the model: {n}")

    # ---- STAGE 2a -------------------------------------------------------------
    rule("STAGE 2a — ANONYMIZE (local)")
    try:
        a = anonymize(u)
    except UnmaskableQuestion as e:
        print(f"\n  REQUEST STOPPED\n  {e}")
        print("\n  Nothing was sent. The database is de-identified, so the question")
        print("  cannot be answered — and a name has no business reaching a provider.")
        return

    print("\n  BEFORE  (stays inside)")
    print(f"      {question}")
    print("\n  AFTER   (this is what leaves)")
    print(f"      {a.masked_question}")

    if a.mapping:
        print("\n  mapping kept locally — NEVER transmitted:")
        for symbol, value in a.mapping.items():
            print(f"      {symbol} = {value!r}   [{a.columns.get(symbol, '?')}]")
    else:
        print("\n  no value to mask in this question")

    if a.unresolved:
        print(f"\n  unresolved mentions: {a.unresolved}")

    # ---- THE GATE -------------------------------------------------------------
    rule("EGRESS GATE — what is checked, and how")
    segments = gen.build_segments(u, a)
    blocked = False
    for s in segments:
        v = gate.check_segment(s, "trace")
        status = "PASS" if v.allowed else "BLOCK"
        preview = " ".join(s.text.split())[:44]
        print(f"  [{status}] {s.origin:<9} via {v.verified_by:<24} {preview}…")
        if not v.allowed:
            blocked = True
            print(f"           refused: {list(v.refused_tokens)[:6]}")

    if blocked:
        print("\n  REQUEST STOPPED — a value would have crossed unmasked.")
        print("  That is the gate working: stage 1 missed it, the gate caught it.")
        return

    # ---- STAGE 2b -------------------------------------------------------------
    rule("STAGE 2b — GENERATE (cloud, sees only the above)")
    g = gen.generate(u, a)
    field("provider", g.target or "—")
    field("calls / repairs", f"{g.calls} / {g.repairs}")
    field("tokens", str(g.tokens))
    field("latency", f"{g.ms:.0f} ms")

    for i, sql in enumerate(g.history, start=1):
        label = "SQL returned" if len(g.history) == 1 else f"attempt {i}"
        print(f"\n  {label}:")
        for line in sql.splitlines():
            print(f"      {line}")

    verdict = sql_validator.validate(g.sql, set(a.mapping))
    print(f"\n  validator: {'accepted' if verdict.valid else 'REJECTED — ' + verdict.reason}")
    print(f"  parameters used by the model: {list(verdict.parameters_used)}")

    swept = gate.sweep_response(g.sql)
    print(f"  output sweep: {'clean' if not swept else 'MODEL WROTE A LITERAL: ' + str(swept)}")

    if not g.valid:
        return

    # ---- STAGE 3 --------------------------------------------------------------
    rule("STAGE 3 — EXECUTE AND ANSWER (local)")
    print(f"\n  bound parameters: {a.parameters()}")
    try:
        columns, rows = execute(g.sql, a.parameters())
    except Exception as e:  # noqa: BLE001
        print(f"\n  execution failed: {type(e).__name__}: {e}")
        return

    print(f"\n  {len(rows)} row(s), columns {columns}")
    for row in rows[:8]:
        print(f"      {row}")
    if len(rows) > 8:
        print(f"      … {len(rows) - 8} more")

    if write:
        from hybridsql.providers import local_model

        t = time.perf_counter()
        written = local_model.write_answer(question, columns, rows)
        print(f"\n  ANSWER  ({written.model}, {time.perf_counter() - t:.1f} s)")
        print(f"      {written.text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="the question to trace")
    parser.add_argument("--demo", action="store_true", help="run four illustrative questions")
    parser.add_argument("--write", action="store_true", help="also run the local answer writer")
    args = parser.parse_args()

    questions = DEMO if args.demo else [" ".join(args.question)]
    if not questions or not questions[0]:
        parser.error("give a question, or use --demo")

    for q in questions:
        trace(q, write=args.write)


if __name__ == "__main__":
    main()
