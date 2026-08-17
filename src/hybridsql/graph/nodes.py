"""The graph's nodes.

Each node is a thin wrapper over a module that already existed and was already
measured. That is the whole design intent of putting LangGraph on top: the
orchestrator is allowed to decide *what runs next*, and nothing else. It does not
build prompts, it does not call providers, it does not touch the database. If a
node were ever to grow logic of its own, the earlier evaluation results would
stop describing the code that runs.

Every node declares its trust zone to the tracer. `cloud` means the node's
payload has crossed — or is about to cross — the boundary; `local` means it has
not. That declaration is what lets a hosted tracing backend be used at all
without contradicting the architecture (see `obs/tracing.py`).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hybridsql.db import schema as sch
from hybridsql.db.connection import QueryTimeout, execute
from hybridsql.graph.state import PipelineState
from hybridsql.obs import tracing
from hybridsql.pipeline import generate as gen
from hybridsql.pipeline.anonymize import UnmaskableQuestion, UnresolvableValue, anonymize
from hybridsql.pipeline.understand import understand
from hybridsql.providers import cloud, local_model
from hybridsql.security import egress_gate, sql_validator
from hybridsql.security.egress_gate import LeakBlocked

_log = logging.getLogger(__name__)


def _timed(state: PipelineState, key: str, ms: float) -> dict[str, float]:
    """Accumulate a stage duration without losing the ones already recorded.

    LangGraph replaces a key rather than merging it, so the dictionary is rebuilt
    on every write. A reducer would be tidier and would also hide the fact that
    two nodes writing the same key is a bug — this stays explicit.
    """
    out = dict(state.get("ms", {}))
    out[key] = round(ms, 1)
    return out


def _fail(state: PipelineState, stage: str, reason: str) -> dict[str, Any]:
    return {"success": False, "failed_stage": stage, "failure_reason": reason}


# ---------------------------------------------------------------------------------
#  Stage 1 — understand (local, no network, all arms)
# ---------------------------------------------------------------------------------
@tracing.node("understand", zone="local")
def understand_node(state: PipelineState) -> dict[str, Any]:
    """Entities, their kind, and the exact stored values behind them.

    Runs in every arm, including Full Cloud, and that is a choice worth defending.
    Full Cloud does not *need* it to answer — but it does need the same table
    selection, otherwise the arms would differ in prompt size as well as in
    privacy, and the benchmark would be measuring two variables at once.
    """
    start = time.perf_counter()
    try:
        u = understand(state["question"])
    except Exception as e:  # noqa: BLE001
        return {
            **_fail(state, "understand", f"{type(e).__name__}: {e}"),
            "ms": _timed(state, "understand", (time.perf_counter() - start) * 1000),
        }

    return {
        "understanding": u,
        "tables": sorted(u.tables),
        "notes": list(u.notes),
        "extractor": u.active_extractor,
        "ms": _timed(state, "understand", (time.perf_counter() - start) * 1000),
    }


# ---------------------------------------------------------------------------------
#  Stage 2a — anonymize (local, hybrid only)
# ---------------------------------------------------------------------------------
@tracing.node("anonymize", zone="local")
def anonymize_node(state: PipelineState) -> dict[str, Any]:
    """Real values become `:v1, :v2`. The mapping stays in this process.

    Only the Hybrid arm runs this. Full Cloud skips it by definition — that is
    what makes it the baseline — and Full Local has nothing to protect against,
    since nothing leaves.
    """
    start = time.perf_counter()
    try:
        anon = anonymize(state["understanding"])
    except UnmaskableQuestion as e:
        # A refusal, not a crash: the database is de-identified, so a question
        # naming a person has no answer, and there is no reason to send anything.
        return {
            **_fail(state, "anonymize", str(e)),
            "refusal": True,
            "ms": _timed(state, "anonymize", (time.perf_counter() - start) * 1000),
        }
    except UnresolvableValue as e:
        # Also a refusal. Carrying the suggestions out with it is what turns a
        # dead end into a next step the analyst can take in one click.
        return {
            **_fail(state, "anonymize", str(e)),
            "refusal": True,
            "suggestions": e.suggestions,
            "ms": _timed(state, "anonymize", (time.perf_counter() - start) * 1000),
        }

    return {
        "anonymization": anon,
        "masked_question": anon.masked_question,
        "symbol_count": anon.symbol_count,
        "ms": _timed(state, "anonymize", (time.perf_counter() - start) * 1000),
    }


# ---------------------------------------------------------------------------------
#  Stage 2b — generate, one node per arm
# ---------------------------------------------------------------------------------
@tracing.node("generate_hybrid", zone="cloud")
def generate_hybrid_node(state: PipelineState) -> dict[str, Any]:
    """Schema + masked question -> SQL. The only outbound call in this arm.

    The egress gate runs inside `gen.generate`, before the socket is opened. A
    `LeakBlocked` here is the system working: stage 1 missed a value and the gate
    caught it.
    """
    start = time.perf_counter()
    u, anon = state["understanding"], state["anonymization"]

    # The gate's verdicts, read before the call for the sake of the interface.
    # This is a second pass over segments that `gen.generate` will check again on
    # its own — the authoritative check stays inside `cloud.call`, where it cannot
    # be skipped. This copy exists only so the demo can *show* the decision, and
    # so a blocked request still has something to display. At 0.033 ms per token
    # the duplication is not worth avoiding.
    verdicts = [
        {
            "origin": s.origin,
            "allowed": v.allowed,
            "verified_by": v.verified_by,
            "tokens": v.token_count,
            "refused": list(v.refused_tokens),
            "preview": " ".join(s.text.split())[:80],
        }
        for s in gen.build_segments(u, anon)
        for v in [egress_gate.check_segment(s, "hybrid")]
    ]

    try:
        g = gen.generate(u, anon)
    except LeakBlocked as e:
        return {
            **_fail(state, "egress_gate", str(e)),
            "refusal": True,
            "egress_segments": verdicts,
            "ms": _timed(state, "generate", (time.perf_counter() - start) * 1000),
        }
    except Exception as e:  # noqa: BLE001
        return {
            **_fail(state, "cloud", f"{type(e).__name__}: {e}"),
            "ms": _timed(state, "generate", (time.perf_counter() - start) * 1000),
        }

    sent = sum(len(m["content"]) for m in gen.build_messages(u, anon))
    out: dict[str, Any] = {
        "sql": g.sql,
        "sql_history": list(g.history),
        "cloud_target": g.target,
        "cloud_tokens": g.tokens,
        "cloud_calls": g.calls,
        "repairs": g.repairs,
        "sql_author": g.target or "cloud",
        "egress_segments": verdicts,
        "egress_chars": sent,
        # Zero by construction, and the claim is not taken on trust: the gate
        # verified every segment, and `tests/test_canary.py` tries to defeat it.
        "egress_values": 0,
        "ms": _timed(state, "generate", (time.perf_counter() - start) * 1000),
    }
    if not g.valid:
        out.update(_fail(state, "sql_validation", g.failure_reason))
    return out


@tracing.node("generate_opaque", zone="cloud")
def generate_opaque_node(state: PipelineState) -> dict[str, Any]:
    """Pseudonymised schema + pseudonymised question -> SQL, translated back here.

    The provider is handed labels and foreign keys. Everything it returns is in
    its own vocabulary, so the restored SQL is produced locally, from a dictionary
    drawn for this request alone.
    """
    start = time.perf_counter()
    u, anon = state["understanding"], state["anonymization"]

    # Built here, not inside `generate_opaque`, and handed down. The pseudonym
    # dictionary is drawn afresh per request, so building a second one to display
    # would show labels that were never sent.
    prompt = gen.build_opaque(u, anon)
    view = prompt.view()
    verdicts = [
        {
            "origin": s.origin,
            "allowed": v.allowed,
            "verified_by": v.verified_by,
            "tokens": v.token_count,
            "refused": list(v.refused_tokens),
            "preview": " ".join(s.text.split())[:80],
        }
        for s in prompt.segments
        for v in [egress_gate.check_segment(s, "opaque")]
    ]

    try:
        g = gen.generate_opaque(u, anon, prompt=prompt)
    except LeakBlocked as e:
        return {
            **_fail(state, "egress_gate", str(e)),
            "refusal": True,
            "egress_segments": verdicts,
            "opaque": view,
            "ms": _timed(state, "generate", (time.perf_counter() - start) * 1000),
        }
    except Exception as e:  # noqa: BLE001
        return {
            **_fail(state, "cloud", f"{type(e).__name__}: {e}"),
            "opaque": view,
            "ms": _timed(state, "generate", (time.perf_counter() - start) * 1000),
        }

    out: dict[str, Any] = {
        "sql": g.sql,
        "sql_history": list(g.history),
        "cloud_target": g.target,
        "cloud_tokens": g.tokens,
        "cloud_calls": g.calls,
        "repairs": g.repairs,
        "sql_author": g.target or "cloud",
        "egress_segments": verdicts,
        "egress_chars": sum(len(m["content"]) for m in prompt.messages),
        "egress_values": 0,
        # What the provider was actually shown: labels, and what each stood for.
        "opaque": g.opaque or view,
        "ms": _timed(state, "generate", (time.perf_counter() - start) * 1000),
    }
    if not g.valid:
        out.update(_fail(state, "sql_validation", g.failure_reason))
    return out


@tracing.node("generate_full_cloud", zone="cloud")
def generate_full_cloud_node(state: PipelineState) -> dict[str, Any]:
    """The unprotected baseline: the question leaves exactly as it was typed.

    This is the arm the whole project argues against, and it has to run for the
    argument to have a number attached. The gate is bypassed explicitly and the
    bypass is journalled — see `providers/cloud.call`.
    """
    start = time.perf_counter()
    u = state["understanding"]
    ddl = sch.ddl(gen.relevant_tables(u), with_row_counts=True)

    notes = ""
    if u.notes:
        notes = "\n\nDomain notes:\n" + "\n".join(f"- {n}" for n in u.notes)
    user = f"Schema:\n{ddl}{notes}\n\nQuestion: {state['question']}\n\nSQL:"
    messages = [
        {"role": "system", "content": gen.INSTRUCTIONS_LITERAL},
        {"role": "user", "content": user},
    ]

    try:
        response = cloud.call(
            messages, context="full-cloud-sql", baseline_unprotected=True
        )
    except Exception as e:  # noqa: BLE001
        return {
            **_fail(state, "cloud", f"{type(e).__name__}: {e}"),
            "ms": _timed(state, "generate", (time.perf_counter() - start) * 1000),
        }

    sql = cloud.extract_sql(response.text)
    # An *empty* expected set, not `None`. This arm's SQL must carry literals —
    # that is the leak being measured — so a `:v1` in it is a defect, and one that
    # would otherwise surface as an unbound-parameter crash at execution instead
    # of a readable rejection here.
    verdict = sql_validator.validate(sql, set())

    out: dict[str, Any] = {
        "sql": sql,
        "sql_history": [sql],
        "cloud_target": response.target,
        "cloud_tokens": response.tokens,
        "cloud_calls": 1,
        "repairs": 0,
        "sql_author": response.target,
        "egress_chars": sum(len(m["content"]) for m in messages),
        # Every value the analyst named went out in clear text.
        "egress_values": len(u.values),
        "ms": _timed(state, "generate", (time.perf_counter() - start) * 1000),
    }
    if not verdict.valid:
        out.update(_fail(state, "sql_validation", verdict.reason))
    return out


@tracing.node("generate_full_local", zone="local")
def generate_full_local_node(state: PipelineState) -> dict[str, Any]:
    """The 1.7B model writes the SQL itself. Nothing leaves, at all.

    Expect this arm to be the weakest. That is the finding, not a failure of the
    experiment: it is the gap this one has to close that justifies renting a
    large model for the SQL step alone.
    """
    start = time.perf_counter()
    u = state["understanding"]
    ddl = sch.ddl(gen.relevant_tables(u), with_row_counts=True)

    try:
        written = local_model.generate_sql(state["question"], ddl, u.notes)
    except Exception as e:  # noqa: BLE001
        return {
            **_fail(state, "local_model", f"{type(e).__name__}: {e}"),
            "ms": _timed(state, "generate", (time.perf_counter() - start) * 1000),
        }

    verdict = sql_validator.validate(written.text, set())
    out: dict[str, Any] = {
        "sql": written.text,
        "sql_history": [written.text],
        "sql_author": f"local:{written.model}",
        "cloud_target": "",
        "cloud_tokens": 0,
        "cloud_calls": 0,
        "egress_chars": 0,
        "egress_values": 0,
        "ms": _timed(state, "generate", (time.perf_counter() - start) * 1000),
    }
    if not verdict.valid:
        out.update(_fail(state, "sql_validation", verdict.reason))
    return out


# ---------------------------------------------------------------------------------
#  Stage 3a — execute (local, read-only, every arm)
# ---------------------------------------------------------------------------------
@tracing.node("execute", zone="local")
def execute_node(state: PipelineState) -> dict[str, Any]:
    """Run the SELECT against read-only SQLite.

    The Hybrid arm binds `:v1` to the real value here, through SQLite's parameter
    API. The query text and the value never meet in one string — not on the way
    out, where the cloud never had it, nor on the way back. Injection is
    impossible by construction rather than by filtering.
    """
    start = time.perf_counter()
    anon = state.get("anonymization")
    params = anon.parameters() if anon is not None else None

    try:
        columns, rows = execute(state["sql"], params)
    except QueryTimeout as e:
        return {
            **_fail(state, "execution", str(e)),
            "ms": _timed(state, "execute", (time.perf_counter() - start) * 1000),
        }
    except Exception as e:  # noqa: BLE001
        return {
            **_fail(state, "execution", f"{type(e).__name__}: {e}"),
            "ms": _timed(state, "execute", (time.perf_counter() - start) * 1000),
        }

    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "ms": _timed(state, "execute", (time.perf_counter() - start) * 1000),
    }


# ---------------------------------------------------------------------------------
#  Stage 3b — write the answer
# ---------------------------------------------------------------------------------
@tracing.node("write_local", zone="local")
def write_local_node(state: PipelineState) -> dict[str, Any]:
    """The local model turns rows into a sentence. Hybrid and Full Local.

    This node is the reason the whole architecture holds: it is the only place
    the real rows meet a language model, and that model is in this process. A
    writer failure degrades to a plain table rather than failing the run — the
    data is already correct by this point, only its phrasing is missing.
    """
    start = time.perf_counter()
    if not state.get("write", True):
        return {
            "answer": _plain(state["columns"], state["rows"]),
            "answer_author": "none (plain rendering)",
            "success": True,
            "ms": _timed(state, "write", 0.0),
        }

    try:
        written = local_model.write_answer(
            state["question"], state["columns"], state["rows"]
        )
        text, author = written.text, f"local:{written.model}"
    except Exception as e:  # noqa: BLE001
        _log.warning("answer writing unavailable, falling back to a table: %s", e)
        text, author = _plain(state["columns"], state["rows"]), "fallback (plain rendering)"

    return {
        "answer": text,
        "answer_author": author,
        "success": True,
        "ms": _timed(state, "write", (time.perf_counter() - start) * 1000),
    }


@tracing.node("write_cloud", zone="cloud")
def write_cloud_node(state: PipelineState) -> dict[str, Any]:
    """Full Cloud only: the provider writes the answer, so it sees the rows.

    This is where the baseline's real cost shows up. The SQL step leaks the
    question; this step leaks the *result* — the actual records. Counting those
    cells is what turns "the cloud sees your data" from a slogan into a column in
    a table.
    """
    start = time.perf_counter()
    columns, rows = state["columns"], state["rows"]
    messages = local_model.build_messages(state["question"], columns, rows)
    cells = sum(1 for row in rows for value in row if value is not None)

    try:
        response = cloud.call(
            messages,
            max_tokens=220,
            context="full-cloud-answer",
            baseline_unprotected=True,
        )
        text, author = response.text.strip(), response.target
    except Exception as e:  # noqa: BLE001
        _log.warning("cloud answer writing failed, falling back to a table: %s", e)
        return {
            "answer": _plain(columns, rows),
            "answer_author": "fallback (plain rendering)",
            "success": True,
            "ms": _timed(state, "write", (time.perf_counter() - start) * 1000),
        }

    return {
        "answer": text,
        "answer_author": author,
        "success": True,
        "egress_chars": state.get("egress_chars", 0)
        + sum(len(m["content"]) for m in messages),
        "egress_values": state.get("egress_values", 0) + cells,
        "ms": _timed(state, "write", (time.perf_counter() - start) * 1000),
    }


def _plain(columns: list[str], rows: list[tuple], max_rows: int = 5) -> str:
    """Render without a model — enough to check a run without loading 1 GB."""
    if not rows:
        return "No matching record."
    if len(rows) == 1 and len(rows[0]) == 1:
        return f"{columns[0]} = {rows[0][0]}"
    header = " | ".join(columns)
    body = "\n".join(
        " | ".join("" if v is None else str(v) for v in row) for row in rows[:max_rows]
    )
    more = f"\n… {len(rows)} rows total" if len(rows) > max_rows else ""
    return f"{header}\n{body}{more}"
