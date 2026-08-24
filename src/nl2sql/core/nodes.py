"""The graph's nodes — thin wrappers over modules that already exist.

That is the whole point of putting LangGraph on top: the orchestrator decides
what runs next and nothing else. No node builds a prompt, calls a provider or
touches the database itself. If one ever grew logic of its own, the measurements
would stop describing the code that runs.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from nl2sql.core import prompt as pr
from nl2sql.core import trace
from nl2sql.core.state import State
from nl2sql.core.steps import track
from nl2sql.db.sqlite import QueryTimeout, execute
from nl2sql.llm import cloud, local
from nl2sql.nlp.understand import understand
from nl2sql.privacy import gate, sqlcheck
from nl2sql.privacy.gate import LeakBlocked
from nl2sql.privacy.mask import UnmaskableQuestion, UnresolvableValue, mask

_log = logging.getLogger(__name__)


def _ms(state: State, key: str, started: float) -> dict[str, float]:
    """Accumulate a stage duration. LangGraph replaces a key rather than merging it."""
    out = dict(state.get("ms", {}))
    out[key] = round((time.perf_counter() - started) * 1000, 1)
    return out


def _fail(stage: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"success": False, "failed_stage": stage, "failure_reason": reason, **extra}


def _steps() -> dict[str, Any]:
    """Hand the interface everything traced so far, after every node."""
    return {"trace": trace.steps_so_far()}


# ---------------------------------------------------------------------------------
#  Stage 1 — understand (local, every arm)
# ---------------------------------------------------------------------------------
@trace.node("understand", zone="local")
def understand_node(state: State) -> dict[str, Any]:
    """Entities, their kind, and the exact stored values behind them.

    Runs in every arm, Full Cloud included: that arm does not need it to answer,
    but it does need the same table selection, or the arms would differ in prompt
    size as well as in privacy and the benchmark would move two variables at once.
    """
    started = time.perf_counter()
    try:
        u = understand(state["question"])
    except Exception as e:  # noqa: BLE001
        return {**_fail("understand", f"{type(e).__name__}: {e}"), "ms": _ms(state, "understand", started), **_steps()}

    return {
        "understanding": u,
        "tables": sorted(u.tables),
        "notes": list(u.notes),
        "extractor": u.extractor,
        "ms": _ms(state, "understand", started),
        **_steps(),
    }


# ---------------------------------------------------------------------------------
#  Stage 2a — mask (local, hybrid arms only)
# ---------------------------------------------------------------------------------
@trace.node("mask", zone="local")
def mask_node(state: State) -> dict[str, Any]:
    """Real values become `:v1`. Full Cloud skips this — its absence *is* that arm."""
    started = time.perf_counter()
    try:
        masked = mask(state["understanding"])
    except UnmaskableQuestion as e:
        return {**_fail("mask", str(e), refusal=True), "ms": _ms(state, "mask", started), **_steps()}
    except UnresolvableValue as e:
        # A refusal too. The suggestions turn a dead end into a next step.
        return {
            **_fail("mask", str(e), refusal=True, suggestions=e.suggestions),
            "ms": _ms(state, "mask", started),
            **_steps(),
        }

    return {
        "masked": masked,
        "masked_question": masked.question,
        "symbol_count": masked.symbol_count,
        "ms": _ms(state, "mask", started),
        **_steps(),
    }


# ---------------------------------------------------------------------------------
#  Stage 2b — generate, one node per arm
# ---------------------------------------------------------------------------------
def _run(state: State, built: pr.Prompt, opaque_arm: bool = False) -> dict[str, Any]:
    """Delegate to the variant's strategy. Imported here to keep the cycle broken."""
    from nl2sql.optimize import variants

    return variants.run(state, built, opaque_arm=opaque_arm)


@trace.node("generate_hybrid", zone="cloud")
def generate_hybrid_node(state: State) -> dict[str, Any]:
    """Schema plus masked question to the provider. The only outbound call here."""
    from nl2sql.optimize.variants import VARIANTS

    started = time.perf_counter()
    variant = VARIANTS[state.get("variant", "baseline")]
    u, masked = state["understanding"], state["masked"]

    with track("prompt", variant=variant.name) as step:
        built = variant.build_prompt(u, masked)
        step.say(
            f"{built.characters} characters going out: {len(built.segments)} parts, "
            f"{len(built.view.get('tables', []))} table(s) described",
            parts=[s.origin for s in built.segments],
            **built.view,
        )

    try:
        out = _run(state, built)
    except LeakBlocked as e:
        return {
            **_fail("gate", str(e), refusal=True),
            "egress_segments": gate.verdicts(built.segments),
            "ms": _ms(state, "generate", started),
            **_steps(),
        }
    except Exception as e:  # noqa: BLE001
        return {**_fail("cloud", f"{type(e).__name__}: {e}"), "ms": _ms(state, "generate", started), **_steps()}

    out["ms"] = _ms(state, "generate", started)
    out.update(_steps())
    return out


@trace.node("generate_opaque", zone="cloud")
def generate_opaque_node(state: State) -> dict[str, Any]:
    """Pseudonymised schema and question; the SQL is translated back here."""
    started = time.perf_counter()
    u, masked = state["understanding"], state["masked"]

    with track("prompt", variant="opaque") as step:
        built = pr.opaque(u, masked)
        step.say(
            f"{built.characters} characters going out, all of it labels: "
            f"{built.view['tables']} table(s) renamed t1..t{built.view['tables']}",
            **{k: v for k, v in built.view.items() if k != "ddl"},
        )

    try:
        out = _run(state, built, opaque_arm=True)
    except LeakBlocked as e:
        return {
            **_fail("gate", str(e), refusal=True),
            "egress_segments": gate.verdicts(built.segments),
            "opaque": built.view,
            "ms": _ms(state, "generate", started),
            **_steps(),
        }
    except Exception as e:  # noqa: BLE001
        return {
            **_fail("cloud", f"{type(e).__name__}: {e}"),
            "opaque": built.view,
            "ms": _ms(state, "generate", started),
            **_steps(),
        }

    out["ms"] = _ms(state, "generate", started)
    out.update(_steps())
    return out


@trace.node("generate_full_cloud", zone="cloud")
def generate_full_cloud_node(state: State) -> dict[str, Any]:
    """The unprotected baseline: the question leaves exactly as it was typed.

    This is the arm the project argues against, and it has to run for the argument
    to have a number attached. The bypass is explicit and journalled.
    """
    started = time.perf_counter()
    u = state["understanding"]

    with track("prompt", variant="full_cloud") as step:
        built = pr.clear(u, state["question"])
        step.say(f"{built.characters} characters going out, values included", **built.view)

    try:
        response = cloud.call(built.messages, context="full-cloud-sql", unprotected=True)
    except Exception as e:  # noqa: BLE001
        return {**_fail("cloud", f"{type(e).__name__}: {e}"), "ms": _ms(state, "generate", started), **_steps()}

    sql = cloud.extract_sql(response.text)
    with track("validate", sql=sql) as step:
        # An *empty* expected set, not None: this arm's SQL must carry literals —
        # that is the leak being measured — so a `:v1` here is a defect.
        verdict = sqlcheck.validate(sql, set())
        step.say("the query is safe to run" if verdict.valid else f"rejected — {verdict.reason}")

    out: dict[str, Any] = {
        "sql": sql,
        "sql_history": [sql],
        "sql_author": response.target,
        "cloud_target": response.target,
        "cloud_tokens": response.tokens,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "cloud_calls": 1,
        "repairs": 0,
        "perplexity": response.perplexity,
        "egress_chars": built.characters,
        "egress_values": len(u.values),   # every value the analyst named went out in clear
        "ms": _ms(state, "generate", started),
        **_steps(),
    }
    if not verdict.valid:
        out.update(_fail("sql_validation", verdict.reason))
    return out


@trace.node("generate_full_local", zone="local")
def generate_full_local_node(state: State) -> dict[str, Any]:
    """The 1.7B model writes the SQL itself. Nothing leaves, at all."""
    started = time.perf_counter()
    u = state["understanding"]
    from nl2sql.db import schema as sch

    with track("prompt", variant="full_local") as step:
        ddl = sch.ddl(pr.relevant_tables(u))
        step.say(f"{len(ddl)} characters of schema, staying on this machine")

    try:
        written = local.generate_sql(state["question"], ddl, u.notes)
    except Exception as e:  # noqa: BLE001
        return {**_fail("local_model", f"{type(e).__name__}: {e}"), "ms": _ms(state, "generate", started), **_steps()}

    with track("validate", sql=written.text) as step:
        verdict = sqlcheck.validate(written.text, set())
        step.say("the query is safe to run" if verdict.valid else f"rejected — {verdict.reason}")

    out: dict[str, Any] = {
        "sql": written.text,
        "sql_history": [written.text],
        "sql_author": f"local:{written.model}",
        "cloud_target": "",
        "cloud_tokens": 0,
        "cloud_calls": 0,
        "completion_tokens": written.tokens,
        "egress_chars": 0,
        "egress_values": 0,
        "ms": _ms(state, "generate", started),
        **_steps(),
    }
    if not verdict.valid:
        out.update(_fail("sql_validation", verdict.reason))
    return out


# ---------------------------------------------------------------------------------
#  Stage 3a — execute (local, read-only, every arm)
# ---------------------------------------------------------------------------------
@trace.node("execute", zone="local")
def execute_node(state: State) -> dict[str, Any]:
    """Run the SELECT against read-only SQLite.

    The hybrid arms bind `:v1` to the real value here, through SQLite's parameter
    API: the query text and the value never meet in one string, on the way out or
    on the way back.
    """
    started = time.perf_counter()
    masked = state.get("masked")
    params = masked.parameters() if masked is not None else None

    with track("execute", sql=state["sql"]) as step:
        try:
            columns, rows = execute(state["sql"], params)
        except QueryTimeout as e:
            step.say(f"the query took too long and was stopped: {e}")
            return {**_fail("execution", str(e)), "ms": _ms(state, "execute", started), **_steps()}
        except Exception as e:  # noqa: BLE001
            step.say(f"the database refused the query: {e}")
            return {
                **_fail("execution", f"{type(e).__name__}: {e}"),
                "ms": _ms(state, "execute", started),
                **_steps(),
            }
        step.say(
            f"{len(rows)} row(s) returned" + (f", bound {len(params)} hidden value(s)" if params else ""),
            columns=columns,
            row_count=len(rows),
        )

    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "ms": _ms(state, "execute", started),
        **_steps(),
    }


# ---------------------------------------------------------------------------------
#  Stage 3b — write the answer
# ---------------------------------------------------------------------------------
def _plain(columns: list[str], rows: list[tuple], max_rows: int = 5) -> str:
    """Render without a model — enough to check a run without loading 1 GB."""
    if not rows:
        return "No matching record."
    if len(rows) == 1 and len(rows[0]) == 1:
        return f"{columns[0]} = {rows[0][0]}"
    header = " | ".join(columns)
    body = "\n".join(" | ".join("" if v is None else str(v) for v in r) for r in rows[:max_rows])
    more = f"\n… {len(rows)} rows total" if len(rows) > max_rows else ""
    return f"{header}\n{body}{more}"


@trace.node("write_local", zone="local")
def write_local_node(state: State) -> dict[str, Any]:
    """The local model turns rows into a sentence.

    This node is why the architecture holds: it is the only place the real rows
    meet a language model, and that model is in this process. A writer failure
    degrades to a plain table rather than failing the run — the data is already
    correct by now, only its phrasing is missing.
    """
    started = time.perf_counter()
    with track("write", zone="local", rows=state.get("row_count", 0)) as step:
        if not state.get("write", True):
            step.say("answer writing was switched off; showing the table as it is")
            return {
                "answer": _plain(state["columns"], state["rows"]),
                "answer_author": "none (plain table)",
                "success": True,
                "ms": _ms(state, "write", started),
                **_steps(),
            }
        try:
            written = local.write_answer(state["question"], state["columns"], state["rows"])
            text, author = written.text, f"local:{written.model}"
            step.say("the answer was written on this machine, from the rows above")
        except Exception as e:  # noqa: BLE001
            _log.warning("answer writing unavailable: %s", e)
            text, author = _plain(state["columns"], state["rows"]), "fallback (plain table)"
            step.say(f"no local model available, showing the table instead ({type(e).__name__})")

    return {
        "answer": text,
        "answer_author": author,
        "success": True,
        "ms": _ms(state, "write", started),
        **_steps(),
    }


@trace.node("write_cloud", zone="cloud")
def write_cloud_node(state: State) -> dict[str, Any]:
    """Full Cloud only: the provider writes the answer, so it sees the rows.

    This is where the baseline's real cost shows up. The SQL step leaks the
    question; this one leaks the result. Counting those cells is what turns "the
    cloud sees your data" into a column in a table.
    """
    started = time.perf_counter()
    columns, rows = state["columns"], state["rows"]
    cells = sum(1 for row in rows for value in row if value is not None)

    with track("write", zone="cloud", rows=len(rows)) as step:
        messages = local.build_messages(state["question"], columns, rows)
        try:
            response = cloud.call(
                messages, max_tokens=220, context="full-cloud-answer", unprotected=True
            )
            step.say(f"the provider wrote the answer and saw {cells} real value(s)", cells=cells)
            return {
                "answer": response.text.strip(),
                "answer_author": response.target,
                "success": True,
                "egress_chars": state.get("egress_chars", 0)
                + sum(len(m["content"]) for m in messages),
                "egress_values": state.get("egress_values", 0) + cells,
                "ms": _ms(state, "write", started),
                **_steps(),
            }
        except Exception as e:  # noqa: BLE001
            _log.warning("cloud answer writing failed: %s", e)
            step.say(f"the provider did not answer, showing the table instead ({type(e).__name__})")
            return {
                "answer": _plain(columns, rows),
                "answer_author": "fallback (plain table)",
                "success": True,
                "ms": _ms(state, "write", started),
                **_steps(),
            }
