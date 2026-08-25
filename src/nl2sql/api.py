"""The REST service — the only door into the trust boundary.

Everything served here runs on the protected side; the one thing that leaves does
so from `llm/cloud.py`, behind the gate. Responses are built from `state.public()`
because a browser is a client, not an insider.

`/ask/stream` emits one event per traced step: the interesting claim is about what
happens between the question and the answer, so the pipeline shows its work.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nl2sql.config import settings
from nl2sql.core import trace
from nl2sql.core.graph import ARMS, compiled, mermaid, run
from nl2sql.core.state import blank, is_refusal, public
from nl2sql.core.steps import catalogue as step_catalogue
from nl2sql.db import schema as sch
from nl2sql.llm import cloud, local
from nl2sql.nlp import extract as ner
from nl2sql.optimize.variants import VARIANTS
from nl2sql.optimize.variants import catalogue as variant_catalogue
from nl2sql.privacy import audit

app = FastAPI(
    title="NL2SQL",
    version="2.0.0",
    description=(
        "Ask a database a question in English without the data leaving the network. "
        "The cloud model sees the schema and a masked question; execution and answer "
        "writing stay local."
    ),
)

# Open by default: this service is read-only, holds a public de-identified research
# database, and is reached over a tunnel whose hostname changes at every restart —
# an origin allowlist would be a list of URLs that no longer exist. On UNIMED's
# network against real data, CORS_ORIGINS pins it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.getenv("CORS_ORIGINS", "*").split(",") if o],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    arm: str = "hybrid"
    variant: str = "baseline"
    write: bool = True


# ---------------------------------------------------------------------------------
#  Status
# ---------------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    """Cheap enough to be polled. Never loads a model to answer."""
    return {
        "status": "ok",
        "database": settings().db_path.exists(),
        "index": settings().index_path.exists(),
        "extractor": ner.state(),
        "local_model": local.state(),
        "tracing": trace.configure(),
    }


@app.get("/meta")
def meta() -> dict[str, Any]:
    """What the interface needs to describe the system without hardcoding it.

    The step labels come from `core/steps.py`, so the wording on screen and the wording
    in the traces cannot drift apart.
    """
    return {
        "arms": list(ARMS),
        "variants": variant_catalogue(),
        "steps": step_catalogue(),
        "schema": sch.summary(),
        "tables": sorted(sch.read_schema()),
        "cloud": cloud.state(),
        "privacy_mode": settings().privacy_mode,
    }


@app.get("/graph/{arm}")
def graph_diagram(arm: str) -> dict[str, str]:
    """The mermaid source of an arm, generated from the compiled graph."""
    if arm not in ARMS:
        raise HTTPException(404, f"unknown arm: {arm}")
    return {"arm": arm, "mermaid": mermaid(arm)}  # type: ignore[arg-type]


@app.get("/egress/report")
def egress_report() -> dict[str, Any]:
    """The audit journal, summarised. The evidence behind the headline claim."""
    return audit.report()


# ---------------------------------------------------------------------------------
#  Asking
# ---------------------------------------------------------------------------------
def _run(body: Ask) -> dict[str, Any]:
    if body.arm not in ARMS:
        raise HTTPException(400, f"unknown arm: {body.arm}. Use one of {list(ARMS)}")
    if body.variant not in VARIANTS:
        raise HTTPException(400, f"unknown variant: {body.variant}. Use one of {list(VARIANTS)}")
    state = run(body.question, arm=body.arm, write=body.write, variant=body.variant)  # type: ignore[arg-type]
    out = public(state)
    out["trace"] = state.get("trace", [])
    out["gate"] = state.get("egress_segments", [])
    return out


@app.post("/ask")
async def ask(body: Ask) -> dict[str, Any]:
    """One question, one answer."""
    return await asyncio.to_thread(_run, body)


@app.post("/compare")
async def compare(body: Ask) -> dict[str, Any]:
    """The same question through all four architectures — the benchmark, live.

    Sequential on purpose: run concurrently they would contend for the same two cores
    and the same llama.cpp instance, and latency is one of the things being measured.
    """
    def _all() -> dict[str, Any]:
        out = {}
        for arm in ARMS:
            started = time.perf_counter()
            try:
                out[arm] = _run(Ask(question=body.question, arm=arm, variant=body.variant, write=body.write))
            except Exception as e:  # noqa: BLE001 — one arm failing is a result too
                out[arm] = {
                    "arm": arm, "success": False, "failed_stage": "arm",
                    "failure_reason": f"{type(e).__name__}: {e}",
                    "ms": {"total": round((time.perf_counter() - started) * 1000, 1)},
                }
        return {"question": body.question, "arms": out}

    return await asyncio.to_thread(_all)


@app.post("/variants")
async def variants(body: Ask) -> dict[str, Any]:
    """The same question through all five hybrid variants, side by side."""
    def _all() -> dict[str, Any]:
        out = {}
        for name in VARIANTS:
            try:
                out[name] = _run(Ask(question=body.question, arm="hybrid", variant=name, write=False))
            except Exception as e:  # noqa: BLE001
                out[name] = {"variant": name, "success": False,
                             "failure_reason": f"{type(e).__name__}: {e}"}
        return {"question": body.question, "variants": out}

    return await asyncio.to_thread(_all)


# ---------------------------------------------------------------------------------
#  Streaming
# ---------------------------------------------------------------------------------
def _event(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _stage_view(node: str, update: dict[str, Any], merged: dict[str, Any]) -> dict[str, Any]:
    """What the interface is shown when a stage completes."""
    view: dict[str, Any] = {"stage": node, "ms": update.get("ms", {}).get(node)}

    if node == "understand":
        u = update.get("understanding")
        view["extractor"] = update.get("extractor", "")
        view["tables"] = update.get("tables", [])
        view["notes"] = update.get("notes", [])
        view["entities"] = [
            {"mention": r.mention, "kind": r.kind, "column": r.column, "value": r.value,
             "score": round(r.score, 2), "masked": r.to_mask}
            for r in (u.resolutions if u is not None else [])
        ]
    elif node == "mask":
        masked = update.get("masked")
        view["before"] = merged.get("question", "")
        view["after"] = update.get("masked_question", "")
        view["mapping"] = dict(masked.mapping) if masked is not None else {}
        view["columns"] = dict(masked.columns) if masked is not None else {}
    elif node == "generate":
        view["sql"] = update.get("sql", "")
        view["provider"] = update.get("sql_author", "")
        view["tokens"] = update.get("cloud_tokens", 0)
        view["prompt_tokens"] = update.get("prompt_tokens", 0)
        view["completion_tokens"] = update.get("completion_tokens", 0)
        view["calls"] = update.get("cloud_calls", 0)
        view["repairs"] = update.get("repairs", 0)
        view["perplexity"] = update.get("perplexity")
        view["difficulty"] = update.get("difficulty", 0)
        view["escalated"] = update.get("escalated", False)
        view["candidates"] = update.get("candidates", [])
        view["egress_chars"] = update.get("egress_chars", 0)
        view["egress_values"] = update.get("egress_values", 0)
        view["gate"] = update.get("egress_segments", [])
        view["opaque"] = update.get("opaque", {})
    elif node == "execute":
        view["columns"] = update.get("columns", [])
        view["row_count"] = update.get("row_count", 0)
        # A sample, not the result set: a query may return 2000 rows.
        view["rows"] = [list(r) for r in update.get("rows", [])[:10]]
    elif node == "write":
        view["answer"] = update.get("answer", "")
        view["author"] = update.get("answer_author", "")

    if update.get("failed_stage"):
        view["failed"] = update["failed_stage"]
        view["reason"] = update["failure_reason"]
        view["refusal"] = is_refusal(update)  # type: ignore[arg-type]
        view["suggestions"] = update.get("suggestions", [])
    return view


async def _stream(body: Ask) -> AsyncIterator[str]:
    trace.configure()
    initial = blank(body.question, arm=body.arm, write=body.write, variant=body.variant)  # type: ignore[arg-type]
    yield _event("start", {"question": body.question, "arm": body.arm, "variant": body.variant})

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def produce() -> None:
        """Drive the graph on a thread, pushing each step and stage as it lands."""
        merged = dict(initial)
        sent = 0
        try:
            with trace.record(body.question, arm=body.arm, variant=body.variant) as recorded:
                for chunk in compiled(body.arm).stream(  # type: ignore[arg-type]
                    initial,
                    stream_mode="updates",
                    config={"run_name": f"nl2sql:{body.arm}:{body.variant}",
                            "tags": [body.arm, body.variant, "stream"]},
                ):
                    for node, update in chunk.items():
                        merged.update(update)
                        # Every small step traced since the last stage, in order.
                        for step in update.get("trace", [])[sent:]:
                            loop.call_soon_threadsafe(queue.put_nowait, ("step", step))
                        sent = len(update.get("trace", [])) or sent
                        loop.call_soon_threadsafe(
                            queue.put_nowait, ("stage", _stage_view(node, update, merged))
                        )
                recorded.result = public(merged)  # type: ignore[arg-type]
            loop.call_soon_threadsafe(queue.put_nowait, ("done", public(merged)))  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            loop.call_soon_threadsafe(
                queue.put_nowait, ("error", {"reason": f"{type(e).__name__}: {e}"})
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    task = asyncio.create_task(asyncio.to_thread(produce))
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield _event(item[0], item[1])
    finally:
        await task


@app.post("/ask/stream")
async def ask_stream(body: Ask) -> StreamingResponse:
    """One server-sent event per traced step, in the order they complete."""
    return StreamingResponse(
        _stream(body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Tunnels and proxies buffer by default, which would deliver every
            # event at once when the run ends — the behaviour this endpoint exists
            # to avoid.
            "X-Accel-Buffering": "no",
        },
    )
