"""The REST API — the only door into the trust boundary.

What this process is
--------------------
Everything this file serves runs on the protected side: the database, the value
index, the entity extractor, the answer writer. The one thing that leaves is the
SQL-generation call, and it leaves from `providers/cloud.py`, behind the gate.
A browser talking to this API is a *client*; it is not inside the boundary, which
is why every response is built from `state.public()` rather than from the state.

Why there is a streaming endpoint
---------------------------------
A demo that returns one JSON blob after nine seconds proves nothing: the
interesting claim is about what happens *between* the question and the answer.
`/ask/stream` emits one server-sent event per stage, so the interface can show
the question being masked, the gate's verdict segment by segment, and the SQL
arriving from the provider — in the order they actually happen. The pipeline is
the product here; hiding it behind a spinner would be hiding the deliverable.

CORS
----
Open by default. This service is read-only, holds a public de-identified research
database, and is reached over an ephemeral tunnel whose hostname changes at every
restart — an origin allowlist would be a list of URLs that no longer exist. When
it runs on UNIMED's network against real data, `CORS_ORIGINS` pins it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from hybridsql.config import settings
from hybridsql.db import schema as sch
from hybridsql.graph import ARMS, compiled, mermaid
from hybridsql.graph.state import blank, is_refusal, public
from hybridsql.obs import tracing
from hybridsql.providers import cloud, extractor, local_model
from hybridsql.security import audit

_log = logging.getLogger(__name__)

app = FastAPI(
    title="UNIMED HybridSQL",
    version="1.0.0",
    description=(
        "Ask a database a question in English without the data leaving the "
        "network. The cloud model sees the schema and a masked question; "
        "execution and answer writing stay local."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.getenv("CORS_ORIGINS", "*").split(",") if o],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------------
#  Schemas
# ---------------------------------------------------------------------------------
class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    arm: str = "hybrid"
    write: bool = True


class Compare(BaseModel):
    question: str = Field(min_length=3, max_length=500)
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
        "extractor": extractor.state(),
        "local_model": local_model.state(),
        "tracing": tracing.configure(),
    }


@app.get("/meta")
def meta() -> dict[str, Any]:
    """What the interface needs to describe the system without hardcoding it."""
    return {
        "arms": list(ARMS),
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
    lines = audit.read()
    return {
        **audit.leak_rate(),
        "bypassed": sum(1 for line in lines if line.get("bypassed")),
        "characters_sent": sum(line.get("length", 0) for line in lines if line.get("allowed")),
    }


# ---------------------------------------------------------------------------------
#  Asking
# ---------------------------------------------------------------------------------
def _run(question: str, arm: str, write: bool) -> dict[str, Any]:
    if arm not in ARMS:
        raise HTTPException(400, f"unknown arm: {arm}. Use one of {list(ARMS)}")
    from hybridsql.graph import run

    return public(run(question, arm=arm, write=write))  # type: ignore[arg-type]


@app.post("/ask")
async def ask(body: Ask) -> dict[str, Any]:
    """One question, one answer.

    Run in a worker thread: the pipeline is synchronous and CPU-bound — llama.cpp
    holds the GIL for seconds at a time — and blocking the event loop would stall
    every other request, including `/health`.
    """
    return await asyncio.to_thread(_run, body.question, body.arm, body.write)


@app.post("/compare")
async def compare(body: Compare) -> dict[str, Any]:
    """The same question through all three architectures — the benchmark, live.

    Sequential on purpose. Running the arms concurrently would have them contend
    for the same two cores and the same llama.cpp instance, and the latency column
    is one of the things being measured.
    """
    def _all() -> dict[str, Any]:
        out = {}
        for arm in ARMS:
            started = time.perf_counter()
            try:
                out[arm] = _run(body.question, arm, body.write)
            except Exception as e:  # noqa: BLE001 — one arm failing is a result too
                out[arm] = {
                    "arm": arm, "success": False, "failed_stage": "arm",
                    "failure_reason": f"{type(e).__name__}: {e}",
                    "ms": {"total": round((time.perf_counter() - started) * 1000, 1)},
                }
        return {"question": body.question, "arms": out}

    return await asyncio.to_thread(_all)


# ---------------------------------------------------------------------------------
#  Streaming
# ---------------------------------------------------------------------------------
def _event(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _stage_view(node: str, update: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """What the interface is shown after each node.

    The mapping `:v1 -> 'aspirin'` is included. That is not a leak: this response
    goes to the analyst who typed the question and already knows the value, and
    showing the substitution side by side is the entire point of the
    demonstration. It is the *cloud* that must not see it, and the cloud is not
    on this connection.
    """
    view: dict[str, Any] = {"stage": node, "ms": update.get("ms", {}).get(node)}

    if node == "understand":
        u = update.get("understanding")
        view["zone"] = "local"
        view["extractor"] = update.get("extractor", "")
        view["tables"] = update.get("tables", [])
        view["entities"] = [
            {
                "mention": r.mention, "kind": r.kind, "column": r.column,
                "value": r.value, "score": round(r.score, 2), "masked": r.to_mask,
            }
            for r in (u.resolutions if u is not None else [])
        ]

    elif node == "anonymize":
        anon = update.get("anonymization")
        view["zone"] = "local"
        view["before"] = state.get("question", "")
        view["after"] = update.get("masked_question", "")
        view["mapping"] = dict(anon.mapping) if anon is not None else {}
        view["columns"] = dict(anon.columns) if anon is not None else {}

    elif node == "generate":
        view["zone"] = "cloud" if state.get("arm") != "full_local" else "local"
        view["sql"] = update.get("sql", "")
        view["provider"] = update.get("sql_author", "")
        view["tokens"] = update.get("cloud_tokens", 0)
        view["calls"] = update.get("cloud_calls", 0)
        view["repairs"] = update.get("repairs", 0)
        view["egress_chars"] = update.get("egress_chars", 0)
        view["egress_values"] = update.get("egress_values", 0)
        view["gate"] = update.get("egress_segments", [])
        # Hybrid Opaque: the question and schema as the provider saw them — t1/c7
        # labels — with the dictionary that turns them back into names.
        view["opaque"] = update.get("opaque", {})

    elif node == "execute":
        view["zone"] = "local"
        view["columns"] = update.get("columns", [])
        view["row_count"] = update.get("row_count", 0)
        # A sample, not the result set: a query may return 2000 rows and the
        # interface only ever renders the first few.
        view["rows"] = [list(r) for r in update.get("rows", [])[:10]]

    elif node == "write":
        view["zone"] = "cloud" if state.get("arm") == "full_cloud" else "local"
        view["answer"] = update.get("answer", "")
        view["author"] = update.get("answer_author", "")

    if update.get("failed_stage"):
        view["failed"] = update["failed_stage"]
        view["reason"] = update["failure_reason"]
        view["refusal"] = is_refusal(update)  # type: ignore[arg-type]

    return view


async def _stream(question: str, arm: str, write: bool) -> AsyncIterator[str]:
    tracing.configure()
    state = blank(question, arm=arm, write=write)  # type: ignore[arg-type]
    yield _event("start", {"question": question, "arm": arm})

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def produce() -> None:
        """Drive the graph on a thread, pushing each node's output as it lands."""
        merged = dict(state)
        try:
            for step in compiled(arm).stream(  # type: ignore[arg-type]
                state,
                stream_mode="updates",
                config={"run_name": f"hybridsql:{arm}", "tags": [arm, "stream"]},
            ):
                for node, update in step.items():
                    merged.update(update)
                    view = _stage_view(node, update, merged)
                    loop.call_soon_threadsafe(queue.put_nowait, ("stage", view))
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
            name, payload = item
            yield _event(name, payload)
    finally:
        await task


@app.post("/ask/stream")
async def ask_stream(body: Ask) -> StreamingResponse:
    """One server-sent event per pipeline stage, in the order they complete."""
    return StreamingResponse(
        _stream(body.question, body.arm, body.write),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Tunnels and reverse proxies buffer by default, which would deliver
            # every event at once when the run ends — exactly the behaviour this
            # endpoint exists to avoid.
            "X-Accel-Buffering": "no",
        },
    )
