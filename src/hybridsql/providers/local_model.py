"""The model that writes the answer — inside the trust boundary.

Why it must be local, without exception
---------------------------------------
This is the only component in the pipeline that sees **the query results**, that
is, the real data. The cloud model only ever saw a schema and a masked question.
If answer writing went out to an API, the rest of the architecture would be
pointless: we would have protected the question in order to export the answer.

That constraint is not left to the caller's vigilance. In `strict` mode a backend
that leaves the network is **refused at load time**, not at call time.

Backend choice
--------------
    llamacpp      in-process, GGUF on CPU. The default. Truly local.
    ollama        when an Ollama server runs on the machine. Local too, but in
                  another process — so a socket to watch.
    hf-inference  network fallback. Refused in strict mode, and loudly logged in
                  demo mode.

The default model is Qwen3-1.7B in Q4_K_M: about 1.1 GB on disk, runs on a few
cores. It is not a brilliant model, and that is the point — the demonstration
must show that a small model suffices to *write*, given that the SQL was authored
by a much larger one and the numbers come from the database, not from the model.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from hybridsql.config import settings

_log = logging.getLogger(__name__)

NETWORK_BACKENDS = {"hf-inference"}
THINKING_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

MAX_PROMPT_ROWS = 30      # beyond this, summarise rather than flood the context

# One to three sentences need well under 200 tokens. The cap is not just about
# cost: a small model given room to ramble will ramble, and repeat itself.
MAX_ANSWER_TOKENS = 180

# SQL needs more room than prose: a join across four eICU tables with a CTE runs
# long, and a query truncated mid-clause is unrecoverable rather than merely bad.
MAX_SQL_TOKENS = 400

# Qwen3 reasons before answering unless told not to, and `/no_think` is the
# model's own documented switch for it. Without this, measured on the SQL task:
# the model spends all 400 tokens deliberating in a `<think>` block, stops, and
# returns no query at all — the Full Local arm scored 0% for a reason that had
# nothing to do with its ability to write SQL. Cheaper and, on a task with no
# reasoning to do, better.
NO_THINK = "/no_think"

_client: Any = None
_lock = threading.Lock()
_state: dict[str, Any] = {"loaded": False, "backend": None, "model": None, "load_ms": None}


class ForbiddenBackend(RuntimeError):
    """The requested backend would leave the trust boundary."""


@dataclass(frozen=True)
class WrittenAnswer:
    text: str
    ms: float
    backend: str
    model: str
    tokens: int = 0


# ---------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------
def _check_boundary(backend: str) -> None:
    cfg = settings()
    if backend in NETWORK_BACKENDS:
        if cfg.privacy_mode == "strict":
            raise ForbiddenBackend(
                f"Backend '{backend}' sends query results outside the trust boundary. "
                "Forbidden in strict mode (PRIVACY_MODE=strict). Use llamacpp or ollama."
            )
        _log.warning(
            "DEMO MODE: backend '%s' sends results outside the boundary. "
            "Never use it to measure the leak rate.",
            backend,
        )


def load() -> Any:
    """Instantiate the backend once for the process."""
    global _client
    if _client is not None:
        return _client

    with _lock:
        if _client is not None:
            return _client

        cfg = settings()
        backend = cfg.local_llm_backend
        _check_boundary(backend)

        start = time.perf_counter()
        if backend == "llamacpp":
            _client = _load_llamacpp()
            model = cfg.local_llm_gguf_file
        elif backend == "ollama":
            _client = _OllamaClient(cfg.ollama_base_url, cfg.ollama_model)
            model = cfg.ollama_model
        elif backend == "hf-inference":
            _client = _HfInferenceClient(cfg.hf_token, cfg.hf_inference_model)
            model = cfg.hf_inference_model
        else:
            raise ValueError(f"unknown backend: {backend}")

        ms = (time.perf_counter() - start) * 1000
        _state.update(loaded=True, backend=backend, model=model, load_ms=round(ms))
        _log.info("local writer ready: %s / %s (%.0f ms)", backend, model, ms)
    return _client


def _load_llamacpp() -> Any:
    """Load the GGUF into memory.

    A file already on disk is preferred (`LOCAL_LLM_GGUF_PATH`, filled by
    `scripts/download_models.sh`). Automatic download remains possible but cannot
    resume after an interruption — best avoided on an unstable connection.
    """
    from llama_cpp import Llama

    cfg = settings()
    common = {
        "n_ctx": cfg.local_llm_ctx,
        "n_threads": cfg.local_llm_threads,
        "verbose": False,
    }

    path = cfg.local_llm_gguf_path
    if path:
        from hybridsql.config import ROOT

        resolved = path if path.is_absolute() else ROOT / path
        if resolved.exists():
            _log.info("local GGUF: %s", resolved)
            return Llama(model_path=str(resolved), **common)

    _log.info("GGUF not on disk, downloading from %s", cfg.local_llm_gguf_repo)
    return Llama.from_pretrained(
        repo_id=cfg.local_llm_gguf_repo, filename=cfg.local_llm_gguf_file, **common
    )


class _OllamaClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def complete(self, prompt: str, max_tokens: int) -> str:
        import httpx

        r = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.2},
            },
            timeout=120.0,
        )
        r.raise_for_status()
        return r.json().get("response", "")


class _HfInferenceClient:
    """Network fallback. Forbidden in strict mode — see `_check_boundary`."""

    def __init__(self, token: str, model: str) -> None:
        self.token = token
        self.model = model

    def complete(self, prompt: str, max_tokens: int) -> str:
        import httpx

        r = httpx.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
            timeout=120.0,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def available() -> bool:
    return _state["loaded"]


def state() -> dict[str, Any]:
    return dict(_state)


# ---------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------
def _table(columns: list[str], rows: list[tuple]) -> str:
    """Render the results in a form the model reads without error.

    Truncation is deliberate: past a few dozen rows, a 1.7-billion-parameter model
    starts copying them wrong. We give it a sample plus the exact count, and
    forbid extrapolation.
    """
    if not columns:
        return "(no column)"
    if not rows:
        return "(no row — the query returned nothing)"

    # A single number is the commonest case ("how many patients…"), and the one a
    # small model most often mishandles: shown as a one-cell table it read the
    # header as the answer and replied "no matching record" for a result of 4.
    # Stated as a fact, it gets it right.
    if len(rows) == 1 and len(rows[0]) == 1:
        return f"{columns[0]} = {rows[0][0]}"

    visible = rows[:MAX_PROMPT_ROWS]
    header = " | ".join(columns)
    body = "\n".join(" | ".join("" if v is None else str(v) for v in row) for row in visible)
    more = ""
    if len(rows) > len(visible):
        more = f"\n... ({len(rows)} rows total, {len(visible)} shown)"
    return f"{header}\n{'-' * len(header)}\n{body}{more}"


SYSTEM_PROMPT = """You are a data analyst assistant. Answer using ONLY the query results given.

Rules:
- Use only the numbers present in the results. Never compute, estimate or invent a value.
- If the results are empty, say plainly that no matching record was found.
- Answer in 1 to 3 sentences, then stop. Do not repeat yourself.
- Do not mention SQL, tables, or column names unless the user asked about them."""


def build_messages(question: str, columns: list[str], rows: list[tuple]) -> list[dict[str, str]]:
    """The answer-writing conversation.

    Two instructions carry the whole weight: compute nothing, invent nothing. The
    numbers come from the database; the model is only there to put them into
    sentences. That is what makes such a small model acceptable.

    Sent as a chat rather than a raw completion. That is not cosmetic: with a raw
    prompt ending in "Answer:", Qwen3 never emits its end-of-turn token and keeps
    going until the token cap — it produced the same sentence four times over and
    took 19 seconds. Applying the model's own chat template makes it stop when it
    is done.
    """
    return [
        # `/no_think` is Qwen3's own switch for skipping the reasoning block. It
        # belongs on every call to this model, not just the SQL one: writing two
        # sentences from a table needs no deliberation, and the block cost 8
        # seconds and a truncated answer every time it ran.
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{NO_THINK}"},
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nQuery results:\n{_table(columns, rows)}",
        },
    ]


def build_prompt(question: str, columns: list[str], rows: list[tuple]) -> str:
    """Flat rendering of the conversation, for backends that take a single string."""
    messages = build_messages(question, columns, rows)
    return f"{messages[0]['content']}\n\n{messages[1]['content']}\n\nAnswer:"


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _strip_thinking(text: str) -> str:
    """Remove reasoning traces, closed or not.

    Qwen3 emits a `<think>…</think>` block before its answer, and the obvious
    regex — match the pair, delete it — has a hole that showed up in the browser:
    when the token budget runs out *inside* the block, no `</think>` is ever
    emitted, the pattern does not match, and the model's scratchpad is rendered to
    the analyst as if it were the answer. Observed verbatim on screen:

        <think> Okay, the user is asking how many patients received aspirin.
        Let me check the query results. The only value provided is patient_count = 0.

    So an unterminated `<think>` truncates everything after it. There is no answer
    hiding behind it — the model never got to write one — and showing nothing is
    better than showing deliberation dressed up as a result. `/no_think` (see
    `NO_THINK`) stops it happening in the first place; this is the net underneath.
    """
    cleaned = THINKING_RE.sub("", text or "")
    opened = cleaned.lower().find("<think>")
    if opened != -1:
        cleaned = cleaned[:opened]
    return cleaned.strip()


def _first_coherent_block(text: str, max_sentences: int = 3) -> str:
    """Keep the answer and drop what the model tacked on afterwards.

    Measured behaviour of Qwen3-1.7B on this task: it produces a correct sentence,
    then keeps going and contradicts itself — "1234 patients received aspirin."
    followed by a bare "1234" and "No matching record was found." The first
    sentence is right; everything after it is the model echoing its own
    instructions.

    So we keep the first paragraph when it is already a complete sentence, and
    otherwise cap at three sentences. Truncating a small model's output is not a
    workaround here — the prompt asks for one to three sentences, and this enforces
    what was asked.
    """
    text = (text or "").strip()
    if not text:
        return ""

    first_paragraph = text.split("\n", 1)[0].strip()
    if len(first_paragraph) >= 25 and first_paragraph[-1] in ".!?":
        return first_paragraph

    sentences = SENTENCE_RE.split(text.replace("\n", " "))
    return " ".join(s.strip() for s in sentences[:max_sentences] if s.strip())


def generate_sql(question: str, ddl: str, notes: list[str] | None = None) -> WrittenAnswer:
    """Have the *local* model write the SQL — the Full Local arm of the benchmark.

    Why this exists
    ---------------
    The benchmark needs a floor as well as a ceiling. Full Cloud shows what a
    frontier model can do when it is handed everything; Full Local shows what the
    same pipeline achieves when nothing leaves at all. The interesting result is
    not that one wins, it is the size of the gap the hybrid architecture has to
    close — and it can only be quoted if this arm actually runs.

    No masking happens here, and none is needed: the question never leaves the
    process. The SQL therefore carries literal values rather than `:v1`, which is
    why the validator is called with `expected_parameters=None` for this arm.

    The model is the same 1.7B used for answer writing. Expect it to be poor at
    this: writing SQL over 31 tables is exactly the task that motivates renting a
    large model in the first place. Reporting that honestly is the point.
    """
    from hybridsql.pipeline.generate import INSTRUCTIONS_LITERAL

    start = time.perf_counter()
    client = load()
    cfg = settings()

    note_block = ""
    if notes:
        note_block = "\n\nDomain notes:\n" + "\n".join(f"- {n}" for n in notes)
    user = f"Schema:\n{ddl}{note_block}\n\nQuestion: {question}\n\nSQL:"

    if cfg.local_llm_backend == "llamacpp":
        out = client.create_chat_completion(
            messages=[
                {"role": "system", "content": f"{INSTRUCTIONS_LITERAL}\n\n{NO_THINK}"},
                {"role": "user", "content": user},
            ],
            max_tokens=MAX_SQL_TOKENS,
            temperature=0.0,
            stop=["<|im_end|>", "\n\nQuestion:", ";"],
        )
        text = out["choices"][0]["message"]["content"]
        tokens = out.get("usage", {}).get("completion_tokens", 0)
    else:
        text = client.complete(f"{INSTRUCTIONS_LITERAL}\n\n{NO_THINK}\n\n{user}", MAX_SQL_TOKENS)
        tokens = 0

    # Reuse the cloud extractor: a small model fences its SQL in markdown just as
    # eagerly as a large one, and the unwrapping problem is identical.
    from hybridsql.providers.cloud import extract_sql

    return WrittenAnswer(
        text=extract_sql(_strip_thinking(text)),
        ms=round((time.perf_counter() - start) * 1000, 1),
        backend=_state["backend"] or cfg.local_llm_backend,
        model=_state["model"] or "",
        tokens=tokens,
    )


def write_answer(
    question: str,
    columns: list[str],
    rows: list[tuple],
    max_tokens: int = MAX_ANSWER_TOKENS,
) -> WrittenAnswer:
    """Turn a query result into a written answer. No network call when the backend
    is `llamacpp` (the default)."""
    start = time.perf_counter()
    client = load()
    cfg = settings()

    if cfg.local_llm_backend == "llamacpp":
        out = client.create_chat_completion(
            messages=build_messages(question, columns, rows),
            max_tokens=max_tokens,
            # Deterministic: there is nothing creative to do here, and sampling
            # made the model wander into repetition.
            temperature=0.0,
            # Qwen3 emits a thinking block by default; `/no_think` is its documented
            # way to skip it. Cheaper and, for a task with no reasoning to do,
            # better — the scratchpad only adds latency and noise to strip.
            stop=["<|im_end|>", "\n\nQuestion:"],
        )
        text = out["choices"][0]["message"]["content"]
        tokens = out.get("usage", {}).get("completion_tokens", 0)
    else:
        text = client.complete(build_prompt(question, columns, rows), max_tokens)
        tokens = 0

    return WrittenAnswer(
        text=_first_coherent_block(_strip_thinking(text)),
        ms=round((time.perf_counter() - start) * 1000, 1),
        backend=_state["backend"] or cfg.local_llm_backend,
        model=_state["model"] or "",
        tokens=tokens,
    )
