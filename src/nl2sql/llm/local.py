"""The local model — inside the trust boundary.

This is the only component that ever sees query *results*, that is, real data. If
answer writing went out to an API the rest of the architecture would be pointless:
we would have protected the question in order to export the answer.

Qwen3-1.7B in Q4_K_M, about 1.1 GB, on CPU. It is not a brilliant model and that
is the point: the SQL was authored by a much larger one and the numbers come from
the database, so writing is all that is left to do here.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from nl2sql.config import ROOT, settings
from nl2sql.core.steps import track

_log = logging.getLogger(__name__)

THINKING_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

MAX_PROMPT_ROWS = 30    # past this a 1.7B model starts copying rows wrong
MAX_ANSWER_TOKENS = 180
MAX_SQL_TOKENS = 400

# Qwen3 reasons before answering unless told not to. Without this it spent all 400
# tokens deliberating and returned no query at all.
NO_THINK = "/no_think"

SYSTEM_PROMPT = """You are a data analyst assistant. Answer using ONLY the query results given.

Rules:
- Use only the numbers present in the results. Never compute, estimate or invent a value.
- If the results are empty, say plainly that no matching record was found.
- Answer in 1 to 3 sentences, then stop. Do not repeat yourself.
- Do not mention SQL, tables, or column names unless the user asked about them."""

_client: Any = None
_lock = threading.Lock()
_state: dict[str, Any] = {"loaded": False, "backend": None, "model": None, "load_ms": None}


@dataclass(frozen=True)
class Written:
    text: str
    ms: float
    model: str
    tokens: int = 0


class _Ollama:
    """Local too, but in another process — so a socket to keep an eye on."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url, self.model = base_url.rstrip("/"), model

    def complete(self, prompt: str, max_tokens: int) -> str:
        import httpx

        r = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.0},
            },
            timeout=120.0,
        )
        r.raise_for_status()
        return r.json().get("response", "")


def load() -> Any:
    """Instantiate the backend once per process."""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        cfg = settings()
        start = time.perf_counter()
        from llama_cpp import Llama

        common = {"n_ctx": cfg.local_ctx, "n_threads": cfg.local_threads, "verbose": False}
        path = cfg.local_gguf_path
        resolved = (path if path and path.is_absolute() else ROOT / path) if path else None
        if resolved and resolved.exists():
            _client = Llama(model_path=str(resolved), logits_all=False, **common)
        else:
            # huggingface_hub cannot resume an interrupted transfer, so a file
            # already on disk is strongly preferred for a 1 GB download.
            _client = Llama.from_pretrained(
                repo_id=cfg.local_gguf_repo, filename=cfg.local_gguf_file, **common
            )
        _state.update(
            loaded=True,
            backend="llamacpp",
            model=cfg.local_gguf_file,
            load_ms=round((time.perf_counter() - start) * 1000),
        )
    return _client


def available() -> bool:
    return bool(_state["loaded"])


def state() -> dict[str, Any]:
    return dict(_state)


def _strip_thinking(text: str) -> str:
    """Remove reasoning traces, closed or not.

    When the token budget runs out inside the block no `</think>` is emitted, the
    pair never matches, and the model's scratchpad gets rendered to the analyst as
    if it were the answer. There is no answer hiding behind it.
    """
    cleaned = THINKING_RE.sub("", text or "")
    opened = cleaned.lower().find("<think>")
    return (cleaned[:opened] if opened != -1 else cleaned).strip()


def _first_block(text: str, max_sentences: int = 3) -> str:
    """Keep the answer, drop what the model tacked on afterwards.

    Qwen3-1.7B writes a correct sentence and then contradicts itself: "1234
    patients received aspirin." followed by a bare "1234" and "No matching record
    was found." The prompt asked for one to three sentences; this enforces it.
    """
    text = (text or "").strip()
    if not text:
        return ""
    paragraph = text.split("\n", 1)[0].strip()
    if len(paragraph) >= 25 and paragraph[-1] in ".!?":
        return paragraph
    sentences = SENTENCE_RE.split(text.replace("\n", " "))
    return " ".join(s.strip() for s in sentences[:max_sentences] if s.strip())


def render_table(columns: list[str], rows: list[tuple]) -> str:
    """Results in a form a small model reads without error.

    A single number is the commonest case and the one it most often mishandles:
    shown as a one-cell table it read the header as the answer and replied "no
    matching record" for a result of 4. Stated as a fact, it gets it right.
    """
    if not columns:
        return "(no column)"
    if not rows:
        return "(no row — the query returned nothing)"
    if len(rows) == 1 and len(rows[0]) == 1:
        return f"{columns[0]} = {rows[0][0]}"

    visible = rows[:MAX_PROMPT_ROWS]
    header = " | ".join(columns)
    body = "\n".join(" | ".join("" if v is None else str(v) for v in row) for row in visible)
    more = f"\n... ({len(rows)} rows total, {len(visible)} shown)" if len(rows) > len(visible) else ""
    return f"{header}\n{'-' * len(header)}\n{body}{more}"


def build_messages(question: str, columns: list[str], rows: list[tuple]) -> list[dict[str, str]]:
    """The answer-writing conversation.

    Sent as a chat, not a raw completion: with a prompt ending in "Answer:" Qwen3
    never emits its end-of-turn token and repeated itself for 19 seconds.
    """
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{NO_THINK}"},
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nQuery results:\n{render_table(columns, rows)}",
        },
    ]


def _chat(messages: list[dict[str, str]], max_tokens: int, stop: list[str]) -> tuple[str, int]:
    """One completion from whichever backend is loaded. Returns (text, tokens)."""
    client = load()
    if isinstance(client, _Ollama):
        flat = "\n\n".join(m["content"] for m in messages)
        return client.complete(flat, max_tokens), 0
    out = client.create_chat_completion(
        messages=messages, max_tokens=max_tokens, temperature=0.0, stop=stop
    )
    return out["choices"][0]["message"]["content"], out.get("usage", {}).get("completion_tokens", 0)


def write_answer(question: str, columns: list[str], rows: list[tuple]) -> Written:
    """Turn a query result into a sentence. No network call."""
    start = time.perf_counter()
    with track("model", zone="local", label="Asking the model here for a sentence", rows=len(rows)) as step:
        text, tokens = _chat(
            build_messages(question, columns, rows),
            MAX_ANSWER_TOKENS,
            ["<|im_end|>", "\n\nQuestion:"],
        )
        written = Written(
            _first_block(_strip_thinking(text)),
            round((time.perf_counter() - start) * 1000, 1),
            _state["model"] or "",
            tokens,
        )
        step.say(
            f"the model on this machine wrote {len(written.text.split())} words "
            f"from {len(rows)} row(s)",
            model=written.model,
            tokens=tokens,
        )
    return written


def generate_sql(question: str, ddl: str, notes: list[str] | None = None) -> Written:
    """The Full Local arm: the 1.7B model writes the SQL itself, nothing leaves.

    Expect it to be poor. That is the finding, not a failure of the experiment:
    the size of the gap is what justifies renting a large model for this one step.
    """
    from nl2sql.core.prompt import INSTRUCTIONS_LITERAL

    start = time.perf_counter()
    note_block = "\n\nDomain notes:\n" + "\n".join(f"- {n}" for n in notes) if notes else ""
    messages = [
        {"role": "system", "content": f"{INSTRUCTIONS_LITERAL}\n\n{NO_THINK}"},
        {"role": "user", "content": f"Schema:\n{ddl}{note_block}\n\nQuestion: {question}\n\nSQL:"},
    ]
    from nl2sql.llm.cloud import extract_sql

    with track(
        "model",
        zone="local",
        label="Asking the model here to write the SQL",
        characters=sum(len(m["content"]) for m in messages),
    ) as step:
        text, tokens = _chat(messages, MAX_SQL_TOKENS, ["<|im_end|>", "\n\nQuestion:", ";"])
        written = Written(
            extract_sql(_strip_thinking(text)),
            round((time.perf_counter() - start) * 1000, 1),
            _state["model"] or "",
            tokens,
        )
        step.say(
            f"the model on this machine wrote a query in {tokens} tokens"
            if written.text
            else "the model on this machine returned no query",
            model=written.model,
            tokens=tokens,
        )
    return written


def perplexity(text: str, prompt: str = "") -> float | None:
    """Score a candidate under the local model: exp(-mean log p) over its tokens.

    This is the fallback for providers that return no log probabilities, and it is
    a fair scorer for ranking candidates against each other because every
    candidate is judged by the same model on the same prompt. Lower is better.
    """
    try:
        client = load()
    except Exception as e:  # noqa: BLE001 — no local model: no local score
        _log.warning("perplexity unavailable: %s", e)
        return None
    if isinstance(client, _Ollama) or not text.strip():
        return None

    full = f"{prompt}\n{text}" if prompt else text
    try:
        out = client.create_completion(full, max_tokens=0, logprobs=0, echo=True)
    except Exception as e:  # noqa: BLE001
        _log.warning("perplexity scoring failed: %s", e)
        return None

    values = (out["choices"][0].get("logprobs") or {}).get("token_logprobs") or []
    scored = [v for v in values[-max(len(text.split()), 1) * 4 :] if isinstance(v, (int, float))]
    if not scored:
        return None
    return round(math.exp(-sum(scored) / len(scored)), 4)
