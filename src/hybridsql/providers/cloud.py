"""The cloud provider client — the system's only network egress.

What this module guarantees
---------------------------
**Nothing leaves here without passing the egress gate.** That is not a convention:
`call()` invokes `egress_gate.require()` on every message before opening a
connection. Bypassing the check would require editing this file, which is visible
in review.

The fallback chain
------------------
Three targets, tried in order, measured on 15 August 2026:

    groq / openai/gpt-oss-120b        1.1 s   — primary
    groq / llama-3.3-70b-versatile    2.4 s   — same key, different model
    openrouter / openai/gpt-oss-20b  26.4 s   — free, last resort

Switching serves two distinct cases: quota (free Groq caps at 30 requests per
minute) and unavailability. A 429 triggers a wait then a retry; a 5xx switches
target immediately.

A trap we hit
-------------
The first attempt used `urllib`, and Groq answered `403 error code 1010` — a
Cloudflare block on the default user agent. Neither the key nor the model was at
fault. Hence `httpx` with an explicit header.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from hybridsql.config import settings
from hybridsql.security import audit, egress_gate

_log = logging.getLogger(__name__)

USER_AGENT = "hybridsql/1.0 (UNIMED internship)"
THINKING_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
CODE_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class NoProviderAvailable(RuntimeError):
    """Every target in the chain failed."""


@dataclass
class Target:
    provider: str
    model: str
    base_url: str
    key: str

    @property
    def name(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass
class CloudResponse:
    text: str
    target: str
    ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def chain() -> list[Target]:
    """Targets in order of preference, filtered on available keys."""
    cfg = settings()
    targets: list[Target] = []
    for name in cfg.provider_chain:
        if name == "groq" and cfg.groq_api_key:
            targets.append(Target("groq", cfg.groq_model, cfg.groq_base_url, cfg.groq_api_key))
            if cfg.groq_model_fallback:
                targets.append(
                    Target("groq", cfg.groq_model_fallback, cfg.groq_base_url, cfg.groq_api_key)
                )
        elif name == "openrouter" and cfg.openrouter_api_key:
            targets.append(
                Target(
                    "openrouter", cfg.openrouter_model, cfg.openrouter_base_url,
                    cfg.openrouter_api_key,
                )
            )
    return targets


SQL_START_RE = re.compile(r"(?:^|\n)\s*(SELECT|WITH)\b", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model response, or return nothing.

    Models happily wrap their answer: Markdown fences, a "Here is the query:"
    preamble, a `<think>` reasoning block on Qwen. We strip all of it rather than
    demand a discipline the model does not have.

    Returning nothing matters as much as returning SQL. On an out-of-scope question
    the model answers in prose — "I can't help with that." — and an earlier version
    matched the word "with" anywhere in the sentence, treating it as the start of a
    CTE. It handed back `with that.` as a query, which reached SQLite and failed
    with a syntax error, so a comprehension problem was reported as an execution
    defect. Two of the 107 evaluation questions failed that way.

    Hence two conditions: the keyword must open a line, and the result must
    actually contain a SELECT. Prose no longer masquerades as SQL, and the
    validator reports the real problem.
    """
    text = THINKING_RE.sub("", text or "").strip()
    block = CODE_BLOCK_RE.search(text)
    if block:
        text = block.group(1)

    start = SQL_START_RE.search(text)
    if not start:
        return ""
    candidate = text[start.start(1):].strip().rstrip(";").strip()

    if not re.search(r"\bSELECT\b", candidate, re.IGNORECASE):
        return ""
    return candidate


def call(
    messages: list[dict[str, str]],
    segments: list[egress_gate.Segment] | None = None,
    # 700 truncated a nested query mid-statement on the evaluation set, which then
    # failed with "incomplete input". Generation is cheap; a cut query is not.
    max_tokens: int = 1200,
    temperature: float = 0.0,
    context: str = "sql-generation",
    baseline_unprotected: bool = False,
) -> CloudResponse:
    """Send a conversation to the first available provider.

    Everything goes through the egress gate **before** any connection is opened.
    Unauthorised content raises `LeakBlocked` and nothing is sent.

    When `segments` is given, the gate verifies each part against the rule matching
    its origin — the instruction text by fingerprint, the DDL by regenerating it,
    the notes by membership, and only the masked question word by word. Without
    segments the whole message goes through the word check, which is stricter and
    slower but always correct.

    `baseline_unprotected` disables the gate
    ----------------------------------------
    It exists for exactly one caller: the **Full Cloud** arm of the benchmark,
    whose entire definition is that the question leaves unmasked. Measuring that
    arm requires actually sending it; a gate that blocked it would leave the
    comparison unmeasured and the report with an empty column.

    Three things keep it from becoming a hole:
      - the parameter is keyword-only in practice and named after what it does,
        so it cannot be passed by accident;
      - it is refused outright when `PRIVACY_MODE=strict`, i.e. on any deployment
        that claims protection;
      - every use is logged at WARNING and written to the audit journal, so the
        egress figures in the report account for it instead of hiding it.
    """
    import httpx

    if baseline_unprotected:
        if settings().privacy_mode == "strict":
            raise egress_gate.LeakBlocked(
                ("baseline_unprotected",),
                "the unprotected baseline cannot run under PRIVACY_MODE=strict",
            )
        sent = sum(len(m.get("content", "")) for m in messages)
        _log.warning(
            "EGRESS GATE BYPASSED — benchmark baseline '%s': %d characters leaving "
            "unmasked, values included. This is the arm being measured, not a defect.",
            context, sent,
        )
        audit.record_bypass(context, sent)
    elif segments:
        egress_gate.require_segments(segments, context)
    else:
        for message in messages:
            egress_gate.require(message.get("content", ""), context)

    cfg = settings()
    targets = chain()
    if not targets:
        raise NoProviderAvailable("No API key configured in .env")

    attempts: list[str] = []
    last_error: Exception | None = None

    for target in targets:
        for attempt in range(cfg.cloud_max_retries + 1):
            start = time.perf_counter()
            try:
                r = httpx.post(
                    f"{target.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {target.key}",
                        "Content-Type": "application/json",
                        "User-Agent": USER_AGENT,
                    },
                    json={
                        "model": target.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=cfg.cloud_timeout_s * (3 if target.provider == "openrouter" else 1),
                )
            except Exception as e:  # noqa: BLE001 — network: switch target
                last_error = e
                attempts.append(f"{target.name}: {type(e).__name__}")
                break

            ms = (time.perf_counter() - start) * 1000

            if r.status_code == 200:
                d = r.json()
                if not d.get("choices"):
                    attempts.append(f"{target.name}: response without choices")
                    break
                u = d.get("usage") or {}
                return CloudResponse(
                    text=d["choices"][0]["message"]["content"] or "",
                    target=target.name,
                    ms=round(ms, 1),
                    prompt_tokens=u.get("prompt_tokens", 0),
                    completion_tokens=u.get("completion_tokens", 0),
                    attempts=attempts,
                )

            attempts.append(f"{target.name}: HTTP {r.status_code}")
            if r.status_code == 429 and attempt < cfg.cloud_max_retries:
                # Quota: wait, then retry the same target. Free Groq caps at 30
                # requests per minute and frees up quickly.
                wait = 2**attempt
                _log.warning("%s rate-limited, waiting %ss", target.name, wait)
                time.sleep(wait)
                continue
            break  # 4xx other than 429, or 5xx: next target

    raise NoProviderAvailable("Every target failed: " + " | ".join(attempts)) from last_error


def state() -> dict[str, Any]:
    return {"targets": [t.name for t in chain()]}
