"""The cloud client — the system's only network egress.

Nothing leaves without passing the egress gate: `call()` runs it before a socket
is opened, so bypassing the check means editing this file, which shows up in
review.

Models are arranged as a **ladder** — small, medium, large — and a caller asks for
a rung rather than for a provider. Within a rung, failure moves to the fallback:
a 429 waits and retries the same target (free Groq caps at 30 requests a minute),
a 5xx or a 404 moves on at once.

Only the small rung reports how sure it was. Groq refuses `logprobs` on every
model it serves; OpenRouter returns them. That is why the cheapest rung is the
OpenRouter one — it is the rung whose confidence decides whether to climb, so it
is the rung that has to be able to say.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from nl2sql.config import settings
from nl2sql.core.steps import track
from nl2sql.privacy import audit, gate

_log = logging.getLogger(__name__)

# Groq answered `403 error code 1010` to urllib: a Cloudflare block on the default
# user agent. Neither the key nor the model was at fault.
USER_AGENT = "nl2sql/2.0 (UNIMED internship)"

THINKING_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
CODE_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
SQL_START_RE = re.compile(r"(?:^|\n)\s*(SELECT|WITH)\b", re.IGNORECASE)

Size = Literal["small", "medium", "large"]

LADDER: tuple[Size, ...] = ("small", "medium", "large")


class NoProviderAvailable(RuntimeError):
    """Every target in the chain failed."""


@dataclass(frozen=True)
class Target:
    provider: str
    model: str
    base_url: str
    key: str

    @property
    def name(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass
class Response:
    text: str
    target: str
    ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    logprobs: list[float] = field(default_factory=list)
    attempts: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def perplexity(self) -> float | None:
        """exp(-mean log p) over the generated tokens: how sure the model was.

        `None` when the provider did not return log probabilities, in which case
        the local scorer in `llm.local` answers instead.
        """
        if not self.logprobs:
            return None
        return round(math.exp(-sum(self.logprobs) / len(self.logprobs)), 4)


def _target(spec: str) -> Target | None:
    """`groq/openai/gpt-oss-120b` -> a Target, or None when its key is missing."""
    cfg = settings()
    provider, _, model = (spec or "").partition("/")
    if not model:
        return None
    if provider == "groq" and cfg.groq_api_key:
        return Target("groq", model, cfg.groq_base_url, cfg.groq_api_key)
    if provider == "openrouter" and cfg.openrouter_api_key:
        return Target("openrouter", model, cfg.openrouter_base_url, cfg.openrouter_api_key)
    return None


def chain(size: Size = "large") -> list[Target]:
    """The targets to try for one rung: the rung itself, then the fallback.

    A rung whose key is not configured falls through to the next one up rather
    than failing — a machine with only a Groq key still runs every variant, it
    simply starts one rung higher and reports no perplexity.
    """
    cfg = settings()
    wanted = {"small": cfg.model_small, "medium": cfg.model_medium, "large": cfg.model_large}
    order = LADDER[LADDER.index(size):]

    targets = [t for t in (_target(wanted[rung]) for rung in order) if t]
    fallback = _target(cfg.model_fallback)
    if fallback and fallback.name not in {t.name for t in targets}:
        targets.append(fallback)
    return targets


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model response, or return nothing.

    Returning nothing matters as much as returning SQL: on an out-of-scope
    question the model answers in prose, and matching "with" anywhere in a
    sentence once handed `with that.` to SQLite, so a comprehension problem was
    reported as an execution defect. Hence: the keyword must open a line, and the
    result must contain a SELECT.
    """
    text = THINKING_RE.sub("", text or "").strip()
    block = CODE_BLOCK_RE.search(text)
    if block:
        text = block.group(1)
    start = SQL_START_RE.search(text)
    if not start:
        return ""
    candidate = text[start.start(1):].strip().rstrip(";").strip()
    return candidate if re.search(r"\bSELECT\b", candidate, re.IGNORECASE) else ""


def _retry_after(response: Any) -> float | None:
    """How long the provider asked us to wait, in seconds."""
    raw = response.headers.get("retry-after") or response.headers.get("x-ratelimit-reset-requests")
    if not raw:
        return None
    try:
        return float(str(raw).rstrip("s"))
    except ValueError:
        return None


def _token_logprobs(choice: dict[str, Any]) -> list[float]:
    content = ((choice.get("logprobs") or {}).get("content")) or []
    return [t["logprob"] for t in content if isinstance(t, dict) and "logprob" in t]


def call(
    messages: list[dict[str, str]],
    segments: list[gate.Segment] | None = None,
    size: Size = "large",
    # 700 truncated a nested query mid-statement; a cut query is unrecoverable.
    max_tokens: int = 1200,
    temperature: float = 0.0,
    logprobs: bool = False,
    context: str = "sql-generation",
    unprotected: bool = False,
) -> Response:
    """Send a conversation to the first available provider.

    `unprotected` disables the gate, and exists for exactly one caller: the Full
    Cloud arm, whose definition is that the question leaves unmasked. It is
    refused under `PRIVACY_MODE=strict`, and every use is logged and journalled so
    the report accounts for it instead of hiding it.
    """
    import httpx

    if unprotected:
        if settings().privacy_mode == "strict":
            raise gate.LeakBlocked(("unprotected",), "the baseline cannot run under strict mode")
        sent = sum(len(m.get("content", "")) for m in messages)
        _log.warning("EGRESS GATE BYPASSED — baseline '%s': %d characters, values included", context, sent)
        audit.record_bypass(context, sent)
    elif segments:
        gate.require_segments(segments, context)
    else:
        for message in messages:
            gate.require(message.get("content", ""), context)

    cfg = settings()
    targets = chain(size)
    if not targets:
        raise NoProviderAvailable("No API key configured in .env")

    attempts: list[str] = []
    last_error: Exception | None = None
    want_logprobs = logprobs

    with track("model", zone="cloud", size=size, characters=sum(len(m["content"]) for m in messages)) as step:
        for target in targets:
            for attempt in range(cfg.cloud_max_retries + 1):
                body: dict[str, Any] = {
                    "model": target.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if want_logprobs:
                    # OpenRouter returns nothing without `top_logprobs`; asking for
                    # one alternative is the cheapest way to get the chosen token's
                    # probability, which is all perplexity needs.
                    body["logprobs"] = True
                    body["top_logprobs"] = 1
                start = time.perf_counter()
                try:
                    r = httpx.post(
                        f"{target.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {target.key}",
                            "Content-Type": "application/json",
                            "User-Agent": USER_AGENT,
                        },
                        json=body,
                        timeout=cfg.cloud_timeout_s * (3 if target.provider == "openrouter" else 1),
                    )
                except Exception as e:  # noqa: BLE001 — network: move to the next target
                    last_error = e
                    attempts.append(f"{target.name}: {type(e).__name__}")
                    break

                ms = round((time.perf_counter() - start) * 1000, 1)
                if r.status_code == 200:
                    payload = r.json()
                    choices = payload.get("choices") or []
                    if not choices:
                        attempts.append(f"{target.name}: response without choices")
                        break
                    usage = payload.get("usage") or {}
                    response = Response(
                        text=choices[0]["message"]["content"] or "",
                        target=target.name,
                        ms=ms,
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        logprobs=_token_logprobs(choices[0]),
                        attempts=attempts,
                    )
                    step.say(
                        f"{target.name} replied in {ms:.0f} ms using {response.tokens} tokens",
                        model=target.name,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        perplexity=response.perplexity,
                        earlier_attempts=attempts,
                    )
                    return response

                # Groq answers 400 to `logprobs` on every model it currently
                # serves. Dropping the field and retrying keeps the code portable
                # to a provider that does support it, with no per-provider table.
                if r.status_code == 400 and want_logprobs:
                    want_logprobs = False
                    continue

                attempts.append(f"{target.name}: HTTP {r.status_code}")
                if r.status_code == 429 and attempt < cfg.cloud_max_retries:
                    # Groq says how long to wait; a doubling guess ignores it and
                    # burns the retry budget before the window has even reset,
                    # which is what makes a long benchmark fail in the middle.
                    wait = _retry_after(r) or 2**attempt
                    _log.warning("%s rate-limited, waiting %.0fs", target.name, wait)
                    time.sleep(min(wait, 30))
                    continue
                break

        step.say("no provider answered", earlier_attempts=attempts)
        raise NoProviderAvailable("Every target failed: " + " | ".join(attempts)) from last_error


def state() -> dict[str, Any]:
    """Which rung resolves to which model, for `/health` and the notebooks."""
    return {rung: [t.name for t in chain(rung)] for rung in LADDER}
