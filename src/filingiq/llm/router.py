"""LLM router: one interface, swappable providers, always-on cost accounting.

WHY A ROUTER
------------
Hardcoding a provider makes the cost/accuracy comparison impossible later, and
that comparison is the most interesting number in the project. Being able to
say "Llama 3.3 70B costs $0.04 per filing at 94% extraction accuracy; the local
8B costs nothing at 87%" is a real engineering result. A single-provider
implementation can only assert that its choice was fine.

Every call records tokens and latency. Cost is not an afterthought bolted on at
the end -- it is a first-class metric from the first call.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

from filingiq.utils.rate_limit import RateLimiter

log = logging.getLogger(__name__)

# USD per 1M tokens (input, output). Provider catalogues churn -- Groq
# deprecated the entire Llama chat family in June 2026 -- so treat this as a
# cache, not a source of truth, and prefer list_models() to discover what your
# key can actually reach today.
PRICING = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3.6-27b": (0.15, 0.60),
    "llama-3.3-70b-versatile": (0.59, 0.79),   # deprecated June 2026
    "llama-3.1-8b-instant": (0.05, 0.08),      # deprecated June 2026
    "_local": (0.0, 0.0),
}

# Ordered preference when auto-selecting. First one the account can reach wins.
GROQ_PREFERRED = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
]

# Reasoning models emit internal chain-of-thought BEFORE the answer, and those
# tokens come out of max_tokens. Give one a 20-token budget with JSON mode on
# and it burns the whole allowance thinking, returns an empty string, and the
# provider rejects it with json_validate_failed and an empty failed_generation.
#
# The symptom looks like a malformed prompt. It is actually a budget problem.
REASONING_MODELS = ("gpt-oss", "qwen3", "deepseek-r1", "o1", "o3")

# Floor for any reasoning model: enough headroom for thinking plus the answer.
REASONING_MIN_TOKENS = 1024


def is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return any(tag in m for tag in REASONING_MODELS)


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_usd: float
    error: str | None = None

    def json(self) -> dict:
        """Parse JSON, tolerating markdown fences the model may add anyway."""
        t = self.text.strip()
        if t.startswith("```"):
            t = t.split("```", 2)[1]
            if t.startswith("json"):
                t = t[4:]
        return json.loads(t.strip())


@dataclass
class UsageTracker:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    errors: int = 0
    latencies: list[int] = field(default_factory=list)

    def record(self, r: LLMResponse) -> None:
        self.calls += 1
        self.tokens_in += r.tokens_in
        self.tokens_out += r.tokens_out
        self.cost_usd += r.cost_usd
        self.latencies.append(r.latency_ms)
        if r.error:
            self.errors += 1

    def summary(self) -> dict:
        lat = sorted(self.latencies)
        return {
            "calls": self.calls,
            "errors": self.errors,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 4),
            "p50_ms": lat[len(lat) // 2] if lat else 0,
            "p95_ms": lat[int(len(lat) * 0.95)] if lat else 0,
        }


def _cost(model: str, tin: int, tout: int) -> float:
    pin, pout = PRICING.get(model, PRICING["_local"])
    return (tin * pin + tout * pout) / 1_000_000


class ModelRouter:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 requests_per_second: float = 0.5, temperature: float = 0.0):
        from filingiq.config import settings

        self.provider = provider or settings.llm_provider
        self.model = model or (settings.groq_model if self.provider == "groq"
                               else settings.ollama_model)
        # Groq's free tier is rate-limited per minute. 0.5 rps stays well under
        # it; exceeding the limit yields 429s that look like model failures.
        self.limiter = RateLimiter(requests_per_second)
        self.temperature = temperature
        self.usage = UsageTracker()
        self._client = None

    def _groq(self):
        if self._client is None:
            from groq import Groq
            key = os.getenv("GROQ_API_KEY", "")
            if not key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to .env "
                    "(get one free at https://console.groq.com/keys)"
                )
            self._client = Groq(api_key=key)
        return self._client

    def list_models(self) -> list[str]:
        """Ask the provider what it actually serves right now.

        Hardcoded model names rot. Groq retired its whole Llama chat family in
        June 2026; code pinned to `llama-3.3-70b-versatile` stopped working
        with a 404 that looks like a bug in your own program. Querying the
        catalogue turns a mystifying failure into a list of valid options.
        """
        if self.provider != "groq":
            return []
        try:
            import requests
            r = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}"},
                timeout=20,
            )
            r.raise_for_status()
            return sorted(m["id"] for m in r.json().get("data", []))
        except Exception as exc:  # noqa: BLE001
            log.debug("could not list models: %s", exc)
            return []

    def autoselect_model(self) -> str | None:
        """Pick the best available model from GROQ_PREFERRED."""
        available = set(self.list_models())
        if not available:
            return None
        for candidate in GROQ_PREFERRED:
            if candidate in available:
                return candidate
        # Fall back to any chat-capable model rather than failing outright.
        chat = [m for m in sorted(available)
                if not any(x in m for x in ("whisper", "guard", "tts", "orpheus"))]
        return chat[0] if chat else None

    def healthcheck(self) -> tuple[bool, str]:
        """Verify the provider answers before starting a long run.

        Without this, a misconfigured provider produces 160 failed calls and a
        report reading "100% abstention" -- a plausible-looking result table
        generated entirely by a wrong environment variable. Failing loudly in
        two seconds beats failing quietly in six minutes.
        """
        try:
            # Budget must accommodate reasoning models; see is_reasoning_model.
            r = self.complete("Reply with JSON only.",
                              'Return exactly {"ok": true}',
                              json_mode=True, max_tokens=1024, retries=1)
            if r.error:
                return False, r.error
            if not r.text.strip():
                return False, "empty response"
            return True, f"{self.provider}/{self.model} responding"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def complete(self, system: str, user: str, json_mode: bool = True,
                 max_tokens: int = 1500, retries: int = 3) -> LLMResponse:
        last_err = None
        for attempt in range(retries):
            self.limiter.acquire()
            t0 = time.time()
            try:
                if self.provider == "groq":
                    r = self._complete_groq(system, user, json_mode, max_tokens)
                else:
                    r = self._complete_ollama(system, user, json_mode, max_tokens)
                r.latency_ms = int((time.time() - t0) * 1000)
                self.usage.record(r)
                return r
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                # 429 and transient 5xx are worth retrying; a bad key is not.
                low = last_err.lower()
                # json_validate_failed is NOT fatal: it usually means the token
                # budget truncated generation, which a retry with the reasoning
                # floor applied can fix.
                fatal = (("api_key" in low or "401" in last_err
                          or "invalid_api_key" in low or "model_not_found" in low
                          or ("404" in last_err and "json" not in low)))
                if fatal:
                    # Configuration errors do not become correct on retry.
                    raise
                wait = 2 ** attempt
                log.warning("LLM call failed (attempt %d/%d): %s -- retrying in %ds",
                            attempt + 1, retries, last_err[:120], wait)
                time.sleep(wait)

        r = LLMResponse(text="", model=self.model, tokens_in=0, tokens_out=0,
                        latency_ms=0, cost_usd=0.0, error=last_err)
        self.usage.record(r)
        return r

    def _complete_groq(self, system, user, json_mode, max_tokens) -> LLMResponse:
        if is_reasoning_model(self.model):
            # Reasoning tokens are billed and counted against max_tokens, so
            # the budget must cover thinking AND the answer.
            max_tokens = max(max_tokens, REASONING_MIN_TOKENS)

        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if is_reasoning_model(self.model):
            # Extraction is a reading task, not a puzzle. Minimal reasoning
            # keeps latency and token spend down with no accuracy cost here --
            # worth re-testing at "medium" as an ablation.
            kwargs["reasoning_effort"] = "low"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._groq().chat.completions.create(**kwargs)
        tin = resp.usage.prompt_tokens
        tout = resp.usage.completion_tokens
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=self.model, tokens_in=tin, tokens_out=tout, latency_ms=0,
            cost_usd=_cost(self.model, tin, tout),
        )

    def _complete_ollama(self, system, user, json_mode, max_tokens) -> LLMResponse:
        import requests
        from filingiq.config import settings

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": self.temperature,
                        "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        base = os.getenv("OLLAMA_URL", "http://localhost:11434")
        resp = requests.post(f"{base}/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            text=data["message"]["content"],
            model=self.model,
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
            latency_ms=0, cost_usd=0.0,
        )
