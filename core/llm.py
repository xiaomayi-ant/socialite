# -*- coding: utf-8 -*-
"""LLMClient — unified async wrapper with OpenAI-first fallback."""

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import anthropic
import httpx

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    AsyncOpenAI = None

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified async LLM client.

    Usage::

        llm = LLMClient(model_name="gpt-4o-mini", max_tokens=512)
        text = await llm.call("Summarise this document …")
        data = await llm.call_json("Return JSON with keys …")
    """

    # ── Token pricing per 1M tokens: (input, output) ──
    _PRICING: Dict[str, Tuple[float, float]] = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "claude-haiku-4-5-20251001": (0.80, 4.00),
    }
    _DEFAULT_PRICING: Tuple[float, float] = (1.00, 3.00)

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        cost_recorder: Optional[Callable] = None,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.cost_recorder = cost_recorder
        self.primary_provider = (
            provider or os.getenv("LLM_PRIMARY_PROVIDER", "openai")
        ).lower()
        self.fallback_provider = os.getenv(
            "LLM_FALLBACK_PROVIDER", "anthropic"
        ).lower()
        self._provider_order = self._build_provider_order()

        proxy_url = os.getenv("LLM_PROXY_URL")
        timeout = float(os.getenv("LLM_HTTP_TIMEOUT_SECONDS", "60"))

        openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        self._openai_client = None
        if AsyncOpenAI is not None and openai_api_key:
            self._openai_client = AsyncOpenAI(
                api_key=openai_api_key,
                http_client=self._build_http_client(proxy_url, timeout),
            )

        self._anthropic_client = None
        if anthropic_api_key:
            self._anthropic_client = anthropic.AsyncAnthropic(
                api_key=anthropic_api_key,
                http_client=self._build_http_client(proxy_url, timeout),
            )

    def _build_provider_order(self) -> List[str]:
        providers: List[str] = []
        for name in (self.primary_provider, self.fallback_provider):
            if name in ("openai", "anthropic") and name not in providers:
                providers.append(name)
        if not providers:
            providers = ["openai", "anthropic"]
        return providers

    def _resolve_model_for_provider(self, provider: str) -> str:
        if provider == "openai":
            if self.model_name.startswith("claude-"):
                return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return self.model_name
        if self.model_name.startswith("gpt-"):
            return os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        return self.model_name

    @staticmethod
    def _build_http_client(proxy_url: Optional[str], timeout_seconds: float) -> httpx.AsyncClient:
        kwargs: Dict[str, Any] = {
            "timeout": timeout_seconds,
            "trust_env": True,
        }
        if proxy_url:
            kwargs["proxy"] = proxy_url
            kwargs["trust_env"] = False
        return httpx.AsyncClient(**kwargs)

    @staticmethod
    def _estimate_cost(
        model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate USD cost from token counts."""
        pricing = LLMClient._PRICING.get(model, LLMClient._DEFAULT_PRICING)
        return (input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000

    async def call(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> str:
        """Send a single-turn prompt and return the assistant text."""
        last_error: Optional[Exception] = None
        for idx, provider in enumerate(self._provider_order):
            try:
                if provider == "openai":
                    text, usage = await self._call_openai(prompt=prompt, system=system)
                else:
                    text, usage = await self._call_anthropic(prompt=prompt, system=system)
                # Fire-and-forget cost recording
                if self.cost_recorder and usage:
                    try:
                        self.cost_recorder(usage)
                    except Exception:
                        logger.debug("Cost recorder failed (non-fatal)")
                return text
            except Exception as e:  # noqa: BLE001
                last_error = e
                if idx < len(self._provider_order) - 1:
                    logger.warning(
                        "LLM provider %s failed (%s); falling back to %s",
                        provider,
                        e,
                        self._provider_order[idx + 1],
                    )
                else:
                    logger.error("LLM call failed on %s (%s): %s", provider, self.model_name, e)

        if last_error is not None:
            raise last_error
        raise RuntimeError("No valid LLM provider configured")

    async def _call_openai(
        self, prompt: str, system: Optional[str] = None
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if self._openai_client is None:
            raise RuntimeError("OpenAI client unavailable (missing OPENAI_API_KEY or openai SDK)")

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        model = self._resolve_model_for_provider("openai")
        resp = await self._openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("Empty OpenAI response")

        usage: Optional[Dict[str, Any]] = None
        if resp.usage:
            input_tokens = resp.usage.prompt_tokens or 0
            output_tokens = resp.usage.completion_tokens or 0
            usage = {
                "model": model,
                "provider": "openai",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": self._estimate_cost(model, input_tokens, output_tokens),
            }
        return text, usage

    async def _call_anthropic(
        self, prompt: str, system: Optional[str] = None
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client unavailable (missing ANTHROPIC_API_KEY)")

        messages = [{"role": "user", "content": prompt}]
        model = self._resolve_model_for_provider("anthropic")
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        resp = await self._anthropic_client.messages.create(**kwargs)
        text = resp.content[0].text

        usage: Optional[Dict[str, Any]] = None
        if resp.usage:
            input_tokens = resp.usage.input_tokens or 0
            output_tokens = resp.usage.output_tokens or 0
            usage = {
                "model": model,
                "provider": "anthropic",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": self._estimate_cost(model, input_tokens, output_tokens),
            }
        return text, usage

    async def call_json(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> Union[Dict[str, Any], List[Any]]:
        """Call the LLM and parse the response as JSON.

        Handles: markdown fences, leading text, trailing garbage,
        and truncated JSON from max_tokens cutoff.
        """
        text = await self.call(prompt, system=system)
        return self._parse_json(text)

    @staticmethod
    def _parse_json(text: str) -> Union[Dict[str, Any], List[Any]]:
        """Extract and parse JSON from potentially messy LLM output."""
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        # Find first { or [
        start = -1
        open_char = ""
        for i, ch in enumerate(text):
            if ch in ("{", "["):
                start = i
                open_char = ch
                break
        if start == -1:
            return json.loads(text)  # let it fail with a clear error

        text = text[start:]

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Trim trailing garbage after the last matching close bracket
        close_char = "}" if open_char == "{" else "]"
        last_close = text.rfind(close_char)
        if last_close > 0:
            try:
                return json.loads(text[:last_close + 1])
            except json.JSONDecodeError:
                pass

        # Truncated JSON: try to repair by closing open brackets/braces
        repaired = _repair_truncated_json(text)
        if repaired is not None:
            return repaired

        # Last resort: original parse to get a clear error
        return json.loads(text)


def _repair_truncated_json(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """Attempt to repair JSON truncated by max_tokens.

    Strategy: strip the last incomplete element, then close all open
    brackets/braces.
    """
    # Remove trailing incomplete string (unterminated quote)
    cleaned = re.sub(r',\s*"[^"]*$', '', text)
    # Remove trailing incomplete key-value pair
    cleaned = re.sub(r',\s*"[^"]*"\s*:\s*[^,\]\}]*$', '', cleaned)
    # Remove trailing incomplete object in array
    cleaned = re.sub(r',\s*\{[^\}]*$', '', cleaned)

    # Count open/close brackets
    stack: List[str] = []
    in_string = False
    escape = False
    for ch in cleaned:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()

    # Close remaining open brackets
    suffix = ""
    for opener in reversed(stack):
        suffix += "}" if opener == "{" else "]"

    if not suffix:
        return None  # nothing to repair

    try:
        return json.loads(cleaned + suffix)
    except json.JSONDecodeError:
        return None
