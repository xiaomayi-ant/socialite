# -*- coding: utf-8 -*-
"""LLMClient — thin async wrapper around Anthropic's API."""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

import anthropic

logger = logging.getLogger(__name__)


class LLMClient:
    """Async Claude client.

    Usage::

        llm = LLMClient(model_name="claude-haiku-4-5-20251001", max_tokens=512)
        text = await llm.call("Summarise this document …")
        data = await llm.call_json("Return JSON with keys …")
    """

    def __init__(
        self,
        model_name: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
        )

    async def call(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> str:
        """Send a single-turn prompt and return the assistant text."""
        messages = [{"role": "user", "content": prompt}]
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        try:
            resp = await self._client.messages.create(**kwargs)
            return resp.content[0].text
        except Exception as e:
            logger.error("LLM call failed (%s): %s", self.model_name, e)
            raise

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
