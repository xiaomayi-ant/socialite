# -*- coding: utf-8 -*-
"""Moltbook Challenge Verifier

Handles the math challenge verification that Moltbook requires after posting comments.
The challenge text is intentionally obfuscated (random case, special chars) and must
be solved within 5 minutes.
"""

import os
import re
import requests
from typing import Optional, Dict, Any


import logging
logger = logging.getLogger(__name__)

class ChallengeVerifier:
    """Solves and submits Moltbook math verification challenges"""

    def __init__(
        self,
        api_key: str,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        """Initialize verifier

        Args:
            api_key: Moltbook API key for submitting answers
            openai_api_key: Optional OpenAI API key for LLM solving
            anthropic_api_key: Optional Anthropic API key for LLM solving
        """
        self.api_key = api_key
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.llm_proxy_url = os.getenv("LLM_PROXY_URL")
        self.base_url = "https://www.moltbook.com/api/v1"

    def solve_and_submit(self, verification_code: str, challenge_text: str) -> bool:
        """Solve challenge and submit answer

        Args:
            verification_code: Code from comment response
            challenge_text: Obfuscated math problem text

        Returns:
            True if verification successful
        """
        # Step 1: Solve with the configured verifier provider/model.
        answer = self._solve_challenge(challenge_text)
        if answer is None:
            logger.warning("Could not solve challenge: %s...", challenge_text[:60])
            return False

        logger.info("   🧮 Challenge solved: %s", answer)

        # Step 2: Submit answer
        success, _message = self._submit_answer(verification_code, answer)
        return success

    def _solve_challenge(self, challenge_text: str) -> Optional[str]:
        """Solve the math challenge

        Args:
            challenge_text: Obfuscated challenge text

        Returns:
            Answer string (e.g. "92.00") or None if unsolvable
        """
        provider = os.getenv("MOLTBOOK_VERIFIER_PROVIDER", "openai").lower()
        if provider == "anthropic":
            if self.anthropic_api_key:
                logger.info(
                    "   Verifier provider=%s model=%s",
                    provider,
                    os.getenv(
                        "MOLTBOOK_VERIFIER_ANTHROPIC_MODEL",
                        "claude-haiku-4-5-20251001",
                    ),
                )
                return self._solve_with_anthropic(challenge_text)
            logger.warning("Verifier provider is anthropic but ANTHROPIC_API_KEY is not configured")
            return None

        if self.openai_api_key:
            logger.info(
                "   Verifier provider=openai model=%s",
                self._openai_model_name(),
            )
            return self._solve_with_openai(challenge_text)
        if provider == "openai":
            logger.warning("Verifier provider is openai but OPENAI_API_KEY is not configured")
            return None
        if self.anthropic_api_key:
            return self._solve_with_anthropic(challenge_text)

        return None

    def _openai_model_name(self) -> str:
        return os.getenv("MOLTBOOK_VERIFIER_OPENAI_MODEL") or os.getenv(
            "OPENAI_MODEL",
            "gpt-4o-mini",
        )

    def _preprocess_for_llm(self, text: str) -> str:
        """Strip noise characters and collapse spaces to help LLM read obfuscated text.

        Removes all characters that are NOT letters, digits, or spaces, then
        collapses any runs of whitespace into a single space.
        """
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _solve_with_openai(
        self,
        challenge_text: str,
        retries: int = 2,
    ) -> Optional[str]:
        """Use OpenAI to solve the challenge."""
        try:
            import time
            import httpx
            from openai import OpenAI

            http_client = None
            if self.llm_proxy_url:
                http_client = httpx.Client(proxy=self.llm_proxy_url, trust_env=False)

            client = OpenAI(
                api_key=self.openai_api_key,
                http_client=http_client,
            )
            cleaned_text = self._preprocess_for_llm(challenge_text)

            model_name = self._openai_model_name()
            for attempt in range(retries):
                try:
                    request_kwargs = {
                        "model": model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "The text below is a math word problem with obfuscated formatting "
                                    "(random caps, noise characters inserted inside words). "
                                    "Reconstruct the numbers, infer the operation, compute the final result, "
                                    "and reply with ONLY the final number in XX.XX format.\n\n"
                                    f"{cleaned_text}"
                                ),
                            }
                        ],
                    }
                    if model_name.startswith("gpt-5"):
                        request_kwargs["max_completion_tokens"] = 16
                    else:
                        request_kwargs["max_tokens"] = 16
                    message = client.chat.completions.create(**request_kwargs)
                    raw = (message.choices[0].message.content or "").strip()
                    match = re.search(r"\d+(?:\.\d+)?", raw)
                    if not match:
                        raise ValueError(f"No number in response: {raw!r}")
                    return f"{float(match.group()):.2f}"
                except Exception as e:
                    if attempt < retries - 1:
                        wait = min(2 ** attempt, 8)
                        logger.warning(
                            "OpenAI solve failed on %s, retrying in %ds: %s",
                            model_name,
                            wait,
                            e,
                        )
                        time.sleep(wait)
                        continue
                    raise
        except Exception as e:
            logger.info("   ⚠️  OpenAI solve failed: %s", e)
            return None

    def _solve_with_anthropic(self, challenge_text: str, retries: int = 5) -> Optional[str]:
        """Use Claude to solve the challenge.

        Args:
            challenge_text: Obfuscated challenge text
            retries: Number of retries on overload

        Returns:
            Answer string or None
        """
        try:
            import anthropic
            import time
            import httpx

            http_client = None
            if self.llm_proxy_url:
                http_client = httpx.Client(proxy=self.llm_proxy_url, trust_env=False)

            client = anthropic.Anthropic(
                api_key=self.anthropic_api_key,
                http_client=http_client,
            )

            # Pre-clean: strip noise characters so LLM can read broken words
            cleaned_text = self._preprocess_for_llm(challenge_text)

            for attempt in range(retries):
                try:
                    message = client.messages.create(
                        model=os.getenv(
                            "MOLTBOOK_VERIFIER_ANTHROPIC_MODEL",
                            "claude-haiku-4-5-20251001",
                        ),
                        max_tokens=16,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "The text below is a math word problem with obfuscated formatting "
                                    "(random caps, noise characters inserted inside words). "
                                    "Step 1: Reconstruct all the numbers mentioned (e.g. 'tw enty fo ur' = 24). "
                                    "Step 2: Identify the operation (add/gains/plus/more = +, "
                                    "multiply/times = *, subtract/loses/minus = -, divide = /). "
                                    "Step 3: Compute the FINAL total after applying all operations. "
                                    "Reply with ONLY the final computed number in XX.XX format.\n\n"
                                    f"{cleaned_text}"
                                ),
                            },
                            {
                                "role": "assistant",
                                "content": "The answer is:",
                            }
                        ],
                    )
                    raw = message.content[0].text.strip()
                    # Extract first number from response (handles trailing text)
                    match = re.search(r"\d+(?:\.\d+)?", raw)
                    if not match:
                        raise ValueError(f"No number in response: {raw!r}")
                    answer = f"{float(match.group()):.2f}"
                    return answer

                except anthropic.APIStatusError as e:
                    if e.status_code == 529 and attempt < retries - 1:
                        wait = min(2 ** attempt * 2, 30)  # 2s, 4s, 8s, 16s, 30s
                        logger.warning("API overloaded, retrying in %ds...", wait)
                        time.sleep(wait)
                        continue
                    raise

        except Exception as e:
            logger.info("   ⚠️  LLM solve failed: %s", e)
            return None

    def _submit_answer(self, verification_code: str, answer: str) -> tuple[bool, Optional[str]]:
        """Submit the answer to Moltbook

        Args:
            verification_code: Verification code from comment
            answer: Calculated answer

        Returns:
            True if accepted
        """
        try:
            resp = requests.post(
                f"{self.base_url}/verify",
                json={"verification_code": verification_code, "answer": answer},
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
            data = resp.json()
            if data.get("success"):
                logger.info("Verification passed!")
                return True, None

            message = data.get("message", "Unknown error")
            logger.info("   ❌ Verification failed: %s", message)
            return False, message

        except Exception as e:
            logger.info("   ❌ Verification submission error: %s", e)
            return False, str(e)
