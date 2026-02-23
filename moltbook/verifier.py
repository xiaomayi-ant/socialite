# -*- coding: utf-8 -*-
"""Moltbook Challenge Verifier

Handles the math challenge verification that Moltbook requires after posting comments.
The challenge text is intentionally obfuscated (random case, special chars) and must
be solved within 5 minutes.
"""

import re
import requests
from typing import Optional, Dict, Any


import logging
logger = logging.getLogger(__name__)

class ChallengeVerifier:
    """Solves and submits Moltbook math verification challenges"""

    def __init__(self, api_key: str, anthropic_api_key: Optional[str] = None):
        """Initialize verifier

        Args:
            api_key: Moltbook API key for submitting answers
            anthropic_api_key: Optional Anthropic API key for LLM solving
        """
        self.api_key = api_key
        self.anthropic_api_key = anthropic_api_key
        self.base_url = "https://www.moltbook.com/api/v1"

    def solve_and_submit(self, verification_code: str, challenge_text: str) -> bool:
        """Solve challenge and submit answer

        Args:
            verification_code: Code from comment response
            challenge_text: Obfuscated math problem text

        Returns:
            True if verification successful
        """
        # Step 1: Try to solve
        answer = self._solve_challenge(challenge_text)
        if answer is None:
            logger.warning("Could not solve challenge: %s...", challenge_text[:60])
            return False

        logger.info("   🧮 Challenge solved: %s", answer)

        # Step 2: Submit answer
        return self._submit_answer(verification_code, answer)

    def _solve_challenge(self, challenge_text: str) -> Optional[str]:
        """Solve the math challenge

        Args:
            challenge_text: Obfuscated challenge text

        Returns:
            Answer string (e.g. "92.00") or None if unsolvable
        """
        # First try local parsing (fast, no API needed)
        answer = self._solve_locally(challenge_text)
        if answer is not None:
            return answer

        # Fall back to LLM if available
        if self.anthropic_api_key:
            return self._solve_with_llm(challenge_text)

        return None

    def _solve_locally(self, text: str) -> Optional[str]:
        """Try to solve using local regex/word parsing

        Args:
            text: Challenge text

        Returns:
            Answer or None
        """
        # Normalize text: lowercase, remove special chars except spaces and *
        clean = re.sub(r"[^a-zA-Z0-9\s\*\+\-\/]", " ", text).lower()
        clean = re.sub(r"\s+", " ", clean).strip()

        # Number word map
        ones = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
            "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
            "eighteen": 18, "nineteen": 19,
        }
        tens = {
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
        }

        def words_to_number(words: list) -> Optional[int]:
            """Convert word list to number"""
            result = 0
            current = 0
            for w in words:
                if w in ones:
                    current += ones[w]
                elif w in tens:
                    current += tens[w]
                elif w == "hundred":
                    current *= 100
                elif w == "thousand":
                    result += current * 1000
                    current = 0
                elif w == "million":
                    result += current * 1_000_000
                    current = 0
            return result + current if (result + current) > 0 else None

        # Try to extract two numbers and an operator
        words = clean.split()
        numbers = []
        operator = None
        i = 0
        while i < len(words):
            # Check for number words
            num_words = []
            j = i
            number_words = set(ones.keys()) | set(tens.keys()) | {"hundred", "thousand", "million"}
            while j < len(words) and words[j] in number_words:
                num_words.append(words[j])
                j += 1
            if num_words:
                n = words_to_number(num_words)
                if n is not None:
                    numbers.append(n)
                i = j
                continue

            # Check for operator keywords
            if words[i] in ("multiplied", "times", "multiply"):
                operator = "*"
            elif words[i] in ("plus", "added", "add", "gains", "gain"):
                operator = "+"
            elif words[i] in ("minus", "subtracted", "subtract", "loses", "lost", "fewer"):
                operator = "-"
            elif words[i] in ("divided", "divide"):
                operator = "/"
            elif words[i] == "*":
                operator = "*"

            i += 1

        if len(numbers) == 2 and operator:
            a, b = numbers[0], numbers[1]
            if operator == "*":
                result = a * b
            elif operator == "+":
                result = a + b
            elif operator == "-":
                result = a - b
            elif operator == "/" and b != 0:
                result = a / b
            else:
                return None
            return f"{result:.2f}"

        return None

    def _preprocess_for_llm(self, text: str) -> str:
        """Strip noise characters and collapse spaces to help LLM read obfuscated text.

        Removes all characters that are NOT letters, digits, or spaces, then
        collapses any runs of whitespace into a single space.
        """
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _solve_with_llm(self, challenge_text: str, retries: int = 5) -> Optional[str]:
        """Use Claude to solve the challenge

        Args:
            challenge_text: Obfuscated challenge text
            retries: Number of retries on overload

        Returns:
            Answer string or None
        """
        try:
            import anthropic
            import time
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)

            # Pre-clean: strip noise characters so LLM can read broken words
            cleaned_text = self._preprocess_for_llm(challenge_text)

            for attempt in range(retries):
                try:
                    message = client.messages.create(
                        model="claude-haiku-4-5-20251001",
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

    def _submit_answer(self, verification_code: str, answer: str) -> bool:
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
                return True
            else:
                logger.info("   ❌ Verification failed: %s", data.get('message', 'Unknown error'))
                return False

        except Exception as e:
            logger.info("   ❌ Verification submission error: %s", e)
            return False
