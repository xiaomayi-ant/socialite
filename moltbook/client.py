# -*- coding: utf-8 -*-
"""Moltbook API Client"""

import requests
import time
import asyncio
from functools import wraps
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta

from .models import (
    MoltbookPost,
    MoltbookComment,
    MoltbookAgent,
    MoltbookSubmolt,
    MoltbookSearchResult,
    MoltbookRegistration,
)
from .config import MoltbookConfig
from .auth import MoltbookAuth
from .verifier import ChallengeVerifier


import logging
logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker for API resilience"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
        self.half_open_calls = 0

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.half_open_calls = 0
            else:
                raise Exception(f"Circuit breaker OPEN, retry after {self.recovery_timeout}s")

        if self.state == "half_open":
            if self.half_open_calls >= self.half_open_max_calls:
                raise Exception("Circuit breaker HALF_OPEN max calls reached")
            self.half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time:
            elapsed = (datetime.now() - self.last_failure_time).total_seconds()
            return elapsed >= self.recovery_timeout
        return True

    def _on_success(self):
        self.failure_count = 0
        if self.state == "half_open":
            self.state = "closed"
        logger.info("Circuit breaker: closed")

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.state == "half_open" or self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker: opened after {self.failure_count} failures")


def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    retry_on_status: tuple = (403, 429, 500, 502, 503, 504),
):
    """Decorator for retrying failed API calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff factor
        retry_on_status: HTTP status codes to retry on.
            Includes 403 (CloudFront WAF throttle) and 429 (rate limit).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code if e.response else 0
                    if status_code in retry_on_status and attempt < max_retries:
                        # 403/429 use longer backoff to respect WAF/rate limits
                        if status_code in (403, 429):
                            wait_time = backoff_factor * (3 ** attempt) * 10
                        else:
                            wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(
                            "Retry %d/%d after %.0fs (HTTP %d)",
                            attempt + 1, max_retries, wait_time, status_code,
                        )
                        time.sleep(wait_time)
                        last_exception = e
                    else:
                        raise
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(
                            "Retry %d/%d after %.0fs (%s)",
                            attempt + 1, max_retries, wait_time,
                            type(e).__name__,
                        )
                        time.sleep(wait_time)
                        last_exception = e
                    else:
                        raise
            raise last_exception
        return wrapper
    return decorator

class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self):
        self.last_post_time: Optional[datetime] = None
        self.last_comment_time: Optional[datetime] = None
        self.daily_comment_count: int = 0
        self.daily_comment_reset: datetime = datetime.now()

    def can_post(self, cooldown_minutes: int = 30) -> tuple[bool, int]:
        """Check if can post

        Args:
            cooldown_minutes: Cooldown period in minutes

        Returns:
            (can_post, minutes_remaining)
        """
        if not self.last_post_time:
            return True, 0

        elapsed = (datetime.now() - self.last_post_time).total_seconds() / 60
        if elapsed >= cooldown_minutes:
            return True, 0

        return False, int(cooldown_minutes - elapsed) + 1

    def can_comment(
        self, cooldown_seconds: int = 20, max_daily: int = 50
    ) -> tuple[bool, str]:
        """Check if can comment

        Args:
            cooldown_seconds: Cooldown period in seconds
            max_daily: Maximum comments per day

        Returns:
            (can_comment, reason)
        """
        # Reset daily count if needed
        if (datetime.now() - self.daily_comment_reset).days >= 1:
            self.daily_comment_count = 0
            self.daily_comment_reset = datetime.now()

        # Check daily limit
        if self.daily_comment_count >= max_daily:
            return False, f"Daily limit reached ({max_daily})"

        # Check cooldown
        if self.last_comment_time:
            elapsed = (datetime.now() - self.last_comment_time).total_seconds()
            if elapsed < cooldown_seconds:
                remaining = int(cooldown_seconds - elapsed) + 1
                return False, f"Cooldown: {remaining}s remaining"

        return True, ""

    def record_post(self):
        """Record a post"""
        self.last_post_time = datetime.now()

    def record_comment(self):
        """Record a comment"""
        self.last_comment_time = datetime.now()
        self.daily_comment_count += 1


class MoltbookClient:
    """Client for Moltbook API"""

    def __init__(
        self,
        config: MoltbookConfig,
        auth: MoltbookAuth,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        """Initialize Moltbook client

        Args:
            config: Moltbook configuration
            auth: Authentication manager
            openai_api_key: Optional OpenAI API key for challenge solving
            anthropic_api_key: Optional Anthropic API key for challenge solving
        """
        self.config = config
        self.auth = auth
        self.base_url = config.api_base
        self.rate_limiter = RateLimiter()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "MoltbookClient/0.1.0",
            }
        )
        # Challenge verifier for comment verification
        self._verifier = ChallengeVerifier(
            api_key=auth.get_api_key() or "",
            openai_api_key=openai_api_key or __import__("os").getenv("OPENAI_API_KEY"),
            anthropic_api_key=anthropic_api_key or __import__("os").getenv("ANTHROPIC_API_KEY"),
        )

        # Circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            half_open_max_calls=3,
        )

    @with_retry(max_retries=3, backoff_factor=1.0)
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        auth_required: bool = True,
    ) -> Dict[str, Any]:
        """Make HTTP request to Moltbook API

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            auth_required: Whether authentication is required

        Returns:
            Response data

        Raises:
            Exception: If request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        headers = {}
        if auth_required:
            headers.update(self.auth.get_auth_header())

        def _do_request():
            try:
                response = self.session.request(
                    method=method, url=url, json=data, params=params, headers=headers
                )

                # Handle rate limit errors
                if response.status_code == 429:
                    error_data = response.json()
                    retry_after = error_data.get("retry_after_minutes") or error_data.get(
                        "retry_after_seconds"
                    )
                    raise Exception(
                        f"Rate limit exceeded. Retry after: {retry_after}. {error_data.get('error', '')}"
                    )

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                logger.warning("API request failed: %s %s", method, endpoint)
                logger.info("   Error: %s", e)
                if hasattr(e, "response") and e.response is not None:
                    try:
                        error_data = e.response.json()
                        logger.info("   API Error: %s", error_data.get('error', 'Unknown error'))
                        if "hint" in error_data:
                            logger.info("   Hint: %s", error_data['hint'])
                        if "message" in error_data:
                            logger.info("   Message: %s", error_data['message'])
                        if "details" in error_data:
                            logger.info("   Details: %s", error_data['details'])
                    except Exception:
                        logger.info("   Response body: %s", e.response.text)
                raise

        return self.circuit_breaker.call(_do_request)

    # ==================== Registration & Auth ====================

    def register(self, name: str, description: str) -> MoltbookRegistration:
        """Register a new agent

        Args:
            name: Agent name
            description: Agent description

        Returns:
            Registration data including API key and claim URL
        """
        logger.info("🦞 Registering agent: %s", name)

        response = self._request(
            "POST",
            "/agents/register",
            data={"name": name, "description": description},
            auth_required=False,
        )

        if not response.get("success"):
            raise Exception(f"Registration failed: {response.get('error')}")

        agent_data = response["agent"]
        registration = MoltbookRegistration(
            api_key=agent_data["api_key"],
            claim_url=agent_data["claim_url"],
            verification_code=agent_data["verification_code"],
            agent_name=name,
        )

        logger.info("✅ Registration successful!")
        logger.info("   Agent: %s", name)
        logger.info("   Verification Code: %s", registration.verification_code)
        logger.info("   Claim URL: %s", registration.claim_url)
        logger.warning("IMPORTANT: Save your API key and send claim URL to your human!")

        return registration

    def check_claim_status(self) -> Dict[str, Any]:
        """Check if agent has been claimed

        Returns:
            Status information
        """
        response = self._request("GET", "/agents/status")
        return response

    def get_me(self) -> MoltbookAgent:
        """Get own profile

        Returns:
            Agent profile
        """
        response = self._request("GET", "/agents/me")
        if response.get("success"):
            return MoltbookAgent(**response["agent"])
        raise Exception("Failed to get profile")

    # ==================== Posts ====================

    def create_post(
        self, submolt: str, title: str, content: Optional[str] = None, url: Optional[str] = None
    ) -> MoltbookPost:
        """Create a new post

        Args:
            submolt: Submolt name (e.g., "general")
            title: Post title
            content: Post content (optional if url provided)
            url: Link URL (optional if content provided)

        Returns:
            Created post
        """
        # Check rate limit
        can_post, minutes_remaining = self.rate_limiter.can_post(
            self.config.post_cooldown_minutes
        )
        if not can_post:
            raise Exception(
                f"Post rate limit: Please wait {minutes_remaining} minutes before posting again"
            )

        logger.info("📝 Creating post in m/{submolt}: %s", title)

        data = {"submolt_name": submolt, "title": title}
        if content:
            data["content"] = content
        if url:
            data["url"] = url

        response = self._request("POST", "/posts", data=data)

        if response.get("success"):
            self.rate_limiter.record_post()
            logger.info("   ✅ Post created successfully!")
            post_data = response.get("post", {})

            # Handle verification challenge (Moltbook anti-spam)
            verification = post_data.get("verification")
            if verification and post_data.get("verificationStatus") == "pending":
                logger.info("Verification challenge required...")
                self._verifier.solve_and_submit(
                    verification_code=verification["verification_code"],
                    challenge_text=verification["challenge_text"],
                )

            # Normalize: API returns submolt as object or string in create response
            if isinstance(post_data.get("submolt"), dict):
                post_data["submolt_name"] = post_data["submolt"].get("name")
                post_data.pop("submolt", None)
            return MoltbookPost(**post_data)

        raise Exception(f"Failed to create post: {response.get('error')}")

    def get_feed(
        self, sort: str = "hot", limit: int = 25, submolt: Optional[str] = None
    ) -> List[MoltbookPost]:
        """Get feed (personalized or global)

        Args:
            sort: Sort order ("hot", "new", "top", "rising")
            limit: Number of posts to fetch
            submolt: Optional submolt filter

        Returns:
            List of posts
        """
        endpoint = "/feed" if not submolt else f"/submolts/{submolt}/feed"
        params = {"sort": sort, "limit": limit}

        response = self._request("GET", endpoint, params=params)

        if response.get("success"):
            posts = [MoltbookPost(**post) for post in response.get("posts", [])]
            return posts

        return []

    def get_post(self, post_id: str) -> Optional[MoltbookPost]:
        """Get a single post

        Args:
            post_id: Post ID

        Returns:
            Post or None
        """
        try:
            response = self._request("GET", f"/posts/{post_id}")
            if response.get("success"):
                return MoltbookPost(**response["post"])
        except:
            pass
        return None

    def delete_post(self, post_id: str) -> bool:
        """Delete own post

        Args:
            post_id: Post ID

        Returns:
            True if deleted
        """
        try:
            response = self._request("DELETE", f"/posts/{post_id}")
            return response.get("success", False)
        except:
            return False

    # ==================== Comments ====================

    def create_comment(
        self, post_id: str, content: str, parent_id: Optional[str] = None
    ) -> MoltbookComment:
        """Create a comment

        Args:
            post_id: Post ID to comment on
            content: Comment content
            parent_id: Parent comment ID for replies

        Returns:
            Created comment
        """
        # Check rate limit
        can_comment, reason = self.rate_limiter.can_comment(
            self.config.comment_cooldown_seconds, self.config.max_comments_per_day
        )
        if not can_comment:
            raise Exception(f"Comment rate limit: {reason}")

        logger.info("💬 Creating comment on post %s", post_id)

        data = {"content": content}
        if parent_id:
            data["parent_id"] = parent_id

        response = self._request("POST", f"/posts/{post_id}/comments", data=data)

        if response.get("success"):
            self.rate_limiter.record_comment()
            logger.info("   ✅ Comment created successfully!")
            comment_data = response["comment"]
            comment = MoltbookComment(**comment_data)

            # Handle verification challenge (Moltbook anti-spam)
            verification = comment_data.get("verification")
            if verification and comment_data.get("verificationStatus") == "pending":
                logger.info("Verification challenge required...")
                self._verifier.solve_and_submit(
                    verification_code=verification["verification_code"],
                    challenge_text=verification["challenge_text"],
                )

            return comment

        raise Exception(f"Failed to create comment: {response.get('error')}")

    def get_comments(
        self, post_id: str, sort: str = "top"
    ) -> List[MoltbookComment]:
        """Get comments on a post

        Args:
            post_id: Post ID
            sort: Sort order ("top", "new", "controversial")

        Returns:
            List of comments
        """
        params = {"sort": sort}
        response = self._request("GET", f"/posts/{post_id}/comments", params=params)

        if response.get("success"):
            return [MoltbookComment(**c) for c in response.get("comments", [])]

        return []

    # ==================== Voting ====================

    def upvote_post(self, post_id: str) -> bool:
        """Upvote a post

        Args:
            post_id: Post ID

        Returns:
            True if successful
        """
        try:
            response = self._request("POST", f"/posts/{post_id}/upvote")
            if response.get("success"):
                logger.info("👍 Upvoted post %s", post_id)

                # Check for follow suggestion
                if not response.get("already_following") and response.get("suggestion"):
                    logger.info("   💡 %s", response['suggestion'])

            return response.get("success", False)
        except Exception as e:
            logger.info("Failed to upvote: %s", e)
            return False

    def downvote_post(self, post_id: str) -> bool:
        """Downvote a post

        Args:
            post_id: Post ID

        Returns:
            True if successful
        """
        try:
            response = self._request("POST", f"/posts/{post_id}/downvote")
            return response.get("success", False)
        except:
            return False

    def upvote_comment(self, comment_id: str) -> bool:
        """Upvote a comment

        Args:
            comment_id: Comment ID

        Returns:
            True if successful
        """
        try:
            response = self._request("POST", f"/comments/{comment_id}/upvote")
            return response.get("success", False)
        except:
            return False

    # ==================== Social ====================

    def follow_agent(self, agent_name: str) -> bool:
        """Follow another molty

        Args:
            agent_name: Agent name to follow

        Returns:
            True if successful
        """
        try:
            response = self._request("POST", f"/agents/{agent_name}/follow")
            if response.get("success"):
                logger.info("🦞 Followed %s", agent_name)
            return response.get("success", False)
        except Exception as e:
            logger.info("Failed to follow {agent_name}: %s", e)
            return False

    def unfollow_agent(self, agent_name: str) -> bool:
        """Unfollow a molty

        Args:
            agent_name: Agent name to unfollow

        Returns:
            True if successful
        """
        try:
            response = self._request("DELETE", f"/agents/{agent_name}/follow")
            return response.get("success", False)
        except:
            return False

    def subscribe_submolt(self, submolt_name: str) -> bool:
        """Subscribe to a submolt

        Args:
            submolt_name: Submolt name

        Returns:
            True if successful
        """
        try:
            response = self._request("POST", f"/submolts/{submolt_name}/subscribe")
            if response.get("success"):
                logger.info("📬 Subscribed to m/%s", submolt_name)
            return response.get("success", False)
        except Exception as e:
            logger.info("Failed to subscribe: %s", e)
            return False

    # ==================== Search ====================

    def semantic_search(
        self, query: str, type: str = "all", limit: int = 20
    ) -> List[MoltbookSearchResult]:
        """Perform semantic search

        Args:
            query: Search query (natural language)
            type: "posts", "comments", or "all"
            limit: Maximum results

        Returns:
            List of search results
        """
        params = {"q": query, "type": type, "limit": limit}
        response = self._request("GET", "/search", params=params)

        if response.get("success"):
            results = [
                MoltbookSearchResult(**r) for r in response.get("results", [])
            ]
            logger.info("Found %d results for: %s", len(results), query)
            return results

        return []

    # ==================== Submolts ====================

    def list_submolts(self) -> List[MoltbookSubmolt]:
        """List all submolts

        Returns:
            List of submolts
        """
        response = self._request("GET", "/submolts")
        if response.get("success"):
            return [MoltbookSubmolt(**s) for s in response.get("submolts", [])]
        return []

    def get_submolt(self, submolt_name: str) -> Optional[MoltbookSubmolt]:
        """Get submolt info

        Args:
            submolt_name: Submolt name

        Returns:
            Submolt or None
        """
        try:
            response = self._request("GET", f"/submolts/{submolt_name}")
            if response.get("success"):
                return MoltbookSubmolt(**response["submolt"])
        except:
            pass
        return None

    # ==================== Profile ====================

    def get_agent_profile(self, agent_name: str) -> Optional[MoltbookAgent]:
        """Get another agent's profile

        Args:
            agent_name: Agent name

        Returns:
            Agent profile or None
        """
        try:
            response = self._request("GET", f"/agents/profile", params={"name": agent_name})
            if response.get("success"):
                return MoltbookAgent(**response["agent"])
        except:
            pass
        return None
