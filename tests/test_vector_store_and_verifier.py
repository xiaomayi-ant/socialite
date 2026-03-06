# -*- coding: utf-8 -*-
"""Regression tests for Qdrant result handling and verifier solver selection."""

from types import SimpleNamespace

import pytest

from moltbook.verifier import ChallengeVerifier
from social_memory.config import VectorStoreConfig
from social_memory.vector_store import VectorStorageManager


@pytest.mark.asyncio
async def test_semantic_search_handles_query_response_points():
    manager = VectorStorageManager(
        VectorStoreConfig(
            collection_name="social_posts",
            storage_strategy="memory",
        )
    )
    manager._initialized = True
    manager.client = SimpleNamespace(
        query_points=lambda **kwargs: SimpleNamespace(
            points=[
                SimpleNamespace(id="post-1", score=0.99, payload={"title": "hello"}),
            ]
        )
    )

    results = await manager.semantic_search([0.1, 0.2], collection_type="posts", limit=1)

    assert results == [
        {"id": "post-1", "score": 0.99, "payload": {"title": "hello"}},
    ]


def test_verifier_uses_configured_openai_solver_directly(monkeypatch):
    verifier = ChallengeVerifier(api_key="test-key", openai_api_key="openai-key")
    monkeypatch.setenv("MOLTBOOK_VERIFIER_PROVIDER", "openai")

    monkeypatch.setattr(
        verifier,
        "_solve_with_openai",
        lambda challenge_text, retries=2: "24.00",
    )
    monkeypatch.setattr(
        verifier,
        "_submit_answer",
        lambda verification_code, answer: (True, None),
    )

    assert verifier.solve_and_submit("code-1", "obfuscated challenge") is True
