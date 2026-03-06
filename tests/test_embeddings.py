# -*- coding: utf-8 -*-
"""Tests for embedding provider request behavior and observability."""

import logging

import requests

from social_memory.embeddings import SiliconFlowBGEEmbeddingModel


class DummySession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.trust_env = True
        self.calls = 0
        self.closed = False

    def post(self, *args, **kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, text="", headers=None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} error",
                response=self,
            )

    def json(self):
        return self._json_data


def test_siliconflow_embedding_disables_env_proxy_by_default():
    model = SiliconFlowBGEEmbeddingModel(
        api_key="key",
        dimension=3,
    )
    session = model._create_session()

    assert session.trust_env is False
    assert model.use_env_proxy is False
    session.close()


def test_siliconflow_embedding_retries_403_and_logs_context(monkeypatch, caplog):
    model = SiliconFlowBGEEmbeddingModel(
        api_key="key",
        dimension=3,
        max_retries=1,
        backoff_factor=0,
        use_env_proxy=False,
    )
    session = DummySession(
        [
            DummyResponse(
                status_code=403,
                text='{"message":"forbidden"}',
                headers={"x-siliconcloud-trace-id": "trace-123"},
            ),
            DummyResponse(
                json_data={"data": [{"embedding": [0.1, 0.2, 0.3]}]},
            ),
        ]
    )
    monkeypatch.setattr(model, "_create_session", lambda: session)
    monkeypatch.setattr("social_memory.embeddings.time.sleep", lambda _: None)

    with caplog.at_level(logging.WARNING):
        embedding = model.encode("hello")

    assert embedding == [0.1, 0.2, 0.3]
    assert session.calls == 2
    assert "status=403" in caplog.text
    assert "trace_id=trace-123" in caplog.text
    assert "via_proxy=False" in caplog.text


def test_siliconflow_embedding_falls_back_after_retry_exhausted(monkeypatch):
    model = SiliconFlowBGEEmbeddingModel(
        api_key="key",
        dimension=3,
        max_retries=1,
        backoff_factor=0,
        use_env_proxy=True,
    )
    session = DummySession(
        [
            DummyResponse(status_code=503, text="unavailable"),
            DummyResponse(status_code=503, text="still unavailable"),
        ]
    )
    monkeypatch.setattr(model, "_create_session", lambda: session)
    monkeypatch.setattr("social_memory.embeddings.time.sleep", lambda _: None)

    embedding = model.encode("hello")

    assert embedding == [0.0, 0.0, 0.0]
    assert session.calls == 2
