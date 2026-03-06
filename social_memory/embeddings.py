import asyncio
import logging
import os
import time
import requests
from typing import List, Optional

logger = logging.getLogger(__name__)


class BaseEmbeddingModel:
    """Common interface for embedding providers."""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def _fallback_embedding(self) -> List[float]:
        return [0.0] * self.dimension

    def encode(self, text: str) -> List[float]:
        raise NotImplementedError

    async def encode_async(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.encode, text)

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(text) for text in texts]

    async def encode_batch_async(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.encode_batch, texts)


class OllamaBGEEmbeddingModel(BaseEmbeddingModel):
    """BGE embedding model using Ollama API."""

    def __init__(
        self,
        model_name: str = "bge-m3",
        api_url: str = "http://localhost:11434/api/embeddings",
        dimension: int = 1024,
        timeout: int = 30,
    ):
        super().__init__(dimension=dimension)
        self.model_name = model_name
        self.api_url = api_url
        self.timeout = timeout

    def encode(self, text: str) -> List[float]:
        """Get embedding for text from local Ollama endpoint."""
        try:
            response = requests.post(
                self.api_url,
                json={"model": self.model_name, "prompt": text},
                timeout=self.timeout,
            )

            if response.status_code == 200:
                embedding = response.json().get("embedding", [])
                if len(embedding) != self.dimension:
                    logger.info(
                        f"Warning: Expected {self.dimension} dimensions, got {len(embedding)}"
                    )
                    return self._fallback_embedding()
                return [float(x) for x in embedding]
            else:
                logger.info(f"Error: {response.status_code}, {response.text}")
                return self._fallback_embedding()

        except Exception as e:
            logger.info(f"Error getting embedding: {e}")
            return self._fallback_embedding()


class SiliconFlowBGEEmbeddingModel(BaseEmbeddingModel):
    """BGE embedding model using SiliconFlow API."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "BAAI/bge-m3",
        api_url: str = "https://api.siliconflow.cn/v1/embeddings",
        dimension: int = 1024,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        use_env_proxy: Optional[bool] = None,
    ):
        super().__init__(dimension=dimension)
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        if use_env_proxy is None:
            env_value = os.getenv("EMBEDDING_USE_ENV_PROXY", "").strip().lower()
            self.use_env_proxy = env_value in {"1", "true", "yes", "on"}
        else:
            self.use_env_proxy = use_env_proxy

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.trust_env = self.use_env_proxy
        return session

    def _retry_wait_time(self, attempt: int, status_code: int) -> float:
        if status_code in (403, 429):
            return self.backoff_factor * (3 ** attempt) * 10
        return self.backoff_factor * (2 ** attempt)

    def encode(self, text: str) -> List[float]:
        """Get embedding for text from SiliconFlow API."""
        if not self.api_key:
            logger.info("SiliconFlow API key is missing, fallback to zero vector")
            return self._fallback_embedding()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "input": text,
            "encoding_format": "float",
        }

        session = self._create_session()
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    response = session.post(
                        self.api_url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()

                    result = response.json()
                    data = result.get("data", [])
                    if not data:
                        logger.info("SiliconFlow response missing data field")
                        return self._fallback_embedding()

                    embedding = data[0].get("embedding", [])
                    if len(embedding) != self.dimension:
                        logger.info(
                            "Warning: Expected %s dimensions, got %s",
                            self.dimension,
                            len(embedding),
                        )
                        return self._fallback_embedding()

                    return [float(x) for x in embedding]
                except requests.exceptions.HTTPError as exc:
                    response = exc.response
                    status_code = response.status_code if response is not None else 0
                    trace_id = ""
                    response_body = ""
                    if response is not None:
                        trace_id = response.headers.get("x-siliconcloud-trace-id", "")
                        response_body = (response.text or "")[:500]
                    logger.warning(
                        "SiliconFlow embedding request failed status=%s trace_id=%s via_proxy=%s attempt=%s/%s body=%s",
                        status_code,
                        trace_id or "-",
                        self.use_env_proxy,
                        attempt + 1,
                        self.max_retries + 1,
                        response_body or "-",
                    )
                    if status_code in {403, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                        wait_time = self._retry_wait_time(attempt, status_code)
                        logger.warning(
                            "Retrying SiliconFlow embedding after %.0fs (HTTP %s)",
                            wait_time,
                            status_code,
                        )
                        time.sleep(wait_time)
                        continue
                    raise
        except Exception as e:
            logger.info("Error getting SiliconFlow embedding: %s", e)
            return self._fallback_embedding()
        finally:
            session.close()


class BGEEmbeddingModel(OllamaBGEEmbeddingModel):
    """Backward-compatible alias for existing local BGE setup."""


def create_embedding_model(
    provider: str = "ollama",
    model_name: str = "bge-m3",
    dimension: int = 1024,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseEmbeddingModel:
    """Factory for embedding model providers.

    Supported providers:
    - ollama: local Ollama embedding API
    - siliconflow: cloud embedding API
    """
    provider_name = (provider or "ollama").strip().lower()

    if provider_name == "siliconflow":
        return SiliconFlowBGEEmbeddingModel(
            api_key=api_key or "",
            model_name=model_name or "BAAI/bge-m3",
            api_url=api_url or "https://api.siliconflow.cn/v1/embeddings",
            dimension=dimension,
        )

    return OllamaBGEEmbeddingModel(
        model_name=model_name or "bge-m3",
        api_url=api_url or "http://localhost:11434/api/embeddings",
        dimension=dimension,
    )
