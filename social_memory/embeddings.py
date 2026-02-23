import logging
logger = logging.getLogger(__name__)
import asyncio
import os
import requests
from typing import List, Optional
from pathlib import Path


class BGEEmbeddingModel:
    """BGE-M3 embedding model using Ollama API"""

    def __init__(
        self,
        model_name: str = "bge-m3",
        api_url: str = "http://localhost:11434/api/embeddings",
        dimension: int = 1024,
    ):
        self.model_name = model_name
        self.api_url = api_url
        self.dimension = dimension

    def encode(self, text: str) -> List[float]:
        """Get embedding for text

        Args:
            text: Text to encode

        Returns:
            Embedding vector
        """
        try:
            response = requests.post(
                self.api_url,
                json={"model": self.model_name, "prompt": text},
                timeout=30,
            )

            if response.status_code == 200:
                embedding = response.json().get("embedding", [])
                if len(embedding) != self.dimension:
                    logger.info(
                        f"Warning: Expected {self.dimension} dimensions, got {len(embedding)}"
                    )
                return embedding
            else:
                logger.info(f"Error: {response.status_code}, {response.text}")
                return [0.0] * self.dimension

        except Exception as e:
            logger.info(f"Error getting embedding: {e}")
            return [0.0] * self.dimension

    async def encode_async(self, text: str) -> List[float]:
        """Get embedding for text asynchronously (non-blocking)

        Args:
            text: Text to encode

        Returns:
            Embedding vector
        """
        return await asyncio.to_thread(self.encode, text)

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors
        """
        return [self.encode(text) for text in texts]

    async def encode_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts asynchronously

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors
        """
        return await asyncio.to_thread(self.encode_batch, texts)
