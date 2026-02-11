"""
Embedding Service Module

Handles embedding generation using OpenAI embedding models.
"""

from typing import List
from openai import OpenAI
from src.config.settings import EMBEDDING_MODEL_NAME


class EmbeddingService:
    """
    A service class responsible for generating embeddings
    for given text inputs using OpenAI models.
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialize EmbeddingService with OpenAI API key.

        Args:
            api_key (str): OpenAI API key.
        """

        self.client: OpenAI = OpenAI(api_key=api_key)

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text input.

        Args:
            text (str): Input text.

        Returns:
            List[float]: Embedding vector.
        """

        response = self.client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=text)

        embedding_vector: List[float] = response.data[0].embedding

        return embedding_vector

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts (List[str]): List of text inputs.

        Returns:
            List[List[float]]: List of embedding vectors.
        """

        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL_NAME, input=texts
        )

        embedding_vectors: List[List[float]] = [
            item.embedding for item in response.data
        ]

        return embedding_vectors
