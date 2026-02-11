"""
Answer Generator Module

Generates grounded conversational answers using:
- Retrieved document chunks
- Short-term conversational memory
"""

from typing import List, Dict
from openai import OpenAI
from src.config.settings import LLM_MODEL_NAME
from src.core.memory_manager import MemoryManager


class AnswerGenerator:
    """
    Generates grounded conversational answers from retrieved context.
    """

    def __init__(self, api_key: str, memory_manager: MemoryManager) -> None:
        """
        Initialize AnswerGenerator.

        Args:
            api_key (str): OpenAI API key.
            memory_manager (MemoryManager): STM manager.
        """

        self.client: OpenAI = OpenAI(api_key=api_key)

        self.memory_manager: MemoryManager = memory_manager

    def generate_answer(
        self, query: str, retrieved_chunks: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Generate conversational grounded answer.

        Args:
            query (str): User question.
            retrieved_chunks (List[Dict[str, str]]): Retrieved chunks.

        Returns:
            Dict[str, str]:
                - answer
                - sources
        """

        context_text: str = self._build_context(retrieved_chunks)

        memory_context: str = self.memory_manager.get_memory_context()

        system_prompt: str = """
        You are a logistics document assistant.

        Answer the question ONLY using:
        1. The document context
        2. Relevant prior conversation (if needed)

        Rules:
        - Do NOT use outside knowledge.
        - If answer is not present in document context, say:
        "Not found in document."
        - Be concise and factual.
        """

        user_prompt: str = f"""
        Conversation History:
        {memory_context}

        Document Context:
        {context_text}

        Current Question:
        {query}
        """

        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        answer: str = response.choices[0].message.content.strip()

        # Add to memory AFTER generation
        self.memory_manager.add_interaction(query, answer)

        return {"answer": answer, "sources": context_text}

    def _build_context(self, retrieved_chunks: List[Dict[str, str]]) -> str:
        """
        Combine retrieved chunks.

        Args:
            retrieved_chunks (List[Dict[str, str]])

        Returns:
            str: Combined context
        """

        context_parts: List[str] = []

        for chunk in retrieved_chunks:
            context_parts.append(chunk.get("content", ""))

        return "\n\n".join(context_parts)
