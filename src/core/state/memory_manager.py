"""
Memory Manager Module

Handles short-term conversational memory (STM).
"""

from typing import List, Dict
from src.config.settings import MAX_SHORT_TERM_MEMORY


class MemoryManager:
    """
    Manages short-term conversational memory.
    """

    def __init__(self) -> None:
        """
        Initialize memory storage.
        """
        self.memory: List[Dict[str, str]] = []

    def add_interaction(self, user_query: str, assistant_response: str) -> None:
        """
        Add new interaction to memory.

        Args:
            user_query (str): User question.
            assistant_response (str): Assistant answer.
        """

        self.memory.append({"user": user_query, "assistant": assistant_response})

        # Enforce memory limit
        if len(self.memory) > MAX_SHORT_TERM_MEMORY:
            self.memory.pop(0)

    def get_memory_context(self) -> str:
        """
        Build formatted conversation history for LLM.

        Returns:
            str: Conversation context string.
        """

        conversation_lines: List[str] = []

        for interaction in self.memory:
            conversation_lines.append(f"User: {interaction['user']}")
            conversation_lines.append(f"Assistant: {interaction['assistant']}")

        return "\n".join(conversation_lines)

    def clear_memory(self) -> None:
        """
        Clear entire memory.
        """
        self.memory.clear()
