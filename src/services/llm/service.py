"""Large Language Model service using vLLM/OpenAI-compatible API."""

from dataclasses import dataclass, field
from typing import AsyncIterator

from openai import AsyncOpenAI

from src.utils.exceptions import LLMError
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """Chat message for LLM conversation."""

    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class ConversationContext:
    """Manages conversation history for a call."""

    system_prompt: str
    messages: list[Message] = field(default_factory=list)
    max_history: int = 10

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append(Message(role=role, content=content))

        # Trim history if too long (keep system + recent messages)
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history :]

    def get_messages(self) -> list[Message]:
        """Get all messages including system prompt."""
        return [Message(role="system", content=self.system_prompt)] + self.messages

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()


class LLMService:
    """LLM service using vLLM with OpenAI-compatible API.

    Connects to a vLLM server running Qwen3-8B-AWQ
    for Turkish language support and streaming generation.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-8B-AWQ",
    ):
        """Initialize the LLM service.

        Args:
            base_url: vLLM server URL
            model: Model identifier
        """
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed",  # vLLM doesn't require API key
        )
        self.model = model
        logger.info("llm_service_initialized", model=model, base_url=base_url)

    async def generate(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            messages: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = response.choices[0].message.content or ""

            logger.debug(
                "generation_complete",
                input_messages=len(messages),
                output_length=len(content),
            )

            return content

        except Exception as e:
            logger.error("generation_failed", error=str(e))
            raise LLMError(f"LLM generation failed: {e}") from e

    async def generate_stream(
        self,
        messages: list[Message],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Stream generated tokens from the LLM.

        Args:
            messages: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Generated text chunks
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("stream_generation_failed", error=str(e))
            raise LLMError(f"Stream generation failed: {e}") from e

    async def generate_with_context(
        self,
        context: ConversationContext,
        user_input: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate response and update conversation context.

        Convenience method that handles context management.

        Args:
            context: Conversation context to use and update
            user_input: User's message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated assistant response
        """
        # Add user message to context
        context.add_message("user", user_input)

        # Generate response
        response = await self.generate(
            messages=context.get_messages(),
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Add assistant response to context
        context.add_message("assistant", response)

        return response

    async def health_check(self) -> bool:
        """Check if the LLM service is available.

        Returns:
            True if service is healthy
        """
        try:
            await self.client.models.list()
            return True
        except Exception as e:
            logger.warning("llm_health_check_failed", error=str(e))
            return False
