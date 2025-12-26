#!/usr/bin/env python3
"""Streaming LLM Service using vLLM/OpenAI-compatible API."""

from typing import AsyncIterator

from openai import AsyncOpenAI

from src.core.streaming_pipeline import StreamingLLMService
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VLLMStreamingService(StreamingLLMService):
    """Streaming LLM service using vLLM's OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-8B-AWQ",
        max_tokens: int = 150,
        temperature: float = 0.7,
        enable_thinking: bool = False,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking

    async def generate_stream(
        self,
        messages: list[dict]
    ) -> AsyncIterator[str]:
        """Generate response tokens in streaming mode.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Yields:
            Individual tokens as they are generated
        """
        try:
            # Prepare extra_body for Qwen3 thinking control
            extra_body = {}
            if not self.enable_thinking:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
                extra_body=extra_body if extra_body else None,
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    yield token

        except Exception as e:
            logger.error("llm_stream_error", error=str(e))
            raise

    async def generate(self, messages: list[dict]) -> str:
        """Generate complete response (non-streaming)."""
        tokens = []
        async for token in self.generate_stream(messages):
            tokens.append(token)
        return "".join(tokens)
