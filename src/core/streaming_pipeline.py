#!/usr/bin/env python3
"""Streaming Pipeline for real-time voice conversations.

This module implements a streaming architecture:
- Audio chunks → STT → partial transcripts
- Transcripts → LLM → streaming tokens
- Tokens → sentence chunker → TTS → audio chunks

Target latency: < 500ms first audio response
"""

import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Callable, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


class StreamState(Enum):
    """Pipeline stream states."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass
class AudioChunk:
    """Audio data chunk."""
    data: bytes
    sample_rate: int
    timestamp: float
    is_final: bool = False


@dataclass
class TranscriptChunk:
    """Partial or final transcript."""
    text: str
    is_final: bool
    confidence: float = 0.0


@dataclass
class TextChunk:
    """LLM text chunk for TTS."""
    text: str
    is_sentence_end: bool
    is_final: bool


class SentenceChunker:
    """Buffers LLM tokens and yields complete sentences for TTS."""

    def __init__(self, min_chars: int = 10):
        self.buffer = ""
        self.min_chars = min_chars
        # Turkish sentence endings
        self.sentence_end_pattern = re.compile(r'[.!?;:]\s*$')

    def add_token(self, token: str) -> Optional[str]:
        """Add token and return sentence if complete."""
        self.buffer += token

        # Check for sentence end
        if len(self.buffer) >= self.min_chars:
            if self.sentence_end_pattern.search(self.buffer):
                sentence = self.buffer.strip()
                self.buffer = ""
                return sentence

        return None

    def flush(self) -> Optional[str]:
        """Flush remaining buffer."""
        if self.buffer.strip():
            sentence = self.buffer.strip()
            self.buffer = ""
            return sentence
        return None


class StreamingPipeline:
    """Real-time streaming voice pipeline.

    Architecture:
    ```
    Audio Input → STT Service → LLM Service → TTS Service → Audio Output
         ↓            ↓              ↓              ↓
      chunks    transcripts      tokens        audio chunks
    ```
    """

    def __init__(
        self,
        stt_service,
        llm_service,
        tts_service,
        system_prompt: str = "Sen yardımcı bir Türkçe asistansın.",
        on_state_change: Optional[Callable[[StreamState], None]] = None,
    ):
        self.stt = stt_service
        self.llm = llm_service
        self.tts = tts_service
        self.system_prompt = system_prompt
        self.on_state_change = on_state_change

        self._state = StreamState.IDLE
        self._cancel_event = asyncio.Event()
        self._conversation_history = []

    @property
    def state(self) -> StreamState:
        return self._state

    def _set_state(self, new_state: StreamState) -> None:
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            logger.info("pipeline_state_change", old=old_state.value, new=new_state.value)
            if self.on_state_change:
                self.on_state_change(new_state)

    async def process_audio_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[AudioChunk]:
        """Process incoming audio and yield response audio.

        This is the main pipeline entry point.
        """
        self._set_state(StreamState.LISTENING)
        self._cancel_event.clear()

        try:
            # Collect audio and transcribe
            transcript = await self._stream_stt(audio_stream)

            if not transcript or self._cancel_event.is_set():
                return

            logger.info("user_transcript", text=transcript)
            self._conversation_history.append({
                "role": "user",
                "content": transcript
            })

            # Generate and stream response
            self._set_state(StreamState.PROCESSING)

            async for audio_chunk in self._stream_response(transcript):
                if self._cancel_event.is_set():
                    logger.info("response_interrupted")
                    self._set_state(StreamState.INTERRUPTED)
                    break

                self._set_state(StreamState.SPEAKING)
                yield audio_chunk

        except asyncio.CancelledError:
            logger.info("pipeline_cancelled")
            raise

        finally:
            self._set_state(StreamState.IDLE)

    async def _stream_stt(self, audio_stream: AsyncIterator[AudioChunk]) -> str:
        """Stream audio to STT and return final transcript."""
        full_transcript = ""

        async for transcript_chunk in self.stt.transcribe_stream(audio_stream):
            if transcript_chunk.is_final:
                full_transcript = transcript_chunk.text
                break

        return full_transcript

    async def _stream_response(self, user_text: str) -> AsyncIterator[AudioChunk]:
        """Generate LLM response and stream TTS audio."""
        sentence_chunker = SentenceChunker()
        assistant_response = ""

        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._conversation_history)

        # Stream LLM tokens
        async for token in self.llm.generate_stream(messages):
            if self._cancel_event.is_set():
                break

            assistant_response += token

            # Check for complete sentence
            sentence = sentence_chunker.add_token(token)
            if sentence:
                logger.debug("sentence_complete", text=sentence)

                # Synthesize sentence immediately
                async for audio_chunk in self.tts.synthesize_stream(sentence):
                    if self._cancel_event.is_set():
                        break
                    yield audio_chunk

        # Flush remaining text
        remaining = sentence_chunker.flush()
        if remaining and not self._cancel_event.is_set():
            async for audio_chunk in self.tts.synthesize_stream(remaining):
                yield audio_chunk

        # Save assistant response to history
        if assistant_response:
            self._conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })

    def interrupt(self) -> None:
        """Interrupt current response (barge-in)."""
        logger.info("interrupt_requested")
        self._cancel_event.set()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()


class StreamingSTTService:
    """Base class for streaming STT services."""

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptChunk]:
        """Transcribe audio stream, yielding partial results."""
        raise NotImplementedError


class StreamingLLMService:
    """Base class for streaming LLM services."""

    async def generate_stream(
        self,
        messages: list[dict]
    ) -> AsyncIterator[str]:
        """Generate response, yielding tokens."""
        raise NotImplementedError


class StreamingTTSService:
    """Base class for streaming TTS services."""

    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncIterator[AudioChunk]:
        """Synthesize text, yielding audio chunks."""
        raise NotImplementedError
