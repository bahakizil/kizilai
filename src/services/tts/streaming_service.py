#!/usr/bin/env python3
"""Streaming TTS Services for real-time synthesis."""

import asyncio
import io
import time
from typing import AsyncIterator, Optional

from src.core.streaming_pipeline import AudioChunk, StreamingTTSService
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EdgeTTSStreamingService(StreamingTTSService):
    """Streaming TTS using Microsoft Edge TTS (free, cloud-based).

    Pros:
    - Very fast (cloud-based)
    - High quality Turkish voices
    - No GPU required
    - Native streaming support

    Cons:
    - Requires internet
    - Limited customization
    """

    def __init__(
        self,
        voice: str = "tr-TR-EmelNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncIterator[AudioChunk]:
        """Synthesize text and yield audio chunks as they arrive."""
        try:
            import edge_tts

            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                pitch=self.pitch,
            )

            timestamp = time.time()
            chunk_count = 0

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunk_count += 1
                    yield AudioChunk(
                        data=chunk["data"],
                        sample_rate=24000,  # Edge TTS uses 24kHz
                        timestamp=timestamp,
                        is_final=False,
                    )

            # Mark last chunk as final
            if chunk_count > 0:
                logger.debug("tts_stream_complete", chunks=chunk_count)

        except ImportError:
            logger.error("edge_tts_not_installed")
            raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")
        except Exception as e:
            logger.error("tts_stream_error", error=str(e))
            raise

    async def synthesize(self, text: str) -> bytes:
        """Synthesize complete audio (non-streaming)."""
        chunks = []
        async for chunk in self.synthesize_stream(text):
            chunks.append(chunk.data)
        return b"".join(chunks)

    @staticmethod
    def available_voices() -> list[dict]:
        """List available Turkish voices."""
        return [
            {"name": "tr-TR-EmelNeural", "gender": "Female", "style": "default"},
            {"name": "tr-TR-AhmetNeural", "gender": "Male", "style": "default"},
        ]


class ChatterboxStreamingService(StreamingTTSService):
    """Streaming TTS using Chatterbox (local GPU).

    Note: Chatterbox doesn't natively support streaming, so we
    synthesize the complete audio and chunk it for output.
    For true streaming, consider using Edge TTS.
    """

    def __init__(
        self,
        device: str = "cuda",
        language_id: str = "tr",
        chunk_duration_ms: int = 100,
    ):
        self.device = device
        self.language_id = language_id
        self.chunk_duration_ms = chunk_duration_ms
        self.model = None

    async def _ensure_model(self):
        """Lazy load the model."""
        if self.model is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            self.model = ChatterboxMultilingualTTS.from_pretrained(
                device=self.device
            )

    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncIterator[AudioChunk]:
        """Synthesize and stream audio in chunks."""
        await self._ensure_model()

        # Generate complete audio (Chatterbox doesn't stream)
        loop = asyncio.get_event_loop()
        wav = await loop.run_in_executor(
            None,
            lambda: self.model.generate(text, language_id=self.language_id)
        )

        # Convert to numpy and chunk
        import numpy as np
        wav_np = wav.cpu().numpy().squeeze()

        sample_rate = 24000
        samples_per_chunk = int(sample_rate * self.chunk_duration_ms / 1000)
        timestamp = time.time()

        # Yield chunks
        for i in range(0, len(wav_np), samples_per_chunk):
            chunk_data = wav_np[i:i + samples_per_chunk]

            # Convert to 16-bit PCM bytes
            audio_bytes = (chunk_data * 32767).astype(np.int16).tobytes()

            is_final = (i + samples_per_chunk) >= len(wav_np)
            yield AudioChunk(
                data=audio_bytes,
                sample_rate=sample_rate,
                timestamp=timestamp,
                is_final=is_final,
            )

    async def synthesize(self, text: str) -> bytes:
        """Synthesize complete audio."""
        chunks = []
        async for chunk in self.synthesize_stream(text):
            chunks.append(chunk.data)
        return b"".join(chunks)
