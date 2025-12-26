#!/usr/bin/env python3
"""Edge TTS Service - Microsoft's free cloud-based TTS.

Features:
- Very fast (cloud-based, no GPU needed)
- High quality Turkish voices
- Native streaming support
- SSML support for fine control

Available Turkish voices:
- tr-TR-EmelNeural (Female, recommended)
- tr-TR-AhmetNeural (Male)
"""

import asyncio
import io
import os
import tempfile
from typing import AsyncIterator, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


class EdgeTTSService:
    """Edge TTS Service for Turkish speech synthesis.

    This is a cloud-based TTS that doesn't require GPU.
    Very fast latency (~100-200ms for first audio).
    """

    # Available Turkish voices
    VOICES = {
        "female": "tr-TR-EmelNeural",
        "male": "tr-TR-AhmetNeural",
    }

    def __init__(
        self,
        voice: str = "tr-TR-EmelNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ):
        """Initialize Edge TTS service.

        Args:
            voice: Voice name (default: tr-TR-EmelNeural)
            rate: Speech rate adjustment (e.g., "+10%", "-5%")
            pitch: Pitch adjustment (e.g., "+10Hz", "-5Hz")
            volume: Volume adjustment (e.g., "+10%", "-5%")
        """
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.volume = volume
        self._edge_tts = None

    async def _ensure_import(self):
        """Lazy import edge_tts."""
        if self._edge_tts is None:
            try:
                import edge_tts
                self._edge_tts = edge_tts
            except ImportError:
                raise RuntimeError(
                    "edge-tts not installed. Run: pip install edge-tts"
                )

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to complete audio bytes.

        Args:
            text: Turkish text to synthesize

        Returns:
            MP3 audio bytes
        """
        await self._ensure_import()

        communicate = self._edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
        )

        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        return audio_data

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio chunks as they are generated.

        Args:
            text: Turkish text to synthesize

        Yields:
            Audio chunks (MP3 format)
        """
        await self._ensure_import()

        communicate = self._edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def synthesize_to_file(
        self,
        text: str,
        output_path: str
    ) -> str:
        """Synthesize text and save to file.

        Args:
            text: Turkish text to synthesize
            output_path: Path to save the audio file

        Returns:
            Path to the saved file
        """
        await self._ensure_import()

        communicate = self._edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
        )

        await communicate.save(output_path)
        return output_path

    async def synthesize_with_timing(
        self,
        text: str
    ) -> tuple[bytes, list[dict]]:
        """Synthesize with word-level timing information.

        Args:
            text: Turkish text to synthesize

        Returns:
            Tuple of (audio_bytes, word_timings)
        """
        await self._ensure_import()

        communicate = self._edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
        )

        audio_data = b""
        word_timings = []

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
            elif chunk["type"] == "WordBoundary":
                word_timings.append({
                    "text": chunk["text"],
                    "offset": chunk["offset"],
                    "duration": chunk["duration"],
                })

        return audio_data, word_timings

    def set_voice(self, gender: str = "female") -> None:
        """Set voice by gender.

        Args:
            gender: "female" or "male"
        """
        if gender.lower() in self.VOICES:
            self.voice = self.VOICES[gender.lower()]
        else:
            raise ValueError(f"Unknown gender: {gender}. Use 'female' or 'male'")

    @classmethod
    async def list_voices(cls, language: str = "tr") -> list[dict]:
        """List available voices for a language.

        Args:
            language: Language code (e.g., "tr" for Turkish)

        Returns:
            List of voice info dictionaries
        """
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            return [v for v in voices if v["Locale"].startswith(language)]
        except ImportError:
            return []


# Convenience function
async def synthesize_turkish(
    text: str,
    voice: str = "female",
    output_path: Optional[str] = None
) -> bytes:
    """Quick helper to synthesize Turkish text.

    Args:
        text: Turkish text to synthesize
        voice: "female" or "male"
        output_path: Optional path to save audio

    Returns:
        Audio bytes (MP3 format)
    """
    service = EdgeTTSService()
    service.set_voice(voice)

    if output_path:
        await service.synthesize_to_file(text, output_path)
        with open(output_path, "rb") as f:
            return f.read()
    else:
        return await service.synthesize(text)
