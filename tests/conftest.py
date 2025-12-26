"""Pytest configuration and fixtures for SesAI tests."""

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.config.settings import Settings
from src.services.llm.service import LLMService, Message
from src.services.stt.service import TranscriptionResult, TurkishSTTService
from src.services.tts.service import EdgeTTSService


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        stt_model="test-model",
        stt_device="cpu",
        stt_compute_type="float32",
        tts_voice="tr-TR-EmelNeural",
        vllm_base_url="http://localhost:8000/v1",
        llm_model="test-model",
        system_prompt="Sen test asistanısın.",
        log_level="DEBUG",
        log_format="console",
    )


@pytest.fixture
def mock_stt() -> MagicMock:
    """Create mock STT service."""
    stt = MagicMock(spec=TurkishSTTService)
    stt.transcribe.return_value = TranscriptionResult(
        text="Merhaba, randevu almak istiyorum",
        confidence=0.95,
        is_final=True,
    )
    return stt


@pytest.fixture
def mock_tts() -> AsyncMock:
    """Create mock TTS service."""
    tts = AsyncMock(spec=EdgeTTSService)

    # Generate some fake audio bytes
    fake_audio = np.zeros(8000, dtype=np.float32)  # 1 second of silence
    tts.synthesize_to_pcm.return_value = fake_audio
    tts.synthesize.return_value = b"\x00" * 1000

    return tts


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Create mock LLM service."""
    llm = AsyncMock(spec=LLMService)
    llm.generate.return_value = "Merhaba! Size nasıl yardımcı olabilirim?"
    llm.generate_with_context.return_value = "Merhaba! Size nasıl yardımcı olabilirim?"
    llm.health_check.return_value = True
    return llm


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Create sample audio for testing."""
    # Generate 1 second of audio at 16kHz
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    # Simple sine wave at 440Hz
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Create sample audio bytes for testing."""
    # Generate 20ms of audio at 8kHz (320 bytes)
    samples = 160
    audio = np.zeros(samples, dtype=np.int16)
    return audio.tobytes()


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
