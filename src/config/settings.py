"""Application settings using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Speech-to-Text (STT) Configuration
    # -------------------------------------------------------------------------
    stt_model: str = "large-v3-turbo"
    stt_device: Literal["cuda", "cpu"] = "cuda"
    stt_compute_type: Literal["int8", "float16", "float32"] = "int8"

    # -------------------------------------------------------------------------
    # Text-to-Speech (TTS) Configuration - Chatterbox
    # -------------------------------------------------------------------------
    tts_device: Literal["cuda", "cpu"] = "cuda"
    tts_language: str = "tr"
    tts_reference_audio: str | None = None  # Path to reference audio for voice cloning

    # -------------------------------------------------------------------------
    # Large Language Model (LLM) Configuration
    # -------------------------------------------------------------------------
    vllm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen3-8B-AWQ"
    llm_max_tokens: int = 256
    llm_temperature: float = 0.7

    # -------------------------------------------------------------------------
    # Audio Configuration
    # -------------------------------------------------------------------------
    telephony_sample_rate: int = 8000  # 8kHz for telephony
    model_sample_rate: int = 16000  # 16kHz for STT model
    tts_sample_rate: int = 24000  # 24kHz from Chatterbox TTS
    frame_duration_ms: int = 20  # 20ms frames

    # -------------------------------------------------------------------------
    # Telephony (AudioSocket) Configuration
    # -------------------------------------------------------------------------
    audiosocket_host: str = "0.0.0.0"
    audiosocket_port: int = 8765

    # -------------------------------------------------------------------------
    # Agent Configuration
    # -------------------------------------------------------------------------
    system_prompt: str = (
        "Sen yardimci bir Turkce musteri hizmetleri asistanisin. "
        "Nazik ve profesyonel bir sekilde kullanicilara yardimci ol."
    )

    # -------------------------------------------------------------------------
    # Logging Configuration
    # -------------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    @property
    def frame_size_bytes(self) -> int:
        """Calculate frame size in bytes for telephony audio (16-bit mono)."""
        samples_per_frame = int(self.telephony_sample_rate * self.frame_duration_ms / 1000)
        return samples_per_frame * 2  # 16-bit = 2 bytes per sample


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
