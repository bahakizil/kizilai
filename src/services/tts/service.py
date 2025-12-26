"""Text-to-Speech service using Chatterbox TTS (MIT License)."""

import numpy as np
import torch

from src.utils.exceptions import TTSError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ChatterboxTTSService:
    """Text-to-Speech service using Chatterbox TTS.

    Provides high-quality multilingual speech synthesis with voice cloning
    capability. Uses the ChatterboxMultilingualTTS model for Turkish support.
    MIT License - free for commercial use.
    """

    def __init__(
        self,
        device: str = "cuda",
        language: str = "tr",
        reference_audio: str | None = None,
    ):
        """Initialize the TTS service.

        Args:
            device: Device to run inference on ('cuda' or 'cpu')
            language: Target language ID (e.g., 'tr' for Turkish)
            reference_audio: Optional path to reference audio for voice cloning
        """
        self.device = device
        self.language = language
        self.reference_audio = reference_audio
        self.model = None
        self.sample_rate = 24000  # Chatterbox outputs at 24kHz

        logger.info(
            "tts_service_initialized",
            device=device,
            language=language,
            reference_audio=reference_audio,
        )

    def load_model(self) -> None:
        """Load the Chatterbox TTS model."""
        if self.model is not None:
            return

        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            logger.info("loading_chatterbox_model", device=self.device)
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            logger.info("chatterbox_model_loaded")

        except Exception as e:
            logger.error("model_load_failed", error=str(e))
            raise TTSError(f"Failed to load Chatterbox model: {e}") from e

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as float32 numpy array at 24kHz
        """
        if not text.strip():
            return np.array([], dtype=np.float32)

        if self.model is None:
            self.load_model()

        try:
            # Generate audio with optional voice cloning
            if self.reference_audio:
                wav = self.model.generate(
                    text,
                    language_id=self.language,
                    audio_prompt_path=self.reference_audio,
                )
            else:
                wav = self.model.generate(
                    text,
                    language_id=self.language,
                )

            # Convert to numpy
            wav_np = wav.cpu().numpy()
            if wav_np.ndim > 1:
                wav_np = wav_np.squeeze()

            wav_np = wav_np.astype(np.float32)

            logger.debug(
                "synthesis_complete",
                text_length=len(text),
                audio_samples=len(wav_np),
                duration_s=len(wav_np) / self.sample_rate,
            )

            return wav_np

        except Exception as e:
            logger.error("synthesis_failed", error=str(e), text=text[:50])
            raise TTSError(f"Speech synthesis failed: {e}") from e

    def synthesize_to_pcm(
        self,
        text: str,
        target_sample_rate: int = 8000,
    ) -> np.ndarray:
        """Synthesize text to PCM audio at specified sample rate.

        Useful for telephony where 8kHz PCM is required.

        Args:
            text: Text to synthesize
            target_sample_rate: Target sample rate (default 8kHz for telephony)

        Returns:
            Audio as float32 numpy array, normalized to [-1, 1]
        """
        if not text.strip():
            return np.array([], dtype=np.float32)

        try:
            # Get audio at native sample rate
            audio = self.synthesize(text)

            if len(audio) == 0:
                return audio

            # Resample if needed
            if self.sample_rate != target_sample_rate:
                from scipy import signal

                num_samples = int(len(audio) * target_sample_rate / self.sample_rate)
                audio = signal.resample(audio, num_samples)

            # Ensure float32
            audio = audio.astype(np.float32)

            logger.debug(
                "pcm_synthesis_complete",
                samples=len(audio),
                sample_rate=target_sample_rate,
            )

            return audio

        except Exception as e:
            logger.error("pcm_synthesis_failed", error=str(e))
            raise TTSError(f"PCM synthesis failed: {e}") from e

    def set_reference_audio(self, path: str | None) -> None:
        """Set the reference audio for voice cloning.

        Args:
            path: Path to reference audio file, or None to disable cloning
        """
        self.reference_audio = path
        logger.info("reference_audio_changed", path=path)

    def set_language(self, language: str) -> None:
        """Change the target language.

        Args:
            language: Language ID (e.g., 'tr', 'en', etc.)
        """
        self.language = language
        logger.info("language_changed", language=language)

    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB.

        Returns:
            GPU memory usage in GB, or 0.0 if not using GPU
        """
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
        return 0.0
