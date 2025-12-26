"""Turkish Speech-to-Text service using faster-whisper."""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from faster_whisper import WhisperModel

from src.utils.exceptions import STTError
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptionResult:
    """Result of speech transcription."""

    text: str
    confidence: float
    is_final: bool
    language: str = "tr"


class TurkishSTTService:
    """Turkish Speech-to-Text service using faster-whisper.

    Uses the large-v3-turbo model with Turkish language setting.
    INT8 quantization for GPU efficiency.
    """

    def __init__(
        self,
        model_path: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "int8",
    ):
        """Initialize the STT service.

        Args:
            model_path: Hugging Face model path or local path
            device: Device to use ('cuda' or 'cpu')
            compute_type: Quantization type ('int8', 'float16', 'float32')
        """
        logger.info(
            "initializing_stt_service",
            model=model_path,
            device=device,
            compute_type=compute_type,
        )

        try:
            self.model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute_type,
            )
            logger.info("stt_service_initialized")
        except Exception as e:
            logger.error("stt_initialization_failed", error=str(e))
            raise STTError(f"Failed to initialize STT model: {e}") from e

    def transcribe(
        self,
        audio: np.ndarray,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio samples as float32 numpy array, normalized to [-1, 1]
            beam_size: Beam size for decoding
            vad_filter: Whether to apply VAD filtering

        Returns:
            TranscriptionResult with transcribed text and confidence
        """
        try:
            segments, info = self.model.transcribe(
                audio,
                language="tr",
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters=dict(
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=300,
                ),
            )

            # Collect all segments
            segment_list = list(segments)

            if not segment_list:
                return TranscriptionResult(
                    text="",
                    confidence=0.0,
                    is_final=True,
                )

            # Combine text and calculate average confidence
            text = " ".join(seg.text.strip() for seg in segment_list)
            avg_confidence = np.mean([seg.avg_logprob for seg in segment_list])

            # Convert log probability to a 0-1 confidence score
            # avg_logprob is typically between -1 and 0, closer to 0 = higher confidence
            confidence = float(np.exp(avg_confidence))

            logger.debug(
                "transcription_complete",
                text=text[:50] + "..." if len(text) > 50 else text,
                confidence=confidence,
                segments=len(segment_list),
            )

            return TranscriptionResult(
                text=text,
                confidence=confidence,
                is_final=True,
            )

        except Exception as e:
            logger.error("transcription_failed", error=str(e))
            raise STTError(f"Transcription failed: {e}") from e

    def transcribe_segments(
        self,
        audio: np.ndarray,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> Iterator[TranscriptionResult]:
        """Transcribe audio and yield results segment by segment.

        Useful for streaming scenarios where you want to process
        results as they become available.

        Args:
            audio: Audio samples as float32 numpy array
            beam_size: Beam size for decoding
            vad_filter: Whether to apply VAD filtering

        Yields:
            TranscriptionResult for each detected speech segment
        """
        try:
            segments, _ = self.model.transcribe(
                audio,
                language="tr",
                beam_size=beam_size,
                vad_filter=vad_filter,
            )

            for segment in segments:
                confidence = float(np.exp(segment.avg_logprob))
                yield TranscriptionResult(
                    text=segment.text.strip(),
                    confidence=confidence,
                    is_final=True,
                )

        except Exception as e:
            logger.error("segment_transcription_failed", error=str(e))
            raise STTError(f"Segment transcription failed: {e}") from e
