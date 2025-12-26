"""Custom exceptions for SesAI platform."""


class SesAIError(Exception):
    """Base exception for all SesAI platform errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class STTError(SesAIError):
    """Speech-to-Text processing error.

    Raised when transcription fails due to audio quality issues,
    model errors, or processing timeouts.
    """

    pass


class TTSError(SesAIError):
    """Text-to-Speech synthesis error.

    Raised when speech synthesis fails due to API errors,
    network issues, or invalid text input.
    """

    pass


class LLMError(SesAIError):
    """Large Language Model generation error.

    Raised when LLM generation fails due to API errors,
    context length issues, or invalid prompts.
    """

    pass


class AudioError(SesAIError):
    """Audio processing error.

    Raised when audio resampling, format conversion,
    or buffer operations fail.
    """

    pass


class TelephonyError(SesAIError):
    """Telephony/AudioSocket error.

    Raised when connection handling, audio streaming,
    or protocol errors occur.
    """

    pass


class PipelineError(SesAIError):
    """Pipeline orchestration error.

    Raised when pipeline initialization, execution,
    or state transitions fail.
    """

    pass
