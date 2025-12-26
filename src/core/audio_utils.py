"""Audio processing utilities."""

import numpy as np
from scipy import signal

from src.utils.exceptions import AudioError
from src.utils.logging import get_logger

logger = get_logger(__name__)


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to target sample rate.

    Args:
        audio: Input audio samples
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio

    if len(audio) == 0:
        return audio

    try:
        num_samples = int(len(audio) * target_sr / orig_sr)
        resampled = signal.resample(audio, num_samples)
        return resampled.astype(audio.dtype)
    except Exception as e:
        raise AudioError(f"Resampling failed: {e}") from e


def bytes_to_numpy(
    audio_bytes: bytes,
    sample_width: int = 2,
    normalize: bool = True,
) -> np.ndarray:
    """Convert raw audio bytes to numpy array.

    Args:
        audio_bytes: Raw PCM audio bytes
        sample_width: Bytes per sample (2 for 16-bit)
        normalize: If True, normalize to [-1, 1] float32

    Returns:
        Audio as numpy array
    """
    if not audio_bytes:
        return np.array([], dtype=np.float32 if normalize else np.int16)

    if sample_width == 2:
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
    elif sample_width == 1:
        audio = np.frombuffer(audio_bytes, dtype=np.int8).astype(np.int16) * 256
    else:
        raise AudioError(f"Unsupported sample width: {sample_width}")

    if normalize:
        return audio.astype(np.float32) / 32768.0
    return audio


def numpy_to_bytes(
    audio: np.ndarray,
    sample_width: int = 2,
    denormalize: bool = True,
) -> bytes:
    """Convert numpy array to raw audio bytes.

    Args:
        audio: Audio samples (float32 normalized or int16)
        sample_width: Bytes per sample (2 for 16-bit)
        denormalize: If True, assume input is float32 [-1, 1]

    Returns:
        Raw PCM bytes
    """
    if len(audio) == 0:
        return b""

    if denormalize and audio.dtype == np.float32:
        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    elif audio.dtype != np.int16:
        audio = audio.astype(np.int16)

    return audio.tobytes()


def calculate_rms(audio: np.ndarray) -> float:
    """Calculate RMS (root mean square) of audio.

    Useful for basic volume/energy detection.

    Args:
        audio: Audio samples (normalized float32)

    Returns:
        RMS value (0.0 to ~1.0 for normalized audio)
    """
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio**2)))


def is_speech_present(
    audio: np.ndarray,
    threshold: float = 0.01,
) -> bool:
    """Simple energy-based speech detection.

    For proper VAD, use Silero VAD instead.

    Args:
        audio: Audio samples (normalized float32)
        threshold: RMS threshold for speech detection

    Returns:
        True if speech is likely present
    """
    return calculate_rms(audio) > threshold


class AudioBuffer:
    """Buffer for accumulating audio chunks.

    Useful for collecting audio until enough samples
    are available for processing.
    """

    def __init__(
        self,
        sample_rate: int,
        min_duration_ms: int = 500,
        max_duration_ms: int = 30000,
    ):
        """Initialize audio buffer.

        Args:
            sample_rate: Sample rate of incoming audio
            min_duration_ms: Minimum audio duration before processing
            max_duration_ms: Maximum buffer size (will flush if exceeded)
        """
        self.sample_rate = sample_rate
        self.min_samples = int(sample_rate * min_duration_ms / 1000)
        self.max_samples = int(sample_rate * max_duration_ms / 1000)
        self._buffer: list[np.ndarray] = []
        self._total_samples = 0

    def append(self, audio: np.ndarray) -> None:
        """Add audio chunk to buffer."""
        self._buffer.append(audio)
        self._total_samples += len(audio)

    def has_enough(self) -> bool:
        """Check if buffer has minimum required audio."""
        return self._total_samples >= self.min_samples

    def is_full(self) -> bool:
        """Check if buffer is at maximum capacity."""
        return self._total_samples >= self.max_samples

    def get_audio(self) -> np.ndarray:
        """Get all buffered audio as single array."""
        if not self._buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._total_samples = 0

    def flush(self) -> np.ndarray:
        """Get audio and clear buffer."""
        audio = self.get_audio()
        self.clear()
        return audio

    @property
    def duration_ms(self) -> float:
        """Current buffer duration in milliseconds."""
        return (self._total_samples / self.sample_rate) * 1000

    def __len__(self) -> int:
        """Total samples in buffer."""
        return self._total_samples
