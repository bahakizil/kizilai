"""Tests for audio utility functions."""

import numpy as np
import pytest

from src.core.audio_utils import (
    AudioBuffer,
    bytes_to_numpy,
    calculate_rms,
    is_speech_present,
    numpy_to_bytes,
    resample_audio,
)


class TestResampleAudio:
    """Tests for resample_audio function."""

    def test_same_sample_rate(self):
        """Should return same array when sample rates match."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = resample_audio(audio, 16000, 16000)
        np.testing.assert_array_equal(result, audio)

    def test_downsample(self):
        """Should downsample 16kHz to 8kHz."""
        audio = np.ones(1600, dtype=np.float32)  # 100ms at 16kHz
        result = resample_audio(audio, 16000, 8000)
        assert len(result) == 800  # 100ms at 8kHz

    def test_upsample(self):
        """Should upsample 8kHz to 16kHz."""
        audio = np.ones(800, dtype=np.float32)  # 100ms at 8kHz
        result = resample_audio(audio, 8000, 16000)
        assert len(result) == 1600  # 100ms at 16kHz

    def test_empty_array(self):
        """Should handle empty array."""
        audio = np.array([], dtype=np.float32)
        result = resample_audio(audio, 16000, 8000)
        assert len(result) == 0


class TestBytesToNumpy:
    """Tests for bytes_to_numpy function."""

    def test_convert_16bit(self):
        """Should convert 16-bit PCM to normalized float32."""
        # Max 16-bit value
        audio_bytes = np.array([32767], dtype=np.int16).tobytes()
        result = bytes_to_numpy(audio_bytes, sample_width=2)
        assert result.dtype == np.float32
        assert abs(result[0] - 1.0) < 0.001

    def test_convert_silence(self):
        """Should convert silence correctly."""
        audio_bytes = np.zeros(100, dtype=np.int16).tobytes()
        result = bytes_to_numpy(audio_bytes)
        np.testing.assert_array_almost_equal(result, np.zeros(100))

    def test_empty_bytes(self):
        """Should handle empty bytes."""
        result = bytes_to_numpy(b"")
        assert len(result) == 0


class TestNumpyToBytes:
    """Tests for numpy_to_bytes function."""

    def test_convert_normalized(self):
        """Should convert normalized float32 to 16-bit PCM."""
        audio = np.array([1.0, -1.0, 0.0], dtype=np.float32)
        result = numpy_to_bytes(audio)

        # Convert back to verify
        back = np.frombuffer(result, dtype=np.int16)
        assert back[0] == 32767  # Max positive
        assert back[1] == -32767  # Max negative (clipped)
        assert back[2] == 0  # Zero

    def test_empty_array(self):
        """Should handle empty array."""
        result = numpy_to_bytes(np.array([], dtype=np.float32))
        assert result == b""


class TestCalculateRMS:
    """Tests for calculate_rms function."""

    def test_silence(self):
        """Should return 0 for silence."""
        audio = np.zeros(100, dtype=np.float32)
        assert calculate_rms(audio) == 0.0

    def test_constant_signal(self):
        """Should calculate RMS correctly for constant signal."""
        audio = np.ones(100, dtype=np.float32) * 0.5
        assert abs(calculate_rms(audio) - 0.5) < 0.001

    def test_empty_array(self):
        """Should return 0 for empty array."""
        assert calculate_rms(np.array([])) == 0.0


class TestIsSpeechPresent:
    """Tests for is_speech_present function."""

    def test_silence(self):
        """Should return False for silence."""
        audio = np.zeros(100, dtype=np.float32)
        assert is_speech_present(audio, threshold=0.01) is False

    def test_loud_signal(self):
        """Should return True for loud signal."""
        audio = np.ones(100, dtype=np.float32) * 0.5
        assert is_speech_present(audio, threshold=0.01) is True


class TestAudioBuffer:
    """Tests for AudioBuffer class."""

    def test_append_and_get(self):
        """Should accumulate audio chunks."""
        buffer = AudioBuffer(sample_rate=16000, min_duration_ms=100)

        chunk1 = np.ones(800, dtype=np.float32)  # 50ms
        chunk2 = np.ones(800, dtype=np.float32)  # 50ms

        buffer.append(chunk1)
        buffer.append(chunk2)

        audio = buffer.get_audio()
        assert len(audio) == 1600

    def test_has_enough(self):
        """Should detect when minimum duration is reached."""
        buffer = AudioBuffer(sample_rate=16000, min_duration_ms=100)

        buffer.append(np.ones(800, dtype=np.float32))  # 50ms
        assert buffer.has_enough() is False

        buffer.append(np.ones(800, dtype=np.float32))  # +50ms = 100ms
        assert buffer.has_enough() is True

    def test_flush(self):
        """Should return audio and clear buffer."""
        buffer = AudioBuffer(sample_rate=16000, min_duration_ms=100)
        buffer.append(np.ones(1600, dtype=np.float32))

        audio = buffer.flush()
        assert len(audio) == 1600
        assert len(buffer) == 0

    def test_duration_ms(self):
        """Should calculate duration correctly."""
        buffer = AudioBuffer(sample_rate=16000, min_duration_ms=100)
        buffer.append(np.ones(1600, dtype=np.float32))  # 100ms
        assert abs(buffer.duration_ms - 100.0) < 0.1
