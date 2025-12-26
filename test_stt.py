#!/usr/bin/env python3
"""Test STT (faster-whisper) with Turkish audio."""

import time
import torch
from faster_whisper import WhisperModel

def test_stt_turkish():
    print("=" * 60)
    print("STT Test - Turkish (faster-whisper)")
    print("=" * 60)

    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("\nLoading faster-whisper model...")
    print("Model: large-v3-turbo (with Turkish language)")
    start = time.time()

    # Use CPU for testing (GPU cuDNN issues on RTX 5080)
    model = WhisperModel(
        "large-v3-turbo",
        device="cpu",
        compute_type="int8"
    )

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s")

    # Test files
    test_files = [
        ("/input/output_tr_1.wav", "Merhaba, ben yapay zeka asistanınızım."),
        ("/input/output_tr_2.wav", "Size nasıl yardımcı olabilirim?"),
        ("/input/output_tr_3.wav", "Bugün hava çok güzel, dışarı çıkmak ister misiniz?"),
    ]

    print("\n" + "-" * 60)
    print("Transcribing Turkish audio files...")
    print("-" * 60)

    total_correct = 0

    for filepath, expected in test_files:
        print(f"\nFile: {filepath}")
        print(f"Expected: {expected}")

        start = time.time()
        segments, info = model.transcribe(
            filepath,
            language="tr",
            beam_size=5,
            vad_filter=True
        )

        text = " ".join(seg.text.strip() for seg in segments)
        transcribe_time = time.time() - start

        print(f"Transcribed: {text}")
        print(f"Time: {transcribe_time:.2f}s")

        # Simple similarity check
        expected_words = set(expected.lower().replace(".", "").replace(",", "").replace("?", "").split())
        transcribed_words = set(text.lower().replace(".", "").replace(",", "").replace("?", "").split())
        overlap = len(expected_words & transcribed_words) / len(expected_words)

        print(f"Word overlap: {overlap:.0%}")
        if overlap >= 0.7:
            print("✓ PASS")
            total_correct += 1
        else:
            print("✗ FAIL")

    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {total_correct}/{len(test_files)} passed")
    print("=" * 60)

    return total_correct == len(test_files)

if __name__ == "__main__":
    test_stt_turkish()
