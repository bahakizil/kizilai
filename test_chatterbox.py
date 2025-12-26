#!/usr/bin/env python3
"""Test Chatterbox TTS with Turkish language."""

import time
import os
import torch
import soundfile as sf
import numpy as np


def test_chatterbox_turkish():
    """Test Chatterbox TTS with Turkish text."""
    print("=" * 60)
    print("Chatterbox TTS - Turkish Test")
    print("=" * 60)

    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Import and load model
    print("\nLoading Chatterbox model...")
    start = time.time()

    from chatterbox.tts import ChatterboxTTS

    # Use multilingual model for Turkish support
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        use_multilingual = True
        print("Using ChatterboxMultilingualTTS (Turkish support)")
    except ImportError:
        model = ChatterboxTTS.from_pretrained(device="cuda")
        use_multilingual = False
        print("Using ChatterboxTTS (English only)")
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s")

    # Test Turkish texts
    test_texts = [
        "Merhaba, ben yapay zeka asistanınızım.",
        "Size nasıl yardımcı olabilirim?",
        "Bugün hava çok güzel, dışarı çıkmak ister misiniz?",
    ]

    print("\n" + "-" * 60)
    print("Generating Turkish speech...")
    print("-" * 60)

    for i, text in enumerate(test_texts, 1):
        print(f"\n[{i}] Text: {text}")

        start = time.time()
        if use_multilingual:
            wav = model.generate(text, language_id="tr")
        else:
            wav = model.generate(text)
        gen_time = time.time() - start

        # Save audio - ensure output dir exists
        output_dir = "/output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/output_tr_{i}.wav"

        # Convert to numpy and ensure correct format
        wav_np = wav.cpu().numpy()
        if wav_np.ndim > 1:
            wav_np = wav_np.squeeze()

        # Ensure float32 in range [-1, 1]
        wav_np = wav_np.astype(np.float32)

        print(f"    Audio shape: {wav_np.shape}, dtype: {wav_np.dtype}, range: [{wav_np.min():.3f}, {wav_np.max():.3f}]")

        sf.write(output_path, wav_np, 24000, subtype='PCM_16')

        duration = len(wav_np) / 24000
        rtf = gen_time / duration  # Real-time factor

        print(f"    Duration: {duration:.2f}s")
        print(f"    Generation time: {gen_time:.2f}s")
        print(f"    RTF: {rtf:.2f}x (< 1 = faster than real-time)")
        print(f"    Saved to: {output_path}")

    # Memory usage
    if torch.cuda.is_available():
        mem_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak GPU memory used: {mem_used:.2f} GB")

    print("\n" + "=" * 60)
    print("Chatterbox Turkish test completed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_chatterbox_turkish()
