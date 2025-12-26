#!/usr/bin/env python3
"""Test Chatterbox voice cloning with Turkish language."""

import time
import os
import torch
import soundfile as sf
import numpy as np

def test_voice_cloning():
    """Test Chatterbox voice cloning."""
    print("=" * 60)
    print("Chatterbox Voice Cloning Test - Turkish")
    print("=" * 60)

    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load model
    print("\nLoading Chatterbox model...")
    start = time.time()

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s")

    # Reference audio for voice cloning (use one of our generated files)
    reference_audio = "/input/output_tr_1.wav"

    if not os.path.exists(reference_audio):
        print(f"\nError: Reference audio not found: {reference_audio}")
        return False

    print(f"\nReference audio: {reference_audio}")

    # Test texts for cloned voice
    test_texts = [
        "Merhaba, ben sizin yeni sesinizim.",
        "Ses klonlama testi basarili oldu.",
        "Bu cumle klonlanmis ses ile soyleniyor.",
    ]

    print("\n" + "-" * 60)
    print("Generating speech with cloned voice...")
    print("-" * 60)

    output_dir = "/output"
    os.makedirs(output_dir, exist_ok=True)

    for i, text in enumerate(test_texts, 1):
        print(f"\n[{i}] Text: {text}")

        start = time.time()

        # Generate with voice cloning
        wav = model.generate(
            text,
            language_id="tr",
            audio_prompt_path=reference_audio
        )

        gen_time = time.time() - start

        # Save audio
        output_path = f"{output_dir}/cloned_tr_{i}.wav"

        wav_np = wav.cpu().numpy()
        if wav_np.ndim > 1:
            wav_np = wav_np.squeeze()
        wav_np = wav_np.astype(np.float32)

        sf.write(output_path, wav_np, 24000, subtype='PCM_16')

        duration = len(wav_np) / 24000
        rtf = gen_time / duration

        print(f"    Duration: {duration:.2f}s")
        print(f"    Generation time: {gen_time:.2f}s")
        print(f"    RTF: {rtf:.2f}x (< 1 = faster than real-time)")
        print(f"    Saved to: {output_path}")

    # Memory usage
    if torch.cuda.is_available():
        mem_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak GPU memory used: {mem_used:.2f} GB")

    print("\n" + "=" * 60)
    print("Voice cloning test completed!")
    print("=" * 60)

    return True

if __name__ == "__main__":
    test_voice_cloning()
