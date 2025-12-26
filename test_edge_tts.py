#!/usr/bin/env python3
"""Test Edge TTS for Turkish speech synthesis.

Edge TTS is Microsoft's free cloud-based TTS with:
- Very fast latency (~100-200ms)
- High quality voices
- No GPU required
"""

import asyncio
import time
import os


async def test_edge_tts():
    """Test Edge TTS synthesis."""
    try:
        import edge_tts
    except ImportError:
        print("Installing edge-tts...")
        os.system("pip install edge-tts")
        import edge_tts

    print("=" * 70)
    print("Edge TTS Test - Turkish")
    print("=" * 70)

    # Test texts
    texts = [
        "Merhaba, size nasıl yardımcı olabilirim?",
        "Bu bir test cümlesidir.",
        "Türkçe konuşan yapay zeka asistanı.",
    ]

    voices = [
        ("tr-TR-EmelNeural", "Female"),
        ("tr-TR-AhmetNeural", "Male"),
    ]

    os.makedirs("output", exist_ok=True)

    for voice, gender in voices:
        print(f"\n{'='*50}")
        print(f"Voice: {voice} ({gender})")
        print("=" * 50)

        for i, text in enumerate(texts):
            print(f"\nText {i+1}: {text}")

            # Measure time
            start = time.time()

            communicate = edge_tts.Communicate(text=text, voice=voice)
            output_file = f"output/edge_tts_{gender.lower()}_{i+1}.mp3"

            await communicate.save(output_file)

            elapsed = time.time() - start
            file_size = os.path.getsize(output_file)

            print(f"  Time: {elapsed*1000:.0f}ms")
            print(f"  Size: {file_size/1024:.1f} KB")
            print(f"  File: {output_file}")

    # Streaming test
    print(f"\n{'='*50}")
    print("Streaming Test")
    print("=" * 50)

    text = "Bu bir streaming testidir. Ses parça parça gelecek."
    print(f"Text: {text}")

    start = time.time()
    first_chunk_time = None
    total_chunks = 0
    total_bytes = 0

    communicate = edge_tts.Communicate(text=text, voice="tr-TR-EmelNeural")

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            if first_chunk_time is None:
                first_chunk_time = time.time() - start
            total_chunks += 1
            total_bytes += len(chunk["data"])

    total_time = time.time() - start

    print(f"  First chunk: {first_chunk_time*1000:.0f}ms")
    print(f"  Total time: {total_time*1000:.0f}ms")
    print(f"  Chunks: {total_chunks}")
    print(f"  Total size: {total_bytes/1024:.1f} KB")

    print("\n" + "=" * 70)
    print("Edge TTS Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_edge_tts())
