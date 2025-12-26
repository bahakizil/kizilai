#!/usr/bin/env python3
"""End-to-end test for SesAI platform."""
import asyncio
import edge_tts
from openai import AsyncOpenAI


async def test_llm():
    """Test LLM connection."""
    print("Testing LLM...")
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    response = await client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        messages=[{"role": "user", "content": "Merhaba!"}],
        max_tokens=30
    )
    result = response.choices[0].message.content
    print(f"  LLM Response: {result}")
    return True


async def test_tts():
    """Test TTS synthesis."""
    print("Testing TTS...")
    communicate = edge_tts.Communicate("Merhaba, ben SesAI.", "tr-TR-EmelNeural")
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    print(f"  TTS Generated: {len(audio_data)} bytes audio")
    return len(audio_data) > 0


async def test_stt():
    """Test STT import (actual transcription needs audio file)."""
    print("Testing STT imports...")
    from faster_whisper import WhisperModel
    print("  STT imports OK (model loading skipped)")
    return True


async def main():
    print("=" * 50)
    print("SesAI End-to-End Test")
    print("=" * 50)

    results = {}

    try:
        results["LLM"] = await test_llm()
    except Exception as e:
        print(f"  LLM Error: {e}")
        results["LLM"] = False

    try:
        results["TTS"] = await test_tts()
    except Exception as e:
        print(f"  TTS Error: {e}")
        results["TTS"] = False

    try:
        results["STT"] = await test_stt()
    except Exception as e:
        print(f"  STT Error: {e}")
        results["STT"] = False

    print("=" * 50)
    print("Results:")
    for component, status in results.items():
        status_str = "PASS" if status else "FAIL"
        print(f"  {component}: {status_str}")

    all_passed = all(results.values())
    print("=" * 50)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
