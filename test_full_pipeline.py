#!/usr/bin/env python3
"""Full Pipeline Test: Edge TTS + STT Container + vLLM"""

import asyncio
import time
import aiohttp
import edge_tts
from openai import AsyncOpenAI

# Configuration
STT_URL = "http://192.168.1.11:8001/transcribe"
VLLM_URL = "http://192.168.1.11:8000/v1"
TTS_VOICE = "tr-TR-EmelNeural"

async def test_full_pipeline(user_input: str):
    """Test the complete voice pipeline."""
    print("=" * 60)
    print("FULL PIPELINE TEST")
    print("=" * 60)

    total_start = time.time()

    # Step 1: Generate user audio (simulate user speaking)
    print(f"\n[1] User says: \"{user_input}\"")
    print("    Generating user audio with Edge TTS...")
    step1_start = time.time()

    communicate = edge_tts.Communicate(user_input, TTS_VOICE)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    step1_time = (time.time() - step1_start) * 1000
    print(f"    User audio generated in {step1_time:.0f}ms ({len(audio_data)/1024:.1f}KB)")

    # Step 2: STT transcription
    print("\n[2] STT: Transcribing audio...")
    step2_start = time.time()

    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('file', audio_data, filename='audio.wav', content_type='audio/wav')
        async with session.post(STT_URL, data=data) as resp:
            result = await resp.json()
            transcribed_text = result["text"]

    step2_time = (time.time() - step2_start) * 1000
    print(f"    Transcribed: \"{transcribed_text}\"")
    print(f"    STT completed in {step2_time:.0f}ms")

    # Step 3: LLM response
    print("\n[3] LLM: Generating response...")
    step3_start = time.time()

    client = AsyncOpenAI(base_url=VLLM_URL, api_key="not-needed")

    messages = [
        {"role": "system", "content": "Sen Türkçe konuşan yardımcı bir asistansın. Kısa ve öz cevaplar ver."},
        {"role": "user", "content": transcribed_text}
    ]

    response = await client.chat.completions.create(
        model="Qwen/Qwen3-8B-AWQ",
        messages=messages,
        max_tokens=100,
        temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    assistant_response = response.choices[0].message.content
    step3_time = (time.time() - step3_start) * 1000
    print(f"    Response: \"{assistant_response}\"")
    print(f"    LLM completed in {step3_time:.0f}ms")

    # Step 4: TTS synthesis
    print("\n[4] TTS: Synthesizing response...")
    step4_start = time.time()
    first_chunk_time = None

    communicate = edge_tts.Communicate(assistant_response, TTS_VOICE)
    response_audio = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            if first_chunk_time is None:
                first_chunk_time = (time.time() - step4_start) * 1000
            response_audio += chunk["data"]

    step4_time = (time.time() - step4_start) * 1000
    print(f"    First audio chunk in {first_chunk_time:.0f}ms")
    print(f"    TTS completed in {step4_time:.0f}ms ({len(response_audio)/1024:.1f}KB)")

    # Summary
    total_time = (time.time() - total_start) * 1000

    # Calculate voice-to-voice latency (STT + LLM + TTS first chunk)
    voice_to_voice = step2_time + step3_time + first_chunk_time

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Step 1 (User TTS):     {step1_time:>6.0f}ms (simulation only)")
    print(f"  Step 2 (STT):          {step2_time:>6.0f}ms")
    print(f"  Step 3 (LLM):          {step3_time:>6.0f}ms")
    print(f"  Step 4 (Response TTS): {step4_time:>6.0f}ms (first chunk: {first_chunk_time:.0f}ms)")
    print("-" * 60)
    print(f"  VOICE-TO-VOICE:        {voice_to_voice:>6.0f}ms (STT + LLM + TTS first)")
    print(f"  TOTAL PIPELINE:        {total_time:>6.0f}ms")
    print("=" * 60)

    return voice_to_voice

async def main():
    # Test with various inputs
    test_inputs = [
        "Merhaba, nasılsın?",
        "Bugün hava nasıl?",
    ]

    results = []
    for text in test_inputs:
        latency = await test_full_pipeline(text)
        results.append(latency)
        print()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    avg = sum(results) / len(results)
    print(f"  Average Voice-to-Voice: {avg:.0f}ms")
    print(f"  Target: <500ms")
    print(f"  Status: {'✅ TARGET MET!' if avg < 500 else '⚠️ NEEDS OPTIMIZATION'}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
