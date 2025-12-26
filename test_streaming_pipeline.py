#!/usr/bin/env python3
"""Streaming Pipeline Test: LLM streaming + sentence chunking + Edge TTS"""

import asyncio
import time
import aiohttp
import edge_tts
from openai import AsyncOpenAI
import re

# Configuration
STT_URL = "http://192.168.1.11:8001/transcribe"
VLLM_URL = "http://192.168.1.11:8000/v1"
TTS_VOICE = "tr-TR-EmelNeural"

def split_sentences(text: str) -> list:
    """Split text into sentences."""
    # Turkish sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

async def test_streaming_pipeline(user_input: str):
    """Test streaming pipeline with sentence chunking."""
    print("=" * 60)
    print("STREAMING PIPELINE TEST")
    print("=" * 60)

    total_start = time.time()

    # Step 1: Generate user audio
    print(f"\n[1] User says: \"{user_input}\"")
    step1_start = time.time()

    communicate = edge_tts.Communicate(user_input, TTS_VOICE)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    step1_time = (time.time() - step1_start) * 1000
    print(f"    Audio generated in {step1_time:.0f}ms")

    # Step 2: STT
    print("\n[2] STT transcribing...")
    step2_start = time.time()

    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('file', audio_data, filename='audio.wav', content_type='audio/wav')
        async with session.post(STT_URL, data=data) as resp:
            result = await resp.json()
            transcribed_text = result["text"]

    step2_time = (time.time() - step2_start) * 1000
    print(f"    Result: \"{transcribed_text}\" ({step2_time:.0f}ms)")

    # Step 3: Streaming LLM + TTS
    print("\n[3] STREAMING: LLM -> Sentence Buffer -> TTS")
    step3_start = time.time()
    first_audio_time = None
    first_token_time = None

    client = AsyncOpenAI(base_url=VLLM_URL, api_key="not-needed")

    messages = [
        {"role": "system", "content": "Sen Türkçe konuşan yardımcı bir asistansın. 2-3 cümle ile kısa cevaplar ver."},
        {"role": "user", "content": transcribed_text}
    ]

    # Stream LLM response
    buffer = ""
    full_response = ""
    sentences_spoken = 0

    stream = await client.chat.completions.create(
        model="Qwen/Qwen3-8B-AWQ",
        messages=messages,
        max_tokens=150,
        temperature=0.7,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            if first_token_time is None:
                first_token_time = (time.time() - step3_start) * 1000
                print(f"    First token in {first_token_time:.0f}ms")

            buffer += token
            full_response += token

            # Check for complete sentences
            sentences = split_sentences(buffer)
            if len(sentences) > 1 or (buffer.strip() and buffer.strip()[-1] in '.!?'):
                for sentence in sentences[:-1] if len(sentences) > 1 else sentences:
                    sentences_spoken += 1
                    print(f"    [Sentence {sentences_spoken}] \"{sentence}\"")

                    # Start TTS for this sentence (simulated timing)
                    tts_start = time.time()
                    communicate = edge_tts.Communicate(sentence, TTS_VOICE)
                    async for audio_chunk in communicate.stream():
                        if audio_chunk["type"] == "audio":
                            if first_audio_time is None:
                                first_audio_time = (time.time() - step3_start) * 1000
                                print(f"    ** FIRST AUDIO at {first_audio_time:.0f}ms (voice-to-voice) **")
                            break  # Just measure first chunk timing

                    buffer = sentences[-1] if len(sentences) > 1 else ""

    # Handle remaining buffer
    if buffer.strip():
        sentences_spoken += 1
        print(f"    [Sentence {sentences_spoken}] \"{buffer.strip()}\"")

        tts_start = time.time()
        communicate = edge_tts.Communicate(buffer.strip(), TTS_VOICE)
        async for audio_chunk in communicate.stream():
            if audio_chunk["type"] == "audio":
                if first_audio_time is None:
                    first_audio_time = (time.time() - step3_start) * 1000
                break

    step3_time = (time.time() - step3_start) * 1000

    # Summary
    total_time = (time.time() - total_start) * 1000

    # Voice-to-voice = STT + time to first audio
    voice_to_voice = step2_time + first_audio_time

    print("\n" + "=" * 60)
    print("STREAMING RESULTS")
    print("=" * 60)
    print(f"  STT:                   {step2_time:>6.0f}ms")
    print(f"  First LLM token:       {first_token_time:>6.0f}ms")
    print(f"  First audio output:    {first_audio_time:>6.0f}ms (from LLM start)")
    print("-" * 60)
    print(f"  VOICE-TO-VOICE:        {voice_to_voice:>6.0f}ms (STT + first audio)")
    print(f"  Total sentences:       {sentences_spoken}")
    print(f"  Full response: \"{full_response}\"")
    print("=" * 60)

    return voice_to_voice

async def main():
    test_inputs = [
        "Merhaba, nasılsın?",
        "Türkiye'nin başkenti neresi?",
    ]

    results = []
    for text in test_inputs:
        latency = await test_streaming_pipeline(text)
        results.append(latency)
        print()

    print("\n" + "=" * 60)
    print("STREAMING FINAL SUMMARY")
    print("=" * 60)
    avg = sum(results) / len(results)
    print(f"  Average Voice-to-Voice: {avg:.0f}ms")
    print(f"  Target: <500ms")
    print(f"  Status: {'✅ TARGET MET!' if avg < 500 else '⚠️ NEEDS WORK'}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
