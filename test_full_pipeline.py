#!/usr/bin/env python3
"""Full pipeline test: STT → LLM → TTS (Turkish)."""

import time
import os
import asyncio
import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from openai import AsyncOpenAI

def test_full_pipeline():
    print("=" * 70)
    print("Full Pipeline Test: STT → LLM → TTS")
    print("=" * 70)

    # Check GPU
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # =========================================================================
    # Step 1: Load STT Model
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 1: Loading STT Model (faster-whisper)")
    print("-" * 70)

    start = time.time()
    stt_model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    print(f"STT model loaded in {time.time() - start:.1f}s")

    # =========================================================================
    # Step 2: Load TTS Model
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 2: Loading TTS Model (Chatterbox)")
    print("-" * 70)

    start = time.time()
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    tts_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    print(f"TTS model loaded in {time.time() - start:.1f}s")

    # =========================================================================
    # Step 3: Initialize LLM Client
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 3: Initializing LLM Client (Qwen3 via vLLM)")
    print("-" * 70)

    llm_client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    print("LLM client initialized")

    # =========================================================================
    # Step 4: Full Pipeline Test
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 4: Running Full Pipeline")
    print("-" * 70)

    # Use existing Turkish audio file as input
    input_audio = "/input/output_tr_2.wav"  # "Size nasıl yardımcı olabilirim?"

    if not os.path.exists(input_audio):
        print(f"Error: Input audio not found: {input_audio}")
        return False

    print(f"\nInput audio: {input_audio}")

    # --- STT ---
    print("\n[STT] Transcribing...")
    start = time.time()
    segments, info = stt_model.transcribe(input_audio, language="tr", beam_size=5)
    user_text = " ".join(seg.text.strip() for seg in segments)
    stt_time = time.time() - start
    print(f"[STT] Transcribed: {user_text}")
    print(f"[STT] Time: {stt_time:.2f}s")

    # --- LLM ---
    print("\n[LLM] Generating response...")

    async def get_llm_response(text):
        response = await llm_client.chat.completions.create(
            model="Qwen/Qwen3-8B-AWQ",
            messages=[
                {"role": "system", "content": "Sen yardımcı bir Türkçe asistansın. Kısa ve öz cevaplar ver."},
                {"role": "user", "content": text}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content

    start = time.time()
    llm_response = asyncio.run(get_llm_response(user_text))
    llm_time = time.time() - start
    print(f"[LLM] Response: {llm_response}")
    print(f"[LLM] Time: {llm_time:.2f}s")

    # --- TTS ---
    print("\n[TTS] Synthesizing speech...")
    start = time.time()
    wav = tts_model.generate(llm_response, language_id="tr")
    tts_time = time.time() - start

    # Save output
    wav_np = wav.cpu().numpy().squeeze().astype(np.float32)
    output_path = "/output/pipeline_response.wav"
    os.makedirs("/output", exist_ok=True)
    sf.write(output_path, wav_np, 24000, subtype='PCM_16')

    duration = len(wav_np) / 24000
    print(f"[TTS] Duration: {duration:.2f}s")
    print(f"[TTS] Time: {tts_time:.2f}s")
    print(f"[TTS] RTF: {tts_time/duration:.2f}x")
    print(f"[TTS] Saved to: {output_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Pipeline Summary")
    print("=" * 70)
    print(f"User said: {user_text}")
    print(f"AI response: {llm_response}")
    print(f"\nLatency breakdown:")
    print(f"  STT: {stt_time*1000:.0f}ms")
    print(f"  LLM: {llm_time*1000:.0f}ms")
    print(f"  TTS: {tts_time*1000:.0f}ms")
    print(f"  TOTAL: {(stt_time + llm_time + tts_time)*1000:.0f}ms")

    if torch.cuda.is_available():
        mem_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak GPU memory: {mem_used:.2f} GB")

    print("=" * 70)
    return True

if __name__ == "__main__":
    test_full_pipeline()
