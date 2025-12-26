#!/usr/bin/env python3
"""Standalone STT Service using faster-whisper on GPU."""

import asyncio
import io
import os
import time
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel

# Configuration
MODEL_SIZE = os.getenv("STT_MODEL", "large-v3-turbo")
DEVICE = os.getenv("STT_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "float16")
LANGUAGE = os.getenv("STT_LANGUAGE", "tr")
PORT = int(os.getenv("STT_PORT", "8001"))

app = FastAPI(title="SesAI STT Service", version="1.0.0")

# Global model
model: Optional[WhisperModel] = None


class TranscriptionResult(BaseModel):
    text: str
    language: str
    duration: float
    processing_time_ms: float
    segments: list


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    cuda_available: bool
    gpu_name: Optional[str]


@app.on_event("startup")
async def load_model():
    """Load the Whisper model on startup."""
    global model
    print(f"Loading STT model: {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})")
    start = time.time()

    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )

    load_time = time.time() - start
    print(f"STT model loaded in {load_time:.1f}s")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "loading",
        model=MODEL_SIZE,
        device=DEVICE,
        cuda_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )


@app.post("/transcribe", response_model=TranscriptionResult)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    beam_size: int = 5
):
    """Transcribe audio file to text."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Read audio file
    audio_data = await file.read()

    # Save to temporary file (faster-whisper needs file path)
    temp_path = f"/tmp/audio_{time.time()}.wav"
    with open(temp_path, "wb") as f:
        f.write(audio_data)

    try:
        start = time.time()

        segments, info = model.transcribe(
            temp_path,
            language=language or LANGUAGE,
            beam_size=beam_size,
            vad_filter=True
        )

        # Collect segments
        segment_list = []
        full_text = []

        for seg in segments:
            segment_list.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "avg_logprob": seg.avg_logprob
            })
            full_text.append(seg.text.strip())

        processing_time = (time.time() - start) * 1000

        return TranscriptionResult(
            text=" ".join(full_text),
            language=info.language,
            duration=info.duration,
            processing_time_ms=processing_time,
            segments=segment_list
        )

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/transcribe/bytes")
async def transcribe_bytes(audio_bytes: bytes, language: Optional[str] = None):
    """Transcribe raw audio bytes."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Save to temporary file
    temp_path = f"/tmp/audio_{time.time()}.wav"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    try:
        start = time.time()

        segments, info = model.transcribe(
            temp_path,
            language=language or LANGUAGE,
            beam_size=5,
            vad_filter=True
        )

        text = " ".join(seg.text.strip() for seg in segments)
        processing_time = (time.time() - start) * 1000

        return {
            "text": text,
            "language": info.language,
            "processing_time_ms": processing_time
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    print("=" * 60)
    print("SesAI STT Service")
    print("=" * 60)
    print(f"Model: {MODEL_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Compute Type: {COMPUTE_TYPE}")
    print(f"Language: {LANGUAGE}")
    print(f"Port: {PORT}")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=PORT)
