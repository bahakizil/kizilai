# SesAI Pipeline Benchmark Results

**Date:** 2025-12-26  
**Hardware:** RTX 5080 16GB (Windows 11 + WSL2 + Docker)  
**Client:** MacBook Pro (via SSH/network)

## Component Performance

### STT (faster-whisper large-v3-turbo)
| Metric | Value |
|--------|-------|
| Model Load | 17.4s |
| Cold Transcription | 6455ms (4.4s audio) |
| Warm Transcription | 263ms (4.4s audio) |
| Real-time Ratio | 17x faster than realtime |
| Accuracy | Perfect Turkish transcription |

### LLM (Qwen3-8B-AWQ via vLLM)
| Metric | Value |
|--------|-------|
| TTFT (isolated) | 156-188ms |
| TTFT (after STT) | 1100-1200ms ⚠️ |
| Cause | GPU contention |

### TTS (Edge TTS - Microsoft Cloud)
| Metric | Value |
|--------|-------|
| First Chunk | 316-450ms |
| Total Synthesis | 400-750ms |
| Voice | tr-TR-EmelNeural (Turkish Female) |

## Full Pipeline Results

### Sequential Pipeline
```
User → STT → LLM → TTS → Response

STT:        ~400ms
LLM TTFT:   ~1200ms (degraded due to GPU contention)
TTS first:  ~400ms
─────────────────────
TOTAL:      ~2100ms
Target:     <500ms
```

### GPU Contention Issue
When STT runs on GPU, LLM performance degrades:
- LLM alone: 160ms TTFT ✅
- LLM after STT: 1200ms TTFT ❌

Both models share the same GPU (RTX 5080), causing interference.

## Optimization Options

### Option 1: Accept Current Latency (MVP)
- 2100ms voice-to-voice
- Good for testing, not production

### Option 2: Separate GPU Resources
- Run STT on CPU (small-medium model)
- Keep LLM on GPU
- Expected: ~600-800ms V2V

### Option 3: Streaming Architecture
- Start TTS while LLM is generating
- Stream first sentence immediately
- Expected: ~700-1000ms to first audio

### Option 4: Cloud STT
- Use Azure Speech or Google STT
- Very fast (~100ms)
- Expected: ~500-600ms V2V

## Recommended Path Forward

1. **Short-term (MVP):** Accept 2s latency, focus on functionality
2. **Medium-term:** Implement sentence streaming (LLM → TTS)
3. **Long-term:** Consider dedicated STT GPU or cloud STT

## Individual Component Benchmarks

### Edge TTS Latency
```
Voice: tr-TR-EmelNeural
First chunk: 316-450ms
Streaming: Native support ✅
```

### STT Container Health
```json
{
  "status": "healthy",
  "model": "large-v3-turbo",
  "device": "cuda",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 5080"
}
```

## Test Commands

```bash
# STT health check
curl http://192.168.1.11:8001/health

# STT transcription
curl -X POST http://192.168.1.11:8001/transcribe -F "file=@audio.wav"

# vLLM health
curl http://192.168.1.11:8000/health

# Full pipeline test
python test_full_pipeline.py
```
