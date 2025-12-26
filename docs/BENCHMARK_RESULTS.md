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

## Optimization: Isolated Resources (TESTED ✅)

Moving STT to CPU eliminates GPU contention:

### Configuration
- **STT:** CPU (small model, int8)
- **LLM:** GPU (Qwen3-8B-AWQ)
- **TTS:** Microsoft Cloud (Edge TTS)

### Results
| Component | Before (Contention) | After (Isolated) |
|-----------|---------------------|------------------|
| STT | 400ms (GPU) | 625ms (CPU small) |
| LLM TTFT | 1200ms ❌ | **137ms** ✅ |
| TTS | 400ms | 517ms |
| **V2V Total** | **2100ms** | **1279ms** |

**Improvement: 39% faster (821ms saved)**

### STT Model Comparison (CPU)
| Model | Processing Time | Accuracy |
|-------|----------------|----------|
| large-v3-turbo | 2240ms | Perfect |
| medium | 2090ms | Perfect |
| small | **625ms** | Good |

### Key Insight
GPU contention was the bottleneck, not individual component speed.
LLM TTFT improved from 1200ms to 137ms (9x faster) by isolating resources.

## Further Optimization Options

### Option 1: Streaming Architecture
- Start TTS on first LLM sentence
- Expected: ~800-900ms to first audio

### Option 2: Cloud STT (Azure/Google)
- ~100ms latency
- Expected: ~650ms V2V

### Option 3: Smaller LLM
- Qwen2.5-3B or Phi-3-mini
- Expected: ~1000ms V2V with current setup

## Recommended Path Forward

1. **Current (MVP):** 1279ms latency, isolated resources ✅
2. **Next:** Implement streaming (LLM → TTS overlap)
3. **Future:** Cloud STT for <500ms target

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
