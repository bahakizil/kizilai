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

### LLM Comparison

| Model | TTFT (isolated) | TTFT (pipeline) | VRAM |
|-------|-----------------|-----------------|------|
| Qwen3-8B-AWQ | 137ms | 1200ms (contention) | ~8GB |
| **Qwen3-4B-AWQ** | **75ms** | **102ms** ✅ | ~3GB |

**Qwen3-4B-AWQ Optimizations:**
- Chunked prefill enabled
- AWQ marlin backend
- max_num_batched_tokens=2048

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

### Configuration v2 (Latest)
- **STT:** CPU (small model, int8)
- **LLM:** GPU (Qwen3-4B-AWQ with chunked prefill)
- **TTS:** Microsoft Cloud (Edge TTS)

### Results Comparison
| Config | STT | LLM TTFT | TTS | V2V Total |
|--------|-----|----------|-----|-----------|
| GPU Contention | 400ms | 1200ms | 400ms | **2100ms** |
| Isolated (8B) | 625ms | 137ms | 517ms | **1279ms** |
| **Isolated (4B)** | 604ms | **102ms** | 456ms | **1161ms** |

**Total Improvement: 45% faster (939ms saved from original)**

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

1. **Current (MVP):** 1161ms latency, Qwen3-4B-AWQ ✅
2. **Next:** Cloud STT (AssemblyAI ~90ms) → ~650ms V2V
3. **Then:** Cartesia Sonic TTS (~50ms) → ~400ms V2V
4. **Future:** Streaming pipeline + sentence chunking → <300ms V2V

### Latency Breakdown to <500ms Target
```
Current:    STT(604) + LLM(102) + TTS(456) = 1161ms
With Cloud: STT(100) + LLM(100) + TTS(450) =  650ms
With Sonic: STT(100) + LLM(100) + TTS(50)  =  250ms ✅
```

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
