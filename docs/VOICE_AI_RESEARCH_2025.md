# Voice AI Pipeline Research - December 2025

## Executive Summary

2025'te voice AI alanında üç temel mimari yaklaşım var:

| Mimari | Latency | Avantajlar | Dezavantajlar |
|--------|---------|------------|---------------|
| **Cascaded (STT→LLM→TTS)** | 500-2000ms | Modüler, esnek | Kümülatif gecikme |
| **Speech-to-Speech (S2S)** | 160-300ms | Ultra düşük latency | Yeni, sınırlı kontrol |
| **Hybrid Streaming** | 300-600ms | En iyi denge | Karmaşık implementasyon |

**Hedef:** <500ms voice-to-voice latency için **Hybrid Streaming** mimarisi önerilir.

---

## 1. Modern Mimari Yaklaşımlar

### 1.1 Cascaded Pipeline (Klasik)
```
Audio → [STT] → Text → [LLM] → Text → [TTS] → Audio
         ↓           ↓            ↓
      100-500ms   200-2000ms   200-800ms
```

**Problem:** Her aşama bir öncekinin bitmesini bekler.

### 1.2 Speech-to-Speech (OpenAI Realtime, Moshi)
```
Audio → [Single Multimodal Model] → Audio
                    ↓
               160-300ms
```

**Örnekler:**
- **OpenAI gpt-realtime:** WebSocket/WebRTC, native audio I/O
- **Moshi (Kyutai):** 160ms theoretical latency, open-source
- **Ultravox:** Audio → LLM extension, text output (şimdilik)

### 1.3 Hybrid Streaming Pipeline (ÖNERILEN)
```
Audio ──┬──→ [Streaming STT] ──┬──→ [Streaming LLM] ──┬──→ [Streaming TTS]
        │         ↓            │         ↓            │         ↓
        │    partial text      │    token stream      │    audio chunks
        └─────────────────────────────────────────────────────────────→ Audio
                              PARALLEL PROCESSING
```

**Key Insight:** Componentler paralel çalışır, birbirini beklemez!

---

## 2. En Hızlı STT Çözümleri (2025)

### 2.1 Benchmark Karşılaştırması

| Provider | Model | Latency | Özellik |
|----------|-------|---------|---------|
| **AssemblyAI** | Universal-Streaming | **90ms** | Immutable transcripts, turn detection |
| **Deepgram** | Nova-3 | 200-300ms | 54% WER reduction, streaming |
| **Deepgram** | Flux | ~200ms | Conversational, built-in turn detection |
| **faster-whisper** | large-v3-turbo | 200-500ms | Local, GPU optimized |

### 2.2 Streaming STT Nasıl Çalışır?

```python
# Chunk-based processing (100-200ms chunks)
async def streaming_stt():
    websocket = await connect("wss://api.stt.provider/stream")
    
    async for audio_chunk in microphone.stream(chunk_ms=100):
        await websocket.send(audio_chunk)
        
        # Partial results arrive immediately
        partial = await websocket.recv()
        if partial.is_final:
            yield partial.text
```

**Kritik Teknikler:**
1. **Chunked Audio:** 100-200ms parçalar halinde gönder
2. **WebSocket:** HTTP overhead yok, persistent connection
3. **Partial Transcripts:** Final olmadan bile LLM'e gönder
4. **Turn Detection:** Silence + prosodic features ile konuşma sonu tespiti

### 2.3 faster-whisper Streaming (Local)

```python
# whisper-streaming implementation
from whisper_streaming import OnlineASRProcessor

processor = OnlineASRProcessor(
    model="large-v3-turbo",
    language="tr",
    min_chunk_size=1.0,  # seconds
    local_agreement_threshold=0.7
)

# Self-adaptive latency based on speech complexity
async for audio_chunk in audio_stream:
    text, is_final = processor.process(audio_chunk)
    if text:
        yield text, is_final
```

---

## 3. En Hızlı TTS Çözümleri (2025)

### 3.1 Benchmark Karşılaştırması

| Provider | Model | TTFB | Streaming | Özellik |
|----------|-------|------|-----------|---------|
| **Cartesia** | Sonic-3 | **40-90ms** | ✅ | State-space model, ultra-fast |
| **Deepgram** | Aura-2 | <200ms | ✅ | Enterprise, reliable |
| **ElevenLabs** | Flash | ~150ms | ✅ | High quality, expensive |
| **Edge TTS** | - | 300-500ms | ✅ | Free, good quality |

### 3.2 Cartesia Sonic Neden Hızlı?

**State-Space Models (SSM):** Transformer'dan farklı, streaming için optimize edilmiş mimari.

```
Traditional Transformer:
Input → [Attention over ALL tokens] → Output
                    ↓
         O(n²) complexity, not streaming-friendly

State-Space Model (Cartesia):
Input → [State Update] → Output
             ↓
    O(n) complexity, native streaming
```

### 3.3 Streaming TTS Implementation

```python
async def streaming_tts(text_stream):
    """Start TTS as soon as first word arrives."""
    buffer = ""
    
    async for token in text_stream:
        buffer += token
        
        # Sentence-level chunking for natural prosody
        if buffer.endswith(('.', '!', '?', ',')):
            audio_stream = await tts.synthesize_stream(buffer)
            async for audio_chunk in audio_stream:
                yield audio_chunk  # First chunk in ~40-90ms
            buffer = ""
```

---

## 4. LLM Inference Optimizasyonu

### 4.1 vLLM Optimizasyon Teknikleri

| Teknik | TTFT İyileştirme | Açıklama |
|--------|------------------|----------|
| **Chunked Prefill** | -30% | Input'u parçalara böl, erken başla |
| **Continuous Batching** | +40% throughput | GPU'yu sürekli meşgul tut |
| **Prefix Caching** | -50% (repeated) | Ortak prefix'leri cache'le |
| **Multi-Step Scheduling** | -20% overhead | CPU-GPU senkronizasyonunu azalt |
| **AWQ Quantization** | 3x speed, 3x memory | 4-bit weight quantization |

### 4.2 Qwen3-4B-AWQ Performans

```bash
# vLLM ile Qwen3-4B-AWQ
vllm serve Qwen/Qwen3-4B-AWQ \
    --quantization awq \
    --max-model-len 4096 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.9
```

**Beklenen Performans:**
- TTFT: 50-100ms (warm)
- Throughput: 100+ tokens/sec
- VRAM: ~3-4GB

### 4.3 Thinking Mode Devre Dışı (Qwen3)

```python
response = await client.chat.completions.create(
    model="Qwen/Qwen3-4B-AWQ",
    messages=messages,
    stream=True,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False}
    }
)
```

---

## 5. Turn Detection & Barge-in

### 5.1 VAD vs Turn Detection

| Özellik | VAD | Turn Detection |
|---------|-----|----------------|
| Algılar | Ses varlığı | Konuşma sonu |
| Latency | ~10ms | 200-600ms |
| Accuracy | Gürültüde düşük | Semantik anlama |
| Kullanım | Barge-in trigger | Response trigger |

### 5.2 Modern Turn Detection

```python
class IntelligentTurnDetector:
    def __init__(self):
        self.vad = SileroVAD()
        self.silence_threshold = 400  # ms, dynamic
        
    async def detect_turn_end(self, audio_stream, partial_text):
        silence_duration = 0
        
        async for chunk in audio_stream:
            if self.vad.is_speech(chunk):
                silence_duration = 0
            else:
                silence_duration += chunk.duration_ms
                
            # Dynamic threshold based on context
            threshold = self.calculate_threshold(partial_text)
            
            if silence_duration > threshold:
                # Check prosodic features (pitch drop, etc.)
                if self.is_sentence_complete(partial_text):
                    return True
        
        return False
```

### 5.3 Barge-in Handling

```python
async def handle_barge_in(self, tts_player, stt_stream):
    """Kullanıcı konuşmaya başladığında TTS'i durdur."""
    
    async for event in stt_stream:
        if event.type == "speech_start":
            # Immediately stop TTS
            await tts_player.stop()
            await tts_player.flush_buffer()
            
            # Start new transcription
            self.state = ConversationState.LISTENING
```

---

## 6. Önerilen SesAI Pipeline Mimarisi

### 6.1 Hedef Latency Breakdown

```
Target: <500ms voice-to-voice

Component         | Target  | Solution
-----------------|---------|---------------------------
STT              | 100ms   | Deepgram Flux / AssemblyAI
LLM TTFT         | 100ms   | Qwen3-4B-AWQ + vLLM optimized
TTS First Chunk  | 100ms   | Cartesia Sonic
Network/Overhead | 100ms   | Local deployment
Turn Detection   | 100ms   | Intelligent endpointing
-----------------|---------|---------------------------
TOTAL            | 500ms   |
```

### 6.2 Streaming Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Audio In ──► VAD ──► Streaming STT ──► Turn Detector             │
│                              │                 │                    │
│                              ▼                 ▼                    │
│                        Partial Text ─────► Response Trigger         │
│                              │                                      │
│                              ▼                                      │
│                    ┌─────────────────────┐                         │
│                    │   Streaming LLM     │                         │
│                    │   (token by token)  │                         │
│                    └─────────┬───────────┘                         │
│                              │                                      │
│                              ▼                                      │
│                    ┌─────────────────────┐                         │
│                    │  Sentence Chunker   │                         │
│                    │  (buffer → sentence)│                         │
│                    └─────────┬───────────┘                         │
│                              │                                      │
│                              ▼                                      │
│                    ┌─────────────────────┐                         │
│                    │   Streaming TTS     │                         │
│                    │  (sentence → audio) │                         │
│                    └─────────┬───────────┘                         │
│                              │                                      │
│                              ▼                                      │
│                         Audio Out ◄─────── Barge-in Handler         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.3 Paralel İşleme

```python
async def optimized_pipeline(audio_stream):
    """
    Key insight: Start each component AS SOON AS possible,
    don't wait for previous to complete.
    """
    
    # Start STT immediately
    stt_task = asyncio.create_task(streaming_stt(audio_stream))
    
    async for partial_text, is_final in stt_task:
        if is_final:
            # Start LLM immediately with current text
            llm_stream = streaming_llm(partial_text)
            
            sentence_buffer = ""
            async for token in llm_stream:
                sentence_buffer += token
                
                # Start TTS as soon as sentence is complete
                if is_sentence_end(sentence_buffer):
                    # Don't await! Fire and stream
                    asyncio.create_task(
                        stream_tts_to_output(sentence_buffer)
                    )
                    sentence_buffer = ""
```

---

## 7. Teknoloji Seçimleri

### 7.1 Önerilen Stack

| Component | Option A (Cloud) | Option B (Local) |
|-----------|-----------------|------------------|
| **STT** | AssemblyAI Universal (90ms) | faster-whisper streaming |
| **LLM** | - | Qwen3-4B-AWQ + vLLM |
| **TTS** | Cartesia Sonic (40ms) | - |
| **VAD** | Silero VAD | Silero VAD |
| **Framework** | Pipecat | Pipecat |

### 7.2 Maliyet Analizi

| Service | Fiyat | 1000 dakika/ay |
|---------|-------|----------------|
| AssemblyAI | $0.15/saat | $2.50 |
| Deepgram Nova-3 | $0.0059/dakika | $5.90 |
| Cartesia Sonic | $0.045/1K chars | ~$5-10 |
| Edge TTS | FREE | $0 |
| Local (GPU) | Elektrik | ~$10-20 |

---

## 8. Implementation Roadmap

### Phase 1: MVP Optimization (Now)
- [ ] Upgrade to Qwen3-4B-AWQ
- [ ] Implement sentence-level TTS streaming
- [ ] Add proper turn detection

### Phase 2: Cloud STT Integration
- [ ] Integrate AssemblyAI or Deepgram
- [ ] Implement WebSocket streaming
- [ ] Target: <600ms V2V

### Phase 3: Advanced Streaming
- [ ] Parallel STT + LLM prefill
- [ ] Predictive response caching
- [ ] Target: <400ms V2V

### Phase 4: Production
- [ ] Cartesia Sonic integration
- [ ] Full barge-in support
- [ ] Target: <300ms V2V

---

## References

### Architectures
- [Real-Time vs Turn-Based Voice Agent Architecture](https://softcery.com/lab/ai-voice-agents-real-time-vs-turn-based-tts-stt-architecture)
- [The Voice AI Stack for 2025](https://www.assemblyai.com/blog/the-voice-ai-stack-for-building-agents)
- [Pipecat Architecture](https://docs.pipecat.ai/guides/learn/overview)

### STT
- [Deepgram Nova-3](https://deepgram.com/learn/nova-2-speech-to-text-api)
- [AssemblyAI Universal-Streaming](https://www.assemblyai.com/universal-streaming)
- [faster-whisper Streaming](https://github.com/ufal/whisper_streaming)

### TTS
- [Cartesia Sonic](https://cartesia.ai/sonic)
- [TTS Voice AI Model Guide 2025](https://layercode.com/blog/tts-voice-ai-model-guide)

### LLM
- [vLLM Optimization](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [Qwen3 Speed Benchmark](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)

### Speech-to-Speech
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- [Moshi (Kyutai)](https://github.com/kyutai-labs/moshi)
- [Ultravox](https://github.com/fixie-ai/ultravox)

### Turn Detection
- [Intelligent Turn Detection](https://www.assemblyai.com/blog/turn-detection-endpointing-voice-agent)
- [Deepgram Flux](https://deepgram.com/learn/introducing-flux-conversational-speech-recognition)
