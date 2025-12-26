# SesAI Voice AI Platform

Enterprise-grade Turkish Voice AI platform for real-time phone conversations.

## Features

- Real-time Turkish speech-to-text (faster-whisper)
- High-quality Turkish text-to-speech (Edge TTS)
- LLM-powered conversations (Qwen2.5 via vLLM)
- FreeSWITCH telephony integration (AudioSocket)
- Barge-in (interruption) handling

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy environment file
cp .env.example .env

# Start vLLM (separate terminal)
docker run --gpus all -p 8000:8000 vllm/vllm-openai --model Qwen/Qwen2.5-7B-Instruct-AWQ

# Run the application
python -m src.main
```

## Docker

```bash
docker-compose -f docker/docker-compose.yml up
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT
