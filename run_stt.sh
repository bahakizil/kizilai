#!/bin/bash
rm -rf /root/.cache/huggingface/hub/models--selimc--whisper-large-v3-turbo-turkish
python3 /test/test_stt.py
