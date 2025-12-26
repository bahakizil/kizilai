# Windows 11 RTX 5080 PC - Uzaktan Kontrol Rehberi

Bu belge, MacBook üzerinden Windows 11 PC'yi (RTX 5080 GPU) tam kontrol ile kullanmak için gerekli tüm bilgileri içerir.

---

## Bağlantı Bilgileri

| Parametre | Değer |
|-----------|-------|
| **Windows IP** | `192.168.1.11` |
| **Kullanıcı** | `kizil` |
| **SSH Port** | `22` (varsayılan) |
| **MacBook IP** | `192.168.1.7` (en7 interface) |

---

## SSH Bağlantısı

### Temel Bağlantı
```bash
ssh kizil@192.168.1.11
```

### SSH Key Kurulumu (Zaten Yapıldı)
SSH key authentication aktif. Key dosyası: `~/.ssh/id_ed25519`

Windows tarafında key şurada kayıtlı:
- `C:\ProgramData\ssh\administrators_authorized_keys`

---

## Komut Çalıştırma Yöntemleri

### 1. Windows CMD Komutu Çalıştırma
```bash
ssh kizil@192.168.1.11 "KOMUT"
```

**Örnekler:**
```bash
# Dizin listele
ssh kizil@192.168.1.11 "dir C:\\Users\\kizil"

# Dosya içeriği oku
ssh kizil@192.168.1.11 "type C:\\Users\\kizil\\dosya.txt"
```

### 2. WSL2 Ubuntu Komutu Çalıştırma (ÖNERİLEN)
```bash
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e KOMUT"
```

**Örnekler:**
```bash
# Linux komutu çalıştır
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e ls -la /home/kizil"

# Docker komutu çalıştır
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker ps"

# GPU durumu kontrol et
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker exec vllm nvidia-smi"
```

### 3. Karmaşık Bash Komutları
```bash
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e bash -c 'KOMUT1 && KOMUT2'"
```

**NOT:** Tırnak işaretleri ve escape karakterleri dikkatli kullanılmalı.

---

## Docker Kullanımı

### Docker Durumu Kontrol
```bash
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker ps -a"
```

### Docker Image Listele
```bash
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker images"
```

### Docker Container Çalıştır
```bash
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker run --gpus all -d --name CONTAINER_ADI IMAGE_ADI"
```

### Docker Logs
```bash
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker logs CONTAINER_ADI"
```

### Docker Container Durdur/Başlat/Sil
```bash
# Durdur
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker stop CONTAINER_ADI"

# Başlat
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker start CONTAINER_ADI"

# Sil
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker rm CONTAINER_ADI"
```

---

## GPU Kullanımı

### GPU Durumu (nvidia-smi)
```bash
# vLLM container üzerinden
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker exec vllm nvidia-smi"

# Veya herhangi bir GPU container içinden
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker run --rm --gpus all nvidia/cuda:12.6.1-runtime-ubuntu22.04 nvidia-smi"
```

### GPU Özellikleri
| Özellik | Değer |
|---------|-------|
| GPU | NVIDIA GeForce RTX 5080 |
| VRAM | 16 GB |
| CUDA Version | 13.1 |
| Driver Version | 591.59 |

---

## vLLM Server

### vLLM Başlatma
```bash
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker run -d --name vllm --gpus all -p 8000:8000 --shm-size=4g vllm/vllm-openai:latest --model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --max-model-len 4096 --gpu-memory-utilization 0.6"
```

### vLLM API Test
```bash
# Model listele
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e curl -s http://localhost:8000/v1/models"

# Chat completion test
ssh kizil@192.168.1.11 'wsl -d Ubuntu -e curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"Qwen/Qwen2.5-7B-Instruct-AWQ\", \"messages\": [{\"role\": \"user\", \"content\": \"Merhaba!\"}], \"max_tokens\": 50}"'
```

### vLLM Durumu Kontrol
```bash
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker ps -a | grep vllm"
```

---

## Dosya Transferi

### Mac → Windows (SCP)
```bash
# Tek dosya
scp /local/path/file.txt kizil@192.168.1.11:/Users/kizil/hedef/

# Dizin (recursive)
scp -r /local/path/dizin/ kizil@192.168.1.11:/Users/kizil/hedef/
```

### Windows → WSL Kopyalama
```bash
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e cp /mnt/c/Users/kizil/kaynak /home/kizil/hedef"
```

### WSL Yolları
| Windows Yolu | WSL Yolu |
|-------------|----------|
| `C:\Users\kizil` | `/mnt/c/Users/kizil` |
| `D:\Data` | `/mnt/d/Data` |

---

## SesAI Platform

### Kod Lokasyonu
- **Windows:** `C:\Users\kizil\sesai\`
- **WSL:** `/home/kizil/sesai/`

### SesAI Docker Image Build
```bash
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker build -t sesai:latest -f /home/kizil/sesai/docker/Dockerfile /home/kizil/sesai"
```

### SesAI Container Çalıştır
```bash
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker run --gpus all --network host -d --name sesai sesai:latest"
```

### SesAI E2E Test
```bash
ssh kizil@192.168.1.11 "wsl -d Ubuntu docker run --rm --network host -v /home/kizil/sesai/test_e2e.py:/app/test_e2e.py sesai:latest python /app/test_e2e.py"
```

---

## Sık Kullanılan Komutlar - Hızlı Referans

```bash
# Bağlantı test
ssh kizil@192.168.1.11 "echo OK"

# GPU durumu
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker exec vllm nvidia-smi"

# Docker container listesi
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker ps -a"

# vLLM health check
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e curl -s http://localhost:8000/v1/models"

# WSL disk kullanımı
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e df -h"

# WSL bellek kullanımı
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e free -h"
```

---

## Sorun Giderme

### SSH Bağlantı Hatası
```bash
# Ağ bağlantısı kontrol
ping 192.168.1.11

# SSH verbose mode
ssh -v kizil@192.168.1.11
```

### Docker Çalışmıyor
```bash
# Docker service durumu (WSL içinde)
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e sudo service docker status"

# Docker'ı başlat
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e sudo service docker start"
```

### GPU Erişim Sorunu
```bash
# NVIDIA Container Toolkit kontrol
ssh kizil@192.168.1.11 "wsl -d Ubuntu -e docker run --rm --gpus all nvidia/cuda:12.6.1-runtime-ubuntu22.04 nvidia-smi"
```

### WSL Başlamıyor
Windows tarafında PowerShell'den:
```powershell
wsl --shutdown
wsl -d Ubuntu
```

---

## Önemli Notlar

1. **Ağ Gereksinimleri:** Her iki cihaz aynı ağda (192.168.1.x) olmalı
2. **SSH Key:** Password-less authentication aktif
3. **WSL2:** Ubuntu dağıtımı kurulu ve varsayılan
4. **Docker:** WSL2 içinde Docker Engine kurulu (Docker Desktop değil)
5. **GPU:** NVIDIA Container Toolkit WSL2'de kurulu

---

## Kurulum Özeti (Referans)

Bu sistem şu adımlarla kuruldu:

1. Windows'ta OpenSSH Server aktifleştirildi
2. SSH key authentication yapılandırıldı (`administrators_authorized_keys`)
3. WSL2 Ubuntu kuruldu
4. Docker Engine WSL2 içine kuruldu
5. NVIDIA Container Toolkit kuruldu
6. vLLM container başlatıldı
7. SesAI kodu deploy edildi ve test edildi

---

*Son güncelleme: 26 Aralık 2025*
