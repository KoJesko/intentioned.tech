# Intentioned - Social Training Platform

A self-hosted, modular AGPL-licensed social training application. Practice conversations, improve communication skills, and receive real-time feedback using Speech-to-Text (STT), a Large Language Model (LLM), and Text-to-Speech (TTS).

## ✨ Features

- **Real-time Voice Interaction**: Speak naturally and get AI responses back in audio
- **Session Analysis**: Detailed feedback on filler words, delivery, tone, microaggressions, and eye contact
- **Eye Contact Tracking**: Webcam-based tracking to improve engagement (using `face-api.js`)
- **Multiple Scenarios**: General chat, Study tutor, Coding help, Creative writing, Parent-Teacher Conference
- **Two Mic Modes**: Push-to-Talk or Voice Activity Detection (VAD)
- **HTTPS/WSS Support**: Secure connections with Let's Encrypt or self-signed certificates
- **Multi-User Support**: Handles up to 5 concurrent sessions (optimized for RTX 5070 Ti)
- **Offline Capable**: Supports offline STT (Vosk) and offline TTS (pyttsx3)

## 🖥️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Uvicorn |
| STT | Vosk (offline) or OpenAI Whisper (AI-based) |
| LLM | Qwen/Qwen2.5-3B-Instruct (4-bit quantized) |
| TTS | Microsoft Edge TTS (natural) or pyttsx3 (offline) |
| Frontend | Vanilla HTML/CSS/JS + face-api.js |
| Protocol | WebSocket (WS/WSS) |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM, optimized for 16GB)
- Node.js (optional, for development)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/intentioned-tech-voice-assistant.git
   cd intentioned-tech-voice-assistant
   ```

2. **Run the server** (auto-creates virtualenv and installs dependencies)
   ```bash
   python server.py
   ```

3. **Access the UI**
   - Local: `http://localhost:6942`
   - With SSL: `https://localhost:6942`

### SSL/HTTPS Setup

For secure connections (required for microphone/webcam outside localhost), place your certificates in the project root:
- `cert.pem` - Certificate file (or fullchain)
- `key.pem` - Private key file

**Using Let's Encrypt:**
```bash
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./cert.pem
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./key.pem
sudo chown $USER:$USER *.pem
chmod 600 key.pem
```

**Generate self-signed (for testing):**
```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \
  -sha256 -days 365 -nodes -subj "/CN=localhost"
```

## 📁 Project Structure

```
intentioned-tech-voice-assistant/
├── server.py          # FastAPI backend (STT + LLM + TTS + Analysis)
├── script.js          # Frontend WebSocket client
├── index.html         # UI with scenario selection and webcam support
├── requirements.txt   # Python dependencies
├── cert.pem          # SSL certificate (not in repo)
├── key.pem           # SSL private key (not in repo)
└── README.md         # This file
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `127.0.0.1` | Bind address |
| `SERVER_PORT` | `6942` | Server port |
| `LLM_MODEL_ID` | `Qwen/Qwen2.5-3B-Instruct` | LLM model to use |
| `USE_VOSK` | `true` | Use offline Vosk STT (faster, no GPU) |
| `USE_PYTTSX3` | `false` | Use offline pyttsx3 TTS instead of Edge TTS |
| `HUGGING_FACE_HUB_TOKEN` | - | HuggingFace token (for gated models) |

### Running on Different Ports

```bash
SERVER_PORT=8080 python server.py
```

## 🔧 API Endpoints

| Endpoint | Type | Description |
|----------|------|-------------|
| `GET /` | HTTP | Serves the web UI |
| `GET /{path}` | HTTP | Static file serving |
| `POST /api/analyze` | HTTP | Analyze completed session |
| `WS /ws/chat` | WebSocket | Real-time audio chat |

### WebSocket Protocol

**Client → Server:**
```json
{
  "type": "audio",
  "audio": "<base64-encoded-audio>",
  "mimeType": "audio/webm"
}
```

**Server → Client:**
```json
{
  "text": "AI response text",
  "audio": "<base64-encoded-mp3>",
  "status": "streaming|complete"
}
```

## 🎨 UI Features

- **Scenario Selection**: Choose context (General, Tutor, Coding, Creative, Parent-Teacher)
- **Session Analysis**: Get scored on filler words, delivery, tone, and more
- **Webcam Integration**: Track eye contact for better engagement scores
- **Mic Mode Toggle**: Push-to-Talk vs Voice Activity Detection
- **Real-time Status**: Connection, recording, and voice activity indicators

## 🐛 Troubleshooting

### "Connection Died" Error
- Check if the server is running: `ss -tulpn | grep 6942`
- Verify WebSocket URL matches the server port
- For HTTPS, ensure certificates are valid

### GPU Out of Memory
- The LLM requires ~2GB VRAM (4-bit quantized Qwen-3B)
- Whisper STT (if enabled) requires ~1-2GB
- Kill other GPU processes: `nvidia-smi` → find PIDs → `kill <pid>`

### Mixed Content Errors
- If serving HTTPS, the WebSocket must also use WSS
- The client auto-detects protocol from `window.location.protocol`

## 📝 License

AGPL License - feel free to use and modify!

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for STT
- [Alpha Cephei Vosk](https://alphacephei.com/vosk/) for offline STT
- [Qwen Team](https://huggingface.co/Qwen) for the Qwen2.5 LLM
- [Edge TTS](https://github.com/rany2/edge-tts) for voice synthesis
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [face-api.js](https://github.com/vladmandic/face-api) for eye contact tracking
