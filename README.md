# 🎤 Pace AI Research - Voice Assistant

A real-time voice-powered AI assistant that uses Speech-to-Text (STT), a Large Language Model (LLM), and Text-to-Speech (TTS) to create a seamless conversational experience.

## ✨ Features

- **Real-time Voice Interaction**: Speak naturally and get AI responses back in audio
- **Multiple Scenarios**: General chat, Study tutor, Coding help, Creative writing
- **Two Mic Modes**: Push-to-Talk or Voice Activity Detection (VAD)
- **HTTPS/WSS Support**: Secure connections with Let's Encrypt or self-signed certificates
- **Edge TTS**: High-quality Microsoft Edge voice synthesis
- **Whisper STT**: OpenAI's Whisper large-v3 for accurate speech recognition
- **Hermes LLM**: NousResearch Hermes-3-Llama-3.1-8B for intelligent responses

## 🖥️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Uvicorn |
| STT | OpenAI Whisper large-v3 |
| LLM | NousResearch/Hermes-3-Llama-3.1-8B (4-bit quantized) |
| TTS | Microsoft Edge TTS (en-US-AriaNeural) |
| Frontend | Vanilla HTML/CSS/JS |
| Protocol | WebSocket (WS/WSS) |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
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

For secure connections, place your certificates in the project root:
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
├── server.py          # FastAPI backend (STT + LLM + TTS)
├── script.js          # Frontend WebSocket client
├── index.html         # UI with scenario selection
├── requirements.txt   # Python dependencies
├── cert.pem          # SSL certificate (not in repo)
├── key.pem           # SSL private key (not in repo)
└── README.md         # This file
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT` | `6942` | Server port |
| `LLM_MODEL_ID` | `NousResearch/Hermes-3-Llama-3.1-8B` | LLM model to use |
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

- **Scenario Selection**: Choose context for AI responses
- **Mic Mode Toggle**: Push-to-Talk vs Voice Activity Detection
- **Audio Visualizer**: Real-time waveform display
- **Connection Status**: Live server connection indicator
- **Reconnect Button**: Manual reconnection option

## 🐛 Troubleshooting

### "Connection Died" Error
- Check if the server is running: `ss -tulpn | grep 6942`
- Verify WebSocket URL matches the server port
- For HTTPS, ensure certificates are valid

### GPU Out of Memory
- The LLM requires ~6-8GB VRAM (4-bit quantized)
- Kill other GPU processes: `nvidia-smi` → find PIDs → `kill <pid>`

### Mixed Content Errors
- If serving HTTPS, the WebSocket must also use WSS
- The client auto-detects protocol from `window.location.protocol`

## 📝 License

AGPL License - feel free to use and modify!

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for STT
- [NousResearch](https://nousresearch.com/) for the Hermes LLM
- [Edge TTS](https://github.com/rany2/edge-tts) for voice synthesis
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
