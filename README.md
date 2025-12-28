# Intentioned - Social Training Platform #

Pace AI Research - Voice Assistant



A self-hosted, open-source voice-powered AI assistant designed for social skills training. Practice conversations, improve communication skills, and receive real-time feedback.A real-time voice-powered AI assistant that uses Speech-to-Text (STT), a Large Language Model (LLM), and Text-to-Speech (TTS) to create a seamless conversational experience.



[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/KoJesko/Honors-Thesis-Conversational-AI-Training)

A self-hosted, open-source voice-powered AI assistant designed for social skills training. Practice conversations, improve communication skills, and receive real-time feedback using Speech-to-Text (STT), Large Language Model (LLM), and Text-to-Speech (TTS).

## ✨ Features

### Core Functionality
- **Real-time Voice Interaction**: Speak naturally and get AI responses back in audio
- **Multiple Training Scenarios**: General chat, Study tutor, Coding help, Creative writing, Parent-Teacher conferences
- **Two Mic Modes**: Push-to-Talk or Voice Activity Detection (VAD)
- **HTTPS/WSS Support**: Secure connections with Let's Encrypt or self-signed certificates
- **Eye Contact Tracking**: Optional webcam-based eye contact analysis
- **Edge TTS**: High-quality Microsoft Edge voice synthesis

### Communication Analysis
- **Filler Word Detection**: Track and reduce "um", "uh", "like", etc.
- **Speaking Pace (WPM)**: Measure words per minute for optimal delivery
- **Response Time Tracking**: Analyze how quickly you respond in conversations
- **Interruption Detection**: Track when you speak over the AI (target: <3)
- **Speech Pacing**: Detect long pauses and maintain conversation flow
- **Tone Analysis**: Get feedback on your conversational tone
- **Microaggression Detection**: Learn to avoid unintentionally harmful phrases

### Safety & Privacy
- **AI Content Moderation**: Intelligent content moderation (no hardcoded blacklists)
- **Safety Violation Logging**: Incidents logged locally for review
- **Self-Hosted**: All data stays on your machine
- **Cross-Platform Support**: Works on Windows, macOS, and Linux

## 🖥️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Uvicorn |
| STT | Vosk (offline, non-AI) or OpenAI Whisper |
| LLM | Qwen2.5-3B-Instruct (4-bit quantized) |
| TTS | Microsoft Edge TTS (en-US-AriaNeural) |
| Eye Contact | face-api.js (vladmandic fork) |
| Frontend | Vanilla HTML/CSS/JS |
| Protocol | WebSocket (WS/WSS) |

## Requirements

- **Python 3.10+**
- **NVIDIA GPU with CUDA support** (recommended: 6GB+ VRAM)
- **Webcam** (optional, for eye contact tracking)

### Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux** | ✅ Fully Supported | Ubuntu 20.04+, Debian 11+ |
| **Windows** | ✅ Fully Supported | Windows 10/11 |
| **macOS** | ✅ Fully Supported | macOS 11+ (Big Sur) |

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/KoJesko/Honors-Thesis-Conversational-AI-Training.git
   cd Honors-Thesis-Conversational-AI-Training
   ```

2. **Run the server** (auto-creates virtualenv and installs dependencies)
   ```bash
   python server.py```

3. **Access the UI**
   - Local: `http://localhost:6942`
   - With SSL: `https://localhost:6942`

## File Structure

Intentioned.tech core

```
Intentioned/
├── server.py              # FastAPI backend (STT + LLM + TTS + Analysis)
├── script.js              # Frontend WebSocket client
├── index.html             # Main UI
├── privacy_policy.html    # Privacy policy (site-specific)
├── terms_of_use.html      # Terms of use (site-specific)
├── code_of_conduct.html   # Code of conduct (universal)
├── requirements.txt       # Python dependencies
├── cert.pem              # SSL certificate (not in repo)
├── key.pem               # SSL private key (not in repo)
└── README.md             # This file
```

## Policies & Documentation

- **[Privacy Policy](privacy_policy.html)**: How data is collected and stored  "mimeType": "audio/webm"

- **[Terms of Use](terms_of_use.html)**: Usage terms and conditions}

- **[Code of Conduct](code_of_conduct.html)**: Community standards and rules```


## Safety Violation Logs**Server → Client:**

When content is flagged by the AI moderation system, logs are stored locally:{

| **Windows** | `%USERPROFILE%\Documents\simulation_safety_violations\` |}

| **macOS** | `~/Documents/simulation_safety_violations/` |```

| **Linux** | `~/Documents/simulation_safety_violations/` |

## UI Features

Logs are JSON files containing timestamps, session IDs, and conversation transcripts.

- **Scenario Selection**: Choose context for AI responses

## Configuration- **Mic Mode Toggle**: Push-to-Talk vs Voice Activity Detection

- **Audio Visualizer**: Real-time waveform display

### Environment Variables- **Connection Status**: Live server connection indicator

- **Reconnect Button**: Manual reconnection option (Readding soon)

| Variable | Default | Description |

|----------|---------|-------------|

| `SERVER_HOST` | `0.0.0.0` | Bind address |

| `SERVER_PORT` | `6942` | Server port |### "Connection Died" Error

| `LLM_MODEL_ID` | `Qwen/Qwen2.5-3B-Instruct` | LLM model to use |- Check if the server is running: `ss -tulpn | grep 6942`

| `HUGGING_FACE_HUB_TOKEN` | - | HuggingFace token (for gated models) |- Verify WebSocket URL matches the server port

- For HTTPS, ensure certificates are valid

### GPU Out of Memory

```bash- The LLM requires ~6-8GB VRAM (4-bit quantized)

# Linux/macOS- Kill other GPU processes: `nvidia-smi` → find PIDs → `kill <pid>`

SERVER_PORT=8080 python server.py

### Mixed Content Errors

# Windows- If serving HTTPS, the WebSocket must also use WSS

set SERVER_PORT=8080 && python server.py- The client auto-detects protocol from `window.location.protocol`

```

## API Endpoints

| Endpoint | Type | Description |
|----------|------|-------------|
| `GET /` | HTTP | Serves the web UI |
| `GET /{path}` | HTTP | Static file serving |
| `POST /api/analyze` | HTTP | Session analysis |
| `WS /ws/chat` | WebSocket | Real-time audio chat |

### WebSocket Protocol

**Client → Server (Audio):**
```json
{
  "audio": "<base64-encoded-audio>",
  "mimeType": "audio/webm",
  "isFinal": true
}
```

**Client → Server (Control):**
```json
{
  "type": "control",
  "scenario": "parent-teacher",
  "ttsMuted": false
}
```

**Server → Client:**
```json
{
  "text": "AI response text",
  "audio": "<base64-encoded-mp3>",
  "status": "streaming|complete|safety_violation"
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT` | `6942` | Server port |
| `LLM_MODEL_ID` | `Qwen/Qwen2.5-3B-Instruct` | LLM model to use |
| `HUGGING_FACE_HUB_TOKEN` | - | HuggingFace token (for gated models) |

### Running on Different Ports

```bash
# Linux/macOS
SERVER_PORT=8080 python server.py

# Windows
set SERVER_PORT=8080 && python server.py
```

## Safety Violation Logs

When content is flagged by the AI moderation system, logs are stored locally:

| Platform | Location |
|----------|----------|
| **Windows** | `%USERPROFILE%\Documents\simulation_safety_violations\` |
| **macOS** | `~/Documents/simulation_safety_violations/` |
| **Linux** | `~/Documents/simulation_safety_violations/` |

Logs are JSON files containing timestamps, session IDs, and conversation transcripts.

## UI Features

- **Scenario Selection**: Choose training context with descriptions
- **How to Use Guide**: Built-in instructions for new users
- **Mic Mode Toggle**: Push-to-Talk vs Voice Activity Detection
- **Eye Contact Tracking**: Real-time feedback on camera engagement
- **Session Analysis**: Comprehensive performance report
- **Connection Status**: Live server connection indicator

## Troubleshooting

### "Connection Died" Error
- Check if the server is running: `ss -tulpn | grep 6942` (Linux) or `netstat -an | findstr 6942` (Windows)
- Verify WebSocket URL matches the server port
- For HTTPS, ensure certificates are valid

### GPU Out of Memory
- The LLM requires ~4-6GB VRAM (4-bit quantized)
- Kill other GPU processes: `nvidia-smi` → find PIDs → `kill <pid>`

### Mixed Content Errors
- If serving HTTPS, the WebSocket must also use WSS
- The client auto-detects protocol from `window.location.protocol`

### Microphone Not Working
- Ensure HTTPS is enabled (required for browser mic access)
- Check browser permissions for microphone
- Try using localhost if on same machine

### Eye Contact Not Tracking
- Ensure webcam is enabled and not in use by other apps
- Allow camera permissions in browser
- Check browser console for face-api.js loading errors

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

This means:
- ✅ You may use, modify, and distribute the software freely
- ✅ You may use it for commercial purposes
- ⚠️ If you modify and deploy the software over a network, you must make your source code available
- ⚠️ You must include the original license and copyright notices

See [LICENSE](LICENSE) for the full license text.

## 🙏 Acknowledgments

- [Vosk](https://alphacephei.com/vosk/) for offline speech recognition
- [OpenAI Whisper](https://github.com/openai/whisper) for STT (fallback)
- [Qwen](https://github.com/QwenLM/Qwen2.5) for the LLM
- [Edge TTS](https://github.com/rany2/edge-tts) for voice synthesis
- [face-api.js](https://github.com/vladmandic/face-api) for eye contact detection
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework

## 🤝 Contributing

Contributions are welcome! Please read our [Code of Conduct](code_of_conduct.html) before contributing.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/KoJesko/Honors-Thesis-Conversational-AI-Training/issues)
- **Discussions**: [GitHub Discussions](https://github.com/KoJesko/Honors-Thesis-Conversational-AI-Training/discussions)
