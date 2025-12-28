# Intentioned - Social Training Platform #

Pace AI Research - Voice Assistant



A self-hosted, open-source voice-powered AI assistant designed for social skills training. Practice conversations, improve communication skills, and receive real-time feedback.A real-time voice-powered AI assistant that uses Speech-to-Text (STT), a Large Language Model (LLM), and Text-to-Speech (TTS) to create a seamless conversational experience.



[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/KoJesko/Honors-Thesis-Conversational-AI-Training)

## Features
- **Real-time Voice Interaction**: Speak naturally and get AI responses back in audio

- **Multiple Scenarios**: General chat, Study tutor, Coding help, Creative writing

- **Two Mic Modes**: Push-to-Talk or Voice Activity Detection (VAD)

## Core Functionality- **HTTPS/WSS Support**: Secure connections with Let's Encrypt or self-signed certificates

- **Real-time Voice Interaction**: Speak naturally and get AI responses back in audio

- **Edge TTS**: High-quality Microsoft Edge voice synthesis

- **Multiple Training Scenarios**: General chat, Study tutor, Coding help, Creative writing, Parent-Teacher conferences

- **Whisper STT Mode**: OpenAI's Whisper large-v3 for accurate speech recognition (disabled by default due to halluinations)

- **Two Mic Modes**: Push-to-Talk or Voice Activity Detection (VAD)

- **Hermes LLM**: NousResearch Hermes-3-Llama-3.1-8B for intelligent responses

- **Eye Contact Tracking**: Optional webcam-based eye contact analysis

## Communication Analysis

- **Filler Word Detection**: Track and reduce "um", "uh", "like", etc.| Component | Technology |

- **Speaking Pace (WPM)**: Measure words per minute for optimal delivery

- **Response Time Tracking**: Analyze how quickly you respond in conversations

- **Interruption Detection**: Track when you speak over the AI (target: <3)

- **Speech Pacing**: Detect long pauses and maintain conversation flow

- **Tone Analysis**: Get feedback on your conversational tone

- **Microaggression Detection**: Learn to avoid unintentionally harmful phrases

## Safety & Privacy

- **AI Content Moderation**: Intelligent content moderation (no hardcoded blacklists)

- **Safety Violation Logging**: Incidents logged locally for review

- **Cross-Platform Support**: Works on Windows, macOS, and Linux

- **Self-Hosted**: All data stays on your machine

- **Python 3.10+**

## 🖥️ Tech Stack- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)

- Node.js (optional, for development)

| Component | Technology |

|-----------|------------|

| Backend | FastAPI + Uvicorn |

| STT | Vosk (offline, non-AI) or OpenAI Whisper |

| LLM | Qwen2.5-3B-Instruct (4-bit quantized) |

| TTS | Microsoft Edge TTS (en-US-AriaNeural) |

| Eye Contact | face-api.js (vladmandic fork) |

| Frontend | Vanilla HTML/CSS/JS |

| Protocol | WebSocket (WS/WSS) |

## 🚀 Quick Start   
```bash

   python server.py

```
- Python 3.10+ **Access the UI**

- NVIDIA GPU with CUDA support (recommended: 6GB+ VRAM)

- Localhost access: `http://localhost:6942`
     *Use SSL for enhanced capabilities (replace "http" with "https")   

- Webcam (optional, for eye contact tracking)


### Supported Platforms



| Platform | Status | Notes |

|----------|--------|-------|

| **Linux** | ✅ Fully Supported | Ubuntu 20.04+, Debian 11+ |

| **Windows** | ✅ Fully Supported | Windows 10/11 |

| **macOS** | ✅ Fully Supported | macOS 11+ (Big Sur) |


1. **Clone the repository**sudo chown $USER:$USER *.pem

   ```bash
   git clone https://github.com/KoJesko/Honors-Thesis-Conversational-AI-Training.git

   cd Honors-Thesis-Conversational-AI-Training``

2. **Run the server** (auto-creates virtualenv and installs dependencies)openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \

   ```bash
   python server.py```

## 📁 Project Structure

3. **Access the UI**

   - Local: `http://localhost:6942`

   - With SSL: `https://localhost:6942`

├── server.py          # FastAPI backend (STT + LLM + TTS)

### SSL/HTTPS Setup
├── script.js          # Frontend WebSocket client

├── index.html         # UI with scenario selection

For secure connections (required for microphone access from non-localhost):

├── cert.pem          # SSL certificate (not in repo)

├── requirements.txt   # Python dependencies

**Generate self-signed certificate:**
├── key.pem           # SSL private key (not in repo)

### GIT Files
└── README.md         # This file



Intentioned/## 🔧 API Endpoints

├── server.py              # FastAPI backend (STT + LLM + TTS + Analysis)

├── script.js              # Frontend WebSocket client| Endpoint | Type | Description |

├── index.html             # Main UI|----------|------|-------------|

├── privacy_policy.html    # Privacy policy (site-specific)| `GET /` | HTTP | Serves the web UI |

├── terms_of_use.html      # Terms of use (site-specific)| `GET /{path}` | HTTP | Static file serving |

├── code_of_conduct.html   # Code of conduct (universal)| `WS /ws/chat` | WebSocket | Real-time audio chat |

├── requirements.txt       # Python dependencies

├── cert.pem              # SSL certificate (not in repo)### WebSocket Protocol

├── key.pem               # SSL private key (not in repo)

└── README.md             # This file**Client → Server:**


## 📋 Policies & Documentation

- **[Privacy Policy](privacy_policy.html)**: How data is collected and stored  "mimeType": "audio/webm"

- **[Terms of Use](terms_of_use.html)**: Usage terms and conditions}

- **[Code of Conduct](code_of_conduct.html)**: Community standards and rules```


## Safety Violation Logs**Server → Client:**

When content is flagged by the AI moderation system, logs are stored locally:{

  "text": "AI response text",

| Platform | Location |  "audio": "<base64-encoded-mp3>",

|----------|----------|  "status": "streaming|complete"

| **Windows** | `%USERPROFILE%\Documents\simulation_safety_violations\` |}

| **macOS** | `~/Documents/simulation_safety_violations/` |```

| **Linux** | `~/Documents/simulation_safety_violations/` |

## 🎨 UI Features

Logs are JSON files containing timestamps, session IDs, and conversation transcripts.

- **Scenario Selection**: Choose context for AI responses

## ⚙️ Configuration- **Mic Mode Toggle**: Push-to-Talk vs Voice Activity Detection

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

### Running on Different Ports

### GPU Out of Memory

```bash- The LLM requires ~6-8GB VRAM (4-bit quantized)

# Linux/macOS- Kill other GPU processes: `nvidia-smi` → find PIDs → `kill <pid>`

SERVER_PORT=8080 python server.py

### Mixed Content Errors

# Windows- If serving HTTPS, the WebSocket must also use WSS

set SERVER_PORT=8080 && python server.py- The client auto-detects protocol from `window.location.protocol`

```

## 📝 License

## 🔧 API Endpoints

AGPL License - feel free to use and modify!

| Endpoint | Type | Description |

|----------|------|-------------|## 🙏 Acknowledgments

| `GET /` | HTTP | Serves the web UI |

| `GET /{path}` | HTTP | Static file serving |- [OpenAI Whisper](https://github.com/openai/whisper) for STT

| `POST /api/analyze` | HTTP | Session analysis |- [NousResearch](https://nousresearch.com/) for the Hermes LLM

| `WS /ws/chat` | WebSocket | Real-time audio chat |- [Edge TTS](https://github.com/rany2/edge-tts) for voice synthesis

- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework

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

## 🎨 UI Features

- **Scenario Selection**: Choose training context with descriptions
- **How to Use Guide**: Built-in instructions for new users
- **Mic Mode Toggle**: Push-to-Talk vs Voice Activity Detection
- **Eye Contact Tracking**: Real-time feedback on camera engagement
- **Session Analysis**: Comprehensive performance report
- **Connection Status**: Live server connection indicator

## 🐛 Troubleshooting

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

## 📝 License

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
