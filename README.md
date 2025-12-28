# 🎤 Intentioned — Social Training Platform# 🎤 Intentioned - Social Training Platform 🎤 #



Pace AI Research — Voice-powered, self-hosted social skills training.Pace AI Research - Voice Assistant



[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/KoJesko/Honors-Thesis-Conversational-AI-Training)

A self-hosted, open-source voice-powered AI assistant designed for social skills training. Practice conversations, improve communication skills, and receive real-time feedback.A real-time voice-powered AI assistant that uses Speech-to-Text (STT), a Large Language Model (LLM), and Text-to-Speech (TTS) to create a seamless conversational experience.

Intentioned is a self-hosted, open-source voice assistant for practicing conversations and improving communication skills. It combines Speech-to-Text (STT), a Large Language Model (LLM), and Text-to-Speech (TTS) for an interactive, real-time experience.



## ✨ Key Features

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)## ✨ Features

- **Real-time Voice Interaction**: Speak naturally and receive audio responses.

- **Multiple Training Scenarios**: General chat, tutor, coding help, roleplay (e.g., parent–teacher conferences).[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/KoJesko/Honors-Thesis-Conversational-AI-Training)

- **Two Mic Modes**: Push-to-Talk and Voice Activity Detection (VAD).

- **High-Quality Audio**: Edge TTS (Microsoft) for speech synthesis.- **Real-time Voice Interaction**: Speak naturally and get AI responses back in audio

- **Robust STT**: Vosk (offline) with Whisper fallback.

- **Communication Analysis**: Speaking pace, filler-word detection, interruption tracking, and eye-contact analysis.## ✨ Features- **Multiple Scenarios**: General chat, Study tutor, Coding help, Creative writing

- **Safety & Privacy**: AI-based content moderation and local safety logging.

- **Two Mic Modes**: Push-to-Talk or Voice Activity Detection (VAD)

## 🖥️ Tech Stack

### Core Functionality- **HTTPS/WSS Support**: Secure connections with Let's Encrypt or self-signed certificates

- **Backend**: FastAPI + Uvicorn

- **LLM**: Qwen / Hermes (configurable; 4-bit quantized models supported)- **Real-time Voice Interaction**: Speak naturally and get AI responses back in audio- **Edge TTS**: High-quality Microsoft Edge voice synthesis

- **STT**: Vosk (offline) with Whisper fallback

- **TTS**: Microsoft Edge TTS (edge-tts)- **Multiple Training Scenarios**: General chat, Study tutor, Coding help, Creative writing, Parent-Teacher conferences- **Whisper STT**: OpenAI's Whisper large-v3 for accurate speech recognition

- **Frontend**: Vanilla HTML/CSS/JS (WebSocket client)

- **Two Mic Modes**: Push-to-Talk or Voice Activity Detection (VAD)- **Hermes LLM**: NousResearch Hermes-3-Llama-3.1-8B for intelligent responses

## 🚀 Quickstart

- **Eye Contact Tracking**: Optional webcam-based eye contact analysis

### Prerequisites

- Python 3.10+## 🖥️ Tech Stack

- Optional: NVIDIA GPU with CUDA (recommended 4–8 GB VRAM for quantized LLM)

- Browser with microphone and camera permissions (for eye contact tracking)### Communication Analysis



### Installation & Run- **Filler Word Detection**: Track and reduce "um", "uh", "like", etc.| Component | Technology |



1. **Run the server** (this will auto-create a virtualenv and install dependencies):- **Speaking Pace (WPM)**: Measure words per minute for optimal delivery

   ```bash

   python server.py- **Response Time Tracking**: Analyze how quickly you respond in conversations| Backend | FastAPI + Uvicorn |

   ```

- **Interruption Detection**: Track when you speak over the AI (target: <3)| STT | OpenAI Whisper large-v3 |

2. **Open the UI**:

   - Local: `http://localhost:6942`- **Speech Pacing**: Detect long pauses and maintain conversation flow| LLM | NousResearch/Hermes-3-Llama-3.1-8B (4-bit quantized) |

   - With HTTPS (required for mic access from non-localhost): `https://localhost:6942`

- **Tone Analysis**: Get feedback on your conversational tone| TTS | Microsoft Edge TTS (en-US-AriaNeural) |

   For HTTPS, place `cert.pem` and `key.pem` in the project root. You can use Let's Encrypt or generate a self-signed cert for testing:

   ```bash- **Microaggression Detection**: Learn to avoid unintentionally harmful phrases| Frontend | Vanilla HTML/CSS/JS |

   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes -subj "/CN=localhost"

   chmod 600 key.pem- **Protocol**: WebSocket (WS/WSS)

   ```

### Safety & Privacy

## 📁 Project Structure

- **AI Content Moderation**: Intelligent content moderation (no hardcoded blacklists)## 🚀 Quick Start

```

TTS/- **Safety Violation Logging**: Incidents logged locally for review

├── server.py            # FastAPI backend (STT + LLM + TTS + analysis)

├── index.html           # Frontend UI- **Cross-Platform Support**: Works on Windows, macOS, and Linux### Prerequisites

├── script.js            # WebSocket client + UI logic

├── privacy_policy.html  # Site-specific privacy policy (gitignored)- **Self-Hosted**: All data stays on your machine

├── terms_of_use.html    # Site-specific terms (gitignored)

├── code_of_conduct.html # Universal code of conduct (tracked)- **Python 3.10+**

├── requirements.txt     # Python dependencies

├── cert.pem             # SSL certificate (not in repo)## 🖥️ Tech Stack- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)

└── key.pem              # SSL private key (not in repo)

```- Node.js (optional, for development)



## 🔧 Configuration### Installation



Environment variables (examples):1. **Run the server** (auto-creates virtualenv and installs dependencies)



| Variable | Default | Description |## 🚀 Quick Start   

|---|---:|---|

| `SERVER_HOST` | `0.0.0.0` | Bind address |```bash

| `SERVER_PORT` | `6942` | Server port |

| `LLM_MODEL_ID` | `Qwen/Qwen2.5-3B-Instruct` | LLM to load |   python server.py

| `HUGGING_FACE_HUB_TOKEN` | - | For gated models |

```

Run on a different port:# Prerequisites  

```bash

SERVER_PORT=8080 python server.py- Python 3.10+3. **Access the UI**

```

- NVIDIA GPU with CUDA support (recommended: 6GB+ VRAM)   - Local: `http://localhost:6942`

## 🔌 API & WebSocket

- Webcam (optional, for eye contact tracking)   - With SSL: `https://localhost:6942`

**Endpoints:**

- `GET /` — serves `index.html`

- Static files served under `/` (policies, assets)

### Supported Platforms### SSL/HTTPS Setup

**WebSocket:**

- `WS /ws/chat` — real-time audio chat (binary audio frames / JSON controls)| Platform | Status | Notes |For secure connections, place your certificates in the project root:



**Example client → server (audio message):**`cert.pem` - Certificate file (or fullchain)

```json

{| **Linux** | ✅ Fully Supported | Ubuntu 20.04+, Debian 11+ | `key.pem` - Private key file

  "type": "audio",

  "audio": "<base64-encoded-audio>",| **Windows** | ✅ Fully Supported | Windows 10/11 |

  "mimeType": "audio/webm",

  "isFinal": true| **macOS** | ✅ Fully Supported | macOS 11+ (Big Sur) |

}

```**Using Let's Encrypt:**



**Example server → client:**

```json### Installationsudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./cert.pem

{

  "text": "AI response text",sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./key.pem

  "audio": "<base64-encoded-mp3>",

  "status": "streaming|complete|safety_violation"1. **Clone the repository**sudo chown $USER:$USER *.pem

}

```   ```bashchmod 600 key.pem



## 🔒 Safety & Privacy   git clone https://github.com/KoJesko/Honors-Thesis-Conversational-AI-Training.git```



- **AI Moderation**: Intelligent content moderation replaces hard-coded blacklists.   cd Honors-Thesis-Conversational-AI-Training

- **Local Logging**: Safety violation logs are stored locally:

  - Windows: `%USERPROFILE%\Documents\simulation_safety_violations\`   ```**Generate self-signed (for testing):**

  - macOS/Linux: `~/Documents/simulation_safety_violations/`

- **Repeated Violations**: If 3 or more similar violations occur in a session, a summary is automatically transmitted to:```bash

  `[...] /simulation_safety_violations/transmitted_to_host/` for host review.

2. **Run the server** (auto-creates virtualenv and installs dependencies)openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \

## 🛠 Troubleshooting

   ```bash  -sha256 -days 365 -nodes -subj "/CN=localhost"

- **"Connection Died"**: Ensure server is running (`ss -tulpn | grep 6942` on Linux).

- **GPU OOM**: LLMs need ~4–8 GB VRAM depending on model/quantization. Use `nvidia-smi` to inspect and kill other GPU processes.   python server.py```

- **Microphone**: HTTPS is required for browser mic access on non-localhost.

   ```

## 📝 License

## 📁 Project Structure

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See `LICENSE` for details.

3. **Access the UI**

## 🤝 Contributing

   - Local: `http://localhost:6942````

Please read `code_of_conduct.html` before contributing.

   - With SSL: `https://localhost:6942`TTS/

1. Fork the repo

2. Create a branch: `git checkout -b feature/my-feature`├── server.py          # FastAPI backend (STT + LLM + TTS)

3. Commit: `git commit -m "Add feature"`

4. Push and open a PR### SSL/HTTPS Setup├── script.js          # Frontend WebSocket client



## 📞 Support├── index.html         # UI with scenario selection



- **Issues**: [GitHub Issues](https://github.com/KoJesko/Honors-Thesis-Conversational-AI-Training/issues)For secure connections (required for microphone access from non-localhost):├── requirements.txt   # Python dependencies

- **Discussions**: [GitHub Discussions](https://github.com/KoJesko/Honors-Thesis-Conversational-AI-Training/discussions)

├── cert.pem          # SSL certificate (not in repo)

## 🙏 Acknowledgments

**Generate self-signed certificate:**├── key.pem           # SSL private key (not in repo)

- [Vosk](https://alphacephei.com/vosk/) — offline STT

- [OpenAI Whisper](https://github.com/openai/whisper) — STT fallback```bash└── README.md         # This file

- [Qwen](https://github.com/QwenLM/Qwen2.5) / [Hermes](https://nousresearch.com/) — LLMs

- [Edge TTS](https://github.com/rany2/edge-tts) — speech synthesis# Linux/macOS```

- [face-api.js](https://github.com/vladmandic/face-api) — eye contact detection

- [FastAPI](https://fastapi.tiangolo.com/) — backend frameworkopenssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \


  -sha256 -days 365 -nodes -subj "/CN=localhost"## ⚙️ Configuration



# Windows (PowerShell as Admin)### Environment Variables

New-SelfSignedCertificate -DnsName "localhost" -CertStoreLocation "cert:\LocalMachine\My"

```| Variable | Default | Description |

|----------|---------|-------------|

**Using Let's Encrypt:**| `SERVER_HOST` | `0.0.0.0` | Bind address |

```bash| `SERVER_PORT` | `6942` | Server port |

sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./cert.pem| `LLM_MODEL_ID` | `NousResearch/Hermes-3-Llama-3.1-8B` | LLM model to use |

sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./key.pem| `HUGGING_FACE_HUB_TOKEN` | - | HuggingFace token (for gated models) |

sudo chown $USER:$USER *.pem

chmod 600 key.pem### Running on Different Ports

```

```bash

## 📁 Project StructureSERVER_PORT=8080 python server.py

```

```

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

``````json

{

## 📋 Policies & Documentation  "type": "audio",

  "audio": "<base64-encoded-audio>",

- **[Privacy Policy](privacy_policy.html)**: How data is collected and stored  "mimeType": "audio/webm"

- **[Terms of Use](terms_of_use.html)**: Usage terms and conditions}

- **[Code of Conduct](code_of_conduct.html)**: Community standards and rules```



### Safety Violation Logs**Server → Client:**

```json

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

- **Reconnect Button**: Manual reconnection option

| Variable | Default | Description |

|----------|---------|-------------|## 🐛 Troubleshooting

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
