# Intentioned | Master Every Conversation

A self-hosted, open-source voice-powered AI assistant designed for social skills training. Practice conversations, improve communication skills, and receive real-time AI feedback.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/KoJesko/intentioned.tech)

## ✨ Features

### Core Functionality
- **Real-time Voice Interaction**: Speak naturally and get AI responses back in audio
- **Multiple Training Scenarios**: General chat, Study tutor, Coding help, Creative writing, Parent-Teacher conferences
- **Two Mic Modes**: Push-to-Talk or Voice Activity Detection (VAD)
- **HTTPS/WSS Support**: Secure connections with Let's Encrypt or self-signed certificates
- **Eye Contact Tracking**: Optional webcam-based eye contact analysis

### Communication Analysis
- **Filler Word Detection**: Track and reduce "um", "uh", "like", etc.
- **Speaking Pace (WPM)**: Measure words per minute for optimal delivery
- **Response Time Tracking**: Analyze how quickly you respond in conversations
- **Interruption Detection**: Track when you speak over the AI (target: <3)
- **Speech Pacing**: Detect long pauses and maintain conversation flow
- **Tone Analysis**: Get feedback on your conversational tone

### Safety & Privacy
- **AI Content Moderation**: Intelligent content moderation (no hardcoded blacklists)
- **Safety Violation Logging**: Incidents logged locally for review
- **Self-Hosted**: All data stays on your machine - no cloud required

## 🛠️ Technology Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Backend** | FastAPI + Uvicorn | Async Python web framework |
| **STT (Primary)** | NVIDIA Parakeet TDT 0.6B v3 | 600M params, 25 languages, CC-BY-4.0 |
| **STT (Fallback)** | Wav2Vec2 / Vosk | CTC-based, zero hallucination |
| **TTS (Primary)** | Kokoro 82M | StyleTTS2 architecture, Apache 2.0, 24kHz |
| **TTS (Fallback)** | Microsoft Edge TTS / pyttsx3 | Cloud / offline options |
| **LLM** | Qwen2.5-3B-Instruct | 4-bit quantized, intelligent responses |
| **Eye Contact** | face-api.js (vladmandic fork) | Real-time webcam analysis |
| **Frontend** | Vanilla HTML/CSS/JS | No build tools required |
| **Protocol** | WebSocket (WS/WSS) | Real-time bidirectional audio |

## 📋 System Requirements

- **Python**: 3.12+ (required for NeMo toolkit compatibility)
- **GPU**: NVIDIA GPU with CUDA 12.x support (recommended: 8GB+ VRAM)
- **VRAM Usage**: ~4-6GB for LLM + ~2GB for STT/TTS models
- **Webcam**: Optional, for eye contact tracking

### Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux** | ✅ Fully Supported | Ubuntu 20.04+, Debian 11+ |
| **Windows** | ✅ Fully Supported | Windows 10/11 |
| **macOS** | ✅ Fully Supported | macOS 11+ (Big Sur) |

## 🚀 Quick Start

## THE MSI FILE IS STILL IN BETA AND MAY NOT WORK PROPERLY. PLEASE FOLLOW MANUAL SETUP INSTEAD.

1. **Clone the repository**
   ```bash
   git clone https://github.com/KoJesko/intentioned.tech
   cd intentioned.tech
   ```

2. **Create Python 3.12 environment**
   ```bash
   # Using uv (recommended - faster)
   uv venv myenv --python 3.12
   
   # Or using standard venv
   python3.12 -m venv myenv
   ```

3. **Activate the environment**
   ```bash
   # Windows
   myenv\Scripts\activate
   
   # Linux/macOS
   source myenv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   # Install PyTorch with CUDA 12.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   
   # Install remaining requirements
   pip install -r requirements.txt
   ```

5. **Run the server**
   ```bash
   python server.py
   ```

6. **Access the UI**
   - Local: `http://localhost:6942`
   - With SSL: `https://localhost:6942`

### SSL/HTTPS Setup

For secure connections (required for microphone access from non-localhost):

**Generate self-signed certificate:**

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \
  -days 365 -nodes -subj "/CN=localhost"
sudo chown $USER:$USER *.pem
```

## 📋 Policies & Documentation

- **[Privacy Policy](privacy_policy.html)**: How data is collected and stored
- **[Terms of Use](terms_of_use.html)**: Usage terms and conditions
- **[Code of Conduct](code_of_conduct.html)**: Community standards and rules

## ⚙️ Configuration

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

## 🔊 Available Voice Models

### Text-to-Speech (TTS)
| Engine | Model | Size | Quality | License |
|--------|-------|------|---------|---------|
| **Kokoro** (Default) | Kokoro 82M | 82M params | ⭐⭐⭐⭐⭐ | Apache 2.0 |
| VibeVoice | VibeVoice 0.5B | 500M params | ⭐⭐⭐⭐⭐ | MIT |
| Edge TTS | Microsoft Neural | Cloud | ⭐⭐⭐⭐ | Proprietary |
| pyttsx3 | System TTS | N/A | ⭐⭐ | MIT |

### Speech-to-Text (STT)
| Engine | Model | Size | Accuracy | License |
|--------|-------|------|----------|---------|
| **Parakeet** (Default) | Parakeet TDT 0.6B v3 | 600M params | ⭐⭐⭐⭐⭐ | CC-BY-4.0 |
| Wav2Vec2 | wav2vec2-large-960h | 315M params | ⭐⭐⭐⭐ | Apache 2.0 |
| Vosk | vosk-model-en-us-0.22 | 1.8GB | ⭐⭐⭐ | Apache 2.0 |

## Project Structure

```
├── server.py           # FastAPI backend (STT + LLM + TTS)
├── config_tool.py      # GUI configuration tool
├── config.json         # Configuration file (generated)
├── script.js           # Frontend WebSocket client
├── index.html          # UI with scenario selection
├── requirements.txt    # Python dependencies
├── privacy_policy.html
├── terms_of_use.html
├── code_of_conduct.html
├── cert.pem            # SSL certificate (not in repo)
├── key.pem             # SSL private key (not in repo)
└── README.md           # This file
```

## 🔧 Configuration Tool

A GUI tool is included to customize scenarios, models, defaults, and moderation settings:

```bash
# Run the configuration tool
python config_tool.py

# Or build as executable
pip install pyinstaller
pyinstaller --onefile --windowed config_tool.py
```

The tool allows you to:
- **Customize Scenarios**: Add, edit, or remove training scenarios
- **Configure Models**: Set default TTS/STT engines and voices
- **Edit Moderation**: Customize the content moderation prompt
- **Manage Voices**: Add or modify available voice options
- **UI Settings**: Change title, subtitle, and feature toggles

Settings are saved to `config.json` which the server loads on startup.

## 🔒 Safety Violation Logs

When content is flagged by the AI moderation system, logs are stored locally:

| Platform | Location |
|----------|----------|
| **Windows** | `%USERPROFILE%\Documents\simulation_safety_violations\` |
| **macOS** | `~/Documents/simulation_safety_violations/` |
| **Linux** | `~/Documents/simulation_safety_violations/` |

Logs are JSON files containing timestamps, session IDs, and conversation transcripts.

## 🖥️ UI Features

- **Scenario Selection**: Choose training context with descriptions
- **How to Use Guide**: Built-in instructions for new users
- **Mic Mode Toggle**: Push-to-Talk vs Voice Activity Detection
- **Eye Contact Tracking**: Real-time feedback on camera engagement
- **Session Analysis**: Comprehensive performance report
- **Voice Selection**: Choose from 11+ Kokoro voices

## 🔌 API Endpoints

| Endpoint | Type | Description |
|----------|------|-------------|
| `GET /` | HTTP | Serves the web UI |
| `GET /{path}` | HTTP | Static file serving |
| `POST /api/analyze` | HTTP | Session analysis |
| `WS /ws/chat` | WebSocket | Real-time audio chat |

## 🔧 Troubleshooting

### "Connection Died" Error
- Check if the server is running: `ss -tulpn | grep 6942` (Linux) or `netstat -an | findstr 6942` (Windows)
- Verify WebSocket URL matches the server port
- For HTTPS, ensure certificates are valid

### GPU Out of Memory
- The LLM requires ~4-6GB VRAM (4-bit quantized)
- Kill other GPU processes: `nvidia-smi` → find PIDs → `kill <pid>`

### NeMo/Parakeet Issues
- Requires Python 3.12+ (not 3.13-3.14 - spacy compatibility)
- If NeMo fails to load, falls back to Wav2Vec2
- Check CUDA version matches PyTorch build

## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

- ✅ You may use, modify, and distribute the software freely
- ✅ You may use it for commercial purposes
- ⚠️ If you modify and deploy the software over a network, you must make your source code available
- ⚠️ You must include the original license and copyright notices

See [LICENSE](LICENSE) for the full license text.

## 🙏 Acknowledgments

- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) - High-quality neural TTS (Apache 2.0)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) - Parakeet STT models (CC-BY-4.0)
- [Qwen](https://github.com/QwenLM/Qwen2.5) - LLM backbone
- [Edge TTS](https://github.com/rany2/edge-tts) - Microsoft neural voices
- [Vosk](https://alphacephei.com/vosk/) - Offline speech recognition
- [face-api.js](https://github.com/vladmandic/face-api) - Eye contact detection
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework

## 🤝 Contributing

Contributions are welcome! Please read our [Code of Conduct](code_of_conduct.html) before contributing.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/KoJesko/intentioned.tech/issues)
- **Discussions**: [GitHub Discussions](https://github.com/KoJesko/intentioned.tech/discussions)
## 📜 Third-Party Licenses & Attribution

This software incorporates the following open-source components:

| Component | License | Attribution |
|-----------|---------|-------------|
| **Google Gemma** | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) | © Google LLC. Gemma is a family of open models by Google. |
| **Qwen 2.5** | Apache 2.0 | © Alibaba Cloud. Qwen2.5 LLM series. |
| **NVIDIA Parakeet** | CC-BY-4.0 | © NVIDIA Corporation. Parakeet TDT speech recognition. |
| **Kokoro TTS** | Apache 2.0 | StyleTTS2-based text-to-speech by hexgrad. |
| **Vosk** | Apache 2.0 | © Alpha Cephei Inc. Offline speech recognition. |
| **face-api.js** | MIT | © vladmandic. Face detection and analysis. |
| **FastAPI** | MIT | © Sebastián Ramírez. Web framework. |
| **Edge TTS** | MIT | © rany2. Microsoft Edge neural voices wrapper. |

### Gemma Model Notice

If using Google Gemma models (e.g., `google/gemma-3-1b-it`), you must comply with the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Commercial use is permitted under 1 million monthly active users.

### Full License Texts

See the [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES) file for complete license texts.

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** - see the [LICENSE](LICENSE) file for details.