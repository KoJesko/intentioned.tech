#!/usr/bin/env python3
"""Speech + LLM server with automatic environment bootstrapping."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# --- Ensure we are running inside the project virtualenv (myenv) ---
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / "myenv"
VENV_BIN = "Scripts" if os.name == "nt" else "bin"
VENV_PYTHON = VENV_DIR / VENV_BIN / ("python.exe" if os.name == "nt" else "python")

if not VENV_DIR.exists():
    print(f"🔧 Creating project virtualenv at {VENV_DIR}...")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])

if Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    if not VENV_PYTHON.exists():
        raise RuntimeError(
            f"Expected virtualenv interpreter at {VENV_PYTHON}, but it was not found."
        )
    print(f"↻ Re-launching inside project venv: {VENV_DIR}")
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

import asyncio
import base64
import io
import json
import re
from threading import Thread

import edge_tts
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    TextIteratorStreamer,
    pipeline,
)

# Initialize the App
app = FastAPI()


# Serve static files (index.html, script.js, etc.)
@app.get("/")
async def serve_index():
    return FileResponse(PROJECT_ROOT / "index.html")


@app.get("/{filename:path}")
async def serve_static(filename: str):
    # Securely resolve the requested path
    try:
        candidate_path = (PROJECT_ROOT / filename).resolve()
        # Only allow serving files that reside inside PROJECT_ROOT
        try:
            # For Python 3.9+, use .is_relative_to; otherwise, use .relative_to
            if not candidate_path.is_relative_to(PROJECT_ROOT):
                return FileResponse(PROJECT_ROOT / "index.html")
        except AttributeError:
            # Fallback for Python <3.9
            try:
                candidate_path.relative_to(PROJECT_ROOT)
            except ValueError:
                return FileResponse(PROJECT_ROOT / "index.html")
        if candidate_path.exists() and candidate_path.is_file():
            return FileResponse(candidate_path)
    except Exception:
        # Any error (invalid path, permission, etc) falls back
        pass
    return FileResponse(PROJECT_ROOT / "index.html")


# --- Conversational tuning knobs ---
STT_MAX_NEW_TOKENS = 256
STT_CHUNK_LENGTH_S = 30
STT_STRIDE_LENGTH_S = (6, 3)

# Using 8B model - fits on 16GB with 4-bit quantization
LLM_DEFAULT_MODEL_ID = "NousResearch/Hermes-3-Llama-3.1-8B"
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", LLM_DEFAULT_MODEL_ID)
HUGGINGFACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

LLM_MAX_NEW_TOKENS = 220
LLM_TEMPERATURE = 0.65
LLM_TOP_P = 0.9
LLM_REPETITION_PENALTY = 1.05

MIN_TTS_CHARS = 60
MAX_TTS_CHARS = 220
SENTENCE_END_REGEX = re.compile(r"[.!?]\s|[.!?]$|\n")
MAX_AUDIO_BUFFER_BYTES = 6 * 1024 * 1024  # 6MB per utterance
MAX_HISTORY_MESSAGES = 12
TARGET_SAMPLE_RATE = 16000
TTS_START_DELAY_SECONDS = 2.0  # matches ollama_tts_app feel
# Default to localhost for security; use SERVER_HOST env var for Docker/container deployments
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "3001"))

# Revision for HuggingFace model pinning (supply chain security)
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "main")

EDGE_TTS_AVAILABLE_VOICES = {
    "female_us": "en-US-AriaNeural",
    "male_us": "en-US-GuyNeural",
    "female_uk": "en-GB-SoniaNeural",
    "male_uk": "en-GB-RyanNeural",
    "female_au": "en-AU-NatashaNeural",
    "male_au": "en-AU-WilliamNeural",
}


def _resolve_edge_voice(preferred: str | None) -> str:
    if not preferred:
        return EDGE_TTS_AVAILABLE_VOICES["female_us"]
    if preferred in EDGE_TTS_AVAILABLE_VOICES:
        return EDGE_TTS_AVAILABLE_VOICES[preferred]
    for pretty_name in EDGE_TTS_AVAILABLE_VOICES.values():
        if preferred.lower() == pretty_name.lower():
            return pretty_name
    print(
        f"⚠️ Voice '{preferred}' not found. Falling back to {EDGE_TTS_AVAILABLE_VOICES['female_us']}"
    )
    return EDGE_TTS_AVAILABLE_VOICES["female_us"]


EDGE_TTS_ENABLED = os.getenv("EDGE_TTS_ENABLED", "true").lower() not in {
    "0",
    "false",
    "no",
}
EDGE_TTS_RATE = os.getenv("EDGE_TTS_RATE", "+0%")
EDGE_TTS_VOLUME = os.getenv("EDGE_TTS_VOLUME", "+0%")
EDGE_TTS_PITCH = os.getenv("EDGE_TTS_PITCH", "+0Hz")
EDGE_TTS_ACTIVE_VOICE = _resolve_edge_voice(os.getenv("EDGE_TTS_VOICE"))


def decode_audio_payload(
    audio_bytes: bytes, mime_type: str | None = None
) -> tuple[np.ndarray, int]:
    if not audio_bytes:
        raise ValueError("No audio payload supplied")

    try:
        data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if samplerate != TARGET_SAMPLE_RATE:
            data = torch.from_numpy(data).to(torch.float32)
            data = (
                torch.nn.functional.interpolate(
                    data.unsqueeze(0).unsqueeze(0),
                    size=int(len(data) * TARGET_SAMPLE_RATE / samplerate),
                    mode="linear",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            samplerate = TARGET_SAMPLE_RATE
        return data.astype(np.float32), samplerate
    except Exception:
        pass

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "pipe:1",
    ]

    process = subprocess.run(
        cmd,
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to decode audio (mime={mime_type}): {process.stderr.decode(errors='ignore')}"
        )

    audio_array = np.frombuffer(process.stdout, dtype=np.float32)
    return audio_array, TARGET_SAMPLE_RATE


def load_llm_stack(model_id: str):
    try:
        print(f"Loading LLM: {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            model_id, token=HUGGINGFACE_TOKEN, revision=HF_MODEL_REVISION
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Use BitsAndBytesConfig for 4-bit quantization (new recommended way)
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            token=HUGGINGFACE_TOKEN,
            revision=HF_MODEL_REVISION,
            max_memory={0: "11GiB", "cpu": "24GiB"},  # Leave room for Whisper
        )
        return tokenizer, model, model_id
    except Exception as exc:
        if model_id == LLM_DEFAULT_MODEL_ID:
            raise
        print(
            f"⚠️ Unable to load '{model_id}' ({exc}). Falling back to '{LLM_DEFAULT_MODEL_ID}'."
        )
        return load_llm_stack(LLM_DEFAULT_MODEL_ID)


# --- HARDWARE SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"🚀 Launching Neural Engine on {device.upper()} (Standard SDPA)...")

# --- 1. STT MODEL (Whisper) ---
# Using large-v3 for best accuracy (cached locally)
stt_model_id = os.getenv("STT_MODEL_ID", "openai/whisper-large-v3")
print(f"Loading STT: {stt_model_id}...")

stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(  # nosec B615
    stt_model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    revision=HF_MODEL_REVISION,
)
stt_model.to(device)
stt_processor = AutoProcessor.from_pretrained(stt_model_id, revision=HF_MODEL_REVISION)  # nosec B615

stt_pipe = pipeline(
    "automatic-speech-recognition",
    model=stt_model,
    tokenizer=stt_processor.tokenizer,
    feature_extractor=stt_processor.feature_extractor,
    max_new_tokens=STT_MAX_NEW_TOKENS,
    chunk_length_s=STT_CHUNK_LENGTH_S,
    stride_length_s=STT_STRIDE_LENGTH_S,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

tokenizer, llm_model, ACTIVE_LLM_MODEL_ID = load_llm_stack(LLM_MODEL_ID)
print(f"🎙️ Edge TTS voice ready: {EDGE_TTS_ACTIVE_VOICE}")


def convert_audio_to_wav(
    audio_bytes: bytes, target_samplerate: int = TARGET_SAMPLE_RATE
) -> bytes:
    """Ensure audio bytes are 16-bit mono WAV at the target sample rate."""
    if not audio_bytes:
        return b""
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "wav",
        "-ac",
        "1",
        "-ar",
        str(target_samplerate),
        "pipe:1",
    ]
    process = subprocess.run(
        cmd,
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to convert Edge TTS audio: {process.stderr.decode(errors='ignore')}"
        )
    return process.stdout


async def synthesize_with_edge_tts(text: str) -> bytes | None:
    if not EDGE_TTS_ENABLED:
        return None
    try:
        communicate = edge_tts.Communicate(
            text,
            EDGE_TTS_ACTIVE_VOICE,
            rate=EDGE_TTS_RATE,
            volume=EDGE_TTS_VOLUME,
            pitch=EDGE_TTS_PITCH,
        )
        audio_chunks = bytearray()
        async for chunk in communicate.stream():
            data = chunk.get("data")
            if chunk.get("type") == "audio" and data:
                audio_chunks.extend(data)
        if not audio_chunks:
            return None
        wav_bytes = await asyncio.to_thread(convert_audio_to_wav, bytes(audio_chunks))
        return wav_bytes
    except Exception as exc:
        print(f"⚠️ Edge TTS synthesis failed: {exc}")
        return None


def trim_history(
    history: list[dict], keep_messages: int = MAX_HISTORY_MESSAGES
) -> None:
    if len(history) <= keep_messages:
        return
    system_prompt = history[0]
    recent = history[-(keep_messages - 1) :]
    history[:] = [system_prompt, *recent]


def clean_transcript_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def aggregate_transcript(stt_output: dict | list) -> str:
    if isinstance(stt_output, dict):
        chunks = stt_output.get("chunks")
        if chunks:
            combined = " ".join(chunk.get("text", "") for chunk in chunks)
            return clean_transcript_text(combined)
        return clean_transcript_text(stt_output.get("text", ""))
    if isinstance(stt_output, list):
        combined = " ".join(
            item.get("text", "") for item in stt_output if isinstance(item, dict)
        )
        return clean_transcript_text(combined)
    return clean_transcript_text(str(stt_output))


def should_emit_chunk(buffer: str) -> bool:
    sample = buffer.strip()
    if not sample:
        return False
    if len(sample) >= MAX_TTS_CHARS:
        return True
    if SENTENCE_END_REGEX.search(sample):
        if len(sample) >= MIN_TTS_CHARS or sample[-1] in ".!?":
            return True
    return False


def normalize_chunk_text(text: str) -> str:
    cleaned = clean_transcript_text(text)
    if not cleaned:
        return ""
    cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
    if cleaned[-1] not in ".?!":
        cleaned += "."
    return cleaned


async def emit_tts_chunk(
    websocket: WebSocket,
    text: str,
    apply_delay: bool = False,
    mute_audio: bool = False,
) -> str:
    normalized = normalize_chunk_text(text)
    if not normalized:
        return ""
    if mute_audio:
        await websocket.send_json(
            {
                "text": normalized,
                "status": "streaming",
                "ttsMuted": True,
            }
        )
        return normalized
    if apply_delay:
        await asyncio.sleep(TTS_START_DELAY_SECONDS)
    print(f"🔊 Generating TTS for: {normalized[:50]}...")
    audio_b64 = await generate_audio_chunk(normalized)
    if audio_b64:
        print(f"✅ TTS audio generated: {len(audio_b64)} bytes (base64)")
        try:
            await websocket.send_json(
                {"text": normalized, "audio": audio_b64, "status": "streaming"}
            )
            print(f"📤 Audio sent successfully")
        except Exception as e:
            print(f"❌ Failed to send audio: {e}")
            raise
        return normalized
    print(f"⚠️ TTS returned no audio for: {normalized[:50]}")
    return ""


async def handle_user_turn(
    audio_payload: bytes,
    websocket: WebSocket,
    conversation_history: list[dict],
    mime_type: str | None = None,
    mute_audio: bool = False,
) -> None:
    if not audio_payload:
        return

    try:
        audio_array, sampling_rate = decode_audio_payload(audio_payload, mime_type)
    except Exception as e:
        print(f"⚠️ Error: {e}")
        await websocket.send_json({"status": "audio-error", "message": "Failed to decode audio"})
        return

    stt_result = stt_pipe({"array": audio_array, "sampling_rate": sampling_rate})
    user_text = aggregate_transcript(stt_result)

    if not user_text:
        print("⚠️ Skipping turn (no speech detected)")
        await websocket.send_json({"status": "no-speech"})
        return

    print(f"User: {user_text}")
    conversation_history.append({"role": "user", "content": user_text})
    trim_history(conversation_history)

    input_ids = tokenizer.apply_chat_template(
        conversation_history,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        repetition_penalty=LLM_REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
    )

    thread = Thread(target=llm_model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    full_response = ""
    tts_delay_pending = True

    print("AI Streaming: ", end="", flush=True)
    for new_text in streamer:
        buffer += new_text
        print(new_text, end="", flush=True)

        if should_emit_chunk(buffer):
            emitted_text = await emit_tts_chunk(
                websocket,
                buffer,
                apply_delay=tts_delay_pending,
                mute_audio=mute_audio,
            )
            if emitted_text:
                full_response += emitted_text + " "
                tts_delay_pending = False
            buffer = ""

    if buffer.strip():
        emitted_text = await emit_tts_chunk(
            websocket,
            buffer,
            apply_delay=tts_delay_pending,
            mute_audio=mute_audio,
        )
        if emitted_text:
            full_response += emitted_text

    print("\n[Done]")
    await websocket.send_json({"status": "complete"})
    conversation_history.append({"role": "assistant", "content": full_response.strip()})
    trim_history(conversation_history)


# --- SCENARIO PROMPTS ---
# Each scenario gets a tailored system prompt
SCENARIO_PROMPTS = {
    "general": """You are a friendly, helpful AI assistant.
Speak naturally, use contractions (like 'don't' instead of 'do not'), and be direct.
Keep responses conversational and short (1-2 sentences max).""",
    "tutor": """You are a patient and encouraging study tutor.
Help students understand concepts by breaking them down into simple steps.
Ask clarifying questions when needed. Be supportive and celebrate small wins.
Speak naturally, use contractions, and keep responses short (1-2 sentences max).""",
    "coding": """You are a knowledgeable programming assistant.
Help debug code, explain concepts, and suggest best practices.
Be concise and practical. Use simple language to explain complex ideas.
Speak naturally, use contractions, and keep responses short (1-2 sentences max).""",
    "creative": """You are a creative writing partner full of ideas.
Help brainstorm stories, develop characters, and overcome writer's block.
Be enthusiastic and imaginative, but stay focused on the user's vision.
Speak naturally, use contractions, and keep responses short (1-2 sentences max).""",
    "parent-teacher": """You will be engaging in a parent-teacher conference. You are a frustrated parent of a 2E (Twice-Exceptional) neurodivergent student.
Your child is gifted but has ADHD and struggles with executive function. You feel the school isn't providing adequate support.
You want your child challenged academically while getting the accommodations they need.
Be emotional but reasonable. Express concerns about IEP implementation and classroom differentiation.
Speak naturally, use contractions (like 'don't' instead of 'do not'), and be direct.
Keep responses conversational and short (1-2 sentences max).""",
}

DEFAULT_SCENARIO = "general"

# Legacy single prompt for backwards compatibility
SYSTEM_PROMPT = SCENARIO_PROMPTS[DEFAULT_SCENARIO]


async def generate_audio_chunk(text: str):
    """Helper to generate audio from a text chunk on the fly."""
    normalized = text.strip()
    if not normalized:
        return None

    wav_bytes = await synthesize_with_edge_tts(normalized)

    if not wav_bytes:
        return None

    return base64.b64encode(wav_bytes).decode("utf-8")


# --- WEBSOCKET ENDPOINT ---
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client Connected (Optimized Pipeline)")

    current_scenario = DEFAULT_SCENARIO
    conversation_history = [
        {"role": "system", "content": SCENARIO_PROMPTS[current_scenario]}
    ]
    processing_lock = asyncio.Lock()
    pending_audio = bytearray()
    client_state = {"tts_muted": False}

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "control":
                # Handle TTS muting
                if "ttsMuted" in message:
                    client_state["tts_muted"] = bool(message.get("ttsMuted", False))
                    print(f"🎚️ Client TTS muted: {client_state['tts_muted']}")

                # Handle scenario changes
                if "scenario" in message:
                    new_scenario = message.get("scenario", DEFAULT_SCENARIO)
                    if new_scenario in SCENARIO_PROMPTS:
                        current_scenario = new_scenario
                        # Reset conversation with new system prompt
                        conversation_history[:] = [
                            {
                                "role": "system",
                                "content": SCENARIO_PROMPTS[current_scenario],
                            }
                        ]
                        print(f"🎭 Switched to scenario: {current_scenario}")
                        await websocket.send_json(
                            {
                                "status": "scenario_changed",
                                "scenario": current_scenario,
                                "text": f"Switched to {current_scenario.replace('-', ' ').title()} mode.",
                            }
                        )
                    else:
                        print(f"⚠️ Unknown scenario: {new_scenario}")
                continue

            if "audio" in message:
                chunk = base64.b64decode(message["audio"])
                mime_type = message.get("mimeType")

                if len(pending_audio) + len(chunk) > MAX_AUDIO_BUFFER_BYTES:
                    print(
                        "⚠️ Audio buffer limit reached, dropping previous data to avoid runaway accumulation."
                    )
                    pending_audio = bytearray()

                pending_audio.extend(chunk)

                if not message.get("isFinal", True):
                    continue

                audio_payload = bytes(pending_audio)
                pending_audio = bytearray()

                if not audio_payload:
                    continue

                async with processing_lock:
                    await handle_user_turn(
                        audio_payload,
                        websocket,
                        conversation_history,
                        mime_type=mime_type,
                        mute_audio=client_state["tts_muted"],
                    )

    except WebSocketDisconnect:
        print("❌ Client Disconnected")
    except Exception as e:
        print(f"⚠️ Error: {e}")
        await websocket.close()


if __name__ == "__main__":
    from pathlib import Path

    import uvicorn

    # Local certs (copied from Let's Encrypt or self-signed)
    local_cert = PROJECT_ROOT / "cert.pem"
    local_key = PROJECT_ROOT / "key.pem"

    # Start server with or without SSL
    if local_cert.exists() and local_key.exists():
        print(f"🔒 Starting HTTPS/WSS server on {SERVER_HOST}:{SERVER_PORT}")
        uvicorn.run(
            app,
            host=SERVER_HOST,
            port=SERVER_PORT,
            ssl_certfile=str(local_cert),
            ssl_keyfile=str(local_key),
        )
    else:
        print(
            f"🔓 Starting HTTP/WS server on {SERVER_HOST}:{SERVER_PORT} (no SSL certs found)"
        )
        uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
