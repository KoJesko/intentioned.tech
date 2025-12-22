#!/usr/bin/env python3
"""Speech + LLM server with automatic environment bootstrapping."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
from pathlib import Path

# --- Ensure we are running inside the project virtualenv (myenv) ---
PROJECT_ROOT = Path(__file__).resolve().parent.resolve()
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
import tempfile
import wave
from collections import Counter
from threading import Thread, Lock
from typing import Optional

import edge_tts
import numpy as np
import pyttsx3
import soundfile as sf
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Vosk for non-AI speech recognition (optional, falls back to Whisper if not available)
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    print("⚠️ Vosk not installed. Will use Whisper for STT.")

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


# Serve static files (index.html, script.js, etc.) with no-cache headers
@app.get("/")
async def serve_index():
    response = FileResponse(PROJECT_ROOT / "index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/{filename:path}")
async def serve_static(filename: str):
    # Securely resolve the requested path
    try:
        # Ensure the provided filename is treated as a relative path segment
        # Strip any leading path separators so we never interpret it as absolute
        safe_filename = filename.lstrip("/\\")
        candidate_path = (PROJECT_ROOT / safe_filename).resolve()
        # Only allow serving files that reside inside PROJECT_ROOT
        try:
            # Use Path.relative_to on resolved paths to enforce containment
            candidate_path.relative_to(PROJECT_ROOT)
        except ValueError:
            # Path is outside PROJECT_ROOT; fall back to index.html
            response = FileResponse(PROJECT_ROOT / "index.html")
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return response
        if candidate_path.exists() and candidate_path.is_file():
            response = FileResponse(candidate_path)
            # No cache for HTML/JS files to ensure latest version
            if safe_filename.endswith(('.html', '.js', '.css')):
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            return response
    except Exception:
        # Any error (invalid path, permission, etc) falls back
        pass
    response = FileResponse(PROJECT_ROOT / "index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# --- Analysis Request/Response Models ---
class AnalysisRequest(BaseModel):
    transcript: list[dict]  # List of {"role": "user"|"assistant", "content": str}
    eye_contact_data: Optional[list[dict]] = None  # Optional webcam data
    speech_timestamps: Optional[list[dict]] = None  # Optional speech timing data {"start": float, "end": float}
    response_times: Optional[list[float]] = None  # Time between AI done speaking and user starts speaking
    scenario: str = "general"


class AnalysisResponse(BaseModel):
    filler_words: dict  # {"count": int, "target": int, "details": dict, "score": str}
    delivery: dict  # {"score": str, "feedback": str}
    tone: dict  # {"score": str, "feedback": str}
    microaggressions: dict  # {"detected": bool, "feedback": str}
    eye_contact: Optional[dict] = None  # {"score": str, "percentage": float, "feedback": str}
    speech_pacing: Optional[dict] = None  # {"score": str, "avg_gap": float, "feedback": str}
    response_time: Optional[dict] = None  # {"score": str, "avg_time": float, "feedback": str}
    overall_score: int = 0  # 0-100 overall score
    ai_summary: str = ""  # AI-generated summary


def score_to_points(score_str: str) -> int:
    """Convert emoji score string to numeric points."""
    if "Excellent" in score_str or "🟢" in score_str:
        return 100
    elif "Good" in score_str or "🟡" in score_str:
        return 75
    elif "Needs Work" in score_str or "🟠" in score_str:
        return 50
    elif "Poor" in score_str or "🔴" in score_str:
        return 25
    return 0


def calculate_overall_score(filler: dict, delivery: dict, tone: dict, 
                           microaggressions: dict, pacing: Optional[dict], 
                           eye_contact: Optional[dict],
                           response_time: Optional[dict] = None) -> int:
    """Calculate overall score out of 100."""
    scores = []
    weights = []
    
    # Filler words (weight: 20%)
    scores.append(score_to_points(filler.get("score", "")))
    weights.append(0.20)
    
    # Delivery (weight: 25%)
    scores.append(score_to_points(delivery.get("score", "")))
    weights.append(0.25)
    
    # Tone (weight: 25%)
    scores.append(score_to_points(tone.get("score", "")))
    weights.append(0.25)
    
    # Microaggressions (weight: 15%) - no detected = 100, detected = 25
    if not microaggressions.get("detected", False):
        scores.append(100)
    else:
        scores.append(25)
    weights.append(0.15)
    
    # Pacing (weight: 8% if available)
    if pacing:
        scores.append(score_to_points(pacing.get("score", "")))
        weights.append(0.08)
    
    # Eye contact (weight: 5% if available)
    if eye_contact:
        scores.append(score_to_points(eye_contact.get("score", "")))
        weights.append(0.05)
    
    # Response time (weight: 7% if available)
    if response_time:
        scores.append(score_to_points(response_time.get("score", "")))
        weights.append(0.07)
    
    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # Calculate weighted average
    overall = sum(s * w for s, w in zip(scores, normalized_weights))
    return round(overall)


def generate_ai_summary(score: int, filler: dict, delivery: dict, tone: dict,
                       microaggressions: dict, pacing: Optional[dict],
                       eye_contact: Optional[dict],
                       response_time: Optional[dict] = None) -> str:
    """Generate a brief AI summary of the performance."""
    summaries = []
    
    if score >= 90:
        summaries.append("Excellent communication skills demonstrated.")
    elif score >= 75:
        summaries.append("Good overall performance with room for improvement.")
    elif score >= 50:
        summaries.append("Developing communication skills.")
    else:
        summaries.append("Focus on core communication fundamentals.")
    
    # Specific feedback
    filler_count = filler.get("count", 0)
    if filler_count > 5:
        summaries.append(f"Reduce filler words (used {filler_count} times).")
    elif filler_count <= 2:
        summaries.append("Minimal filler words - excellent clarity.")
    
    if "Poor" in delivery.get("score", "") or "Needs Work" in delivery.get("score", ""):
        summaries.append("Work on expanding responses with more detail.")
    
    if microaggressions.get("detected", False):
        summaries.append("Review language for more inclusive phrasing.")
    
    if pacing and ("Poor" in pacing.get("score", "") or "Needs Work" in pacing.get("score", "")):
        summaries.append("Practice smoother speech pacing.")
    
    if eye_contact:
        percentage = eye_contact.get("percentage", 0)
        if percentage < 50:
            summaries.append("Improve eye contact with the camera.")
        elif percentage >= 70:
            summaries.append("Strong eye contact maintained.")
    
    if response_time:
        avg_time = response_time.get("avg_time", 0)
        if avg_time > 5:
            summaries.append("Work on responding more quickly to maintain conversation flow.")
        elif avg_time <= 2:
            summaries.append("Excellent response speed shows active engagement.")
    
    return " ".join(summaries)


def count_filler_words(text: str) -> dict:
    """Count filler words in text and return detailed breakdown."""
    words = text.lower().split()
    text_lower = text.lower()
    
    filler_counts = Counter()
    
    # Count single-word fillers
    for word in words:
        # Clean punctuation
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in FILLER_WORDS:
            filler_counts[clean_word] += 1
    
    # Count multi-word fillers
    multi_word_fillers = ["you know", "i mean", "sort of", "kind of"]
    for phrase in multi_word_fillers:
        count = text_lower.count(phrase)
        if count > 0:
            filler_counts[phrase] += count
    
    total = sum(filler_counts.values())
    
    # Determine score
    if total <= 2:
        score = "🟢 Excellent"
    elif total <= 5:
        score = "🟡 Good"
    elif total <= 10:
        score = "🟠 Needs Work"
    else:
        score = "🔴 Poor"
    
    return {
        "count": total,
        "target": 5,
        "details": dict(filler_counts.most_common(10)),
        "score": score
    }


def analyze_delivery_and_tone(transcript: list[dict], scenario: str) -> tuple[dict, dict]:
    """Analyze delivery and tone of user's messages."""
    user_messages = [msg["content"] for msg in transcript if msg.get("role") == "user"]
    
    if not user_messages:
        return (
            {"score": "N/A", "feedback": "No user messages to analyze."},
            {"score": "N/A", "feedback": "No user messages to analyze."}
        )
    
    combined_text = " ".join(user_messages)
    word_count = len(combined_text.split())
    sentence_count = len(re.findall(r'[.!?]+', combined_text)) or 1
    avg_sentence_length = word_count / sentence_count
    
    # Delivery analysis (based on message structure)
    delivery_issues = []
    delivery_positives = []
    
    # Check for very short responses (might indicate lack of elaboration)
    short_responses = sum(1 for msg in user_messages if len(msg.split()) < 5)
    if short_responses > len(user_messages) * 0.5:
        delivery_issues.append("Many responses were very brief. Try elaborating more.")
    else:
        delivery_positives.append("Good response length and elaboration.")
    
    # Check for question engagement
    questions_asked = sum(1 for msg in user_messages if "?" in msg)
    if questions_asked > 0:
        delivery_positives.append(f"Asked {questions_asked} clarifying question(s) - shows engagement.")
    
    # Average sentence length feedback
    if avg_sentence_length > 25:
        delivery_issues.append("Some sentences were quite long. Consider breaking them up for clarity.")
    elif avg_sentence_length < 8:
        delivery_issues.append("Sentences were very short. Try connecting ideas more smoothly.")
    else:
        delivery_positives.append("Good sentence structure and flow.")
    
    delivery_score = "🟢 Excellent" if len(delivery_issues) == 0 else ("🟡 Good" if len(delivery_issues) <= 1 else "🟠 Needs Work")
    delivery_feedback = " ".join(delivery_positives + delivery_issues)
    
    # Tone analysis
    tone_issues = []
    tone_positives = []
    
    # Check for aggressive language patterns
    aggressive_patterns = ["you always", "you never", "that's wrong", "that's stupid", "obviously"]
    aggressive_count = sum(combined_text.lower().count(p) for p in aggressive_patterns)
    if aggressive_count > 0:
        tone_issues.append("Some phrases could come across as confrontational.")
    
    # Check for positive/professional language
    positive_patterns = ["thank you", "please", "i appreciate", "i understand", "that makes sense"]
    positive_count = sum(combined_text.lower().count(p) for p in positive_patterns)
    if positive_count >= 2:
        tone_positives.append("Good use of polite and professional language.")
    
    # Scenario-specific tone feedback
    if scenario == "parent-teacher":
        if "concern" in combined_text.lower() or "worried" in combined_text.lower():
            tone_positives.append("Appropriately expressed concerns as a parent would.")
    
    tone_score = "🟢 Excellent" if len(tone_issues) == 0 and len(tone_positives) > 0 else ("🟡 Good" if len(tone_issues) <= 1 else "🟠 Needs Work")
    tone_feedback = " ".join(tone_positives + tone_issues) or "Tone was neutral."
    
    return (
        {"score": delivery_score, "feedback": delivery_feedback},
        {"score": tone_score, "feedback": tone_feedback}
    )


def check_microaggressions(transcript: list[dict]) -> dict:
    """Check for potential microaggressions in user messages."""
    user_messages = [msg["content"] for msg in transcript if msg.get("role") == "user"]
    combined_text = " ".join(user_messages).lower()
    
    # Common microaggression patterns (educational context)
    patterns = {
        "where are you really from": "Questioning someone's belonging",
        "you speak english well": "Surprise at language ability can be othering",
        "you're so articulate": "Can imply low expectations based on identity",
        "i don't see color": "Dismisses experiences of people of color",
        "all lives matter": "Dismisses specific concerns",
        "you people": "Othering language",
        "that's so gay": "Using identity as negative descriptor",
        "man up": "Reinforces harmful gender stereotypes",
        "you're too sensitive": "Dismisses valid concerns",
    }
    
    detected = []
    for pattern, explanation in patterns.items():
        if pattern in combined_text:
            detected.append(f"'{pattern}' - {explanation}")
    
    if detected:
        return {
            "detected": True,
            "feedback": "⚠️ Some phrases may be perceived as microaggressions: " + "; ".join(detected)
        }
    
    return {
        "detected": False,
        "feedback": "✅ No common microaggression patterns detected. Good job maintaining inclusive language!"
    }


def analyze_eye_contact(eye_contact_data: list[dict]) -> Optional[dict]:
    """Analyze eye contact data from webcam."""
    if not eye_contact_data:
        return None
    
    total_frames = len(eye_contact_data)
    if total_frames == 0:
        return None
    
    looking_at_camera = sum(1 for frame in eye_contact_data if frame.get("looking_at_camera", False))
    percentage = (looking_at_camera / total_frames) * 100
    
    if percentage >= 70:
        score = "🟢 Excellent"
        feedback = f"Maintained eye contact {percentage:.1f}% of the time. Great engagement!"
    elif percentage >= 50:
        score = "🟡 Good"
        feedback = f"Eye contact at {percentage:.1f}%. Try to look at the camera more consistently."
    elif percentage >= 30:
        score = "🟠 Needs Work"
        feedback = f"Eye contact at {percentage:.1f}%. Practice looking at the camera while speaking."
    else:
        score = "🔴 Poor"
        feedback = f"Eye contact at {percentage:.1f}%. Focus on maintaining eye contact for better connection."
    
    return {
        "score": score,
        "percentage": round(percentage, 1),
        "feedback": feedback
    }


def analyze_speech_pacing(speech_timestamps: list[dict]) -> Optional[dict]:
    """Analyze gaps/pauses between speech segments."""
    if not speech_timestamps or len(speech_timestamps) < 2:
        return None
    
    # Sort by start time
    sorted_timestamps = sorted(speech_timestamps, key=lambda x: x.get("start", 0))
    
    # Calculate gaps between speech segments
    gaps = []
    long_pauses = []  # Pauses over 3 seconds
    
    for i in range(1, len(sorted_timestamps)):
        prev_end = sorted_timestamps[i - 1].get("end", 0)
        curr_start = sorted_timestamps[i].get("start", 0)
        gap = curr_start - prev_end
        
        if gap > 0.1:  # Ignore tiny gaps (less than 100ms)
            gaps.append(gap)
            if gap > 3.0:
                long_pauses.append(gap)
    
    if not gaps:
        return {
            "score": "🟢 Excellent",
            "avg_gap": 0,
            "max_gap": 0,
            "long_pauses": 0,
            "feedback": "Smooth, continuous speech with no significant pauses."
        }
    
    avg_gap = sum(gaps) / len(gaps)
    max_gap = max(gaps)
    long_pause_count = len(long_pauses)
    
    # Ideal speaking has natural pauses of 0.5-2 seconds
    if avg_gap <= 1.5 and long_pause_count == 0:
        score = "🟢 Excellent"
        feedback = f"Great pacing! Average pause of {avg_gap:.1f}s between thoughts. Natural and confident."
    elif avg_gap <= 2.5 and long_pause_count <= 2:
        score = "🟡 Good"
        feedback = f"Good pacing with {avg_gap:.1f}s average pause. {long_pause_count} longer pause(s) noted - consider smoother transitions."
    elif avg_gap <= 4.0 or long_pause_count <= 4:
        score = "🟠 Needs Work"
        feedback = f"Average pause of {avg_gap:.1f}s. {long_pause_count} long pause(s) detected. Try to maintain momentum in your speech."
    else:
        score = "🔴 Poor"
        feedback = f"Frequent long pauses ({avg_gap:.1f}s avg). This can lose your audience's attention. Practice speaking more fluidly."
    
    return {
        "score": score,
        "avg_gap": round(avg_gap, 2),
        "max_gap": round(max_gap, 2),
        "long_pauses": long_pause_count,
        "feedback": feedback
    }


def analyze_response_time(response_times: list[float]) -> dict:
    """Analyze how quickly user responds after AI finishes speaking."""
    if not response_times:
        return {
            "score": "🟡 Not Enough Data",
            "avg_time": 0,
            "min_time": 0,
            "max_time": 0,
            "feedback": "Not enough conversation turns to measure response time."
        }
    
    avg_time = sum(response_times) / len(response_times)
    min_time = min(response_times)
    max_time = max(response_times)
    
    # Ideal response time is 0.5-2 seconds (natural conversation pace)
    if avg_time <= 1.5:
        score = "🟢 Excellent"
        feedback = f"Quick and engaged! Average response time of {avg_time:.1f}s shows active listening."
    elif avg_time <= 3.0:
        score = "🟡 Good"
        feedback = f"Good response time of {avg_time:.1f}s. You're engaged in the conversation."
    elif avg_time <= 5.0:
        score = "🟠 Needs Work"
        feedback = f"Average response time of {avg_time:.1f}s. Try to respond more promptly to maintain conversation flow."
    else:
        score = "🔴 Slow"
        feedback = f"Long response time ({avg_time:.1f}s avg). This can make conversations feel disconnected. Practice active listening."
    
    return {
        "score": score,
        "avg_time": round(avg_time, 2),
        "min_time": round(min_time, 2),
        "max_time": round(max_time, 2),
        "count": len(response_times),
        "feedback": feedback
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_session(request: AnalysisRequest):
    """Analyze a completed conversation session."""
    print(f"📊 Analyzing session with {len(request.transcript)} messages...")
    
    # Extract user text for filler word analysis
    user_text = " ".join(
        msg["content"] for msg in request.transcript if msg.get("role") == "user"
    )
    
    # Perform analyses
    filler_analysis = count_filler_words(user_text)
    delivery_analysis, tone_analysis = analyze_delivery_and_tone(request.transcript, request.scenario)
    microaggression_analysis = check_microaggressions(request.transcript)
    eye_contact_analysis = analyze_eye_contact(request.eye_contact_data) if request.eye_contact_data else None
    speech_pacing_analysis = analyze_speech_pacing(request.speech_timestamps) if request.speech_timestamps else None
    response_time_analysis = analyze_response_time(request.response_times) if request.response_times else None
    
    # Calculate overall score (include response time)
    overall = calculate_overall_score(
        filler_analysis, delivery_analysis, tone_analysis,
        microaggression_analysis, speech_pacing_analysis, eye_contact_analysis,
        response_time_analysis
    )
    
    # Generate AI summary
    summary = generate_ai_summary(
        overall, filler_analysis, delivery_analysis, tone_analysis,
        microaggression_analysis, speech_pacing_analysis, eye_contact_analysis,
        response_time_analysis
    )
    
    print(f"✅ Analysis complete: {filler_analysis['count']} filler words, overall score: {overall}/100")
    
    return AnalysisResponse(
        filler_words=filler_analysis,
        delivery=delivery_analysis,
        tone=tone_analysis,
        microaggressions=microaggression_analysis,
        eye_contact=eye_contact_analysis,
        speech_pacing=speech_pacing_analysis,
        response_time=response_time_analysis,
        overall_score=overall,
        ai_summary=summary
    )


# --- Conversational tuning knobs ---
STT_MAX_NEW_TOKENS = 256
STT_CHUNK_LENGTH_S = 30
STT_STRIDE_LENGTH_S = (6, 3)

# Using smaller models optimized for RTX 5070 Ti (16GB VRAM) with multi-user support
# STT: Vosk (non-AI, traditional acoustic models) or Whisper as fallback
# LLM: Qwen2.5-3B-Instruct (~2GB in 4-bit) - reliable and well-supported
LLM_DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", LLM_DEFAULT_MODEL_ID)
HUGGINGFACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# STT Configuration
# USE_VOSK=true uses Vosk (non-AI, traditional speech recognition)
# USE_VOSK=false uses Whisper (AI-based, more accurate but uses GPU)
USE_VOSK = os.getenv("USE_VOSK", "true").lower() in {"1", "true", "yes"}
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", str(PROJECT_ROOT / "vosk-model-en-us-0.22"))

# Fallback Whisper model if Vosk not available
STT_DEFAULT_MODEL_ID = "openai/whisper-small"  # ~500MB vs ~3GB for large-v3

LLM_MAX_NEW_TOKENS = 150  # Reduced for faster responses with smaller model
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
SERVER_PORT = int(os.getenv("SERVER_PORT", "6942"))

# Multi-user support configuration
MAX_CONCURRENT_SESSIONS = 5
active_sessions: dict[str, dict] = {}  # session_id -> session data
session_lock = Lock()

# Filler words to detect (common speech disfluencies)
FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "er", "ah", "ahh", "hmm", "hm",
    "like", "you know", "i mean", "sort of", "kind of", "basically",
    "actually", "literally", "honestly", "right", "so", "well",
    "anyway", "whatever", "stuff", "things", "yeah"
}

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

# --- pyttsx3 (offline) TTS Configuration ---
# Set USE_PYTTSX3=true to use offline TTS instead of Edge TTS
# Default is FALSE - Edge TTS sounds more natural
USE_PYTTSX3 = os.getenv("USE_PYTTSX3", "false").lower() in {"1", "true", "yes"}
PYTTSX3_RATE = int(os.getenv("PYTTSX3_RATE", "175"))  # Words per minute
PYTTSX3_VOLUME = float(os.getenv("PYTTSX3_VOLUME", "1.0"))  # 0.0 to 1.0

# Thread-safe pyttsx3 engine (created per-call due to threading issues)
_pyttsx3_lock = Lock()


def synthesize_with_pyttsx3(text: str) -> bytes | None:
    """Synthesize speech using pyttsx3 (offline, non-AI TTS)."""
    if not text.strip():
        return None
    
    try:
        with _pyttsx3_lock:
            engine = pyttsx3.init()
            engine.setProperty('rate', PYTTSX3_RATE)
            engine.setProperty('volume', PYTTSX3_VOLUME)
            
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            engine.stop()
            
            # Read the file and convert to proper format
            with open(tmp_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            
            # Convert to consistent format (16kHz mono WAV)
            wav_bytes = convert_audio_to_wav(audio_bytes)
            return wav_bytes
            
    except Exception as exc:
        print(f"⚠️ pyttsx3 synthesis failed: {exc}")
        return None


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
        # Ensure pad token is different from eos token to avoid attention mask issues
        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token

        # Use BitsAndBytesConfig for 4-bit quantization (new recommended way)
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        # Optimized memory allocation for RTX 5070 Ti (16GB)
        # Reserve ~3GB for Whisper-small, ~2GB for Qwen2.5-3B
        # Leaves ~10GB for inference and multi-user headroom
        model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            token=HUGGINGFACE_TOKEN,
            revision=HF_MODEL_REVISION,
            max_memory={0: "8GiB", "cpu": "16GiB"},  # Conservative for multi-user
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


# --- MODEL MANAGER (for multi-user support) ---
class ModelManager:
    """Manages model loading/unloading for multi-user support."""
    
    def __init__(self):
        self.stt_model = None
        self.stt_processor = None
        self.stt_pipe = None
        self.vosk_model = None  # Vosk model for non-AI STT
        self.vosk_recognizer = None
        self.use_vosk = USE_VOSK and VOSK_AVAILABLE
        self.llm_model = None
        self.llm_tokenizer = None
        self.active_model_id = None
        self._lock = Lock()
        self._ref_count = 0
    
    def load_models(self):
        """Load models if not already loaded."""
        with self._lock:
            self._ref_count += 1
            if self.llm_model is not None:
                print(f"📊 Models already loaded (ref_count: {self._ref_count})")
                return
            
            print("🔄 Loading models (optimized for RTX 5070 Ti / 16GB VRAM)...")
            
            # Load STT - use Vosk (non-AI) if available and enabled, otherwise Whisper
            if self.use_vosk:
                self._load_vosk_model()
            else:
                self._load_whisper_model()
            
            # Load LLM
            self.llm_tokenizer, self.llm_model, self.active_model_id = load_llm_stack(LLM_MODEL_ID)
            print(f"✅ Models loaded (ref_count: {self._ref_count})")
    
    def _load_vosk_model(self):
        """Load Vosk model for non-AI speech recognition."""
        from pathlib import Path
        vosk_path = Path(VOSK_MODEL_PATH)
        
        if not vosk_path.exists():
            print(f"⚠️ Vosk model not found at {vosk_path}")
            print("📥 Downloading Vosk model...")
            self._download_vosk_model(vosk_path)
        
        if vosk_path.exists():
            print(f"Loading STT: Vosk (non-AI) from {vosk_path}...")
            self.vosk_model = VoskModel(str(vosk_path))
            print("✅ Vosk model loaded (non-AI STT)")
        else:
            print("⚠️ Vosk model download failed, falling back to Whisper")
            self.use_vosk = False
            self._load_whisper_model()
    
    def _download_vosk_model(self, vosk_path):
        """Download Vosk model if not present."""
        import urllib.request
        import zipfile
        
        # Use the small English model (~50MB) - good balance of speed and accuracy
        model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        zip_path = vosk_path.parent / "vosk-model.zip"
        
        try:
            print(f"📥 Downloading from {model_url}...")
            urllib.request.urlretrieve(model_url, str(zip_path))
            
            print("📦 Extracting model...")
            with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
                zip_ref.extractall(vosk_path.parent)
            
            # Rename extracted folder to expected path
            extracted_name = "vosk-model-small-en-us-0.15"
            extracted_path = vosk_path.parent / extracted_name
            if extracted_path.exists():
                extracted_path.rename(vosk_path)
            
            # Clean up zip
            zip_path.unlink()
            print("✅ Vosk model downloaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to download Vosk model: {e}")
            if zip_path.exists():
                zip_path.unlink()
    
    def _load_whisper_model(self):
        """Load Whisper model as fallback."""
        stt_model_id = os.getenv("STT_MODEL_ID", STT_DEFAULT_MODEL_ID)
        print(f"Loading STT: {stt_model_id} (Whisper)...")
        
        self.stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            stt_model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            revision=HF_MODEL_REVISION,
        )
        self.stt_model.to(device)
        self.stt_processor = AutoProcessor.from_pretrained(stt_model_id, revision=HF_MODEL_REVISION)
        
        self.stt_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.stt_model,
            tokenizer=self.stt_processor.tokenizer,
            feature_extractor=self.stt_processor.feature_extractor,
            max_new_tokens=STT_MAX_NEW_TOKENS,
            chunk_length_s=STT_CHUNK_LENGTH_S,
            stride_length_s=STT_STRIDE_LENGTH_S,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )
    
    def transcribe_audio(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio using the loaded STT model (Vosk or Whisper)."""
        if self.use_vosk and self.vosk_model is not None:
            return self._transcribe_with_vosk(audio_array, sample_rate)
        elif self.stt_pipe is not None:
            stt_result = self.stt_pipe({"array": audio_array, "sampling_rate": sample_rate})
            return aggregate_transcript(stt_result)
        else:
            raise RuntimeError("No STT model loaded")
    
    def _transcribe_with_vosk(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """Transcribe using Vosk (non-AI)."""
        # Convert float32 audio to int16 PCM
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # Create recognizer for this transcription
        recognizer = KaldiRecognizer(self.vosk_model, sample_rate)
        recognizer.SetWords(True)
        
        # Process audio
        recognizer.AcceptWaveform(audio_int16.tobytes())
        result = json.loads(recognizer.FinalResult())
        
        text = result.get("text", "").strip()
        
        # Apply hallucination filter
        if is_whisper_hallucination(text):
            return ""
        
        return text
    
    def unload_models(self, force: bool = False):
        """Unload models to free GPU memory."""
        with self._lock:
            self._ref_count = max(0, self._ref_count - 1)
            
            if self._ref_count > 0 and not force:
                print(f"📊 Other sessions active, keeping models (ref_count: {self._ref_count})")
                return
            
            print("🧹 Unloading models to free memory...")
            
            # Clear Vosk STT
            if self.vosk_model is not None:
                del self.vosk_model
                self.vosk_model = None
            if self.vosk_recognizer is not None:
                del self.vosk_recognizer
                self.vosk_recognizer = None
            
            # Clear Whisper STT
            if self.stt_pipe is not None:
                del self.stt_pipe
                self.stt_pipe = None
            if self.stt_model is not None:
                self.stt_model.cpu()
                del self.stt_model
                self.stt_model = None
            if self.stt_processor is not None:
                del self.stt_processor
                self.stt_processor = None
            
            # Clear LLM
            if self.llm_model is not None:
                self.llm_model.cpu()
                del self.llm_model
                self.llm_model = None
            if self.llm_tokenizer is not None:
                del self.llm_tokenizer
                self.llm_tokenizer = None
            
            self.active_model_id = None
            
            # Force garbage collection and clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print("✅ Models unloaded, GPU memory freed")
    
    @property
    def is_loaded(self) -> bool:
        # Loaded if we have either Vosk or Whisper STT, plus LLM
        has_stt = self.vosk_model is not None or self.stt_model is not None
        return has_stt and self.llm_model is not None


# Global model manager
model_manager = ModelManager()

# Note: Models are now loaded lazily on first connection to support multi-user mode.
# This allows the server to start quickly and manage GPU memory efficiently.

# STT Engine info
if USE_VOSK and VOSK_AVAILABLE:
    print(f"🎤 STT Engine: Vosk (non-AI, offline)")
else:
    print(f"🎤 STT Engine: Whisper (AI-based)")

# TTS Engine info
if USE_PYTTSX3:
    print(f"🎙️ TTS Engine: pyttsx3 (offline) - Rate: {PYTTSX3_RATE} WPM")
else:
    print(f"🎙️ TTS Engine: Edge TTS (natural) - Voice: {EDGE_TTS_ACTIVE_VOICE}")
print(f"📊 Server configured for up to {MAX_CONCURRENT_SESSIONS} concurrent sessions")
print("⏳ Models will be loaded on first client connection...")


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


# Known Whisper hallucination phrases (generated on silence/noise)
WHISPER_HALLUCINATIONS = {
    # Thank you variants
    "thank you", "thanks", "thank you.", "thanks.", "thank you!", "thanks!",
    "thank you for watching", "thanks for watching", "thank you for listening",
    "thanks for listening", "thank you so much", "thanks so much",
    # Subscribe/YouTube hallucinations
    "subscribe", "like and subscribe", "please subscribe", "please like and subscribe",
    "don't forget to subscribe", "hit the subscribe button",
    # Goodbye variants
    "bye", "goodbye", "bye bye", "bye.", "see you", "see you next time",
    "see you later", "talk to you later", "take care",
    # Short meaningless responses
    "you", "you.", "me too", "me too!", "me too.", "okay", "ok",
    "yes", "no", "yeah", "yep", "nope", "sure", "right",
    "i see", "oh", "oh.", "ah", "um", "uh", "hmm",
    # Punctuation only
    "...", "…", ".", "", " ",
    # Music/sound indicators
    "♪", "♪♪", "♪ ♪", "[music]", "(music)", "[applause]", "[laughter]",
    "[silence]", "[inaudible]", "[background noise]",
    # Korean hallucinations
    "음악", "자막", "구독", "시청", "감사합니다", "안녕하세요",
    # Chinese hallucinations
    "谢谢", "再见", "订阅", "谢谢观看",
    # Other languages
    "merci", "gracias", "danke", "arigato", "arigatou",
    # Subtitles indicators
    "subtitles", "captions", "subtitles by", "translated by",
    # End indicators
    "the end", "the end.", "fin", "end",
    # Common short hallucinations
    "you're welcome", "sure thing", "no problem", "of course",
    "that's right", "i know", "i understand", "got it",
}

# Patterns that indicate hallucination (regex)
HALLUCINATION_PATTERNS = [
    r"^[\s\.\,\!\?\-]+$",  # Only punctuation/whitespace
    r"^\[.*\]$",  # [anything in brackets]
    r"^\(.*\)$",  # (anything in parens)
    r"^♪+$",  # Music symbols
    r"^MBC\s+뉴스",  # Korean news hallucination
    r"이준범입니다",  # Korean name hallucination
]


def is_whisper_hallucination(text: str) -> bool:
    """Check if text is a known Whisper hallucination."""
    if not text:
        return True
    
    cleaned = text.strip().lower()
    
    # Check exact matches
    if cleaned in WHISPER_HALLUCINATIONS:
        return True
    
    # Check if too short (likely noise)
    if len(cleaned) < 2:
        return True
    
    # Check patterns
    for pattern in HALLUCINATION_PATTERNS:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True
    
    # Check if contains mostly non-ASCII (likely wrong language detection)
    ascii_chars = sum(1 for c in cleaned if ord(c) < 128 and c.isalpha())
    non_ascii = sum(1 for c in cleaned if ord(c) >= 128)
    if non_ascii > ascii_chars and non_ascii > 3:
        print(f"⚠️ Filtering non-English hallucination: {text}")
        return True
    
    return False


def aggregate_transcript(stt_output: dict | list) -> str:
    if isinstance(stt_output, dict):
        chunks = stt_output.get("chunks")
        if chunks:
            combined = " ".join(chunk.get("text", "") for chunk in chunks)
            result = clean_transcript_text(combined)
            if is_whisper_hallucination(result):
                return ""
            return result
        result = clean_transcript_text(stt_output.get("text", ""))
        if is_whisper_hallucination(result):
            return ""
        return result
    if isinstance(stt_output, list):
        combined = " ".join(
            item.get("text", "") for item in stt_output if isinstance(item, dict)
        )
        result = clean_transcript_text(combined)
        if is_whisper_hallucination(result):
            return ""
        return result
    result = clean_transcript_text(str(stt_output))
    if is_whisper_hallucination(result):
        return ""
    return result


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


# Minimum audio duration to avoid Whisper hallucinations (in seconds)
# Increased to 0.8s to better filter out noise/clicks
MIN_AUDIO_DURATION_SEC = 0.8

async def handle_user_turn(
    audio_payload: bytes,
    websocket: WebSocket,
    conversation_history: list[dict],
    mime_type: str | None = None,
    mute_audio: bool = False,
) -> None:
    """Process user audio input and generate AI response."""
    if not audio_payload:
        return
    
    # Ensure models are loaded
    if not model_manager.is_loaded:
        print("⚠️ Models not loaded, loading now...")
        model_manager.load_models()

    try:
        audio_array, sampling_rate = decode_audio_payload(audio_payload, mime_type)
    except Exception as e:
        print(f"⚠️ Error: {e}")
        await websocket.send_json({"status": "audio-error", "message": "Failed to decode audio"})
        return

    # Check minimum audio duration to avoid hallucinations
    audio_duration = len(audio_array) / sampling_rate
    if audio_duration < MIN_AUDIO_DURATION_SEC:
        print(f"⚠️ Skipping short audio ({audio_duration:.2f}s < {MIN_AUDIO_DURATION_SEC}s)")
        await websocket.send_json({"status": "no-speech", "message": "Audio too short"})
        return

    # Use model manager's transcribe method (Vosk or Whisper)
    user_text = model_manager.transcribe_audio(audio_array, sampling_rate)

    if not user_text:
        print("⚠️ Skipping turn (no speech detected)")
        await websocket.send_json({"status": "no-speech"})
        return

    print(f"User: {user_text}")
    
    # Send user's transcribed text back to client for display
    await websocket.send_json({"user_text": user_text})
    
    conversation_history.append({"role": "user", "content": user_text})
    trim_history(conversation_history)

    # Use model manager's tokenizer and LLM
    input_ids = model_manager.llm_tokenizer.apply_chat_template(
        conversation_history,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)
    
    # Create attention mask to avoid warnings
    attention_mask = torch.ones_like(input_ids).to(device)

    streamer = TextIteratorStreamer(
        model_manager.llm_tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        repetition_penalty=LLM_REPETITION_PENALTY,
        pad_token_id=model_manager.llm_tokenizer.pad_token_id,
    )

    thread = Thread(target=model_manager.llm_model.generate, kwargs=generation_kwargs)
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

    # Use pyttsx3 (offline) if enabled, otherwise fall back to Edge TTS
    if USE_PYTTSX3:
        wav_bytes = await asyncio.to_thread(synthesize_with_pyttsx3, normalized)
    else:
        wav_bytes = await synthesize_with_edge_tts(normalized)

    if not wav_bytes:
        return None

    return base64.b64encode(wav_bytes).decode("utf-8")


# --- WEBSOCKET ENDPOINT ---
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Generate unique session ID
    import uuid
    session_id = str(uuid.uuid4())
    
    # Track session for multi-user support
    with session_lock:
        if len(active_sessions) >= MAX_CONCURRENT_SESSIONS:
            await websocket.send_json({
                "status": "error",
                "message": f"Server at capacity ({MAX_CONCURRENT_SESSIONS} users). Please try again later."
            })
            await websocket.close()
            return
        active_sessions[session_id] = {"connected_at": asyncio.get_event_loop().time()}
    
    # Load models for this session
    model_manager.load_models()
    
    print(f"✅ Client Connected (Session: {session_id[:8]}..., Active: {len(active_sessions)})")

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
        print(f"❌ Client Disconnected (Session: {session_id[:8]}...)")
    except Exception as e:
        print(f"⚠️ Error: {e}")
        await websocket.close()
    finally:
        # Clean up session
        with session_lock:
            if session_id in active_sessions:
                del active_sessions[session_id]
        
        # Unload models if no more sessions
        model_manager.unload_models()
        print(f"📊 Active sessions: {len(active_sessions)}")


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
