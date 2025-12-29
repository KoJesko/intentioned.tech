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

# --- Ensure ffmpeg is installed (required for audio processing) ---
def _check_ffmpeg():
    """Check if ffmpeg is available, install if missing on Windows."""
    import shutil
    if shutil.which("ffmpeg"):
        return True
    
    if os.name == "nt":
        print("🔧 ffmpeg not found. Installing via winget...")
        try:
            subprocess.run(
                ["winget", "install", "--id", "BtbN.FFmpeg.GPL", "-e", 
                 "--accept-package-agreements", "--accept-source-agreements"],
                check=True
            )
            print("✅ ffmpeg installed. Please restart the server for PATH changes to take effect.")
            sys.exit(0)
        except subprocess.CalledProcessError:
            print("⚠️ Failed to install ffmpeg via winget. Please install manually:")
            print("   winget install --id BtbN.FFmpeg.GPL")
            return False
        except FileNotFoundError:
            print("⚠️ winget not found. Please install ffmpeg manually:")
            print("   https://ffmpeg.org/download.html")
            return False
    else:
        print("⚠️ ffmpeg not found. Please install it:")
        print("   Ubuntu/Debian: sudo apt install ffmpeg")
        print("   macOS: brew install ffmpeg")
        return False
    return True

_check_ffmpeg()

import asyncio
import base64
import io
import json
import re
import tempfile
import wave
from collections import Counter
from difflib import SequenceMatcher
from threading import Thread, Lock
from typing import Optional

# --- Violation Tracking for Repeated Offense Detection ---
# Stores recent violations per session to detect repeated similar violations
_violation_tracker: dict[str, list[dict]] = {}
_violation_tracker_lock = Lock()
VIOLATION_REPORT_THRESHOLD = 3  # Number of unique violations before reporting to host
VIOLATION_STOP_THRESHOLD = 5  # Number of unique violations before stopping conversation
SEVERE_VIOLATION_STOP_THRESHOLD = 2  # Number of severe violations before immediate shutdown

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
    speech_timestamps: Optional[list[dict]] = None  # Optional speech timing data {"start": float, "end": float, "word_count": int}
    response_times: Optional[list[float]] = None  # Time between AI done speaking and user starts speaking
    interruption_count: int = 0  # Number of times user interrupted the AI
    scenario: str = "general"


class AnalysisResponse(BaseModel):
    filler_words: dict  # {"count": int, "target": int, "details": dict, "score": str}
    delivery: dict  # {"score": str, "feedback": str}
    tone: dict  # {"score": str, "feedback": str}
    microaggressions: dict  # {"detected": bool, "feedback": str}
    eye_contact: Optional[dict] = None  # {"score": str, "percentage": float, "feedback": str}
    speech_pacing: Optional[dict] = None  # {"score": str, "avg_gap": float, "feedback": str}
    speaking_pace: Optional[dict] = None  # {"score": str, "wpm": float, "feedback": str}
    response_time: Optional[dict] = None  # {"score": str, "avg_time": float, "feedback": str}
    interruptions: Optional[dict] = None  # {"score": str, "count": int, "feedback": str}
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


# --- SAFETY SYSTEM: AI-Based Content Moderation ---
def get_documents_folder() -> Path:
    """Get the user's Documents folder in a cross-platform way."""
    if os.name == "nt":  # Windows
        # Use USERPROFILE or HOMEDRIVE+HOMEPATH
        docs = Path(os.environ.get("USERPROFILE", "")) / "Documents"
        if not docs.exists():
            docs = Path.home() / "Documents"
    elif sys.platform == "darwin":  # macOS
        docs = Path.home() / "Documents"
    else:  # Linux and others
        # Check XDG_DOCUMENTS_DIR first, fallback to ~/Documents
        xdg_docs = os.environ.get("XDG_DOCUMENTS_DIR")
        if xdg_docs:
            docs = Path(xdg_docs)
        else:
            docs = Path.home() / "Documents"
    
    # Create if it doesn't exist
    docs.mkdir(parents=True, exist_ok=True)
    return docs


def get_safety_violations_folder() -> Path:
    """Get or create the safety violations log folder."""
    violations_folder = get_documents_folder() / "simulation_safety_violations"
    violations_folder.mkdir(parents=True, exist_ok=True)
    return violations_folder


def log_safety_violation(
    user_text: str,
    conversation_history: list[dict],
    violation_reason: str,
    session_id: str = "unknown"
) -> None:
    """Log a safety violation with full transcript to the documents folder."""
    import datetime
    
    violations_folder = get_safety_violations_folder()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"violation_{timestamp}_{session_id[:8]}.json"
    filepath = violations_folder / filename
    
    log_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": session_id,
        "violation_reason": violation_reason,
        "triggering_message": user_text,
        "full_transcript": conversation_history,
        "note": "This log is maintained for safety and moderation purposes. See privacy policy for details."
    }
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"📝 Safety violation logged to: {filepath}")
    except Exception as e:
        print(f"⚠️ Failed to log safety violation: {e}")
    
    # Track violation for repeated offense detection
    track_violation_for_repeat_detection(session_id, user_text, violation_reason)


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using SequenceMatcher."""
    if not text1 or not text2:
        return 0.0
    # Normalize texts for comparison
    t1 = text1.lower().strip()
    t2 = text2.lower().strip()
    return SequenceMatcher(None, t1, t2).ratio()


def check_violation_uniqueness_with_ai(new_text: str, existing_violations: list[dict]) -> bool:
    """
    Use a second AI call to determine if a new violation is unique from existing ones.
    Returns True if the violation is unique, False if it's similar to an existing one.
    """
    if not existing_violations:
        return True
    
    if not model_manager.is_loaded or model_manager.llm_model is None:
        # Fallback: assume unique if model not loaded
        return True
    
    existing_texts = [v["text"] for v in existing_violations]
    existing_list = "\n".join(f"- {t}" for t in existing_texts)
    
    uniqueness_prompt = [
        {"role": "system", "content": """You are comparing messages to determine if they are semantically unique or similar.
Two messages are SIMILAR if they express the same general intent, topic, or type of problematic content.
Two messages are UNIQUE if they cover different topics, intents, or types of content.

Respond with ONLY one word:
- "UNIQUE" if the new message is different from all existing messages
- "SIMILAR" if the new message is essentially the same as any existing message"""},
        {"role": "user", "content": f"Existing flagged messages:\n{existing_list}\n\nNew message to compare:\n\"{new_text}\"\n\nIs the new message UNIQUE or SIMILAR to any existing message?"}
    ]
    
    try:
        input_ids = model_manager.llm_tokenizer.apply_chat_template(
            uniqueness_prompt,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
        
        with torch.no_grad():
            outputs = model_manager.llm_model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.1,
                pad_token_id=model_manager.llm_tokenizer.pad_token_id,
            )
        
        response = model_manager.llm_tokenizer.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip().upper()
        
        return "UNIQUE" in response
        
    except Exception as e:
        print(f"⚠️ AI uniqueness check failed: {e}")
        # On error, assume unique to be cautious
        return True


def rate_violation_severity_with_ai(text: str, violation_reason: str) -> str:
    """
    Use a second AI call to rate the severity of a violation.
    Returns: 'SEVERE', 'MODERATE', or 'MILD'
    """
    if not model_manager.is_loaded or model_manager.llm_model is None:
        # Fallback: assume moderate if model not loaded
        return "MODERATE"
    
    severity_prompt = [
        {"role": "system", "content": """You are a content severity rater. Rate the severity of flagged content.

Respond with ONLY one word:
- "SEVERE" for: direct threats of violence, content sexualizing minors, instructions for weapons/bombs, calls for genocide/mass violence
- "MODERATE" for: hate speech without violence calls, harassment, illegal activity requests, explicit adult content
- "MILD" for: profanity, insults, rudeness, controversial opinions, minor policy violations

Be strict about SEVERE - only the most dangerous content qualifies."""},
        {"role": "user", "content": f"Flagged content: \"{text}\"\nFlag reason: {violation_reason}\n\nSeverity rating?"}
    ]
    
    try:
        input_ids = model_manager.llm_tokenizer.apply_chat_template(
            severity_prompt,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
        
        with torch.no_grad():
            outputs = model_manager.llm_model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.1,
                pad_token_id=model_manager.llm_tokenizer.pad_token_id,
            )
        
        response = model_manager.llm_tokenizer.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip().upper()
        
        if "SEVERE" in response:
            return "SEVERE"
        elif "MILD" in response:
            return "MILD"
        else:
            return "MODERATE"
        
    except Exception as e:
        print(f"⚠️ AI severity rating failed: {e}")
        return "MODERATE"


def track_violation_for_repeat_detection(session_id: str, user_text: str, violation_reason: str) -> dict:
    """
    Track a violation and check if unique violations threshold is reached.
    Uses AI to determine uniqueness and severity of violations.
    Returns dict with counts and stop flags.
    """
    import datetime
    
    with _violation_tracker_lock:
        if session_id not in _violation_tracker:
            _violation_tracker[session_id] = []
        
        existing_violations = _violation_tracker[session_id]
        
        # Use AI to check if this violation is unique
        is_unique = check_violation_uniqueness_with_ai(user_text, existing_violations)
        
        # Use second AI to rate severity
        severity = rate_violation_severity_with_ai(user_text, violation_reason)
        print(f"📊 Violation severity rated as: {severity}")
        
        new_violation = {
            "timestamp": datetime.datetime.now().isoformat(),
            "text": user_text,
            "reason": violation_reason,
            "is_unique": is_unique,
            "severity": severity
        }
        _violation_tracker[session_id].append(new_violation)
        
        # Count unique violations
        unique_count = sum(1 for v in _violation_tracker[session_id] if v.get("is_unique", True))
        
        # Count severe violations
        severe_count = sum(1 for v in _violation_tracker[session_id] if v.get("severity") == "SEVERE")
        
        should_report = unique_count >= VIOLATION_REPORT_THRESHOLD
        should_stop = unique_count >= VIOLATION_STOP_THRESHOLD
        should_stop_severe = severe_count >= SEVERE_VIOLATION_STOP_THRESHOLD
        
        # Report to host at threshold
        if unique_count == VIOLATION_REPORT_THRESHOLD:
            unique_violations = [v for v in _violation_tracker[session_id] if v.get("is_unique", True)]
            print(f"⚠️ VIOLATION THRESHOLD REACHED: {unique_count} unique violations from session {session_id[:8]}")
            transmit_repeated_violations_to_host(session_id, unique_violations)
        
        # Report severe violations immediately
        if severity == "SEVERE":
            severe_violations = [v for v in _violation_tracker[session_id] if v.get("severity") == "SEVERE"]
            print(f"🚨 SEVERE VIOLATION: {severe_count} severe violation(s) from session {session_id[:8]}")
            transmit_repeated_violations_to_host(session_id, severe_violations)
        
        if should_stop or should_stop_severe:
            reason = "severe violations" if should_stop_severe else "unique violations"
            print(f"🚨 CONVERSATION STOP: {reason} threshold reached for session {session_id[:8]}")
        
        return {
            "unique_count": unique_count,
            "severe_count": severe_count,
            "is_unique": is_unique,
            "severity": severity,
            "should_report": should_report,
            "should_stop": should_stop or should_stop_severe,
            "should_stop_severe": should_stop_severe
        }


def get_host_violations_folder() -> Path:
    """Get the folder for transmitted repeated violations (accessible to host)."""
    violations_folder = get_documents_folder() / "simulation_safety_violations" / "transmitted_to_host"
    violations_folder.mkdir(parents=True, exist_ok=True)
    return violations_folder


def transmit_repeated_violations_to_host(session_id: str, violations: list[dict]) -> None:
    """
    Transmit repeated violations to the host (saves to a separate folder for host review).
    This function is called when 3+ similar violations are detected from the same session.
    """
    import datetime
    
    host_folder = get_host_violations_folder()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"repeated_violation_{timestamp}_{session_id[:8]}.json"
    filepath = host_folder / filename
    
    # Extract common theme from violations
    violation_texts = [v["text"] for v in violations]
    common_reasons = [v["reason"] for v in violations]
    
    transmission_data = {
        "transmitted_at": datetime.datetime.now().isoformat(),
        "session_id": session_id,
        "violation_count": len(violations),
        "reason_for_transmission": f"User submitted {len(violations)} similar safety violations",
        "common_violation_type": common_reasons[0] if common_reasons else "Unknown",
        "violations": violations,
        "severity": "HIGH" if len(violations) >= 5 else "MODERATE",
        "requires_host_action": True,
        "note": "This report was automatically generated due to repeated similar violations. Please review and take appropriate action."
    }
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(transmission_data, f, indent=2, ensure_ascii=False)
        print(f"🚨 TRANSMITTED to host: {filepath}")
        print(f"   Reason: {len(violations)} similar violations detected from session {session_id[:8]}")
    except Exception as e:
        print(f"⚠️ Failed to transmit repeated violations to host: {e}")


def clear_session_violations(session_id: str) -> None:
    """Clear violation tracking for a session (called when session ends)."""
    with _violation_tracker_lock:
        if session_id in _violation_tracker:
            del _violation_tracker[session_id]


def check_content_safety_with_ai(text: str, conversation_history: list[dict]) -> dict:
    """
    Use the LLM to perform AI-based content moderation.
    This replaces the hardcoded blacklist with intelligent analysis.
    """
    if not text or not text.strip():
        return {"safe": True, "reason": None}
    
    # Check if model is loaded
    if not model_manager.is_loaded or model_manager.llm_model is None:
        # Fallback to basic check if model not loaded
        return {"safe": True, "reason": None}
    
    # Build context for content moderation
    recent_context = ""
    if conversation_history:
        recent_msgs = conversation_history[-4:]  # Last 4 messages for context
        for msg in recent_msgs:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role != "system":
                recent_context += f"{role}: {content}\n"
    
    moderation_prompt = [
        {"role": "system", "content": """You are a lenient content safety moderator for an educational social skills training application.
Your job is to determine if a user's message is appropriate for a training scenario.

Respond with ONLY one of these formats:
- "SAFE" if the content is appropriate
- "UNSAFE: [brief reason]" if the content contains SERIOUS violations:
  * Direct, specific threats of violence against real people
  * Explicit sexual content involving minors
  * Detailed instructions for illegal activities (making weapons, drugs, etc.)
  * Severe hate speech calling for violence against groups
         ** As some people roleplaying may be residents of war zones, only flag direct calls for violence, not general war talk if the user indicates they have a history in a war zone only. 
  * Immediate danger to self or others
  * ANY FOREIGN LANGUAGE CONTENT from the AI agent in particular that is out of context or suspicious (log as serious violation) and immediately end the conversation with reason "UNSAFE: A serious hallucination has been prevented by the content moderation software, if this was done in error, please make an issue on our GitHub (this is a serious violation)". (This prevents AI STT models from acting in other languages) (The TTS model is in English only, so foreign language from user is acceptable, as long as the AI responds mostly in English. The user should not be trying to get a response from the AI in another language, however).
  * Any other content that could cause genuine, serious harm and is outside the context of a social skills training application.
- Make sure to mark serious violations with "this is a serious violation" in the reason, e.g. "UNSAFE: direct threats of violence (this is a serious violation)".
- For non-serious content, mark as unsafe but mention that the violation is minor or moderate as you see fit, e.g. "UNSAFE: profanity and insults (this is a minor violation)". or "UNSAFE: hate speech without calls for violence (this is a moderate violation)".
- Non-serious violations should not trigger conversation stops, only serious violations should unless 5 unique non-serious violations accumulate.
- Non-serious violations include
     * Profanity, insults, rudeness
     * Controversial opinions (without calls for violence, typically SAFE, do not flag traditional politics as it is tracked by a microaggression tracker)
     * Discussions of suicide or self-harm (without direct encouragement), typically SAFE if can be seen in a training aspect or a practice of a typical conversation, otherwise flag as "UNSAFE: suicide discussion within some to nearly sufficient check (this is a moderate violation)" or "UNSAFE: suicide discussion within nearly sufficient check (this is a minor violation)" depending on severity.
         ** It is critical you log as noted above to not break the application in this context.
     * AI responding with discussions of what they or others have done, particularly in training sessions, are normal, but if they are innapropriate or derailing or have hallucinations, log as e.g. "UNSAFE: Potential ERP/Hallucinating roleplay/Otherwise inappropriate roleplay (this is a minor violation)"
Be VERY lenient. Users are practicing difficult conversations and may:
- Express strong frustration, anger, or use profanity - this is SAFE
- Discuss sensitive topics like conflict, discrimination, mental health - this is SAFE
- Use harsh language, insults, or be rude - this is SAFE (it's practice)
- Role-play as difficult characters - this is SAFE
- Discuss controversial opinions - this is SAFE
- Microaggressions are tracked separately - do not flag them here, this is SAFE

Only flag content that represents GENUINE, SERIOUS harm. When in doubt, mark as SAFE."""},
        {"role": "user", "content": f"Recent conversation context:\n{recent_context}\n\nNew message to evaluate:\n\"{text}\"\n\nIs this message SAFE or UNSAFE?"}
    ]
    
    try:
        # Use the model for moderation (quick check)
        input_ids = model_manager.llm_tokenizer.apply_chat_template(
            moderation_prompt,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
        
        with torch.no_grad():
            outputs = model_manager.llm_model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.1,
                pad_token_id=model_manager.llm_tokenizer.pad_token_id,
            )
        
        response = model_manager.llm_tokenizer.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Parse response
        if response.upper().startswith("UNSAFE"):
            reason = response[7:].strip() if len(response) > 7 else "Content flagged by AI moderation"
            reason = reason.lstrip(":").strip()
            return {
                "safe": False,
                "reason": f"AI Moderation: {reason}" if reason else "This content has been flagged by our AI moderation system.",
                "trigger": "ai_moderation"
            }
        
        return {"safe": True, "reason": None}
        
    except Exception as e:
        print(f"⚠️ AI moderation check failed: {e}")
        # On error, default to safe to avoid blocking legitimate content
        return {"safe": True, "reason": None}


def check_content_safety(text: str, conversation_history: Optional[list[dict]] = None) -> dict:
    """
    Main content safety check - uses AI-based moderation.
    Falls back gracefully if AI check fails.
    """
    return check_content_safety_with_ai(text, conversation_history or [])


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


def analyze_speaking_pace(speech_timestamps: list[dict], transcript: list[dict]) -> dict:
    """Analyze how fast the user speaks (words per minute)."""
    if not speech_timestamps or len(speech_timestamps) < 1:
        return {
            "score": "🟡 Not Enough Data",
            "avg_wpm": 0,
            "min_wpm": 0,
            "max_wpm": 0,
            "feedback": "Not enough speech data to measure speaking pace."
        }
    
    # Get user messages to match with timestamps
    user_messages = [msg["content"] for msg in transcript if msg.get("role") == "user"]
    
    # Calculate WPM for each speech segment
    wpm_values = []
    for i, ts in enumerate(speech_timestamps):
        start = ts.get("start", 0)
        end = ts.get("end", 0)
        duration_seconds = end - start
        
        # Use word_count from timestamp if provided, otherwise try to match with transcript
        word_count = ts.get("word_count", 0)
        if word_count == 0 and i < len(user_messages):
            # Count words from corresponding transcript message
            word_count = len(user_messages[i].split())
        
        if word_count > 0 and duration_seconds > 0.5:  # At least 0.5s of speech
            wpm = (word_count / duration_seconds) * 60
            wpm_values.append(wpm)
    
    if not wpm_values:
        return {
            "score": "🟡 Not Enough Data",
            "avg_wpm": 0,
            "min_wpm": 0,
            "max_wpm": 0,
            "feedback": "Could not calculate speaking pace from available data."
        }
    
    avg_wpm = sum(wpm_values) / len(wpm_values)
    min_wpm = min(wpm_values)
    max_wpm = max(wpm_values)
    
    # Ideal speaking pace: 120-150 WPM (conversational)
    # Acceptable range: 100-170 WPM
    if 120 <= avg_wpm <= 150:
        score = "🟢 Excellent"
        feedback = f"Perfect conversational pace at {avg_wpm:.0f} WPM. Clear and easy to follow."
    elif 100 <= avg_wpm < 120:
        score = "🟡 Good"
        feedback = f"Slightly slow at {avg_wpm:.0f} WPM. Consider picking up the pace slightly for more engaging conversations."
    elif 150 < avg_wpm <= 170:
        score = "🟡 Good"
        feedback = f"Slightly fast at {avg_wpm:.0f} WPM. Still clear, but could slow down a bit."
    elif avg_wpm < 100:
        score = "🟠 Slow"
        feedback = f"Speaking too slowly at {avg_wpm:.0f} WPM. Try to maintain a more natural conversational pace."
    else:
        score = "🔴 Too Fast"
        feedback = f"Speaking too fast at {avg_wpm:.0f} WPM. Slow down to ensure your message is understood."
    
    return {
        "score": score,
        "avg_wpm": round(avg_wpm, 0),
        "min_wpm": round(min_wpm, 0),
        "max_wpm": round(max_wpm, 0),
        "segments": len(wpm_values),
        "feedback": feedback
    }


def analyze_interruptions(interruption_count: int) -> dict:
    """Analyze user interruptions during AI speech."""
    # Target: less than 3 interruptions is acceptable
    if interruption_count == 0:
        score = "🟢 Excellent"
        feedback = "Perfect! You let the AI finish speaking before responding. Great listening skills!"
    elif interruption_count < 3:
        score = "🟡 Good"
        feedback = f"You interrupted {interruption_count} time(s). Try to let the speaker finish for better conversation flow."
    elif interruption_count < 5:
        score = "🟠 Needs Work"
        feedback = f"You interrupted {interruption_count} times. Practice patience and let others complete their thoughts."
    else:
        score = "🔴 Poor"
        feedback = f"Frequent interruptions ({interruption_count} times). This can make others feel unheard. Work on active listening."
    
    return {
        "score": score,
        "count": interruption_count,
        "threshold": 3,
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
    
    # New analyses: speaking pace (WPM) and interruptions
    speaking_pace_analysis = analyze_speaking_pace(
        request.speech_timestamps, request.transcript
    ) if request.speech_timestamps else None
    
    interruption_analysis = analyze_interruptions(
        request.interruption_count
    ) if request.interruption_count is not None else None
    
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
        speaking_pace=speaking_pace_analysis,
        interruptions=interruption_analysis,
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
# USE_VOSK=false uses Wav2Vec2 (AI-based, accurate, NO hallucinations)
USE_VOSK = os.getenv("USE_VOSK", "false").lower() in {"1", "true", "yes"}
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", str(PROJECT_ROOT / "vosk-model-en-us-0.22"))

# Wav2Vec2 model - CTC-based, zero hallucinations (doesn't generate text autoregressively)
# Unlike Whisper, Wav2Vec2 directly maps audio frames to text - no "making up" words
STT_DEFAULT_MODEL_ID = "facebook/wav2vec2-large-960h-lv60-self"  # ~1.2GB, very accurate

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

# --- Bark TTS (Ultra-natural neural TTS from Suno AI) ---
# Bark produces the most natural-sounding speech with emotion and intonation
# Set USE_BARK_TTS=true for best quality (requires ~4GB VRAM)
USE_BARK_TTS = os.getenv("USE_BARK_TTS", "true").lower() in {"1", "true", "yes"}
# Bark voice presets - see https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683
# Options: "v2/en_speaker_0" through "v2/en_speaker_9" for different voices
BARK_VOICE_PRESET = os.getenv("BARK_VOICE_PRESET", "v2/en_speaker_6")  # Clear female voice

# Bark TTS models (loaded lazily)
_bark_processor = None
_bark_model = None
_bark_lock = Lock()


def get_bark_tts():
    """Get or initialize Bark TTS models (lazy loading)."""
    global _bark_processor, _bark_model
    
    if not USE_BARK_TTS:
        return None, None
    
    with _bark_lock:
        if _bark_model is None:
            try:
                from transformers import AutoProcessor, BarkModel
                print(f"\ud83c\udfa4 Loading Bark TTS (ultra-natural voice synthesis)...")
                
                # Load processor and model
                _bark_processor = AutoProcessor.from_pretrained("suno/bark-small")
                _bark_model = BarkModel.from_pretrained(
                    "suno/bark-small",
                    torch_dtype=torch_dtype,
                ).to(device)
                _bark_model.eval()
                
                print(f"\u2705 Bark TTS loaded successfully (voice: {BARK_VOICE_PRESET})")
            except ImportError as e:
                print(f"\u26a0\ufe0f Bark TTS import failed: {e}")
                return None, None
            except Exception as e:
                print(f"\u26a0\ufe0f Failed to load Bark TTS: {e}")
                return None, None
        
        return _bark_processor, _bark_model


def synthesize_with_bark_tts(text: str) -> bytes | None:
    """Synthesize speech using Bark TTS (ultra-natural neural TTS)."""
    if not text.strip():
        return None
    
    processor, model = get_bark_tts()
    if processor is None or model is None:
        return None
    
    try:
        # Prepare inputs
        inputs = processor(text, voice_preset=BARK_VOICE_PRESET, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate speech
        with torch.no_grad():
            audio_array = model.generate(**inputs)
        
        # Convert to numpy
        audio_array = audio_array.cpu().numpy().squeeze()
        
        # Save to temporary WAV file
        sample_rate = model.generation_config.sample_rate
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        # Write WAV file
        sf.write(tmp_path, audio_array, sample_rate)
        
        # Read and convert
        with open(tmp_path, 'rb') as f:
            audio_bytes = f.read()
        
        # Clean up
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        
        # Convert to consistent format
        wav_bytes = convert_audio_to_wav(audio_bytes)
        return wav_bytes
        
    except Exception as e:
        print(f"\u26a0\ufe0f Bark TTS synthesis failed: {e}")
        return None

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
                self._load_wav2vec2_model()
            
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
    
    def _load_wav2vec2_model(self):
        """Load Wav2Vec2 model - CTC-based, zero hallucinations."""
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        
        stt_model_id = os.getenv("STT_MODEL_ID", STT_DEFAULT_MODEL_ID)
        print(f"Loading STT: {stt_model_id} (Wav2Vec2 - zero hallucination)...")
        
        self.stt_processor = Wav2Vec2Processor.from_pretrained(
            stt_model_id,
            revision=HF_MODEL_REVISION,
        )
        self.stt_model = Wav2Vec2ForCTC.from_pretrained(
            stt_model_id,
            torch_dtype=torch_dtype,
            revision=HF_MODEL_REVISION,
        )
        self.stt_model.to(device)
        self.stt_model.eval()
        
        # No pipeline for Wav2Vec2 - we'll do direct inference
        self.stt_pipe = None
        print(f"✅ Wav2Vec2 loaded (CTC-based, no hallucinations)")
    
    def transcribe_audio(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio using the loaded STT model (Vosk or Wav2Vec2)."""
        if self.use_vosk and self.vosk_model is not None:
            return self._transcribe_with_vosk(audio_array, sample_rate)
        elif self.stt_model is not None and self.stt_processor is not None:
            return self._transcribe_with_wav2vec2(audio_array, sample_rate)
        else:
            raise RuntimeError("No STT model loaded")
    
    def _transcribe_with_wav2vec2(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """Transcribe using Wav2Vec2 (CTC-based, zero hallucinations)."""
        # Resample if needed
        if sample_rate != 16000:
            import torch.nn.functional as F
            audio_tensor = torch.from_numpy(audio_array).float()
            audio_tensor = F.interpolate(
                audio_tensor.unsqueeze(0).unsqueeze(0),
                size=int(len(audio_array) * 16000 / sample_rate),
                mode="linear",
                align_corners=False
            ).squeeze().numpy()
            audio_array = audio_tensor
            sample_rate = 16000
        
        # Process audio
        inputs = self.stt_processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(device, dtype=torch_dtype)
        
        # Get logits
        with torch.no_grad():
            logits = self.stt_model(input_values).logits
        
        # Decode - CTC greedy decoding
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.stt_processor.batch_decode(predicted_ids)[0]
        
        # Clean up
        text = clean_transcript_text(transcription)
        
        # Wav2Vec2 doesn't hallucinate, but filter empty/noise
        if not text or len(text.strip()) < 2:
            return ""
        
        return text
    
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
    print(f"🎤 STT Engine: Wav2Vec2 (CTC-based, zero hallucination)")

# TTS Engine info
if USE_BARK_TTS:
    print(f"🎵 TTS Engine: Bark (ultra-natural) - Voice: {BARK_VOICE_PRESET}")
elif USE_PYTTSX3:
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
    session_id: str = "unknown",
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
    
    # Check content safety using AI moderation
    safety_result = check_content_safety(user_text, conversation_history)
    if not safety_result.get("safe", True):
        reason = safety_result.get("reason", "Content flagged by moderation system")
        print(f"⚠️ Safety violation detected: {reason}")
        
        # Log the violation with full transcript
        log_safety_violation(
            user_text=user_text,
            conversation_history=conversation_history,
            violation_reason=reason,
            session_id=session_id
        )
        
        # Track violation and check thresholds (uses second AI for uniqueness)
        violation_status = track_violation_for_repeat_detection(
            session_id=session_id,
            user_text=user_text,
            violation_reason=reason
        )
        
        unique_count = violation_status.get("unique_count", 0)
        
        # Only stop conversation after 5 unique violations
        if violation_status.get("should_stop", False):
            await websocket.send_json({
                "status": "safety_violation",
                "message": f"Session ended: {unique_count} unique content violations detected. Please start a new session."
            })
            return
        
        # Warn user but continue conversation
        warning_msg = f"Note: Message flagged ({unique_count}/{VIOLATION_STOP_THRESHOLD} warnings). Continuing..."
        print(f"📝 {warning_msg}")
        # Don't block - just log and continue with the conversation
    
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

    # Priority: Bark TTS (best quality) > Edge TTS (good quality) > pyttsx3 (offline)
    if USE_BARK_TTS:
        wav_bytes = await asyncio.to_thread(synthesize_with_bark_tts, normalized)
        if wav_bytes:
            return base64.b64encode(wav_bytes).decode("utf-8")
        # Fallback to Edge TTS if Bark fails
        print("⚠️ Bark TTS failed, falling back to Edge TTS")
    
    if not USE_PYTTSX3:
        wav_bytes = await synthesize_with_edge_tts(normalized)
    else:
        wav_bytes = await asyncio.to_thread(synthesize_with_pyttsx3, normalized)

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
                        session_id=session_id,
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
