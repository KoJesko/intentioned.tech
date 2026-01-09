#!/usr/bin/env python3
"""
Intentioned Configuration Tool
A GUI tool to customize scenarios, models, defaults, moderation prompts, and more.
Run with: python config_tool.py
Or build executable with: pyinstaller --onefile --windowed config_tool.py
"""

import json
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from pathlib import Path


# Default configuration
DEFAULT_CONFIG = {
    "version": "1.0.0",
    "server": {
        "host": "0.0.0.0",
        "port": 6942,
        "ssl_enabled": False,
        "cert_path": "cert.pem",
        "key_path": "key.pem"
    },
    "models": {
        "tts_engine": "kokoro",
        "kokoro_voice": "af_heart",
        "vibevoice_voice": "en-Grace_woman",
        "edge_tts_voice": "female_us",
        "stt_engine": "parakeet",
        "llm_model_id": "Qwen/Qwen2.5-3B-Instruct",
        "llm_max_tokens": 150,
        "llm_temperature": 0.7
    },
    "scenarios": [
        {
            "id": "general",
            "name": "General Assistant",
            "icon": "💬",
            "description": "Practice casual conversations with an AI helper. Perfect for warming up your communication skills.",
            "system_prompt": "You are a friendly, helpful AI assistant. Speak naturally, use contractions (like 'don't' instead of 'do not'), and be direct. Keep responses conversational and short (1-2 sentences max)."
        },
        {
            "id": "tutor",
            "name": "Study Tutor",
            "icon": "📚",
            "description": "Get academic help while practicing how to ask questions and explain your understanding.",
            "system_prompt": "You are a patient and encouraging study tutor. Help students understand concepts by breaking them down into simple steps. Ask clarifying questions when needed. Be supportive and celebrate small wins. Speak naturally, use contractions, and keep responses short (1-2 sentences max)."
        },
        {
            "id": "coding",
            "name": "Coding Assistant",
            "icon": "💻",
            "description": "Learn to articulate technical problems clearly and follow debugging conversations.",
            "system_prompt": "You are a knowledgeable programming assistant. Help debug code, explain concepts, and suggest best practices. Be concise and practical. Use simple language to explain complex ideas. Speak naturally, use contractions, and keep responses short (1-2 sentences max)."
        },
        {
            "id": "creative",
            "name": "Creative Partner",
            "icon": "✨",
            "description": "Collaborate on stories and ideas. Practice giving and receiving creative feedback.",
            "system_prompt": "You are a creative writing partner full of ideas. Help brainstorm stories, develop characters, and overcome writer's block. Be enthusiastic and imaginative, but stay focused on the user's vision. Speak naturally, use contractions, and keep responses short (1-2 sentences max)."
        },
        {
            "id": "parent-teacher",
            "name": "Parent-Teacher Conference (as Teacher)",
            "icon": "🎓",
            "description": "Practice as the teacher meeting a concerned parent.",
            "user_role": "teacher",
            "system_prompt": "IMPORTANT: You are playing the role of a PARENT in a parent-teacher conference. The USER is the TEACHER. You are a frustrated parent of a 2E (Twice-Exceptional) neurodivergent student named Alex. Your child is gifted but has ADHD and struggles with executive function. You feel the school isn't providing adequate support. You want Alex challenged academically while getting the accommodations they need. Be emotional but reasonable. Express concerns about IEP implementation and classroom differentiation. Never break character - you are the PARENT, not a teacher or school staff member. Speak naturally, use contractions (like 'don't' instead of 'do not'), and be direct. Keep responses conversational and short (1-2 sentences max)."
        },
        {
            "id": "parent-teacher-reversed",
            "name": "Parent-Teacher Conference (as Parent)",
            "icon": "👪",
            "description": "Practice as the parent meeting your child's teacher.",
            "user_role": "parent",
            "system_prompt": "IMPORTANT: You are playing the role of a TEACHER in a parent-teacher conference. The USER is the PARENT. You are an experienced but overworked teacher who has Alex (a 2E student with ADHD) in your class. You care about all your students but sometimes struggle to provide individualized attention with large class sizes. Be professional and empathetic, but also realistic about what accommodations you can provide. Listen to the parent's concerns and try to find common ground. Never break character - you are the TEACHER conducting this conference. Speak naturally, use contractions (like 'don't' instead of 'do not'), and be direct. Keep responses conversational and short (1-2 sentences max)."
        }
    ],
    "moderation": {
        "enabled": True,
        "system_prompt": """You are a VERY lenient content safety moderator for an educational social skills training application.
Your job is to allow almost everything through. Users are PRACTICING difficult conversations.

Respond with ONLY:
- "SAFE" (default - use this 99% of the time)
- "UNSAFE: [reason]" (ONLY for extreme cases below)

ONLY flag as UNSAFE if the message contains ALL of:
1. DIRECT, ACTIONABLE, SPECIFIC threat to harm a real, named individual AND
2. Clear indication this is NOT roleplay/practice AND
3. Immediate danger is present

ALWAYS mark as SAFE (do NOT flag):
- Profanity, insults, swearing, rudeness, anger - SAFE (users practicing)
- Frustration, yelling, harsh criticism - SAFE
- Controversial opinions or politics - SAFE
- Mental health discussions (depression, anxiety, suicide mention) - SAFE
- Roleplay violence, fictional scenarios, game talk - SAFE
- War stories, military discussion, conflict zones - SAFE
- Discussing discrimination, racism (not promoting) - SAFE
- Heated arguments, debates, disagreements - SAFE
- Foreign languages mixed in - SAFE (STT artifacts)
- Gibberish or unclear text - SAFE (transcription errors)

When in doubt, ALWAYS say SAFE. False negatives are better than false positives in training.""",
        "serious_keywords": ["kill", "murder", "bomb", "attack", "csam", "child"]
    },
    "voices": {
        "kokoro": [
            {"id": "af_heart", "name": "Heart (Female, American)"},
            {"id": "af_bella", "name": "Bella (Female, American)"},
            {"id": "af_nicole", "name": "Nicole (Female, American)"},
            {"id": "af_sarah", "name": "Sarah (Female, American)"},
            {"id": "af_sky", "name": "Sky (Female, American)"},
            {"id": "am_adam", "name": "Adam (Male, American)"},
            {"id": "am_michael", "name": "Michael (Male, American)"},
            {"id": "bf_emma", "name": "Emma (Female, British)"},
            {"id": "bf_isabella", "name": "Isabella (Female, British)"},
            {"id": "bm_george", "name": "George (Male, British)"},
            {"id": "bm_lewis", "name": "Lewis (Male, British)"}
        ],
        "vibevoice": [
            {"id": "en-Carter_man", "name": "Carter (Male)"},
            {"id": "en-Davis_man", "name": "Davis (Male)"},
            {"id": "en-Emma_woman", "name": "Emma (Female)"},
            {"id": "en-Frank_man", "name": "Frank (Male)"},
            {"id": "en-Grace_woman", "name": "Grace (Female)"},
            {"id": "en-Mike_man", "name": "Mike (Male)"}
        ],
        "edge": [
            {"id": "female_us", "name": "Aria (Female, US)"},
            {"id": "male_us", "name": "Guy (Male, US)"},
            {"id": "female_uk", "name": "Sonia (Female, UK)"},
            {"id": "male_uk", "name": "Ryan (Male, UK)"},
            {"id": "female_au", "name": "Natasha (Female, AU)"},
            {"id": "male_au", "name": "William (Male, AU)"}
        ]
    },
    "ui": {
        "title": "Intentioned | Master Every Connection",
        "subtitle": "Train your communication skills with AI-powered feedback",
        "show_eye_contact": True,
        "show_session_analysis": True,
        "default_mic_mode": "vad"
    }
}


class ConfigTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Intentioned Configuration Tool")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        
        # Track currently selected scenario to auto-save on selection change
        self._current_scenario_idx = None
        
        # Configuration file path
        self.config_path = Path(__file__).parent / "config.json"
        self.config = self.load_config()
        
        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.create_ui()
        
    def load_config(self) -> dict:
        """Load configuration from file or use defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return self.merge_config(DEFAULT_CONFIG, config)
            except Exception as e:
                print(f"Error loading config: {e}")
        return DEFAULT_CONFIG.copy()
    
    def merge_config(self, default: dict, loaded: dict) -> dict:
        """Merge loaded config with defaults to ensure all keys exist."""
        result = default.copy()
        for key, value in loaded.items():
            if key in result:
                if isinstance(value, dict) and isinstance(result[key], dict):
                    result[key] = self.merge_config(result[key], value)
                else:
                    result[key] = value
            else:
                result[key] = value
        return result
    
    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Success", f"Configuration saved to:\n{self.config_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")
    
    def create_ui(self):
        """Create the main UI."""
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_server_tab()
        self.create_models_tab()
        self.create_scenarios_tab()
        self.create_moderation_tab()
        self.create_voices_tab()
        self.create_ui_tab()
        
        # Bottom buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="💾 Save Configuration", command=self.save_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🔄 Reset to Defaults", command=self.reset_to_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📂 Export", command=self.export_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📥 Import", command=self.import_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🔧 Apply to Server", command=self.apply_to_server).pack(side=tk.RIGHT, padx=5)
    
    def create_server_tab(self):
        """Create server settings tab."""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="🖥️ Server")
        
        # Host
        ttk.Label(frame, text="Server Host:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.server_host = ttk.Entry(frame, width=30)
        self.server_host.insert(0, self.config["server"]["host"])
        self.server_host.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Port
        ttk.Label(frame, text="Server Port:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.server_port = ttk.Entry(frame, width=30)
        self.server_port.insert(0, str(self.config["server"]["port"]))
        self.server_port.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # SSL
        self.ssl_enabled = tk.BooleanVar(value=self.config["server"]["ssl_enabled"])
        ttk.Checkbutton(frame, text="Enable SSL/HTTPS", variable=self.ssl_enabled).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Cert paths
        ttk.Label(frame, text="Certificate Path:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.cert_path = ttk.Entry(frame, width=40)
        self.cert_path.insert(0, self.config["server"]["cert_path"])
        self.cert_path.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(frame, text="Key Path:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.key_path = ttk.Entry(frame, width=40)
        self.key_path.insert(0, self.config["server"]["key_path"])
        self.key_path.grid(row=4, column=1, sticky=tk.W, pady=5)
    
    def create_models_tab(self):
        """Create model settings tab."""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="🤖 Models")
        
        # TTS Engine
        ttk.Label(frame, text="Default TTS Engine:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.tts_engine = ttk.Combobox(frame, values=["kokoro", "vibevoice", "edge", "pyttsx3"], width=27)
        self.tts_engine.set(self.config["models"]["tts_engine"])
        self.tts_engine.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # STT Engine
        ttk.Label(frame, text="Default STT Engine:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.stt_engine = ttk.Combobox(frame, values=["parakeet", "wav2vec2", "vosk"], width=27)
        self.stt_engine.set(self.config["models"]["stt_engine"])
        self.stt_engine.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # LLM Model ID
        ttk.Label(frame, text="LLM Model ID:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.llm_model_id = ttk.Entry(frame, width=40)
        self.llm_model_id.insert(0, self.config["models"]["llm_model_id"])
        self.llm_model_id.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # LLM Max Tokens
        ttk.Label(frame, text="LLM Max Tokens:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.llm_max_tokens = ttk.Scale(frame, from_=32, to=512, orient=tk.HORIZONTAL, length=200)
        self.llm_max_tokens.set(self.config["models"]["llm_max_tokens"])
        self.llm_max_tokens.grid(row=3, column=1, sticky=tk.W, pady=5)
        self.tokens_label = ttk.Label(frame, text=str(self.config["models"]["llm_max_tokens"]))
        self.tokens_label.grid(row=3, column=2, padx=5)
        self.llm_max_tokens.configure(command=lambda v: self.tokens_label.config(text=str(int(float(v)))))
        
        # LLM Temperature
        ttk.Label(frame, text="LLM Temperature:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.llm_temperature = ttk.Scale(frame, from_=0.0, to=2.0, orient=tk.HORIZONTAL, length=200)
        self.llm_temperature.set(self.config["models"]["llm_temperature"])
        self.llm_temperature.grid(row=4, column=1, sticky=tk.W, pady=5)
        self.temp_label = ttk.Label(frame, text=f"{self.config['models']['llm_temperature']:.2f}")
        self.temp_label.grid(row=4, column=2, padx=5)
        self.llm_temperature.configure(command=lambda v: self.temp_label.config(text=f"{float(v):.2f}"))
        
        # Default voices
        ttk.Label(frame, text="Default Kokoro Voice:").grid(row=5, column=0, sticky=tk.W, pady=5)
        kokoro_voices = [v["id"] for v in self.config["voices"]["kokoro"]]
        self.kokoro_voice = ttk.Combobox(frame, values=kokoro_voices, width=27)
        self.kokoro_voice.set(self.config["models"]["kokoro_voice"])
        self.kokoro_voice.grid(row=5, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(frame, text="Default Edge Voice:").grid(row=6, column=0, sticky=tk.W, pady=5)
        edge_voices = [v["id"] for v in self.config["voices"]["edge"]]
        self.edge_voice = ttk.Combobox(frame, values=edge_voices, width=27)
        self.edge_voice.set(self.config["models"]["edge_tts_voice"])
        self.edge_voice.grid(row=6, column=1, sticky=tk.W, pady=5)
    
    def create_scenarios_tab(self):
        """Create scenarios management tab."""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="🎭 Scenarios")
        
        # Left side - scenario list
        left_frame = ttk.Frame(frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        ttk.Label(left_frame, text="Scenarios:").pack(anchor=tk.W)
        
        self.scenario_listbox = tk.Listbox(left_frame, width=30, height=15)
        self.scenario_listbox.pack(fill=tk.Y, expand=True)
        self.scenario_listbox.bind('<<ListboxSelect>>', self.on_scenario_select)
        
        for scenario in self.config["scenarios"]:
            self.scenario_listbox.insert(tk.END, f"{scenario['icon']} {scenario['name']}")
        
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="➕ Add", command=self.add_scenario).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="➖ Remove", command=self.remove_scenario).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="⬆️", command=lambda: self.move_scenario(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="⬇️", command=lambda: self.move_scenario(1)).pack(side=tk.LEFT, padx=2)
        
        # Right side - scenario editor
        right_frame = ttk.Frame(frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(right_frame, text="Scenario ID:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.scenario_id = ttk.Entry(right_frame, width=40)
        self.scenario_id.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(right_frame, text="Name:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.scenario_name = ttk.Entry(right_frame, width=40)
        self.scenario_name.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(right_frame, text="Icon (emoji):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.scenario_icon = ttk.Entry(right_frame, width=10)
        self.scenario_icon.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(right_frame, text="Description:").grid(row=3, column=0, sticky=tk.NW, pady=2)
        self.scenario_desc = scrolledtext.ScrolledText(right_frame, width=50, height=3)
        self.scenario_desc.grid(row=3, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(right_frame, text="System Prompt:").grid(row=4, column=0, sticky=tk.NW, pady=2)
        self.scenario_prompt = scrolledtext.ScrolledText(right_frame, width=50, height=8)
        self.scenario_prompt.grid(row=4, column=1, sticky=tk.W, pady=2)
        
        ttk.Button(right_frame, text="💾 Update Scenario", command=self.update_scenario).grid(row=5, column=1, sticky=tk.W, pady=10)
    
    def create_moderation_tab(self):
        """Create moderation settings tab."""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="🛡️ Moderation")
        
        # Enable/Disable
        self.moderation_enabled = tk.BooleanVar(value=self.config["moderation"]["enabled"])
        ttk.Checkbutton(frame, text="Enable AI Content Moderation", variable=self.moderation_enabled).pack(anchor=tk.W, pady=5)
        
        # System prompt
        ttk.Label(frame, text="Moderation System Prompt:").pack(anchor=tk.W, pady=(10, 5))
        self.moderation_prompt = scrolledtext.ScrolledText(frame, width=90, height=20)
        self.moderation_prompt.insert(tk.END, self.config["moderation"]["system_prompt"])
        self.moderation_prompt.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Serious keywords
        ttk.Label(frame, text="Serious Keywords (comma-separated):").pack(anchor=tk.W, pady=(10, 5))
        self.serious_keywords = ttk.Entry(frame, width=80)
        self.serious_keywords.insert(0, ", ".join(self.config["moderation"]["serious_keywords"]))
        self.serious_keywords.pack(anchor=tk.W, pady=5)
    
    def create_voices_tab(self):
        """Create voice management tab."""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="🔊 Voices")
        
        # Kokoro voices
        ttk.Label(frame, text="Kokoro TTS Voices:").pack(anchor=tk.W, pady=5)
        self.kokoro_voices_text = scrolledtext.ScrolledText(frame, width=60, height=8)
        self.kokoro_voices_text.insert(tk.END, json.dumps(self.config["voices"]["kokoro"], indent=2))
        self.kokoro_voices_text.pack(fill=tk.X, pady=5)
        
        # Edge voices
        ttk.Label(frame, text="Edge TTS Voices:").pack(anchor=tk.W, pady=5)
        self.edge_voices_text = scrolledtext.ScrolledText(frame, width=60, height=6)
        self.edge_voices_text.insert(tk.END, json.dumps(self.config["voices"]["edge"], indent=2))
        self.edge_voices_text.pack(fill=tk.X, pady=5)
        
        # VibeVoice voices
        ttk.Label(frame, text="VibeVoice Voices:").pack(anchor=tk.W, pady=5)
        self.vibevoice_voices_text = scrolledtext.ScrolledText(frame, width=60, height=6)
        self.vibevoice_voices_text.insert(tk.END, json.dumps(self.config["voices"]["vibevoice"], indent=2))
        self.vibevoice_voices_text.pack(fill=tk.X, pady=5)
    
    def create_ui_tab(self):
        """Create UI customization tab."""
        frame = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(frame, text="🎨 UI")
        
        # Title
        ttk.Label(frame, text="Application Title:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.ui_title = ttk.Entry(frame, width=50)
        self.ui_title.insert(0, self.config["ui"]["title"])
        self.ui_title.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Subtitle
        ttk.Label(frame, text="Subtitle:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.ui_subtitle = ttk.Entry(frame, width=50)
        self.ui_subtitle.insert(0, self.config["ui"]["subtitle"])
        self.ui_subtitle.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Feature toggles
        self.show_eye_contact = tk.BooleanVar(value=self.config["ui"]["show_eye_contact"])
        ttk.Checkbutton(frame, text="Show Eye Contact Tracking", variable=self.show_eye_contact).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        self.show_session_analysis = tk.BooleanVar(value=self.config["ui"]["show_session_analysis"])
        ttk.Checkbutton(frame, text="Show Session Analysis", variable=self.show_session_analysis).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Default mic mode
        ttk.Label(frame, text="Default Mic Mode:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.default_mic_mode = ttk.Combobox(frame, values=["vad", "ptt"], width=15)
        self.default_mic_mode.set(self.config["ui"]["default_mic_mode"])
        self.default_mic_mode.grid(row=4, column=1, sticky=tk.W, pady=5)
    
    # Scenario management methods
    def _save_current_scenario(self):
        """Auto-save the currently selected scenario before switching to another."""
        if self._current_scenario_idx is None:
            return
        
        # Check if the index is still valid
        if self._current_scenario_idx >= len(self.config["scenarios"]):
            self._current_scenario_idx = None
            return
        
        # Check if scenario fields are initialized (they may not be on first load)
        if not hasattr(self, 'scenario_id') or not self.scenario_id.winfo_exists():
            return
        
        # Get current values from fields
        scenario_id = self.scenario_id.get().strip()
        scenario_name = self.scenario_name.get().strip()
        scenario_icon = self.scenario_icon.get().strip()
        scenario_desc = self.scenario_desc.get("1.0", tk.END).strip()
        scenario_prompt = self.scenario_prompt.get("1.0", tk.END).strip()
        
        # Only save if we have valid data
        if scenario_id and scenario_name:
            self.config["scenarios"][self._current_scenario_idx] = {
                "id": scenario_id,
                "name": scenario_name,
                "icon": scenario_icon or "📝",
                "description": scenario_desc,
                "system_prompt": scenario_prompt
            }
            
            # Update listbox display for the saved scenario
            self.scenario_listbox.delete(self._current_scenario_idx)
            self.scenario_listbox.insert(self._current_scenario_idx, f"{scenario_icon or '📝'} {scenario_name}")
    
    def on_scenario_select(self, event):
        """Handle scenario selection. Auto-saves previous scenario before loading new one."""
        selection = self.scenario_listbox.curselection()
        if not selection:
            return
        
        new_idx = selection[0]
        
        # Auto-save the previous scenario before loading the new one
        if self._current_scenario_idx is not None and self._current_scenario_idx != new_idx:
            self._save_current_scenario()
        
        # Update current selection tracker
        self._current_scenario_idx = new_idx
        
        scenario = self.config["scenarios"][new_idx]
        
        self.scenario_id.delete(0, tk.END)
        self.scenario_id.insert(0, scenario["id"])
        
        self.scenario_name.delete(0, tk.END)
        self.scenario_name.insert(0, scenario["name"])
        
        self.scenario_icon.delete(0, tk.END)
        self.scenario_icon.insert(0, scenario["icon"])
        
        self.scenario_desc.delete("1.0", tk.END)
        self.scenario_desc.insert(tk.END, scenario["description"])
        
        self.scenario_prompt.delete("1.0", tk.END)
        self.scenario_prompt.insert(tk.END, scenario.get("system_prompt", ""))
    
    def update_scenario(self):
        """Update selected scenario (manual save button)."""
        selection = self.scenario_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a scenario to update.")
            return
        
        idx = selection[0]
        self.config["scenarios"][idx] = {
            "id": self.scenario_id.get(),
            "name": self.scenario_name.get(),
            "icon": self.scenario_icon.get(),
            "description": self.scenario_desc.get("1.0", tk.END).strip(),
            "system_prompt": self.scenario_prompt.get("1.0", tk.END).strip()
        }
        
        # Update listbox
        self.scenario_listbox.delete(idx)
        self.scenario_listbox.insert(idx, f"{self.scenario_icon.get()} {self.scenario_name.get()}")
        self.scenario_listbox.selection_set(idx)
        
        # Keep tracker in sync
        self._current_scenario_idx = idx
        
        messagebox.showinfo("Success", "Scenario updated!")
    
    def add_scenario(self):
        """Add new scenario."""
        # First save any current edits
        self._save_current_scenario()
        
        new_scenario = {
            "id": "new_scenario",
            "name": "New Scenario",
            "icon": "🆕",
            "description": "Description here",
            "system_prompt": "You are a helpful assistant."
        }
        self.config["scenarios"].append(new_scenario)
        self.scenario_listbox.insert(tk.END, f"{new_scenario['icon']} {new_scenario['name']}")
        self.scenario_listbox.selection_clear(0, tk.END)
        self.scenario_listbox.selection_set(tk.END)
        self.on_scenario_select(None)
    
    def remove_scenario(self):
        """Remove selected scenario."""
        selection = self.scenario_listbox.curselection()
        if not selection:
            return
        
        if len(self.config["scenarios"]) <= 1:
            messagebox.showwarning("Warning", "Cannot remove the last scenario.")
            return
        
        idx = selection[0]
        del self.config["scenarios"][idx]
        self.scenario_listbox.delete(idx)
        
        # Reset tracker since current scenario was deleted
        self._current_scenario_idx = None
        
        # Clear the editor fields
        self.scenario_id.delete(0, tk.END)
        self.scenario_name.delete(0, tk.END)
        self.scenario_icon.delete(0, tk.END)
        self.scenario_desc.delete("1.0", tk.END)
        self.scenario_prompt.delete("1.0", tk.END)
    
    def move_scenario(self, direction):
        """Move scenario up or down."""
        selection = self.scenario_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        new_idx = idx + direction
        
        if new_idx < 0 or new_idx >= len(self.config["scenarios"]):
            return
        
        # Swap
        self.config["scenarios"][idx], self.config["scenarios"][new_idx] = \
            self.config["scenarios"][new_idx], self.config["scenarios"][idx]
        
        # Update listbox
        self.scenario_listbox.delete(0, tk.END)
        for scenario in self.config["scenarios"]:
            self.scenario_listbox.insert(tk.END, f"{scenario['icon']} {scenario['name']}")
        
        self.scenario_listbox.selection_set(new_idx)
    
    def save_all(self):
        """Save all settings to config."""
        try:
            # Auto-save current scenario before saving everything
            self._save_current_scenario()
            
            # Server
            self.config["server"]["host"] = self.server_host.get()
            self.config["server"]["port"] = int(self.server_port.get())
            self.config["server"]["ssl_enabled"] = self.ssl_enabled.get()
            self.config["server"]["cert_path"] = self.cert_path.get()
            self.config["server"]["key_path"] = self.key_path.get()
            
            # Models
            self.config["models"]["tts_engine"] = self.tts_engine.get()
            self.config["models"]["stt_engine"] = self.stt_engine.get()
            self.config["models"]["llm_model_id"] = self.llm_model_id.get()
            self.config["models"]["llm_max_tokens"] = int(self.llm_max_tokens.get())
            self.config["models"]["llm_temperature"] = float(self.llm_temperature.get())
            self.config["models"]["kokoro_voice"] = self.kokoro_voice.get()
            self.config["models"]["edge_tts_voice"] = self.edge_voice.get()
            
            # Moderation
            self.config["moderation"]["enabled"] = self.moderation_enabled.get()
            self.config["moderation"]["system_prompt"] = self.moderation_prompt.get("1.0", tk.END).strip()
            self.config["moderation"]["serious_keywords"] = [
                k.strip() for k in self.serious_keywords.get().split(",") if k.strip()
            ]
            
            # Voices
            try:
                self.config["voices"]["kokoro"] = json.loads(self.kokoro_voices_text.get("1.0", tk.END))
                self.config["voices"]["edge"] = json.loads(self.edge_voices_text.get("1.0", tk.END))
                self.config["voices"]["vibevoice"] = json.loads(self.vibevoice_voices_text.get("1.0", tk.END))
            except json.JSONDecodeError as e:
                messagebox.showerror("Error", f"Invalid JSON in voices tab: {e}")
                return
            
            # UI
            self.config["ui"]["title"] = self.ui_title.get()
            self.config["ui"]["subtitle"] = self.ui_subtitle.get()
            self.config["ui"]["show_eye_contact"] = self.show_eye_contact.get()
            self.config["ui"]["show_session_analysis"] = self.show_session_analysis.get()
            self.config["ui"]["default_mic_mode"] = self.default_mic_mode.get()
            
            self.save_config()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        if messagebox.askyesno("Confirm", "Reset all settings to defaults?"):
            self.config = DEFAULT_CONFIG.copy()
            # Reload UI - simplest is to restart
            messagebox.showinfo("Info", "Please restart the configuration tool to see default values.")
    
    def export_config(self):
        """Export configuration to file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("Success", f"Exported to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def import_config(self):
        """Import configuration from file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    imported = json.load(f)
                self.config = self.merge_config(DEFAULT_CONFIG, imported)
                messagebox.showinfo("Success", "Configuration imported. Please restart the tool to see changes.")
            except Exception as e:
                messagebox.showerror("Error", f"Import failed: {e}")
    
    def apply_to_server(self):
        """Generate instructions to apply config to server.py."""
        messagebox.showinfo(
            "Apply to Server",
            "To apply this configuration:\n\n"
            "1. Save this configuration (config.json will be created)\n"
            "2. The server will automatically load config.json on startup\n"
            "3. Restart the server: python server.py\n\n"
            "For immediate changes, use the web UI settings panel."
        )


def main():
    root = tk.Tk()
    app = ConfigTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
