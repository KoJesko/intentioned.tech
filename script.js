// Configuration
const IS_LOCAL_CONTEXT = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const HAS_SECURE_CONTEXT = window.isSecureContext || IS_LOCAL_CONTEXT;

// Auto-detect WebSocket scheme based on page protocol
const WS_SCHEME = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
// If we're behind Cloudflare (or any external HTTPS on 443), do NOT append an explicit port.
// Otherwise, use explicit ports for direct-host connections.
const DEFAULT_HTTPS_PORT = 6942;
const DEFAULT_HTTP_PORT = 3001;

const PAGE_PORT = window.location.port; // '' when default (80/443)
const IS_HTTPS = window.location.protocol === 'https:';
const IS_DEFAULT_HTTPS_PORT = IS_HTTPS && (PAGE_PORT === '' || PAGE_PORT === '443');

// Heuristic: external/cloudflare-style access = https + default port + not localhost.
const IS_EXTERNAL_HTTPS = IS_DEFAULT_HTTPS_PORT && !IS_LOCAL_CONTEXT;

function buildWebSocketUrl() {
    const host = window.location.hostname;

    if (IS_HTTPS) {
        // When on standard HTTPS (443 via Cloudflare), the WS endpoint should be wss://host/ws/chat
        // and Cloudflare will proxy to the origin/port.
        if (IS_EXTERNAL_HTTPS) {
            return `${WS_SCHEME}${host}/ws/chat`;
        }
        return `${WS_SCHEME}${host}:${DEFAULT_HTTPS_PORT}/ws/chat`;
    }

    // HTTP dev/local
    return `${WS_SCHEME}${host}:${DEFAULT_HTTP_PORT}/ws/chat`;
}

const CONFIG = {
    WS_URL: buildWebSocketUrl(),
    CHUNK_SIZE_MS: 500,
    SAMPLE_RATE: 16000
};

const MODES = {
    PTT: 'ptt',
    VAD: 'vad'
};

// Improved VAD settings for better detection
const VAD_THRESHOLD = 0.02;      // Increased sensitivity threshold (was 0.015)
const VAD_RELEASE_MS = 1500;     // Time to wait after voice stops before ending capture (was 1200)
const VAD_MIN_ACTIVE_MS = 300;   // Minimum speaking duration before we consider it speech (was 200)
const VAD_SMOOTHING_FRAMES = 3;  // Number of frames to average for smoother detection
let vadSmoothingBuffer = [];

// State
let socket = null;
let mediaRecorder = null;
let audioContext = null;
let isRecording = false;
let audioQueue = [];
let isPlaying = false;
let recordedChunks = [];
let activeStream = null;
let inputMode = MODES.PTT;
let isPaused = false;
let micWarningShown = false;

// Session tracking for analysis
let conversationTranscript = [];  // Store {role, content} pairs
let currentScenario = 'general';
let speechTimestamps = [];  // Track {start, end} of each speech segment
let currentSpeechStart = null;  // Track when user starts speaking

// Response time tracking (time between AI done speaking and user starts speaking)
let aiSpeechEndTime = null;  // Timestamp when AI finishes speaking
let responseTimes = [];  // Array of response times in seconds
let isAiSpeaking = false;  // Track if AI is currently speaking
let interruptionCount = 0;  // Count of times user interrupted AI

// Webcam / Eye Contact state
let webcamEnabled = false;
let webcamStream = null;
let eyeContactData = [];
let eyeContactInterval = null;

// Voice Activity state
let vadAudioContext = null;
let vadSource = null;
let vadProcessor = null;
let vadGainNode = null;
let vadStream = null;
let vadRecorder = null;
let vadChunks = [];
let vadRecording = false;
let vadSpeechStart = 0;
let vadLastVoiceTs = 0;

// DOM Elements - updated for new professional UI
const statusEl = document.getElementById('statusText');
const statusBadge = document.getElementById('statusBadge');
const recordBtn = document.getElementById('recordBtn');
const chatLog = document.getElementById('chatLog');
const muteBtn = document.getElementById('muteBtn');
const resetBtn = document.getElementById('resetBtn');
const webcamBtn = document.getElementById('webcamBtn');
const vadStatusEl = document.getElementById('vadStatusText');
const vadIndicator = document.getElementById('vadIndicator');

function init() {
    console.log('=== init() STARTING ===');
    validateMicrophoneSupport();
    initWebSocket();
    setupUI();
    console.log('=== setupUI() COMPLETED ===');
    updateRecordButtonForMode();
    setVadStatus(null);
    console.log('=== init() COMPLETED ===');
}

function initWebSocket() {
    updateStatus('Connecting...');
    
    socket = new WebSocket(CONFIG.WS_URL);

    socket.onopen = (e) => {
        console.log("[open] Connection established to " + CONFIG.WS_URL);
        updateStatus('Connected', 'connected');
        sendControlMessage({ ttsMuted: isPaused });
    };

    socket.onmessage = async (event) => {
        try {
            const response = JSON.parse(event.data);
            
            // Handle safety violation - stop the conversation
            if (response.status === 'safety_violation') {
                console.log("🚨 Safety violation detected:", response.violation_type);
                showSafetyViolation(response.message);
                return;
            }
            
            // Handle User's transcribed text
            if (response.user_text) {
                appendChat('You', response.user_text);
            }
            
            // Handle AI Text Response (The Chat Log)
            if (response.text) {
                appendChat('AI', response.text);
            }

            // Handle Audio Response (The Voice)
            if (response.audio) {
                // Determine playback strategy based on status
                // "streaming" means we queue it up. "complete" might mean end of turn.
                queueAudio(response.audio);
            }
            
            if (response.status === 'complete') {
                console.log("AI turn complete");
            }
            
            if (response.status === 'no-speech') {
                console.log("No speech detected in audio");
            }

        } catch (error) {
            console.error("Error parsing message:", error);
        }
    };

    socket.onclose = (event) => {
        if (event.wasClean) {
            updateStatus(`Disconnected`);
        } else {
            updateStatus('Connection Lost');
            // Optional: Auto-reconnect logic could go here
        }
    };

    socket.onerror = (error) => {
        console.error(`[error] ${error.message}`);
        updateStatus('Error');
    };
}

function reconnect() {
    console.log("[reconnect] Attempting to reconnect...");
    updateStatus('Reconnecting...');
    
    // Close existing socket if open
    if (socket) {
        try {
            socket.close();
        } catch (e) {
            console.warn("Error closing socket:", e);
        }
        socket = null;
    }
    
    // Small delay before reconnecting
    setTimeout(() => {
        initWebSocket();
    }, 500);
}

// --- Audio Handling ---

async function startRecording() {
    if (inputMode !== MODES.PTT) return;
    if (!validateMicrophoneSupport(true)) {
        return;
    }

    // Track speech start time for pacing analysis
    currentSpeechStart = Date.now() / 1000;  // in seconds
    
    // Check for interruption (user speaking while AI is still speaking)
    if (isAiSpeaking) {
        interruptionCount++;
        console.log(`⚠️ Interruption detected! Total: ${interruptionCount}`);
    }
    
    // Calculate response time (time since AI finished speaking)
    if (aiSpeechEndTime !== null) {
        const responseTime = currentSpeechStart - aiSpeechEndTime;
        // Only count reasonable response times (0.1s to 60s)
        if (responseTime > 0.1 && responseTime < 60) {
            responseTimes.push(responseTime);
            console.log(`Response time: ${responseTime.toFixed(2)}s`);
        }
        aiSpeechEndTime = null;  // Reset for next measurement
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        activeStream = stream;
        
        // Use MediaRecorder to grab chunks
        // Note: For 'true' real-time low latency, we might eventually want AudioWorklets,
        // but MediaRecorder is fine for MVP.
        const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4';
        mediaRecorder = new MediaRecorder(stream, { mimeType });
        recordedChunks = [];

        mediaRecorder.ondataavailable = async (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            await transmitRecording();
        };

        // Record continuously without timeslice to get valid WebM container
        // The complete file is delivered on stop() as a single chunk
        mediaRecorder.start(); 
        isRecording = true;
        updateRecordButtonForMode();

    } catch (err) {
        console.error("Error accessing microphone:", err);
        alert("Could not access microphone.");
    }
}

function validateMicrophoneSupport(showAlert = false) {
    if (!HAS_SECURE_CONTEXT) {
        const message = 'Microphone access requires HTTPS or localhost.';
        updateStatus(message);
        if (showAlert && !micWarningShown) {
            alert('Microphone access requires HTTPS or localhost. Please use a secure connection.');
            micWarningShown = true;
        }
        return false;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const message = 'Microphone not supported';
        updateStatus(message);
        if (showAlert && !micWarningShown) {
            alert('This browser does not support microphone access. Please use Chrome or Edge.');
            micWarningShown = true;
        }
        return false;
    }
    return true;
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        // Track speech end time for pacing analysis
        if (currentSpeechStart !== null) {
            const speechEnd = Date.now() / 1000;  // in seconds
            speechTimestamps.push({
                start: currentSpeechStart,
                end: speechEnd
            });
            currentSpeechStart = null;
        }
        
        mediaRecorder.stop();
        if (activeStream) {
            activeStream.getTracks().forEach(track => track.stop());
            activeStream = null;
        }
        isRecording = false;
        updateRecordButtonForMode();
    }
}

async function transmitRecording() {
    await transmitChunks(recordedChunks, mediaRecorder?.mimeType || 'audio/webm');
    recordedChunks = [];
}

function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const chunkSize = 0x8000;

    for (let i = 0; i < bytes.length; i += chunkSize) {
        const chunk = bytes.subarray(i, i + chunkSize);
        binary += String.fromCharCode.apply(null, chunk);
    }

    return window.btoa(binary);
}

// --- Audio Playback Queue ---

async function queueAudio(base64String) {
    // Decode Base64 to ArrayBuffer
    const binaryString = window.atob(base64String);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    // Init AudioContext only on user interaction (browser policy)
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        if (isPaused) {
            try {
                await audioContext.suspend();
            } catch (err) {
                console.warn('Unable to initialize paused audio context', err);
            }
        }
    }

    // Decode and push to queue
    try {
        const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
        audioQueue.push(audioBuffer);
        if (!isPlaying) {
            playNextChunk();
        }
    } catch (e) {
        console.error("Error decoding audio chunk", e);
    }
}

function playNextChunk() {
    if (audioQueue.length === 0) {
        isPlaying = false;
        isAiSpeaking = false;  // AI finished speaking
        // Record when AI finishes speaking for response time measurement
        aiSpeechEndTime = Date.now() / 1000;
        console.log('AI finished speaking at:', aiSpeechEndTime);
        return;
    }

    isPlaying = true;
    isAiSpeaking = true;  // AI is speaking
    const buffer = audioQueue.shift();
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.onended = playNextChunk;
    source.start(0);
}

// --- UI Helpers ---

function setupUI() {
    // Record button (for PTT mode)
    if (recordBtn) {
        recordBtn.addEventListener('mousedown', () => {
            if (inputMode === MODES.PTT) startRecording();
        });
        recordBtn.addEventListener('mouseup', () => {
            if (inputMode === MODES.PTT) stopRecording();
        });
        // Touch support for mobile
        recordBtn.addEventListener('touchstart', (e) => {
            if (inputMode !== MODES.PTT) return;
            e.preventDefault();
            startRecording();
        });
        recordBtn.addEventListener('touchend', (e) => {
            if (inputMode !== MODES.PTT) return;
            e.preventDefault();
            stopRecording();
        });
    }

    // Mode toggle buttons (VAD vs PTT)
    const modeToggle = document.getElementById('modeToggle');
    if (modeToggle) {
        modeToggle.querySelectorAll('.toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const mode = btn.dataset.mode;
                modeToggle.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                if (mode === 'vad') {
                    inputMode = MODES.VAD;
                    startVAD();
                } else {
                    inputMode = MODES.PTT;
                    stopVAD();
                }
                updateRecordButtonForMode();
            });
        });
    }

    // Mute button
    if (muteBtn) {
        muteBtn.addEventListener('click', toggleMute);
    }

    // Reset button
    if (resetBtn) {
        resetBtn.addEventListener('click', startNewSession);
    }

    // Webcam button
    if (webcamBtn) {
        webcamBtn.addEventListener('click', toggleWebcam);
    }

    // Scenario buttons
    document.querySelectorAll('.scenario-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const scenario = btn.dataset.scenario;
            currentScenario = scenario;
            sendControlMessage({ scenario });
        });
    });
    
    // End Session button
    const endSessionBtn = document.getElementById('endSessionBtn');
    console.log('End Session Button found:', endSessionBtn);
    if (endSessionBtn) {
        endSessionBtn.addEventListener('click', endSessionAndAnalyze);
        console.log('End Session click listener attached');
    } else {
        console.error('End Session Button NOT FOUND!');
    }
    
    // Modal close button
    const closeModalBtn = document.getElementById('closeModal');
    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', () => {
            document.getElementById('analysisModal').classList.remove('active');
        });
    }
    
    // Click outside modal to close
    const modalOverlay = document.getElementById('analysisModal');
    if (modalOverlay) {
        modalOverlay.addEventListener('click', (e) => {
            if (e.target === modalOverlay) {
                modalOverlay.classList.remove('active');
            }
        });
    }
}

// Toggle mute function
let isMuted = false;
function toggleMute() {
    isMuted = !isMuted;
    if (muteBtn) {
        muteBtn.textContent = isMuted ? '🔇 Unmute AI' : '🔊 Mute AI';
    }
    sendControlMessage({ mute: isMuted });
}

// Toggle webcam function
let webcamActive = false;
function toggleWebcam() {
    if (webcamActive) {
        disableWebcam();
    } else {
        enableWebcam();
    }
}

function sendControlMessage(payload = {}) {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
        return;
    }
    socket.send(JSON.stringify({ type: 'control', ...payload }));
}

function updateStatus(text, state = 'default') {
    if (statusEl) {
        statusEl.textContent = text;
    }
    if (statusBadge) {
        statusBadge.classList.remove('connected', 'recording', 'error');
        if (state === 'connected') {
            statusBadge.classList.add('connected');
        } else if (state === 'recording') {
            statusBadge.classList.add('recording');
        }
    }
}

function updateRecordButtonForMode() {
    if (!recordBtn) return;

    if (inputMode === MODES.PTT) {
        recordBtn.disabled = false;
        if (isRecording) {
            recordBtn.innerHTML = '🔴 Recording...';
            recordBtn.classList.remove('primary');
            recordBtn.classList.add('danger');
        } else {
            recordBtn.innerHTML = '🎤 Hold to Speak';
            recordBtn.classList.remove('danger');
            recordBtn.classList.add('primary');
        }
    } else {
        recordBtn.disabled = true;
        recordBtn.innerHTML = '🎤 VAD Active';
    }
}

function updatePauseButton() {
    if (!pauseBtn) return;
    pauseBtn.textContent = isPaused ? 'Resume Audio' : 'Pause Audio';
    pauseBtn.classList.toggle('bg-green-600', isPaused);
    pauseBtn.classList.toggle('bg-gray-700', !isPaused);
}

async function togglePausePlayback() {
    isPaused = !isPaused;

    if (!audioContext) {
        sendControlMessage({ ttsMuted: isPaused });
        return;
    }

    try {
        if (isPaused) {
            await audioContext.suspend();
        } else {
            await audioContext.resume();
        }
        sendControlMessage({ ttsMuted: isPaused });
    } catch (error) {
        console.error('Failed to toggle playback state', error);
    }
}

function handleModeChange(event) {
    const selectedMode = event.target.value;
    if (selectedMode === inputMode) return;
    inputMode = selectedMode;

    if (inputMode === MODES.VAD) {
        enableVadMode();
    } else {
        disableVadMode();
    }

    updateRecordButtonForMode();
}

async function enableVadMode() {
    try {
        vadStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4';
        vadRecorder = new MediaRecorder(vadStream, { mimeType });
        vadChunks = [];

        vadRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                vadChunks.push(event.data);
            }
        };

        vadRecorder.onstop = async () => {
            await transmitChunks(vadChunks, vadRecorder?.mimeType || 'audio/webm');
            vadChunks = [];
            vadRecording = false;
            setVadStatus('listening');
            updateRecordButtonForMode();
        };

        vadAudioContext = new (window.AudioContext || window.webkitAudioContext)();
        vadSource = vadAudioContext.createMediaStreamSource(vadStream);
        vadProcessor = vadAudioContext.createScriptProcessor(2048, 1, 1);
        vadGainNode = vadAudioContext.createGain();
        vadGainNode.gain.value = 0;

        vadSource.connect(vadProcessor);
        vadProcessor.connect(vadGainNode);
        vadGainNode.connect(vadAudioContext.destination);

        vadProcessor.onaudioprocess = handleVadAudio;
        vadSpeechStart = 0;
        vadLastVoiceTs = 0;
        setVadStatus('listening');
    } catch (error) {
        console.error('Failed to enable VAD mode', error);
        updateStatus('Microphone Error');
        inputMode = MODES.PTT;
        disableVadMode();
        updateRecordButtonForMode();
    }
}

function disableVadMode() {
    if (vadProcessor) {
        vadProcessor.disconnect();
        vadProcessor.onaudioprocess = null;
        vadProcessor = null;
    }
    if (vadSource) {
        vadSource.disconnect();
        vadSource = null;
    }
    if (vadGainNode) {
        vadGainNode.disconnect();
        vadGainNode = null;
    }
    if (vadAudioContext) {
        vadAudioContext.close();
        vadAudioContext = null;
    }
    if (vadRecorder) {
        if (vadRecorder.state !== 'inactive') {
            vadRecorder.stop();
        }
        vadRecorder = null;
    }
    if (vadStream) {
        vadStream.getTracks().forEach(track => track.stop());
        vadStream = null;
    }
    vadChunks = [];
    vadRecording = false;
    vadSmoothingBuffer = [];
    setVadStatus(null);
}

function handleVadAudio(event) {
    const inputBuffer = event.inputBuffer.getChannelData(0);
    let sumSquares = 0;
    for (let i = 0; i < inputBuffer.length; i++) {
        const sample = inputBuffer[i];
        sumSquares += sample * sample;
    }
    const rms = Math.sqrt(sumSquares / inputBuffer.length);
    const now = performance.now();
    
    // Apply smoothing to reduce false triggers
    vadSmoothingBuffer.push(rms);
    if (vadSmoothingBuffer.length > VAD_SMOOTHING_FRAMES) {
        vadSmoothingBuffer.shift();
    }
    const smoothedRms = vadSmoothingBuffer.reduce((a, b) => a + b, 0) / vadSmoothingBuffer.length;

    if (smoothedRms > VAD_THRESHOLD) {
        vadLastVoiceTs = now;
        if (!vadRecording && vadRecorder?.state === 'inactive') {
            startVadCapture(now);
        } else if (vadRecording) {
            setVadStatus('speaking');
        }
    } else if (vadRecording && now - vadLastVoiceTs > VAD_RELEASE_MS && now - vadSpeechStart > VAD_MIN_ACTIVE_MS) {
        stopVadCapture();
    } else if (!vadRecording) {
        // Show listening status when not recording
        setVadStatus('listening');
    }
}

function startVadCapture(startTimestamp) {
    if (!vadRecorder || vadRecorder.state !== 'inactive') return;
    vadChunks = [];
    // Don't use timeslice - record continuously and get complete file on stop()
    // This ensures we get a valid WebM container with proper headers
    vadRecorder.start();
    vadRecording = true;
    vadSpeechStart = startTimestamp || performance.now();
    
    // Track speech start for pacing analysis
    currentSpeechStart = Date.now() / 1000;
    
    // Check for interruption (user speaking while AI is still speaking)
    if (isAiSpeaking) {
        interruptionCount++;
        console.log(`⚠️ Interruption detected! Total: ${interruptionCount}`);
    }
    
    // Calculate response time (time since AI finished speaking)
    if (aiSpeechEndTime !== null) {
        const responseTime = currentSpeechStart - aiSpeechEndTime;
        // Only count reasonable response times (0.1s to 60s)
        if (responseTime > 0.1 && responseTime < 60) {
            responseTimes.push(responseTime);
            console.log(`Response time: ${responseTime.toFixed(2)}s`);
        }
        aiSpeechEndTime = null;  // Reset for next measurement
    }
    
    setVadStatus('speaking');
    updateRecordButtonForMode();
}

function stopVadCapture() {
    if (!vadRecorder || vadRecorder.state !== 'recording') return;
    
    // Track speech end for pacing analysis
    if (currentSpeechStart !== null) {
        const speechEnd = Date.now() / 1000;
        speechTimestamps.push({
            start: currentSpeechStart,
            end: speechEnd
        });
        currentSpeechStart = null;
    }
    
    vadRecording = false;
    setVadStatus('processing');
    vadRecorder.stop();
    updateRecordButtonForMode();
}

function setVadStatus(state) {
    if (!vadStatusEl) return;
    
    const messages = {
        listening: 'Voice Activity: listening',
        speaking: 'Voice Activity: capturing speech…',
        processing: 'Voice Activity: processing clip',
        idle: 'Voice Activity: idle'
    };
    vadStatusEl.textContent = messages[state] || 'Voice Activity: idle';
    
    // Update the indicator dot
    if (vadIndicator) {
        vadIndicator.classList.remove('speaking');
        if (state === 'speaking') {
            vadIndicator.classList.add('speaking');
        }
    }
}

// Minimum audio duration to prevent Whisper hallucinations on short clips
// Increased to 800ms to better filter noise
const MIN_AUDIO_DURATION_MS = 800;

async function transmitChunks(chunks, mimeType = 'audio/webm') {
    if (!chunks.length || !socket || socket.readyState !== WebSocket.OPEN) {
        return;
    }

    const blob = new Blob(chunks, { type: mimeType });
    
    // Estimate duration: for 16kHz mono 16-bit audio, ~32 bytes per ms
    // For compressed formats, we use blob.size / 4 as rough estimate
    const estimatedDurationMs = blob.size / 4;
    
    // Skip very short audio clips (likely noise/accidental clicks)
    if (estimatedDurationMs < MIN_AUDIO_DURATION_MS) {
        console.log(`Skipping short audio clip (${Math.round(estimatedDurationMs)}ms < ${MIN_AUDIO_DURATION_MS}ms)`);
        return;
    }

    try {
        const arrayBuffer = await blob.arrayBuffer();
        const base64Audio = arrayBufferToBase64(arrayBuffer);
        socket.send(JSON.stringify({
            audio: base64Audio,
            mimeType: blob.type,
            sampleRate: CONFIG.SAMPLE_RATE,
            isFinal: true,
            durationMs: Math.round(estimatedDurationMs)
        }));
    } catch (error) {
        console.error('Failed to transmit recording', error);
    }
}

function appendChat(role, text) {
    if (!chatLog) return;
    const msgDiv = document.createElement('div');
    const isAI = role === 'AI';
    msgDiv.className = `message ${isAI ? 'assistant' : 'user'}`;

    // Create role element safely
    const roleDiv = document.createElement('div');
    roleDiv.className = 'role';
    roleDiv.textContent = role;

    // Create text container safely to avoid XSS
    const textSpan = document.createElement('span');
    textSpan.textContent = text;

    msgDiv.appendChild(roleDiv);
    msgDiv.appendChild(textSpan);

    chatLog.appendChild(msgDiv);
    chatLog.scrollTop = chatLog.scrollHeight;
    
    // Track for analysis
    conversationTranscript.push({
        role: isAI ? 'assistant' : 'user',
        content: text
    });
}

// Show safety violation warning and force user to restart
function showSafetyViolation(message) {
    // Stop all recording immediately
    if (isRecording) {
        stopRecording();
    }
    if (vadRecording) {
        stopVadRecording();
    }
    
    // Clear the audio queue
    audioQueue = [];
    isPlaying = false;
    
    // Create a prominent warning message in the chat
    const warningDiv = document.createElement('div');
    warningDiv.className = 'message safety-warning';
    warningDiv.innerHTML = `
        <div class="role" style="color: #ef4444;">⚠️ Safety System</div>
        <div style="background: rgba(239, 68, 68, 0.15); padding: 16px; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);">
            <strong>Conversation Stopped</strong><br>
            <p style="margin: 8px 0;">${message}</p>
            <p style="margin: 8px 0; font-size: 0.9em; color: #a0a0b0;">
                Please click the <strong>🔄 Reset</strong> button to start a new conversation.
            </p>
        </div>
    `;
    chatLog.appendChild(warningDiv);
    chatLog.scrollTop = chatLog.scrollHeight;
    
    // Update status
    updateStatus('Safety Violation - Please Reset', 'error');
    
    // Disable the record button until reset
    if (recordBtn) {
        recordBtn.disabled = true;
        recordBtn.textContent = '🚫 Disabled';
    }
    
    // Highlight the reset button
    if (resetBtn) {
        resetBtn.style.animation = 'pulse 1s infinite';
        resetBtn.style.background = 'var(--danger)';
        resetBtn.style.borderColor = 'var(--danger)';
        resetBtn.style.color = 'white';
    }
}

// --- Webcam / Eye Contact Functions with Face-API.js ---
let faceApiLoaded = false;
let faceDetectionRunning = false;

async function loadFaceApi() {
    if (faceApiLoaded) return true;
    
    try {
        // Check if face-api is loaded (with retries)
        let attempts = 0;
        while (typeof faceapi === 'undefined' && attempts < 5) {
            console.log(`Waiting for face-api.js to load (attempt ${attempts + 1}/5)...`);
            await new Promise(resolve => setTimeout(resolve, 500));
            attempts++;
        }
        
        if (typeof faceapi === 'undefined') {
            console.error('face-api.js failed to load after 5 attempts');
            return false;
        }
        
        console.log('face-api.js library loaded, loading models...');
        
        // Load the tiny face detector model from CDN (vladmandic fork)
        const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/model';
        
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        console.log('TinyFaceDetector model loaded');
        
        await faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL);
        console.log('FaceLandmark68TinyNet model loaded');
        
        faceApiLoaded = true;
        console.log('✅ Face-API models loaded successfully');
        return true;
    } catch (err) {
        console.error('Failed to load face-api models:', err);
        return false;
    }
}

async function enableWebcam() {
    try {
        const webcamVideo = document.getElementById('webcamPreview');
        const webcamCard = document.getElementById('webcamCard');
        const eyeContactDot = document.getElementById('eyeContactDot');
        const eyeContactLabel = document.getElementById('eyeContactLabel');
        
        // Request webcam access
        webcamStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 320, height: 240, facingMode: 'user' } 
        });
        
        webcamVideo.srcObject = webcamStream;
        webcamCard.style.display = 'block';
        
        webcamEnabled = true;
        webcamActive = true;
        eyeContactData = [];
        
        if (webcamBtn) {
            webcamBtn.textContent = '📷 Disable Webcam';
        }
        
        // Try to load face-api for real detection
        const faceApiReady = await loadFaceApi();
        
        if (faceApiReady) {
            // Real face detection
            faceDetectionRunning = true;
            detectFace(webcamVideo, eyeContactDot, eyeContactLabel);
        } else {
            // Fallback: simple detection based on video stream activity
            eyeContactInterval = setInterval(() => {
                // Basic fallback - assume eye contact if video is playing
                const isLooking = webcamVideo.readyState >= 2;
                eyeContactData.push({
                    timestamp: Date.now(),
                    looking_at_camera: isLooking
                });
                eyeContactLabel.textContent = isLooking ? 'Tracking' : 'No face';
                eyeContactDot.classList.toggle('active', isLooking);
            }, 1000);
        }
        
        console.log('Webcam enabled for eye contact tracking');
    } catch (err) {
        console.error('Failed to enable webcam:', err);
        webcamEnabled = false;
        webcamActive = false;
        if (webcamBtn) {
            webcamBtn.textContent = '📷 Webcam';
        }
        alert('Could not access webcam. Please ensure camera permissions are granted.');
    }
}

async function detectFace(video, dotEl, labelEl) {
    if (!faceDetectionRunning || !webcamEnabled) return;
    
    // Wait for video to be ready
    if (video.readyState < 2) {
        requestAnimationFrame(() => detectFace(video, dotEl, labelEl));
        return;
    }
    
    try {
        const detection = await faceapi.detectSingleFace(
            video, 
            new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.4 })
        ).withFaceLandmarks(true);
        
        const isLooking = detection !== undefined && detection !== null;
        
        eyeContactData.push({
            timestamp: Date.now(),
            looking_at_camera: isLooking,
            confidence: detection ? detection.detection.score : 0
        });
        
        if (dotEl) {
            dotEl.classList.toggle('active', isLooking);
        }
        if (labelEl) {
            if (isLooking) {
                const confidence = Math.round(detection.detection.score * 100);
                labelEl.textContent = `Tracking (${confidence}%)`;
            } else {
                labelEl.textContent = 'No face detected';
            }
        }
    } catch (err) {
        console.error('Face detection error:', err);
        if (labelEl) {
            labelEl.textContent = 'Detection error';
        }
    }
    
    // Continue detection loop (throttle to ~10fps for performance)
    if (faceDetectionRunning && webcamEnabled) {
        setTimeout(() => {
            requestAnimationFrame(() => detectFace(video, dotEl, labelEl));
        }, 100);
    }
}

function disableWebcam() {
    faceDetectionRunning = false;
    
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    if (eyeContactInterval) {
        clearInterval(eyeContactInterval);
        eyeContactInterval = null;
    }
    
    const webcamVideo = document.getElementById('webcamPreview');
    const webcamCard = document.getElementById('webcamCard');
    
    if (webcamVideo) {
        webcamVideo.srcObject = null;
    }
    if (webcamCard) {
        webcamCard.style.display = 'none';
    }
    
    webcamEnabled = false;
    webcamActive = false;
    
    if (webcamBtn) {
        webcamBtn.textContent = '📷 Webcam';
    }
}

// --- End Session & Analysis Functions ---
async function endSessionAndAnalyze() {
    console.log('=== endSessionAndAnalyze CALLED ===');
    console.log('Ending session and analyzing...');
    
    // Show modal with loading state
    const modal = document.getElementById('analysisModal');
    const contentEl = document.getElementById('analysisContent');
    
    modal.classList.add('active');
    contentEl.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <span>Analyzing your session...</span>
        </div>
    `;
    
    // Disable webcam if enabled
    disableWebcam();
    
    // Prepare analysis request
    const analysisRequest = {
        transcript: conversationTranscript,
        eye_contact_data: eyeContactData.length > 0 ? eyeContactData : null,
        speech_timestamps: speechTimestamps.length > 0 ? speechTimestamps : null,
        response_times: responseTimes.length > 0 ? responseTimes : null,
        interruption_count: interruptionCount,
        scenario: currentScenario
    };
    
    console.log('Speech timestamps:', speechTimestamps);
    console.log('Response times:', responseTimes);
    console.log('Interruption count:', interruptionCount);
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(analysisRequest)
        });
        
        if (!response.ok) {
            throw new Error('Analysis request failed');
        }
        
        const results = await response.json();
        displayAnalysisResults(results, contentEl);
        
    } catch (error) {
        console.error('Analysis failed:', error);
        contentEl.innerHTML = `
            <div class="loading">
                <span style="color: var(--danger);">❌ Analysis failed. Please try again.</span>
            </div>
        `;
    }
}

function getRatingClass(score) {
    if (score === 'Excellent' || score === 'Good' || score >= 80) return 'excellent';
    if (score === 'Fair' || score >= 60) return 'good';
    if (score === 'Needs Improvement' || score >= 40) return 'fair';
    return 'needs-work';
}

function displayAnalysisResults(results, contentEl) {
    const score = results.overall_score || 0;
    const scorePercent = score;
    
    let fillerTags = '';
    if (results.filler_words && results.filler_words.details) {
        fillerTags = Object.entries(results.filler_words.details)
            .map(([word, count]) => `<span class="filler-tag">"${word}" × ${count}</span>`)
            .join('');
    }
    
    let pacingHtml = '';
    if (results.speech_pacing) {
        pacingHtml = `
            <div class="analysis-card">
                <h4>🎵 Speech Pacing</h4>
                <span class="rating ${getRatingClass(results.speech_pacing.score)}">${results.speech_pacing.score}</span>
                <div class="feedback">${results.speech_pacing.feedback}</div>
                <div class="suggestion">
                    Avg pause: ${results.speech_pacing.avg_gap}s | 
                    Max pause: ${results.speech_pacing.max_gap}s | 
                    Long pauses: ${results.speech_pacing.long_pauses}
                </div>
            </div>
        `;
    }
    
    let responseTimeHtml = '';
    if (results.response_time) {
        responseTimeHtml = `
            <div class="analysis-card">
                <h4>⏱️ Response Time</h4>
                <span class="rating ${getRatingClass(results.response_time.score)}">${results.response_time.score}</span>
                <div class="feedback">${results.response_time.feedback}</div>
                <div class="suggestion">
                    Avg: ${results.response_time.avg_time}s | 
                    Min: ${results.response_time.min_time}s | 
                    Max: ${results.response_time.max_time}s
                </div>
            </div>
        `;
    }
    
    let speakingPaceHtml = '';
    if (results.speaking_pace) {
        speakingPaceHtml = `
            <div class="analysis-card">
                <h4>🏃 Speaking Pace</h4>
                <span class="rating ${getRatingClass(results.speaking_pace.score)}">${results.speaking_pace.score}</span>
                <div class="feedback">${results.speaking_pace.feedback}</div>
                <div class="suggestion">
                    Avg: ${results.speaking_pace.avg_wpm} WPM | 
                    Min: ${results.speaking_pace.min_wpm} WPM | 
                    Max: ${results.speaking_pace.max_wpm} WPM
                </div>
            </div>
        `;
    }
    
    let interruptionsHtml = '';
    if (results.interruptions) {
        interruptionsHtml = `
            <div class="analysis-card">
                <h4>🤐 Interruptions</h4>
                <span class="rating ${getRatingClass(results.interruptions.score)}">${results.interruptions.score}</span>
                <div class="feedback">${results.interruptions.feedback}</div>
                <div class="suggestion">
                    Count: ${results.interruptions.count} (Target: &lt;${results.interruptions.threshold})
                </div>
            </div>
        `;
    }
    
    let eyeContactHtml = '';
    if (results.eye_contact) {
        eyeContactHtml = `
            <div class="analysis-card">
                <h4>👁️ Eye Contact</h4>
                <span class="rating ${getRatingClass(results.eye_contact.score)}">${results.eye_contact.score}</span>
                <div class="feedback">${results.eye_contact.feedback}</div>
            </div>
        `;
    }
    
    contentEl.innerHTML = `
        <!-- Overall Score -->
        <div class="score-section">
            <div class="score-circle" style="--score-percent: ${scorePercent}%;">
                <div class="score-value">${score}<span>/100</span></div>
            </div>
            <div class="score-label">Overall Performance Score</div>
        </div>
        
        <!-- AI Summary -->
        ${results.ai_summary ? `
        <div class="analysis-card full-width" style="margin-bottom: 24px;">
            <h4>🤖 AI Summary</h4>
            <div class="ai-summary">${results.ai_summary}</div>
        </div>
        ` : ''}
        
        <!-- Analysis Grid -->
        <div class="analysis-grid">
            <!-- Filler Words -->
            <div class="analysis-card">
                <h4>💬 Filler Words</h4>
                <div class="value">${results.filler_words?.count || 0}</div>
                <span class="rating ${getRatingClass(results.filler_words?.score)}">${results.filler_words?.score || 'N/A'}</span>
                <div class="filler-list">${fillerTags || '<span class="filler-tag">None detected ✓</span>'}</div>
                <div class="suggestion">Target: under ${results.filler_words?.target || 5} per session</div>
            </div>
            
            <!-- Delivery -->
            <div class="analysis-card">
                <h4>🎤 Delivery</h4>
                <span class="rating ${getRatingClass(results.delivery?.score)}">${results.delivery?.score || 'N/A'}</span>
                <div class="feedback">${results.delivery?.feedback || 'No feedback available'}</div>
            </div>
            
            <!-- Tone -->
            <div class="analysis-card">
                <h4>🎭 Tone</h4>
                <span class="rating ${getRatingClass(results.tone?.score)}">${results.tone?.score || 'N/A'}</span>
                <div class="feedback">${results.tone?.feedback || 'No feedback available'}</div>
            </div>
            
            <!-- Microaggressions -->
            <div class="analysis-card">
                <h4>⚠️ Microaggressions</h4>
                <span class="rating ${results.microaggressions?.detected ? 'needs-work' : 'excellent'}">${results.microaggressions?.detected ? 'Detected' : 'None'}</span>
                <div class="feedback">${results.microaggressions?.feedback || 'No issues detected'}</div>
            </div>
            
            ${pacingHtml}
            ${responseTimeHtml}
            ${speakingPaceHtml}
            ${interruptionsHtml}
            ${eyeContactHtml}
        </div>
        
        <!-- New Session Button -->
        <div style="text-align: center; margin-top: 24px;">
            <button class="control-btn primary" onclick="startNewSession()">
                🔄 Start New Session
            </button>
        </div>
    `;
}

function startNewSession() {
    // Reset state
    conversationTranscript = [];
    eyeContactData = [];
    speechTimestamps = [];
    responseTimes = [];
    aiSpeechEndTime = null;
    isAiSpeaking = false;
    interruptionCount = 0;
    currentScenario = 'general';
    currentSpeechStart = null;
    
    // Clear chat log
    if (chatLog) {
        chatLog.innerHTML = '';
    }
    
    // Re-enable record button (in case it was disabled by safety violation)
    if (recordBtn) {
        recordBtn.disabled = false;
    }
    
    // Reset the reset button style (in case it was highlighted)
    if (resetBtn) {
        resetBtn.style.animation = '';
        resetBtn.style.background = '';
        resetBtn.style.borderColor = '';
        resetBtn.style.color = '';
    }
    
    // Hide modal
    document.getElementById('analysisModal').classList.remove('active');
    
    // Reset scenario selection
    document.querySelectorAll('.scenario-btn').forEach(c => c.classList.remove('active'));
    document.querySelector('.scenario-btn[data-scenario="general"]')?.classList.add('active');
    
    // Reconnect websocket to reset server-side conversation
    reconnect();
    
    // Update status
    updateStatus('Connected', 'connected');
}

// Start
window.addEventListener('load', init);
window.addEventListener('beforeunload', () => {
    disableVadMode();
    disableWebcam();
});
