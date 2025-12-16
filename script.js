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

const VAD_THRESHOLD = 0.015;
const VAD_RELEASE_MS = 1200;
const VAD_MIN_ACTIVE_MS = 200;

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

// DOM Elements
const statusEl = document.getElementById('status');
const recordBtn = document.getElementById('recordBtn');
const chatLog = document.getElementById('chatLog');
const inputModeSelect = document.getElementById('inputModeSelect');
const pauseBtn = document.getElementById('pauseBtn');
const vadStatusEl = document.getElementById('vadStatus');

function init() {
    validateMicrophoneSupport();
    initWebSocket();
    setupUI();
    updateRecordButtonForMode();
    updatePauseButton();
    setVadStatus(null);
}

function initWebSocket() {
    updateStatus('Connecting...', 'text-yellow-500');
    
    socket = new WebSocket(CONFIG.WS_URL);

    socket.onopen = (e) => {
        console.log("[open] Connection established to " + CONFIG.WS_URL);
        updateStatus('Connected (Ready)', 'text-green-500');
        sendControlMessage({ ttsMuted: isPaused });
    };

    socket.onmessage = async (event) => {
        try {
            const response = JSON.parse(event.data);
            
            // Handle Text Response (The Chat Log)
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

        } catch (error) {
            console.error("Error parsing message:", error);
        }
    };

    socket.onclose = (event) => {
        if (event.wasClean) {
            updateStatus(`Closed cleanly, code=${event.code}`, 'text-gray-500');
        } else {
            updateStatus('Connection Died', 'text-red-500');
            // Optional: Auto-reconnect logic could go here
        }
    };

    socket.onerror = (error) => {
        console.error(`[error] ${error.message}`);
        updateStatus('Error', 'text-red-600');
    };
}

function reconnect() {
    console.log("[reconnect] Attempting to reconnect...");
    updateStatus('Reconnecting...', 'text-yellow-500');
    
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

        // Slice audio into small chunks for streaming upload
        mediaRecorder.start(CONFIG.CHUNK_SIZE_MS); 
        isRecording = true;
        updateRecordButtonForMode();

    } catch (err) {
        console.error("Error accessing microphone:", err);
        alert("Could not access microphone.");
    }
}

function validateMicrophoneSupport(showAlert = false) {
    if (!HAS_SECURE_CONTEXT) {
        const message = 'Microphone access requires HTTPS or localhost. Please visit https://pascacktechnology.ddns.net or use localhost with a secure tunnel.';
        updateStatus(message, 'text-red-500');
        if (showAlert && !micWarningShown) {
            alert(message);
            micWarningShown = true;
        }
        return false;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const message = 'This browser does not support navigator.mediaDevices.getUserMedia. Please use a modern browser like Chrome or Edge.';
        updateStatus(message, 'text-red-500');
        if (showAlert && !micWarningShown) {
            alert(message);
            micWarningShown = true;
        }
        return false;
    }
    return true;
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
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
        return;
    }

    isPlaying = true;
    const buffer = audioQueue.shift();
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.onended = playNextChunk;
    source.start(0);
}

// --- UI Helpers ---

function setupUI() {
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

    if (inputModeSelect) {
        inputModeSelect.addEventListener('change', handleModeChange);
    }

    if (pauseBtn) {
        pauseBtn.addEventListener('click', togglePausePlayback);
    }

    // Reconnect button
    const reconnectBtn = document.getElementById('reconnectBtn');
    if (reconnectBtn) {
        reconnectBtn.addEventListener('click', reconnect);
    }
}

function sendControlMessage(payload = {}) {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
        return;
    }
    socket.send(JSON.stringify({ type: 'control', ...payload }));
}

function updateStatus(text, colorClass) {
    if (statusEl) {
        statusEl.textContent = text;
        statusEl.className = `text-sm font-bold ${colorClass}`;
    }
}

function updateRecordButtonForMode() {
    if (!recordBtn) return;

    recordBtn.disabled = inputMode === MODES.VAD;
    recordBtn.classList.remove('cursor-not-allowed', 'opacity-80');

    if (inputMode === MODES.PTT) {
        recordBtn.classList.remove('bg-gray-700');
        if (isRecording) {
            recordBtn.classList.remove('bg-blue-600');
            recordBtn.classList.add('bg-red-600', 'animate-pulse');
            recordBtn.textContent = 'Release to Send';
        } else {
            recordBtn.classList.remove('bg-red-600', 'animate-pulse');
            recordBtn.classList.add('bg-blue-600');
            recordBtn.textContent = 'Hold to Speak';
        }
    } else {
        recordBtn.classList.remove('bg-blue-600', 'bg-red-600', 'animate-pulse');
        recordBtn.classList.add('bg-gray-700', 'cursor-not-allowed', 'opacity-80');
        recordBtn.textContent = vadRecording ? 'Listening (Voice Activity)' : 'Voice Activity Enabled';
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
    updatePauseButton();

    if (!audioContext) {
        if (isPaused) {
            updateStatus('Playback paused (will resume when audio starts)', 'text-yellow-400');
        }
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
        updateStatus('VAD Error (microphone unavailable)', 'text-red-500');
        inputMode = MODES.PTT;
        if (inputModeSelect) {
            inputModeSelect.value = MODES.PTT;
        }
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

    if (rms > VAD_THRESHOLD) {
        vadLastVoiceTs = now;
        if (!vadRecording && vadRecorder?.state === 'inactive') {
            startVadCapture(now);
        } else if (vadRecording) {
            setVadStatus('speaking');
        }
    } else if (vadRecording && now - vadLastVoiceTs > VAD_RELEASE_MS && now - vadSpeechStart > VAD_MIN_ACTIVE_MS) {
        stopVadCapture();
    }
}

function startVadCapture(startTimestamp) {
    if (!vadRecorder || vadRecorder.state !== 'inactive') return;
    vadChunks = [];
    vadRecorder.start(CONFIG.CHUNK_SIZE_MS);
    vadRecording = true;
    vadSpeechStart = startTimestamp || performance.now();
    setVadStatus('speaking');
    updateRecordButtonForMode();
}

function stopVadCapture() {
    if (!vadRecorder || vadRecorder.state !== 'recording') return;
    vadRecording = false;
    setVadStatus('processing');
    vadRecorder.stop();
    updateRecordButtonForMode();
}

function setVadStatus(state) {
    if (!vadStatusEl) return;
    if (!state) {
        vadStatusEl.classList.add('hidden');
        return;
    }

    vadStatusEl.classList.remove('hidden');
    const messages = {
        listening: 'Voice Activity: listening',
        speaking: 'Voice Activity: capturing speech…',
        processing: 'Voice Activity: processing clip',
        idle: 'Voice Activity: idle'
    };
    vadStatusEl.textContent = messages[state] || 'Voice Activity: idle';
}

async function transmitChunks(chunks, mimeType = 'audio/webm') {
    if (!chunks.length || !socket || socket.readyState !== WebSocket.OPEN) {
        return;
    }

    const blob = new Blob(chunks, { type: mimeType });

    try {
        const arrayBuffer = await blob.arrayBuffer();
        const base64Audio = arrayBufferToBase64(arrayBuffer);
        socket.send(JSON.stringify({
            audio: base64Audio,
            mimeType: blob.type,
            sampleRate: CONFIG.SAMPLE_RATE,
            isFinal: true,
            durationMs: Math.round(blob.size / 2)
        }));
    } catch (error) {
        console.error('Failed to transmit recording', error);
    }
}

function appendChat(role, text) {
    if (!chatLog) return;
    const msgDiv = document.createElement('div');
    msgDiv.className = `p-2 my-2 rounded ${role === 'AI' ? 'bg-gray-100 text-gray-800 self-start' : 'bg-blue-100 text-blue-800 self-end'}`;
    msgDiv.textContent = `${role}: ${text}`;
    chatLog.appendChild(msgDiv);
    chatLog.scrollTop = chatLog.scrollHeight;
}

// Start
window.addEventListener('load', init);
window.addEventListener('beforeunload', () => {
    disableVadMode();
});
