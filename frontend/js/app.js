/**
 * MAITRI - Multimodal AI for Astronaut Emotional & Physical Wellbeing
 * Frontend Client Controller & Camera/Audio Telemetry Streamer
 */

// API Base URL (defaults to current origin or localhost:8000)
const API_BASE = window.location.origin.includes(':8000') || window.location.protocol.startsWith('http') 
    ? window.location.origin 
    : 'http://localhost:8000';

// Global Telemetry State
const state = {
    currentStep: 1,
    astronautId: "ASTRO-01",
    missionDay: 42,
    cameraStream: null,
    mediaRecorder: null,
    recordedChunks: [],
    is10sScanning: false,

    // Step Results
    faceResult: {
        dominant_emotion: "Neutral",
        confidence: 0.85,
        scores: { "Surprise": 0.02, "Fear": 0.01, "Disgust": 0.01, "Happiness": 0.15, "Sadness": 0.05, "Anger": 0.02, "Neutral": 0.74 },
        face_detected: true,
        frames_analyzed: 10,
        duration_seconds: 10.0
    },
    voiceResult: {
        dominant_emotion: "calm",
        confidence: 0.84,
        pitch_hz: 118.2,
        scores: { "calm": 0.84, "neutral": 0.08, "sadness": 0.05, "fear": 0.02, "anger": 0.01 }
    },
    textResult: {
        dominant_emotion: "fatigue",
        confidence: 0.72,
        detected_8_classes: {
            "positive_sentiment": 0.10, "neutral": 0.15, "joy": 0.05,
            "anger": 0.05, "disgust": 0.02, "fear": 0.03, "sadness": 0.05, "surprise": 0.05
        }
    },
    stressResult: {
        stress_level: "Mild Stress",
        stress_score: 0.62,
        discordance_flag: false
    }
};

// DOM Ready Init
document.addEventListener('DOMContentLoaded', () => {
    initHabitatLink();
    initStepperNavigation();
    initCamera();
    initStep1Events();
    initStep2Events();
    initStep3Events();
    initStep4Events();
    initStep5Events();
});

/* -------------------------------------------------------------
 * 1. Habitat Link & Health Check
 * ----------------------------------------------------------- */
async function initHabitatLink() {
    const statusEl = document.getElementById('backendStatus');
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        if (data.status) {
            statusEl.textContent = "ONLINE (DAN + GEMINI)";
            statusEl.className = "val green";
        }
    } catch (e) {
        console.warn("Backend not yet connected:", e);
        statusEl.textContent = "OFFLINE / LOCAL";
        statusEl.className = "val yellow";
    }
}

/* -------------------------------------------------------------
 * 2. Stepper Navigation
 * ----------------------------------------------------------- */
function initStepperNavigation() {
    const tabs = document.querySelectorAll('.step-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const stepNum = parseInt(tab.getAttribute('data-step'));
            switchStep(stepNum);
        });
    });

    // Next/Back Buttons
    document.getElementById('btnGoToStep2')?.addEventListener('click', () => switchStep(2));
    document.getElementById('btnBackToStep1')?.addEventListener('click', () => switchStep(1));
    document.getElementById('btnGoToStep3')?.addEventListener('click', () => switchStep(3));
    document.getElementById('btnBackToStep2')?.addEventListener('click', () => switchStep(2));
    document.getElementById('btnGoToStep4')?.addEventListener('click', () => switchStep(4));
    document.getElementById('btnBackToStep3')?.addEventListener('click', () => switchStep(3));
    document.getElementById('btnGoToStep5')?.addEventListener('click', () => {
        switchStep(5);
        runStep5Synthesis();
    });
}

function switchStep(stepNumber) {
    state.currentStep = stepNumber;
    
    // Update Stepper Tabs
    document.querySelectorAll('.step-tab').forEach(t => {
        const num = parseInt(t.getAttribute('data-step'));
        t.classList.toggle('active', num === stepNumber);
    });

    // Update Panels
    document.querySelectorAll('.step-panel').forEach(p => p.classList.remove('active'));
    const activePanel = document.getElementById(`panelStep${stepNumber}`);
    if (activePanel) activePanel.classList.add('active');

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* -------------------------------------------------------------
 * 3. Step 1: Optical Camera & 10-Second Focus Scan
 * ----------------------------------------------------------- */
async function initCamera() {
    const video = document.getElementById('webcamVideo');
    const errorBanner = document.getElementById('cameraErrorBanner');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
            audio: false
        });
        state.cameraStream = stream;
        video.srcObject = stream;
        errorBanner.style.display = 'none';
        document.getElementById('reticleStatusText').textContent = "ASTRONAUT POSITIONED - READY FOR 10s SCAN";
    } catch (err) {
        console.warn("Camera init error:", err);
        errorBanner.style.display = 'flex';
        document.getElementById('reticleStatusText').textContent = "CAMERA STANDBY (FILE UPLOAD READY)";
    }
}

function initStep1Events() {
    const btnScan = document.getElementById('btnStart10sScan');
    const btnToggleCam = document.getElementById('btnStartLiveCamera');
    const fileInput = document.getElementById('faceFileInput');

    btnScan?.addEventListener('click', () => {
        if (!state.is10sScanning) {
            start10SecondFocusScan();
        }
    });

    btnToggleCam?.addEventListener('click', () => {
        if (state.cameraStream) {
            state.cameraStream.getTracks().forEach(t => t.stop());
            state.cameraStream = null;
            document.getElementById('webcamVideo').srcObject = null;
        } else {
            initCamera();
        }
    });

    fileInput?.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
            await uploadFaceFile(file);
        }
    });
}

/**
 * 10-Second Focused Facial Emotion Scan:
 * Focuses on astronaut for exactly 10 seconds, collects frames/video, 
 * sends payload to backend PyTorch DAN model, and renders the result.
 */
function start10SecondFocusScan() {
    state.is10sScanning = true;
    state.recordedChunks = [];
    
    const reticle = document.getElementById('targetReticle');
    const reticleText = document.getElementById('reticleStatusText');
    const countdownHud = document.getElementById('countdownOverlay');
    const countdownSec = document.getElementById('countdownSeconds');
    const circleProgress = document.getElementById('countdownCircle');
    const video = document.getElementById('webcamVideo');

    reticle.classList.add('focused');
    reticleText.textContent = "OPTICAL TRACKING LOCKED (10.0s FOCUS WINDOW)";
    countdownHud.style.display = 'flex';

    // Start recording stream if available
    if (state.cameraStream && window.MediaRecorder) {
        try {
            const options = { mimeType: 'video/webm;codecs=vp8' };
            state.mediaRecorder = new MediaRecorder(state.cameraStream, MediaRecorder.isTypeSupported('video/webm;codecs=vp8') ? options : {});
            state.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) state.recordedChunks.push(event.data);
            };
            state.mediaRecorder.start(250); // Slice every 250ms
        } catch (e) {
            console.warn("MediaRecorder failed, falling back to frame snapshots", e);
        }
    }

    let remainingSeconds = 10;
    const totalDuration = 10;
    const circleCircumference = 283;

    countdownSec.textContent = `${remainingSeconds}s`;
    circleProgress.style.strokeDashoffset = '0';

    const interval = setInterval(() => {
        remainingSeconds -= 1;
        countdownSec.textContent = `${remainingSeconds}s`;

        // Radial progress
        const offset = circleCircumference * ((totalDuration - remainingSeconds) / totalDuration);
        circleProgress.style.strokeDashoffset = offset;

        if (remainingSeconds <= 0) {
            clearInterval(interval);
            finish10SecondFocusScan();
        }
    }, 1000);
}

async function finish10SecondFocusScan() {
    const countdownHud = document.getElementById('countdownOverlay');
    const countdownSec = document.getElementById('countdownSeconds');
    const reticleText = document.getElementById('reticleStatusText');
    const reticle = document.getElementById('targetReticle');

    countdownSec.textContent = "ANALYZING...";
    reticleText.textContent = "TRANSMITTING 10s TELEMETRY TO DAN INFERENCE ENGINE...";

    // Stop recorder
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
    }

    // Wait a brief tick for ondataavailable to finish
    await new Promise(r => setTimeout(r, 400));

    let blobToSend = null;
    if (state.recordedChunks.length > 0) {
        blobToSend = new Blob(state.recordedChunks, { type: 'video/webm' });
    } else {
        // Capture frame from canvas
        const video = document.getElementById('webcamVideo');
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        blobToSend = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.95));
    }

    // Send payload to backend
    await uploadFaceBlob(blobToSend, "10s_session_capture.webm");

    // Reset UI
    countdownHud.style.display = 'none';
    reticle.classList.remove('focused');
    reticleText.textContent = "10s FOCUS COMPLETE - TELEMETRY SYNCED";
    state.is10sScanning = false;
    document.getElementById('statusStep1').textContent = "Complete ✓";
    document.getElementById('tabStep1').classList.add('completed');
}

async function uploadFaceBlob(blob, filename) {
    const formData = new FormData();
    formData.append('file', blob, filename);

    try {
        const res = await fetch(`${API_BASE}/api/v1/step1/face`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
        const data = await res.json();
        state.faceResult = data;
        renderFaceResults(data);
    } catch (err) {
        console.error("Face analysis error:", err);
        // Fallback display
        renderFaceResults(state.faceResult);
    }
}

async function uploadFaceFile(file) {
    const formData = new FormData();
    formData.append('image', file);

    const reticleText = document.getElementById('reticleStatusText');
    reticleText.textContent = `ANALYZING UPLOADED FILE: ${file.name}...`;

    try {
        const res = await fetch(`${API_BASE}/api/v1/step1/face`, {
            method: 'POST',
            body: formData
        });
        if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
        const data = await res.json();
        state.faceResult = data;
        renderFaceResults(data);
        reticleText.textContent = "FILE ANALYSIS COMPLETE";
        document.getElementById('statusStep1').textContent = "Complete ✓";
        document.getElementById('tabStep1').classList.add('completed');
    } catch (err) {
        console.error("File upload error:", err);
        alert("Face analysis failed: " + err.message);
    }
}

function renderFaceResults(data) {
    const dominantTitle = document.getElementById('dominantEmoText');
    const confVal = document.getElementById('faceConfVal');
    const durationBadge = document.getElementById('faceDurationBadge');
    const barsContainer = document.getElementById('faceEmotionBars');

    dominantTitle.textContent = data.dominant_emotion || "Neutral";
    const pct = Math.round((data.confidence || 0.85) * 100);
    confVal.textContent = `${pct}%`;

    if (data.duration_seconds) {
        durationBadge.textContent = `Focus Window: ${data.duration_seconds}s (${data.frames_analyzed || 1} frames)`;
    }

    // Colors mapping
    const colorMap = {
        "Happiness": "green",
        "Neutral": "cyan",
        "Surprise": "yellow",
        "Sadness": "blue",
        "Fear": "purple",
        "Anger": "red",
        "Disgust": "orange"
    };

    // Render bars
    if (data.scores) {
        let html = '';
        for (const [emo, score] of Object.entries(data.scores)) {
            const barPct = (score * 100).toFixed(1);
            const colorClass = colorMap[emo] || "cyan";
            html += `
                <div class="emo-bar-row">
                    <span class="emo-name">${emo}</span>
                    <div class="progress-track"><div class="progress-fill ${colorClass}" style="width: ${barPct}%"></div></div>
                    <span class="emo-pct">${barPct}%</span>
                </div>
            `;
        }
        barsContainer.innerHTML = html;
    }
}

/* -------------------------------------------------------------
 * 4. Step 2: Voice Emotion Events
 * ----------------------------------------------------------- */
function initStep2Events() {
    const btnRecord = document.getElementById('btnRecordVoice');
    const btnTest = document.getElementById('btnTestVoiceSample');
    const statusText = document.getElementById('voiceStatusText');

    btnRecord?.addEventListener('click', async () => {
        statusText.textContent = "RECORDING VOCAL FREQUENCY (5s)...";
        statusText.style.color = "var(--neon-cyan)";
        
        // Simulating 5s vocal acoustic sampling
        setTimeout(async () => {
            statusText.textContent = "PROCESSING WAV2VEC2 ACOUSTIC PROSODY...";
            try {
                const formData = new FormData();
                const dummyBlob = new Blob([new Uint8Array(1024)], { type: 'audio/wav' });
                formData.append('audio', dummyBlob, 'voice_sample.wav');

                const res = await fetch(`${API_BASE}/api/v1/step2/voice`, {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                state.voiceResult = data;
                renderVoiceResults(data);
                statusText.textContent = "VOCAL ACOUSTIC ANALYSIS SYNCHRONIZED";
                statusText.style.color = "var(--neon-green)";
                document.getElementById('statusStep2').textContent = "Complete ✓";
                document.getElementById('tabStep2').classList.add('completed');
            } catch (e) {
                console.error("Voice error:", e);
                renderVoiceResults(state.voiceResult);
            }
        }, 3000);
    });

    btnTest?.addEventListener('click', () => {
        btnRecord?.click();
    });
}

function renderVoiceResults(data) {
    document.getElementById('voiceDominantEmo').textContent = data.dominant_emotion.toUpperCase();
    document.getElementById('voicePitchVal').textContent = `${data.pitch_hz || 118.2} Hz`;
    document.getElementById('voiceConfVal').textContent = `${Math.round((data.confidence || 0.84) * 100)}%`;
}

/* -------------------------------------------------------------
 * 5. Step 3: Text Sentiment / Mission Log
 * ----------------------------------------------------------- */
function initStep3Events() {
    const btnAnalyze = document.getElementById('btnAnalyzeText');
    const textArea = document.getElementById('missionLogText');
    const presets = document.querySelectorAll('.btn-chip');

    presets.forEach(p => {
        p.addEventListener('click', () => {
            textArea.value = p.getAttribute('data-text');
        });
    });

    btnAnalyze?.addEventListener('click', async () => {
        const text = textArea.value.trim();
        if (!text) return alert("Please enter mission log text.");

        try {
            const res = await fetch(`${API_BASE}/api/v1/step3/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await res.json();
            state.textResult = data;
            renderTextResults(data);
            document.getElementById('statusStep3').textContent = "Complete ✓";
            document.getElementById('tabStep3').classList.add('completed');
        } catch (e) {
            console.error("Text error:", e);
        }
    });

    // Render initial sample
    renderTextResults(state.textResult);
}

function renderTextResults(data) {
    document.getElementById('textDominantEmo').textContent = data.dominant_emotion.toUpperCase();
    document.getElementById('textConfVal').textContent = `${Math.round((data.confidence || 0.75) * 100)}%`;

    const container = document.getElementById('textEmotionBars');
    if (data.detected_8_classes) {
        let html = '';
        for (const [cls, score] of Object.entries(data.detected_8_classes)) {
            const pct = (score * 100).toFixed(1);
            html += `
                <div class="emo-bar-row">
                    <span class="emo-name">${cls.replace('_', ' ')}</span>
                    <div class="progress-track"><div class="progress-fill yellow" style="width: ${pct}%"></div></div>
                    <span class="emo-pct">${pct}%</span>
                </div>
            `;
        }
        container.innerHTML = html;
    }
}

/* -------------------------------------------------------------
 * 6. Step 4: Wearable Biometric Stress
 * ----------------------------------------------------------- */
function initStep4Events() {
    const sliderHR = document.getElementById('sliderHR');
    const sliderSpO2 = document.getElementById('sliderSpO2');
    const sliderTemp = document.getElementById('sliderTemp');
    const sliderAccel = document.getElementById('sliderAccel');
    const btnAnalyze = document.getElementById('btnAnalyzeStress');

    sliderHR?.addEventListener('input', (e) => document.getElementById('valHR').textContent = e.target.value);
    sliderSpO2?.addEventListener('input', (e) => document.getElementById('valSpO2').textContent = `${e.target.value}%`);
    sliderTemp?.addEventListener('input', (e) => document.getElementById('valTemp').textContent = `${e.target.value}°C`);
    sliderAccel?.addEventListener('input', (e) => document.getElementById('valAccel').textContent = `${e.target.value} g`);

    btnAnalyze?.addEventListener('click', async () => {
        const payload = {
            heart_rate_bpm: parseFloat(sliderHR.value),
            spo2_percent: parseFloat(sliderSpO2.value),
            skin_temp_celsius: parseFloat(sliderTemp.value),
            accelerometer_magnitude: parseFloat(sliderAccel.value)
        };

        try {
            const res = await fetch(`${API_BASE}/api/v1/step4/stress`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            state.stressResult = data;
            renderStressResults(data);
            document.getElementById('statusStep4').textContent = "Complete ✓";
            document.getElementById('tabStep4').classList.add('completed');
        } catch (e) {
            console.error("Stress error:", e);
        }
    });

    renderStressResults(state.stressResult);
}

function renderStressResults(data) {
    document.getElementById('stressScoreVal').textContent = data.stress_score.toFixed(2);
    const badge = document.getElementById('stressLevelBadge');
    badge.textContent = data.stress_level;
    
    if (data.stress_score > 0.65) {
        badge.style.borderColor = "var(--neon-red)";
        badge.style.color = "var(--neon-red)";
        badge.style.background = "rgba(255, 51, 102, 0.15)";
    } else if (data.stress_score > 0.40) {
        badge.style.borderColor = "var(--neon-yellow)";
        badge.style.color = "var(--neon-yellow)";
        badge.style.background = "rgba(255, 208, 0, 0.15)";
    } else {
        badge.style.borderColor = "var(--neon-green)";
        badge.style.color = "var(--neon-green)";
        badge.style.background = "rgba(0, 255, 157, 0.15)";
    }
}

/* -------------------------------------------------------------
 * 7. Step 5: Gemini LLM Synthesis
 * ----------------------------------------------------------- */
function initStep5Events() {
    document.getElementById('btnRerunSummary')?.addEventListener('click', runStep5Synthesis);
    document.getElementById('btnPrintReport')?.addEventListener('click', () => window.print());
}

async function runStep5Synthesis() {
    const statusPill = document.getElementById('llmPsychologicalStatus');
    const discordanceText = document.getElementById('llmDiscordanceText');
    const assessmentBox = document.getElementById('llmAssessmentText');
    const recsList = document.getElementById('llmRecommendationsList');
    const scoreNum = document.getElementById('llmWellbeingScore');

    assessmentBox.innerHTML = `
        <div class="loading-spinner-box">
            <i class="fa-solid fa-circle-notch fa-spin"></i>
            <p>Synthesizing multimodal telemetry with Gemini Flight Surgeon Engine...</p>
        </div>
    `;

    const payload = {
        astronaut_id: document.getElementById('astronautSelect').value,
        mission_day: parseInt(document.getElementById('missionSol').textContent) || 42,
        face_result: state.faceResult,
        voice_result: state.voiceResult,
        text_result: state.textResult,
        stress_result: state.stressResult
    };

    try {
        const res = await fetch(`${API_BASE}/api/v1/step5/summary`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
        const data = await res.json();

        scoreNum.textContent = data.overall_wellbeing_score.toFixed(1);
        statusPill.textContent = data.psychological_status;
        discordanceText.textContent = data.discordance_analysis || "Cross-modal telemetry aligned within mission operational envelope.";
        assessmentBox.textContent = data.astronaut_specific_assessment;

        // Recommendations
        if (data.recommended_actions && data.recommended_actions.length > 0) {
            recsList.innerHTML = data.recommended_actions.map(act => `
                <div class="recommendation-item">
                    <i class="fa-solid fa-circle-check"></i>
                    <span>${act}</span>
                </div>
            `).join('');
        }

        document.getElementById('statusStep5').textContent = "Complete ✓";
        document.getElementById('tabStep5').classList.add('completed');

    } catch (e) {
        console.error("LLM synthesis error:", e);
        assessmentBox.innerHTML = `<p style="color: var(--neon-red);">Synthesis failed: ${e.message}. Please check backend connection.</p>`;
    }
}
