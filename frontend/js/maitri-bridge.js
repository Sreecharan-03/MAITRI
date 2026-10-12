/**
 * MAITRI Telemetry Bridge & API Client
 * Centralized connector between Frontend and FastAPI Backend on port 8000.
 */

const API_BASE = (window.location.port === '8000') 
  ? window.location.origin 
  : 'http://localhost:8000';

const API_V1 = `${API_BASE}/api/v1`;

// Default / Initial Session State
const DEFAULT_TELEMETRY = {
    astronaut_id: "ASTRO-01",
    mission_day: 42,
    face_result: {
        dominant_emotion: "Neutral",
        confidence: 0.89,
        scores: { "Surprise": 0.02, "Fear": 0.01, "Disgust": 0.01, "Happiness": 0.15, "Sadness": 0.04, "Anger": 0.01, "Neutral": 0.77 },
        face_detected: true,
        frames_analyzed: 10,
        duration_seconds: 10.0
    },
    voice_result: {
        dominant_emotion: "calm",
        confidence: 0.84,
        pitch_hz: 118.2,
        scores: { "calm": 0.84, "neutral": 0.08, "sadness": 0.05, "fear": 0.02, "anger": 0.01 }
    },
    text_result: {
        dominant_emotion: "fatigue",
        confidence: 0.72,
        detected_8_classes: {
            "positive_sentiment": 0.10, "neutral": 0.15, "joy": 0.05,
            "anger": 0.05, "disgust": 0.02, "fear": 0.03, "sadness": 0.05, "surprise": 0.05
        }
    },
    stress_result: {
        stress_level: "Mild Stress",
        stress_score: 0.62,
        discordance_flag: false,
        details: { hr_bpm: 78, hrv_ms: 38, eda_us: 3.2, skin_temp_c: 36.5 }
    },
    summary_result: null
};

// Retrieve saved state or initialize
function getTelemetryState() {
    const raw = localStorage.getItem('maitri_telemetry');
    if (!raw) {
        localStorage.setItem('maitri_telemetry', JSON.stringify(DEFAULT_TELEMETRY));
        return DEFAULT_TELEMETRY;
    }
    try {
        return JSON.parse(raw);
    } catch {
        return DEFAULT_TELEMETRY;
    }
}

function updateTelemetryState(key, data) {
    const state = getTelemetryState();
    state[key] = data;
    localStorage.setItem('maitri_telemetry', JSON.stringify(state));
}

// Check Backend Health
async function checkBackendHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        return await res.json();
    } catch (e) {
        console.warn("MAITRI Backend offline or unreachable:", e);
        return null;
    }
}

/**
 * Universal API Client for all 5 Multimodal Telemetry Assessment Steps
 */
const maitriApi = {
    // Step 1: Send base64 image or video FormData to DAN PyTorch model
    analyzeFace: async (formData) => {
        const response = await fetch(`${API_V1}/step1/face`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error(`Face API HTTP ${response.status}`);
        return await response.json();
    },

    // Step 2: Send recorded audio blob to Wav2Vec2
    analyzeVoice: async (audioFormData) => {
        const response = await fetch(`${API_V1}/step2/voice`, {
            method: 'POST',
            body: audioFormData
        });
        if (!response.ok) throw new Error(`Voice API HTTP ${response.status}`);
        return await response.json();
    },

    // Step 3: Send log text to GoEmotions RoBERTa
    analyzeText: async (textPayload) => {
        const response = await fetch(`${API_V1}/step3/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(textPayload)
        });
        if (!response.ok) throw new Error(`Text API HTTP ${response.status}`);
        return await response.json();
    },

    // Step 4: Send WESAD biosensors to XGBoost / Stress model
    analyzeStress: async (vitalsPayload) => {
        const response = await fetch(`${API_V1}/step4/stress`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(vitalsPayload)
        });
        if (!response.ok) throw new Error(`Stress API HTTP ${response.status}`);
        return await response.json();
    },

    // Step 5: Send unified assessment data + API key to Gemini Synthesis
    synthesizeAssessment: async (summaryPayload) => {
        const response = await fetch(`${API_V1}/step5/summary`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(summaryPayload)
        });
        if (!response.ok) throw new Error(`Summary API HTTP ${response.status}`);
        return await response.json();
    }
};

// Export to global window scope
window.maitriApi = maitriApi;
window.API_BASE = API_BASE;
window.API_V1 = API_V1;
