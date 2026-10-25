/**
 * MAITRI Telemetry Bridge & API Client
 * Cloud-Ready Connector for Local, Vercel, and Render Deployments.
 */

// 1. Resolve Backend Base URL
function resolveApiBase() {
    // A. Explicit global variable override (e.g., injected during build/HTML)
    if (typeof window !== 'undefined' && window.MAITRI_API_BASE) {
        return window.MAITRI_API_BASE.replace(/\/+$/, '');
    }

    // B. User-configured override saved in localStorage
    if (typeof window !== 'undefined' && window.localStorage) {
        var saved = window.localStorage.getItem('maitri_api_base');
        if (saved && saved.trim()) {
            return saved.trim().replace(/\/+$/, '');
        }
    }

    // C. Host-based automatic detection
    if (typeof window !== 'undefined' && window.location) {
        var loc = window.location;
        // Running on localhost/127.0.0.1
        if (loc.hostname === 'localhost' || loc.hostname === '127.0.0.1') {
            return loc.port === '8000' ? loc.origin : 'http://localhost:8000';
        }
        // Running on Vercel or cloud domain:
        // Use relative path by default so Vercel rewrites in vercel.json proxy directly to Render,
        // or fallback to preconfigured default Render backend.
        var fallbackCloud = window.__DEFAULT_RENDER_URL__ || 'https://maitri-backend-lona.onrender.com';
        return fallbackCloud.replace(/\/+$/, '');
    }

    return 'http://localhost:8000';
}

var API_BASE = resolveApiBase();
var API_V1 = `${API_BASE}/api/v1`;

window.API_BASE = API_BASE;
window.API_V1 = API_V1;

// Function to update Backend API URL dynamically from UI
window.setMaitriBackendUrl = function (newUrl) {
    if (typeof window !== 'undefined' && window.localStorage) {
        if (!newUrl || !newUrl.trim()) {
            window.localStorage.removeItem('maitri_api_base');
        } else {
            var cleaned = newUrl.trim().replace(/\/+$/, '');
            window.localStorage.setItem('maitri_api_base', cleaned);
        }
        window.location.reload();
    }
};

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

// Check Backend Health with timing
async function checkBackendHealth() {
    try {
        const start = performance.now();
        const url = `${API_BASE}/health`;
        const res = await fetch(url);
        const latency = Math.round(performance.now() - start);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        return { ok: true, latency, data };
    } catch (e) {
        console.warn("MAITRI Backend offline or unreachable:", e);
        return { ok: false, error: e.message };
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
    },

    // Multimodal End-to-End Prediction (Entropy-Gated Fusion & CSI)
    predictMultimodal: async (formDataOrJson) => {
        const isFormData = typeof FormData !== 'undefined' && formDataOrJson instanceof FormData;
        const headers = isFormData ? {} : { 'Content-Type': 'application/json' };
        const body = isFormData ? formDataOrJson : JSON.stringify(formDataOrJson);
        const response = await fetch(`${API_V1}/predict_multimodal`, {
            method: 'POST',
            headers: headers,
            body: body
        });
        if (!response.ok) throw new Error(`Predict Multimodal API HTTP ${response.status}`);
        return await response.json();
    }
};

// Export to global window scope
window.maitriApi = maitriApi;
window.getTelemetryState = getTelemetryState;
window.updateTelemetryState = updateTelemetryState;
window.checkBackendHealth = checkBackendHealth;

// ==============================================================================
// Sleek Floating Aerospace Gateway Badge & Modal (Auto-mounted)
// ==============================================================================
(function initTelemetryGatewayWidget() {
    if (typeof document === 'undefined') return;

    window.addEventListener('DOMContentLoaded', async () => {
        // Prevent duplicate injection
        if (document.getElementById('maitri-gateway-root')) return;

        const root = document.createElement('div');
        root.id = 'maitri-gateway-root';
        root.innerHTML = `
            <style>
                .maitri-gw-pill {
                    position: fixed;
                    bottom: 16px;
                    right: 16px;
                    z-index: 9999;
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    background: rgba(2, 4, 10, 0.85);
                    border: 1px solid rgba(6, 182, 212, 0.35);
                    border-radius: 9999px;
                    padding: 6px 14px;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 11px;
                    color: #94a3b8;
                    cursor: pointer;
                    backdrop-filter: blur(12px);
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5), 0 0 12px rgba(6, 182, 212, 0.15);
                    transition: all 0.25s ease;
                    user-select: none;
                }
                .maitri-gw-pill:hover {
                    border-color: rgba(6, 182, 212, 0.8);
                    color: #e2e8f0;
                    transform: translateY(-1px);
                    box-shadow: 0 6px 24px rgba(6, 182, 212, 0.25);
                }
                .maitri-gw-dot {
                    width: 7px;
                    height: 7px;
                    border-radius: 50%;
                    background: #64748b;
                }
                .maitri-gw-dot.active {
                    background: #10b981;
                    box-shadow: 0 0 8px #10b981;
                }
                .maitri-gw-dot.warn {
                    background: #f59e0b;
                    box-shadow: 0 0 8px #f59e0b;
                    animation: maitri-pulse 1.8s infinite;
                }
                @keyframes maitri-pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.4; }
                }
                .maitri-gw-modal-backdrop {
                    position: fixed;
                    inset: 0;
                    background: rgba(2, 4, 10, 0.8);
                    backdrop-filter: blur(8px);
                    z-index: 10000;
                    display: none;
                    align-items: center;
                    justify-content: center;
                    padding: 16px;
                }
                .maitri-gw-modal {
                    background: #02040a;
                    border: 1px solid rgba(6, 182, 212, 0.4);
                    border-radius: 16px;
                    max-width: 480px;
                    width: 100%;
                    padding: 24px;
                    color: #f1f5f9;
                    font-family: 'Inter', sans-serif;
                    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8), 0 0 30px rgba(6, 182, 212, 0.15);
                }
            </style>
            <div id="maitri-gw-pill" class="maitri-gw-pill" title="Click to inspect or change MAITRI Backend connection">
                <span id="maitri-gw-dot" class="maitri-gw-dot"></span>
                <span id="maitri-gw-label">Checking Link...</span>
            </div>

            <div id="maitri-gw-modal" class="maitri-gw-modal-backdrop">
                <div class="maitri-gw-modal">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 16px; font-weight: 700; color: #06b6d4; display: flex; align-items: center; gap: 8px;">
                            <span>🛰️</span>
                            <span>MAITRI Telemetry Gateway</span>
                        </div>
                        <button id="maitri-gw-close" style="background: none; border: none; color: #64748b; font-size: 20px; cursor: pointer;">✕</button>
                    </div>

                    <p style="font-size: 13px; color: #94a3b8; line-height: 1.5; margin-bottom: 16px;">
                        Configure the FastAPI telemetry backend URL. When deployed on <b>Vercel</b>, link your <b>Render</b> deployment endpoint here.
                    </p>

                    <label style="display: block; font-size: 11px; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; color: #cbd5e1; margin-bottom: 6px; letter-spacing: 0.05em;">
                        Backend API URL (Render / Local):
                    </label>
                    <input id="maitri-gw-input" type="text" placeholder="https://maitri-backend-lona.onrender.com"
                        style="width: 100%; box-sizing: border-box; background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 8px; padding: 10px 12px; color: #f8fafc; font-family: 'JetBrains Mono', monospace; font-size: 13px; margin-bottom: 14px; outline: none;">

                    <div id="maitri-gw-health-report" style="font-size: 12px; font-family: 'JetBrains Mono', monospace; background: rgba(15, 23, 42, 0.5); border: 1px solid #1e293b; border-radius: 8px; padding: 10px; margin-bottom: 16px; color: #94a3b8;">
                        Status: Ready to test...
                    </div>

                    <div style="display: flex; gap: 10px; justify-content: flex-end;">
                        <button id="maitri-gw-reset" style="background: rgba(30, 41, 59, 0.8); border: 1px solid #334155; color: #cbd5e1; border-radius: 8px; padding: 8px 14px; font-size: 12px; cursor: pointer; font-weight: 500;">
                            Reset to Default
                        </button>
                        <button id="maitri-gw-save" style="background: #06b6d4; border: none; color: #02040a; border-radius: 8px; padding: 8px 16px; font-size: 12px; cursor: pointer; font-weight: 600;">
                            Connect & Save
                        </button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(root);

        const pill = document.getElementById('maitri-gw-pill');
        const dot = document.getElementById('maitri-gw-dot');
        const label = document.getElementById('maitri-gw-label');
        const modal = document.getElementById('maitri-gw-modal');
        const closeBtn = document.getElementById('maitri-gw-close');
        const input = document.getElementById('maitri-gw-input');
        const saveBtn = document.getElementById('maitri-gw-save');
        const resetBtn = document.getElementById('maitri-gw-reset');
        const report = document.getElementById('maitri-gw-health-report');

        // Set current input value
        input.value = API_BASE || window.location.origin;

        // Health check on load
        async function runHealthCheck() {
            const res = await checkBackendHealth();
            if (res.ok) {
                dot.className = 'maitri-gw-dot active';
                label.textContent = `Telemetry Live (${res.latency}ms)`;
                report.innerHTML = `<span style="color: #10b981;">✓ Online</span> • Latency: ${res.latency}ms<br>Models: Active • Gemini: ${res.data?.models?.gemini_api_configured ? 'Configured' : 'No Key'}`;
            } else {
                dot.className = 'maitri-gw-dot warn';
                label.textContent = `Link Offline • Configure`;
                report.innerHTML = `<span style="color: #f59e0b;">⚠ Unreachable:</span> ${res.error || 'Server not responding'}<br>Tip: Enter your Render backend URL above.`;
            }
        }

        runHealthCheck();

        // Modal toggle
        pill.addEventListener('click', () => {
            modal.style.display = 'flex';
        });

        closeBtn.addEventListener('click', () => {
            modal.style.display = 'none';
        });

        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.style.display = 'none';
        });

        saveBtn.addEventListener('click', () => {
            const val = input.value.trim();
            window.setMaitriBackendUrl(val);
        });

        resetBtn.addEventListener('click', () => {
            window.setMaitriBackendUrl('');
        });
    });
})();
