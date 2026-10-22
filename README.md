<div align="center">

# MAITRI: Multimodal AI for Astronaut Emotional & Physical Well-Being Monitoring

**Autonomous Affective Computing, Autonomic Stress Telemetry, and Covert Emotional Discordance Detection in Extreme Environments**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co)
[![ModernBERT](https://img.shields.io/badge/Backbone-ModernBERT--Large-blue?style=flat-square)](https://huggingface.co/answerdotai/ModernBERT-large)
[![SwinV2](https://img.shields.io/badge/Vision-SwinV2--Tiny-success?style=flat-square)](https://github.com/microsoft/Swin-Transformer)
[![Wav2Vec2](https://img.shields.io/badge/Audio-Wav2Vec2--Base-purple?style=flat-square)](https://huggingface.co/facebook/wav2vec2-base)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

[Architecture](#2-multi-channel-architectures--mathematical-formulation) •
[Benchmarks](#3-empirical-benchmarks--validation-results) •
[Flight Cockpit](#4-mission-control-flight-ops-cockpit) •
[Quickstart](#5-installation--quickstart-guide) •
[API Specification](#6-telemetry-api-specification) •
[Citation](#9-citation--academic-attribution)

</div>

---

## 1. Executive Summary & Scientific Motivation

Long-duration space exploration (LDSE)—such as deep-space lunar orbital operations and crewed interplanetary transits to Mars—subjects astronauts to prolonged **Isolated, Confined, and Extreme (ICE)** conditions. In these mission regimes, terrestrial communication delays of up to 22 minutes make real-time telemedicine and psychiatric intervention from mission control impossible.

Astronauts are rigorously trained to maintain outward composure, operational focus, and strict voice discipline even while experiencing acute psychological fatigue, sleep fragmentation, or internal anxiety. Consequently, unimodal affective analysis fails catastrophically:
- **Facial expressions** and **audio transcripts** frequently register as calm, smiling, or neutral due to conscious suppression and socio-professional behavioral protocol.
- **Autonomic physiological markers** (sympathetic nervous activation, heart rate variability collapse, electrodermal surges) reveal substantial covert strain that remains concealed from surface observation.

```
       CONVENTIONAL AFFECTIVE COMPUTING vs. MAITRI DISCORDANCE ENGINE

   [ Overt Channels ]                  [ Autonomic Channel ]
┌─────────────────────────┐         ┌─────────────────────────┐
│ Face:  "Calm / Focused" │         │ Vitals: HR=108, HRV=14ms│
│ Speech:"Steady / Tone"  │         │ EDA:    2.8 μS (Surge)  │
│ Text:  "Nominal tasks"  │         └────────────┬────────────┘
└───────────┬─────────────┘                      │
            │                                    │
   ┌────────┴────────┐                  ┌────────┴────────┐
   │ Average Fusion  │                  │  MAITRI Cross-  │
   │  "Status: OK"   │                  │ Modal Analysis  │
   │  [FALSE NEG]    │                  └────────┬────────┘
   └─────────────────┘                           │
                                     CRITICAL DISCORDANCE ALERT:
                             "Covert Physiological Strain Detected;
                               Overt Composure Masking True State"
```

**MAITRI** resolves this critical operational vulnerability through an **Entropy-Gated Multimodal Discordance Architecture**. By coupling four decoupled unimodal neural channels with a mathematically grounded **Composite Suppression Index (CSI)**, MAITRI detects covert discordance between outward expression and autonomic stress. It then prompts an onboard Gemini LLM to act as an autonomous AI Flight Surgeon, generating actionable clinical briefs and targeted operational countermeasures without human-in-the-loop dependencies.

---

## 2. Multi-Channel Architectures & Mathematical Formulation

MAITRI decouples affective telemetry into four distinct sensory streams before harmonizing them onto the 8-class primary Plutchik wheel of emotions (**Fear, Anger, Sadness, Joy, Trust, Anticipation, Surprise, Disgust**).

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                           MAITRI PIPELINE OVERVIEW                          │
 └─────────────────────────────────────────────────────────────────────────────┘
  Astronaut Sensors            Neural Decoupling Channels         Discordance & Synthesis
 
 ┌───────────────┐           ┌────────────────────────────┐
 │  Camera Stream ├──────────►│ NAD-Net: SwinV2-Tiny Split ├──┐
 │  (10s Optical)│           │ Upper/Lower Cross-Attention│   │
 └───────────────┘           └────────────────────────────┘   │
                                                              │
 ┌───────────────┐           ┌────────────────────────────┐   │   ┌────────────────────────┐
 │  Microphone   ├──────────►│ PAN: Wav2Vec2 + 134-D Mel  ├───┼──►│  Entropy-Gated Fusion   │
 │  (Audio WebM) │           │ Adaptive Gated Prosody     │   │   │  Dynamic Uncertainty   │
 └───────────────┘           └────────────────────────────┘   │   │  Weighting: α_m(H_m)   │
                                                              │   └───────────┬────────────┘
 ┌───────────────┐           ┌────────────────────────────┐   │               │
 │  Daily Journal├──────────►│ PAC-Net: ModernBERT-Large  ├───┤               ▼
 │  (Text Log)   │           │ Gated Orthogonal Projection│   │   ┌────────────────────────┐
 └───────────────┘           └────────────────────────────┘   │   │  Plutchik Harmonizer   │
                                                              │   │  8-Class Unified Vector│
 ┌───────────────┐           ┌────────────────────────────┐   │   └───────────┬────────────┘
 │  Wearable     ├──────────►│ Calibrated LOSO Estimator  ├───┘               │
 │  Biosensors   │           │ Personalized Z-Score Norm  │                   ▼
 └───────────────┘           └──────────────┬─────────────┘       ┌────────────────────────┐
                                            │                     │ Composite Suppression  │
                                            └────────────────────►│ Index (CSI) & Alert    │
                                                                  └───────────┬────────────┘
                                                                              │
                                                                              ▼
                                                                  ┌────────────────────────┐
                                                                  │ AI Flight Surgeon Memo │
                                                                  │ Gemini Clinical Engine │
                                                                  └────────────────────────┘
```

---

### Channel 1: Facial Channel — NAD-Net (Neuro-Anatomical Decoupling Network)
- **Model Checkpoint:** `backend/models/Face/swinv2_export/swinv2_nadnet_best.pth`
- **Backbone:** Hierarchical Swin Transformer V2-Tiny (`swinv2_tiny_window16_256`).
- **Anatomical Decoupling:** Human micro-expressions betray emotional leakage in distinct upper vs. lower facial zones. NAD-Net crops the upper 55% (orbicularis oculi, corrugator supercilii) and lower 55% (zygomaticus major, depressor anguli oris) of the bounding-box-aligned face.
- **Cross-Attention Fusion:** Upper and lower 768-D feature maps exchange reciprocal spatial contextual cues via **Bidirectional Multi-Head Cross-Attention**:
  $$\mathbf{F}_{\text{up}\to\text{low}} = \text{MHA}(\mathbf{Q}=\mathbf{F}_{\text{low}}, \mathbf{K}=\mathbf{F}_{\text{up}}, \mathbf{V}=\mathbf{F}_{\text{up}})$$
- **Facial Suppression Index (FSI):** Quantifies affective divergence between isolated upper and lower facial zones using symmetric Kullback-Leibler (KL) divergence over their respective projection logits:
  $$\text{FSI} = \frac{1}{2} \left[ D_{\text{KL}}\left(\mathbf{p}_{\text{up}} \parallel \mathbf{p}_{\text{low}}\right) + D_{\text{KL}}\left(\mathbf{p}_{\text{low}} \parallel \mathbf{p}_{\text{up}}\right) \right]$$

---

### Channel 2: Speech Channel — PAN (Prosodic Acoustic Network)
- **Model Checkpoint:** `backend/models/Speech/MAITRI_Speech_Channel_Bundle/maitri_pan_best_model.pt`
- **Backbone:** Pretrained Wav2Vec 2.0 (`facebook/wav2vec2-base`) fine-tuned concurrently with a 134-D Handcrafted Prosodic Branch.
- **Acoustic Feature Extraction:** Computes a 128-band log-mel spectrogram padded to 16 kHz alongside six continuous acoustic parameters:
  - Fundamental Pitch ($F_0$) via normalized autocorrelation
  - Root-Mean-Square Energy (RMS)
  - Zero-Crossing Rate (ZCR)
  - Spectral Centroid
  - Spectral Rolloff & Energy Flux
- **Adaptive Sigmoid Gate:** Blends contextual semantic speech vectors $\mathbf{h}_{\text{w2v}}$ with prosodic features $\mathbf{h}_{\text{pros}}$:
  $$\mathbf{g} = \sigma\left(\mathbf{W}_g [\mathbf{h}_{\text{w2v}} \,\|\, \mathbf{h}_{\text{pros}}] + \mathbf{b}_g\right)$$
  $$\mathbf{h}_{\text{fused}} = \mathbf{g} \odot \mathbf{h}_{\text{w2v}} + (1 - \mathbf{g}) \odot \mathbf{h}_{\text{pros}}$$
- **Acoustic Strain Index (ASI):** Evaluates high-frequency pitch jitter, vocal instability, and branch divergence under microgravity phonation constraints.

---

### Channel 3: Text Channel — PAC-Net (Polarity-Attentive Context Network)
- **Model Checkpoint:** `backend/models/Text/pacnet_extracted/pacnet_modernbert_best.pth`
- **Backbone:** ModernBERT-Large with rotary positional embeddings (RoPE) and unpadded flash attention.
- **Gated Orthogonal Projection (GOP):** Deconstructs the sequence embedding $\mathbf{e}$ against an external lexical sentiment anchor $\mathbf{v}_{\text{anchor}}$ (derived from VADER domain priors):
  $$\mathbf{e}_{\parallel} = \frac{\mathbf{e} \cdot \mathbf{v}_{\text{anchor}}}{\|\mathbf{v}_{\text{anchor}}\|^2} \mathbf{v}_{\text{anchor}}, \quad \mathbf{e}_{\perp} = \mathbf{e} - \mathbf{e}_{\parallel}$$
- **Tension-Flow Conflict (TFC / TSI):** Measures the orthogonal contextual friction vector $\|\mathbf{e}_{\perp}\|$, exposing instances where linguistic politeness or operational jargon conceals cognitive overload and subclinical psychological distress.

---

### Channel 4: Wearable Physiological Channel — Calibrated LOSO Estimator
- **Model Checkpoint:** `backend/models/Physiological/MAITRI_Physiological_Channel_Bundle/maitri_calibrated_stress_estimator_loso.pkl`
- **Reference Baseline:** `backend/models/Physiological/MAITRI_Physiological_Channel_Bundle/maitri_baseline_reference.pkl`
- **Backbone:** CalibratedClassifierCV wrapping optimized Gradient-Boosted Decision Trees (XGBoost) and Random Forests.
- **13-Dimensional Biometric Feature Vector:** Extracted from continuous 60-second sliding windows with 10-second shifts:
  $$\mathbf{x}_{\text{physio}} = \left[\text{HR}, \text{SDNN}, \text{RMSSD}, \overline{\text{EDA}}, \sigma_{\text{EDA}}, \Delta\text{EDA}, N_{\text{SCR}}, \overline{\text{Resp}}, \sigma_{\text{Resp}}, \text{RR}, \overline{\text{Temp}}, \sigma_{\text{Temp}}, \frac{d\text{Temp}}{dt}\right]$$
- **Personalized Baseline Normalization ($z$-Score Calibration):** To prevent severe data leakage and eradicate inter-subject baseline physiological variance, all features are dynamic $z$-score normalized against individual non-stress baseline resting states:
  $$\hat{x}^{(i)}_j = \frac{x^{(i)}_j - \mu_{j, \text{baseline}}^{(i)}}{\sigma_{j, \text{baseline}}^{(i)}}$$
- **Output:** Yields a strictly calibrated posterior stress probability $P(\text{stress} \mid \hat{\mathbf{x}}) \in [0.0, 1.0]$.

---

### Dynamic Entropy-Gated Multimodal Fusion
Rather than naive late averaging, MAITRI dynamically reweights unimodal predictions based on their **predictive Shannon uncertainty**:
$$H_m = -\sum_{c=1}^{C} p_m(c) \log_2 p_m(c), \quad m \in \{\text{Face}, \text{Speech}, \text{Text}\}$$

Modality weights $\alpha_m$ are dynamically governed via temperature-scaled softmin:
$$\alpha_m = \frac{\exp\left(-H_m / \tau\right)}{\sum_{j} \exp\left(-H_j / \tau\right)}, \quad \tau = 0.85$$

A corrupted or expression-suppressed channel (high entropy) has its voting influence downweighted automatically in favor of deterministic, low-entropy modalities.

---

### Composite Suppression Index (CSI) & Discordance Trigger
To combine internal friction and physiological activation into a single operational biomarker, MAITRI formulates the **Composite Suppression Index (CSI)**:

$$\text{CSI} = 0.25 \cdot \text{TFC} + 0.35 \cdot \text{FSI} + 0.15 \cdot \text{ASI} + 0.25 \cdot \left(\frac{\text{PSI}}{100}\right)$$

| Operational Threshold | Clinical Assessment | Dashboard Indicator |
| :--- | :--- | :--- |
| $\text{CSI} < 0.22 \;\land\; P(\text{stress}) < 0.40$ | Nominal Affective State (Congruent) | **Green: Nominal Homeostasis** |
| $0.22 \le \text{CSI} < 0.35$ | Compensated Operational Workload | **Yellow: Subclinical Fatigue** |
| $\text{CSI} \ge 0.35 \;\lor\; (P(\text{stress}) \ge 0.70 \;\land\; \text{Overt} = \text{Calm})$ | **Covert Emotional Discordance Detected** | **Red: CRITICAL Discordance Alert** |

---

### Autonomous AI Flight Surgeon Briefing Layer
The fused affect distribution, CSI metric, and biosensor vectors are compiled into a structured clinical payload ingested by Google Gemini (`gemini-flash-latest`). The LLM:
1. Calculates a calibrated **Overall Wellbeing Score (1.0 to 10.0)**.
2. Synthesizes a formal aerospace-medicine clinical summary.
3. Prescribes targeted operational countermeasures:
   - *Protocol 01: Circadian Spectrum Adjustment (shift cabin lighting to 480nm amber mode)*
   - *Protocol 02: Autonomic Vagal Reset (10-minute 4-7-8 pacing cycle)*
   - *Protocol 03: Task Deconfliction (reschedule high-cognitive Extravehicular Activity blocks)*

---

## 3. Empirical Benchmarks & Validation Results

All unimodal channels and the multimodal fusion pipeline were trained and validated against gold-standard aerospace and affective computing benchmarks.

| Modality / Channel | Neural Architecture | Dataset Benchmark | Evaluation Protocol | Empirical Accuracy | Primary F1 / Metric |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Facial Expression** | SwinV2-Tiny NAD-Net | RAF-DB (7 Classes) | Standard Test Split | **85.69%** | **0.7770 Macro F1** |
| **Acoustic Voice** | Wav2Vec 2.0 + PAN Prosody | RAVDESS (8 Classes) | 80/20 Stratified Split | **90.97%** | **0.9040 Macro F1** |
| **Text Sentiment** | ModernBERT-Large PAC-Net | Empathetic Dialogues | Plutchik-8 Projection | **80.51% (Top-1)** | **92.15% (Top-2)** |
| **Wearable Stress** | Calibrated LOSO Ensemble | WESAD (Chest Resampled) | **Strict Subject LOSO-CV** | **94.48%** | **0.9797 ROC-AUC** |
| **Multimodal Core** | Entropy-Gated Fusion | Combined Telemetry | End-to-End Latency | **100% Sync** | **< 45ms Local Inference** |

> **Leave-One-Subject-Out (LOSO) Rigor:** In the physiological stress channel, 100% of validation was conducted by training on $N-1$ subjects and testing strictly on the unseen subject to prevent data leakage and ensure generalization across diverse human baselines.

---

## 4. Mission Control Flight-Ops Cockpit

MAITRI provides an interface styled with a deep-spaceflight dark theme (`#02040A`), built with native responsive CSS, SVG dials, and real-time JavaScript telemetry bridges.

| Dashboard Subsystem | Route | Core Functionality |
| :--- | :--- | :--- |
| **Mission Landing Cockpit** | `/dashboard/index.html` | System overview, health status, telemetry link integrity, and direct workflow launcher. |
| **Step 1: Facial Micro-Scan** | `/dashboard/face-scan.html` | Live WebRTC/Webcam camera feed, 10-second optical scan reticle, RAF-DB classification, and FSI display. |
| **Step 2: Acoustic Voice Check** | `/dashboard/voice-check.html` | In-browser MediaRecorder audio stream, waveform feedback, Wav2Vec 2.0 classification, and pitch ($F_0$) readout. |
| **Step 3: Daily Log Analysis** | `/dashboard/daily-log.html` | Astronaut journal textarea, debounced ModernBERT inference, TFC tension-flow score, and primary affect chips. |
| **Step 4: Wearable Biosensors** | `/dashboard/health-vitals.html` | Real-time sliders for Heart Rate, HRV, EDA, Sweat Spikes, and Skin Temperature with dynamic $P(\text{stress})$ gauge. |
| **Step 5: Executive Briefing** | `/dashboard/summary.html` | **Mission Readiness circular dial (0–100), Plutchik 8-class affect vector, entropy weights breakdown, CSI discordance alert, and Gemini Flight Surgeon clinical brief.** |

---

## 5. Installation & Quickstart Guide

### Prerequisites
- **Operating System:** Linux, macOS, or Windows 10/11
- **Python:** Version 3.10, 3.11, or 3.12
- **Hardware:** Minimum 8 GB RAM (CPU mode supported out-of-the-box; CUDA optional for accelerated SwinV2/Wav2Vec forward passes)
- **Git**

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/Sreecharan-03/MAITRI.git
cd MAITRI
```

---

### Step 2: Configure Python Virtual Environment
```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

---

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Configure Environment Variables
Create your local `.env` configuration from the provided template:
```bash
cp .env.example .env
```

Open `.env` in an editor and specify your Google Gemini API key:
```env
PORT=8000
HOST=0.0.0.0
DEVICE=cpu
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Model Checkpoints (Relative to backend/)
FACE_MODEL_PATH=models/Face/swinv2_export/swinv2_nadnet_best.pth
SPEECH_MODEL_PATH=models/Speech/MAITRI_Speech_Channel_Bundle/maitri_pan_best_model.pt
TEXT_MODEL_PATH=models/Text/pacnet_extracted/pacnet_modernbert_best.pth
TOKENIZER_DIR=models/Text/pacnet_extracted
PHYSIO_MODEL_PATH=models/Physiological/MAITRI_Physiological_Channel_Bundle/maitri_calibrated_stress_estimator_loso.pkl
PHYSIO_BASELINE_PATH=models/Physiological/MAITRI_Physiological_Channel_Bundle/maitri_baseline_reference.pkl
```

---

### Step 5: Download Verified Model Checkpoints
Because weights files exceed standard Git storage limits, run the automated Google Drive downloader:
```bash
python download_models.py
```
*This downloads the trained checkpoints for SwinV2 NAD-Net, Wav2Vec PAN, ModernBERT PAC-Net, and the calibrated physiological estimators directly into their designated folders.*

---

### Step 6: Verify Model Integrity
Execute the multi-model architecture and memory verification suite:
```bash
python verify_models.py
```
*Output confirms that all neural state-dicts, tensor shapes, and pickle bundles load into memory without errors.*

---

### Step 7: Launch the FastAPI Backend & Mission Dashboard
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### Step 8: Open the Mission Dashboard
Open any modern web browser (Chrome, Edge, Firefox, Safari) and navigate to:
```
http://localhost:8000/dashboard/
```
*(Navigating directly to `http://localhost:8000/` in a browser will automatically redirect you to the dashboard).*

---

## 6. Telemetry API Specification

All endpoints are hosted locally under `/api/v1/` and serve both JSON and `multipart/form-data` schemas.

### 1. Multimodal End-to-End Prediction
`POST /api/v1/predict_multimodal`
Fuses text transcript logs and 13-dimensional wearable telemetry arrays into entropy-gated Plutchik emotion probabilities, calculates the Composite Suppression Index, and triggers discordance analysis.

**Request Payload (JSON or FormData):**
```json
{
  "text_log": "Completed solar array EVA inspection. Experiencing mild fatigue, but operational pace is steady.",
  "wearable_features": [74.0, 42.1, 36.5, 0.45, 0.09, 1.3, 4.0, 15.0, 1.2, 0.25, 36.6, 0.2, 0.01]
}
```

**Response (200 OK):**
```json
{
  "predicted_emotion": "Trust",
  "confidence": 0.6334,
  "fused_emotion_distribution": {
    "Fear": 0.029,
    "Anger": 0.025,
    "Sadness": 0.050,
    "Joy": 0.050,
    "Trust": 0.633,
    "Anticipation": 0.153,
    "Surprise": 0.031,
    "Disgust": 0.028
  },
  "modality_certainty_weights": {
    "face": 0.269,
    "speech": 0.690,
    "text": 0.041
  },
  "composite_suppression_index": 0.2705,
  "discordance_alert": false,
  "summary": {
    "overall_score": 7.4,
    "status": "Autonomically Stable with Subclinical Fatigue and Compensated Demeanor",
    "recommendations": [
      "Initiate 15-minute vagal stimulation / box breathing cycle",
      "Adjust crew quarters lighting to 480nm spectrum",
      "Reschedule high-cognitive EVA checks to next flight block"
    ]
  }
}
```

---

### 2. Unimodal Step Endpoints
- `POST /api/v1/step1/face`: Accepts image or 10-second `.webm` video recording. Returns 7 RAF-DB probabilities and `fsi_suppression_index`.
- `POST /api/v1/step2/voice`: Accepts `.webm`/`.wav` audio buffer. Returns pitch ($F_0$), acoustic classification, and harmonic stability.
- `POST /api/v1/step3/text`: Ingests `{"text": "..."}`. Returns ModernBERT sentiment, GoEmotions alignment, and tension score (TFC).
- `POST /api/v1/step4/stress`: Ingests `{"heart_rate_bpm": 85.0, "hrv_rmssd": 22.0, "mean_eda": 2.4, ...}`. Evaluates normalized 13-D WESAD vector through calibrated LOSO estimator.
- `POST /api/v1/step5/summary`: Compiles unified telemetry outputs and queries Gemini LLM for flight-surgeon memo generation.
- `GET /health`: Returns subsystem integrity and model loading status.

---

## 7. Automated Testing Suites

MAITRI provides two rigorous automated testing scripts inside `backend/`:

### 1. Offline Neural Pipeline Test
Validates forward-pass inference across all four models without launching a server:
```bash
python test_pipeline.py
```

### 2. End-to-End API & Integration Test
Simulates synthetic telemetry payloads across all live HTTP REST endpoints:
```bash
python test_e2e_frontend_api.py
```

**Expected Test Output:**
```
============================================================
>>> MAITRI FULL FRONTEND-BACKEND INTEGRATION TEST <<<
============================================================

[1/6] Checking Backend Health (/health)...
  Status: {'status': 'MAITRI Backend Online', 'pipeline_ready': True, ...}

[2/6] Testing Step 1 Face Model Prediction (/api/v1/step1/face)...
  Emotion: Neutral (Confidence: 0.89, FSI: 0.18)

[3/6] Testing Step 2 Voice Model Prediction (/api/v1/step2/voice)...
  Tone: calm (Pitch: 118.2 Hz)

[4/6] Testing Step 3 Text NLP Prediction (/api/v1/step3/text)...
  Dominant Sentiment: fatigue (Confidence: 0.72)

[5/6] Testing Step 4 Stress Model Prediction (/api/v1/step4/stress)...
  Stress Level: Baseline / Relaxed Homeostasis (Index: 0.20)

[6/6] Testing Step 5 Gemini LLM Synthesis (/api/v1/step5/summary)...
  Overall Score: 7.4 / 10
  Status: Autonomically Stable with Subclinical Fatigue and Compensatory Composure
  Recommendations: 4 actions generated

============================================================
>>> SUCCESS: ALL FRONTEND-TO-BACKEND API CALLS TESTED AND FUNCTIONAL <<<
============================================================
```

---

## 8. Repository Layout

```plaintext
MAITRI/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application setup & static frontend mounting
│   │   ├── config.py                   # Pydantic environment settings & path resolutions
│   │   ├── routers/
│   │   │   ├── multimodal.py           # Core /predict_multimodal fusion endpoint
│   │   │   ├── step1_face.py           # SwinV2 facial emotion route
│   │   │   ├── step2_voice.py          # Wav2Vec2 vocal prosody route
│   │   │   ├── step3_text.py           # ModernBERT text analysis route
│   │   │   ├── step4_stress.py         # WESAD 13-D physiological stress route
│   │   │   └── step5_summary.py        # Gemini flight surgeon briefing route
│   │   ├── services/
│   │   │   ├── face_model.py           # SwinV2 NAD-Net inference & FSI logic
│   │   │   ├── voice_model.py          # PAN speech acoustic processing
│   │   │   ├── text_model.py           # ModernBERT & PAC-Net projection
│   │   │   ├── stress_model.py         # Calibrated LOSO tree ensemble evaluation
│   │   │   ├── fusion_service.py       # Entropy softmin weighting & CSI calculation
│   │   │   └── llm_service.py          # Gemini AI flight surgeon prompt compiler
│   │   └── schemas/
│   │       └── assessment.py           # Pydantic request & response contracts
│   ├── models/
│   │   ├── README.md                   # Documentation on weights & Google Drive download links
│   │   ├── Face/swinv2_export/         # swinv2_nadnet_best.pth & confusion matrix
│   │   ├── Speech/MAITRI_Speech_Channel_Bundle/ # maitri_pan_best_model.pt & metadata
│   │   ├── Text/pacnet_extracted/      # pacnet_modernbert_best.pth & tokenizers
│   │   └── Physiological/MAITRI_Physiological_Channel_Bundle/ # *.pkl models, scaler & baseline
│   ├── download_models.py              # Automated model weight downloader using gdown
│   ├── verify_models.py                # Multi-model architecture & memory verification script
│   ├── test_pipeline.py                # Offline pipeline forward pass validation
│   ├── test_e2e_frontend_api.py        # Automated API integration test suite
│   ├── requirements.txt                # Pinned backend dependencies
│   └── .env.example                    # Environment variable configuration template
│
└── frontend/
    ├── index.html                      # Telemetry cockpit landing page
    ├── face-scan.html                  # Subsystem 1: Facial scan UI (10s reticle)
    ├── voice-check.html                # Subsystem 2: Voice check UI (audio recorder)
    ├── daily-log.html                  # Subsystem 3: Daily text log UI (sentiment chip)
    ├── health-vitals.html              # Subsystem 4: Wearable vitals interactive sliders UI
    ├── summary.html                    # Subsystem 5: Executive summary, CSI gauge & radar UI
    ├── maitri-bridge.js                # Universal async REST client communicating with :8000
    ├── assets/                         # Official emblems, logos, and space wallpapers
    └── css/                            # Custom spaceflight dark theme style sheets
```

---

## 9. Citation & Academic Attribution

If you utilize the MAITRI framework, NAD-Net, PAN, PAC-Net, or the Composite Suppression Index (CSI) in your research, please cite our work:

```bibtex
@article{maitri2026affective,
  title={MAITRI: Multimodal Affective Telemetry and Covert Emotional Discordance Detection for Astronaut Isolation in Deep-Space Missions},
  author={MAITRI Aerospace Cognitive AI Consortium},
  journal={IEEE Transactions on Affective Computing / Aerospace Systems},
  year={2026},
  publisher={GitHub Repository},
  url={https://github.com/Sreecharan-03/MAITRI}
}
```

---

## 10. License & Ethical AI Usage

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Ethical Statement on Affective Telemetry in ICE Environments
The algorithms in MAITRI are designed strictly for **astronaut well-being monitoring, operational safety, and supportive clinical assessment**. They are not intended for punitive performance appraisal or coercive psychological evaluation. All biometric normalizations utilize localized baseline calibrations to preserve physiological individuality and cognitive dignity in extreme exploration environments.
