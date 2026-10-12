# MAITRI

Multimodal AI-driven Telemetric and Real-time Interface for astronaut mental health, affective computing, and cognitive wellness monitoring.

## Overview
MAITRI integrates multimodal assessment streams:
- **Facial Emotion Recognition**: Real-time DAN (Deep Attention Network) architecture
- **Voice Emotion & Prosody Analysis**
- **Text Sentiment & Daily Journal Analysis**
- **Physiological Stress Monitoring**
- **AI Mental Health Copilot**: Multimodal fusion and personalized wellness recommendations powered by Google Gemini

## Setup & Running

### 1. Backend
```bash
cd backend
python -m venv venv
# Activate virtual environment:
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env  # Add your GEMINI_API_KEY
uvicorn app.main:app --reload --port 8000
```

> **Note on Model Weights**: Model checkpoint files (`*.pth`) exceed GitHub file size limits and are ignored by git. Place model weights (such as `rafdb_epoch21_acc0.897_bacc0.8275.pth`) in `backend/models/face_detection/`.

### 2. Frontend
Serve or open `frontend/index.html` directly in your browser or with a local static file server.
