import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.routers import (
    step1_face,
    step2_voice,
    step3_text,
    step4_stress,
    step5_summary
)
from app.config import settings

app = FastAPI(
    title="MAITRI Astronaut Wellbeing API",
    version=settings.VERSION,
    description="Multimodal pipeline for astronaut emotion, stress, and discordance detection (Face, Voice, Text, Vitals & Gemini LLM)"
)

# Enable CORS for frontend integration (Vite / React / Localhost / Static Web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register step-by-step routes
app.include_router(step1_face.router)
app.include_router(step2_voice.router)
app.include_router(step3_text.router)
app.include_router(step4_stress.router)
app.include_router(step5_summary.router)

# Check if frontend folder exists to mount static demo if needed
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/dashboard", StaticFiles(directory=frontend_dir, html=True), name="frontend")

@app.get("/")
def root():
    return {
        "system": "MAITRI Multimodal Astronaut Health & Emotion Assessment System",
        "status": "ONLINE",
        "docs_url": "/docs",
        "dashboard_url": "/dashboard/",
        "modules": {
            "step1_face": "/api/v1/step1/face (Real DAN RAF-DB Model with 10s temporal focus)",
            "step2_voice": "/api/v1/step2/voice (Wav2Vec2 Stub)",
            "step3_text": "/api/v1/step3/text (GoEmotions Stub)",
            "step4_stress": "/api/v1/step4/stress (WESAD Telemetry Stub)",
            "step5_summary": "/api/v1/step5/summary (Gemini Flight Surgeon LLM Engine)"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "MAITRI Backend Online",
        "pipeline_ready": True,
        "models": {
            "face_dan_loaded": True,
            "gemini_api_configured": bool(settings.GEMINI_API_KEY)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
