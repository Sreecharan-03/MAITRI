import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.routers import (
    step1_face,
    step2_voice,
    step3_text,
    step4_stress,
    step5_summary,
    multimodal_router
)
from app.config import settings

logger = logging.getLogger("uvicorn.info")

def background_model_downloader():
    try:
        from download_models import run_download
        run_download()
        import gc
        gc.collect()
        print("INFO:     Background model check complete. Models will lazy-load on demand.")
    except Exception as e:
        print(f"WARN:     Background downloader note: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading
    t = threading.Thread(target=background_model_downloader, daemon=True)
    t.start()
    yield

app = FastAPI(
    title="MAITRI Astronaut Wellbeing API",
    version=settings.VERSION,
    description="Multimodal pipeline for astronaut emotion, stress, and discordance detection (Face, Voice, Text, Vitals & Gemini LLM)",
    lifespan=lifespan
)

# Enable CORS for frontend integration (Vite / React / Localhost / Static Web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prevent browser from serving stale 304 cached HTML/JS
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/dashboard"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Register step-by-step and multimodal routes
app.include_router(step1_face.router)
app.include_router(step2_voice.router)
app.include_router(step3_text.router)
app.include_router(step4_stress.router)
app.include_router(step5_summary.router)
app.include_router(multimodal_router)

# Check if frontend folder exists to mount static demo if needed
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/dashboard", StaticFiles(directory=frontend_dir, html=True), name="frontend")

@app.get("/")
def root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(url="/dashboard/")
    return {
        "system": "MAITRI Multimodal Astronaut Health & Emotion Assessment System",
        "status": "ONLINE",
        "docs_url": "/docs",
        "dashboard_url": "/dashboard/",
        "modules": {
            "predict_multimodal": "/api/v1/predict_multimodal (Entropy-Gated Fusion & CSI)",
            "step1_face": "/api/v1/step1/face (SwinV2 NAD-Net Model with FSI)",
            "step2_voice": "/api/v1/step2/voice (Wav2Vec2 PAN Stream)",
            "step3_text": "/api/v1/step3/text (PAC-Net ModernBERT Stream)",
            "step4_stress": "/api/v1/step4/stress (WESAD 13-D Calibrated LOSO Estimator)",
            "step5_summary": "/api/v1/step5/summary (Gemini Flight Surgeon LLM Engine)"
        }
    }

@app.get("/health")
def health_check():
    from app.services.face_model import get_face_engine
    engine = get_face_engine()
    ckpt = str(settings.FACE_MODEL_CHECKPOINT)
    return {
        "status": "MAITRI Backend Online",
        "pipeline_ready": True,
        "models": {
            "face_nadnet_loaded": engine.is_loaded,
            "face_checkpoint_exists": os.path.exists(ckpt),
            "multimodal_fusion_ready": True,
            "gemini_api_configured": bool(settings.GEMINI_API_KEY)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
