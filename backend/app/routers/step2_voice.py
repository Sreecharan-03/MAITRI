from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
from app.schemas.assessment import VoiceResponse
from app.services.voice_model import predict_voice_emotion

router = APIRouter(prefix="/api/v1/step2", tags=["Step 2: Voice Emotion"])

@router.post("/voice", response_model=VoiceResponse, summary="Analyze Voice Emotion from Audio Clip")
async def analyze_voice(
    audio: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None)
):
    upload = audio or file
    if upload is None:
        raise HTTPException(status_code=400, detail="No audio file provided in request.")

    contents = await upload.read()
    try:
        result = await predict_voice_emotion(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice analysis error: {str(e)}")
