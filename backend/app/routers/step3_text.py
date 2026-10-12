from fastapi import APIRouter, HTTPException
from app.schemas.assessment import TextRequest, TextResponse
from app.services.text_model import predict_text_emotion

router = APIRouter(prefix="/api/v1/step3", tags=["Step 3: Text Emotion"])

@router.post("/text", response_model=TextResponse, summary="Analyze Sentiment and Emotion from Log Text")
async def analyze_text(payload: TextRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")
    try:
        result = await predict_text_emotion(payload.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis error: {str(e)}")
