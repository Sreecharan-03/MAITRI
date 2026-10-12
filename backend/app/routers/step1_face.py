from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from app.schemas.assessment import FaceResponse
from app.services.face_model import predict_face_emotion, predict_face_video_or_frames

router = APIRouter(prefix="/api/v1/step1", tags=["Step 1: Face Emotion"])

@router.post("/face", response_model=FaceResponse, summary="Analyze Face Emotion from Image or 10-Second Video Clip")
async def analyze_face(
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Accepts an astronaut face image or a 10-second video recording.
    Detects face ROI, focuses on facial expressions, and runs DAN neural network inference.
    """
    upload = image or file
    if upload is None:
        raise HTTPException(status_code=400, detail="No file or image provided in request.")

    contents = await upload.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = await predict_face_emotion(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face analysis error: {str(e)}")

@router.post("/face/10s-focus", response_model=FaceResponse, summary="Process 10-Second Focused Astronaut Camera Capture")
async def analyze_face_10s_focus(
    video: UploadFile = File(..., description="10-second webm/mp4 video clip captured from astronaut camera")
):
    """
    Specifically processes a 10-second focused astronaut recording session.
    Extracts multi-frame temporal slices, tracks facial bounding boxes, 
    and outputs aggregated emotional stability and dominant state.
    """
    contents = await video.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Video payload is empty.")

    try:
        result = await predict_face_video_or_frames(contents, duration_sec=10.0)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"10-Second focus video analysis error: {str(e)}")
