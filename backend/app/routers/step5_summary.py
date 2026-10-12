from fastapi import APIRouter, Header, HTTPException
from typing import Optional
from app.schemas.assessment import AstronautSummaryRequest, AstronautSummaryResponse
from app.services.llm_service import generate_astronaut_report

router = APIRouter(prefix="/api/v1/step5", tags=["Step 5: Final Astronaut Summary & LLM Evaluation"])

@router.post("/summary", response_model=AstronautSummaryResponse, summary="Generate Multimodal Flight-Surgeon Report with Gemini LLM")
async def analyze_summary(
    payload: AstronautSummaryRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
):
    """
    Ingests multimodal outputs from Face, Voice, Text, and Wearable stress telemetry.
    Sends prompt to Gemini LLM to produce clinical flight surgeon assessment,
    discordance detection, and countermeasure recommendations.
    """
    key = x_api_key or payload.custom_api_key
    try:
        data_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        report = await generate_astronaut_report(data_dict, api_key=key)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation error: {str(e)}")
