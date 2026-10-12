from fastapi import APIRouter, HTTPException
from app.schemas.assessment import WearableDataRequest, StressResponse
from app.services.stress_model import predict_wearable_stress

router = APIRouter(prefix="/api/v1/step4", tags=["Step 4: Wearable Stress"])

@router.post("/stress", response_model=StressResponse, summary="Analyze Physiological Telemetry for Autonomic Stress")
async def analyze_stress(payload: WearableDataRequest):
    try:
        hr = payload.get_hr()
        spo2 = payload.get_spo2()
        temp = payload.get_temp()
        accel = payload.get_accel()

        result = await predict_wearable_stress(
            hr=hr,
            spo2=spo2,
            temp=temp,
            accel=accel
        )
        
        # Include custom details if HRV/EDA provided
        if payload.hrv_rmssd or payload.hrvRmssd or payload.mean_eda or payload.meanEda:
            result.setdefault("details", {})
            if payload.hrv_rmssd or payload.hrvRmssd:
                result["details"]["hrv_rmssd"] = payload.hrv_rmssd or payload.hrvRmssd
            if payload.mean_eda or payload.meanEda:
                result["details"]["mean_eda"] = payload.mean_eda or payload.meanEda
            if payload.scr_count or payload.scrCount:
                result["details"]["scr_count"] = payload.scr_count or payload.scrCount
            if payload.respiration_rate or payload.respirationRate:
                result["details"]["respiration_rate"] = payload.respiration_rate or payload.respirationRate

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stress analysis error: {str(e)}")
