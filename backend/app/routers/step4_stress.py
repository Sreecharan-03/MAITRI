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

        rmssd = payload.hrv_rmssd or payload.hrvRmssd
        eda = payload.mean_eda or payload.meanEda

        features_13d = None
        if rmssd is not None or eda is not None:
            hr_val = float(hr if hr is not None else 78.0)
            rmssd_val = float(rmssd if rmssd is not None else 38.0)
            eda_val = float(eda if eda is not None else 1.2)
            temp_val = float(temp if temp is not None else 36.5)
            scr_val = float(payload.scr_count or payload.scrCount or (4.0 if eda_val < 1.0 else 9.0))
            resp_val = float(payload.respiration_rate or payload.respirationRate or 15.0)

            features_13d = [
                hr_val,
                rmssd_val * 1.15,
                rmssd_val,
                eda_val,
                eda_val * 0.15,
                eda_val * 1.6,
                scr_val,
                resp_val,
                1.2,
                0.25,
                temp_val,
                0.2,
                0.01
            ]

        result = await predict_wearable_stress(
            hr=hr,
            spo2=spo2,
            temp=temp,
            accel=accel,
            features_13d=features_13d
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
