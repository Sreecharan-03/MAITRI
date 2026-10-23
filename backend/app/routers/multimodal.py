import json
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from app.services.face_model import predict_face_emotion
from app.services.voice_model import predict_voice_emotion
from app.services.text_model import predict_text_emotion
from app.services.stress_model import predict_wearable_stress
from app.services.fusion_service import fuse_multimodal_streams

router = APIRouter(prefix="/api/v1", tags=["Multimodal Emotion & Covert Stress Inference"])

class MultimodalPayload(BaseModel):
    astronaut_id: Optional[str] = Field(default="ASTRO-01", description="Astronaut crew member identifier")
    text: Optional[str] = Field(
        default="Everything is nominal on the station, proceeding with standard routine maintenance.",
        description="Astronaut communication transcript or mission journal log"
    )
    heart_rate_bpm: Optional[float] = Field(default=82.5, description="Telemetry heart rate (BPM)")
    spo2_percent: Optional[float] = Field(default=96.5, description="Telemetry SpO2 percentage")
    skin_temp_celsius: Optional[float] = Field(default=35.8, description="Telemetry peripheral skin temperature (°C)")
    accelerometer_magnitude: Optional[float] = Field(default=1.05, description="Inertial accelerometer magnitude (g)")
    features_13d: Optional[List[float]] = Field(
        default=None,
        description="Optional 13-D normalized/extracted WESAD window"
    )
    face_result: Optional[Dict[str, Any]] = None
    voice_result: Optional[Dict[str, Any]] = None

@router.post("/predict_multimodal", summary="Execute Multimodal Inference & Entropy-Gated Fusion")
async def predict_multimodal(request: Request):
    """
    Core MAITRI multimodal inference endpoint.
    Supports JSON body or multipart/form-data payloads.
    """
    try:
        content_type = request.headers.get("content-type", "")
        text_input = "Everything is nominal on the station, proceeding with standard routine maintenance."
        features_13d = None
        hr = 82.5
        spo2 = 96.5
        temp = 35.8
        face_res = None
        voice_res = None

        if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            form = await request.form()
            text_input = str(form.get("text_log") or form.get("text") or text_input)
            wearable_raw = form.get("wearable_features") or form.get("features_13d")
            if wearable_raw:
                try:
                    if isinstance(wearable_raw, str):
                        parts = [float(x.strip()) for x in wearable_raw.split(",") if x.strip()]
                    else:
                        parts = list(wearable_raw)
                    if len(parts) == 13:
                        features_13d = parts
                        hr = parts[0]
                        temp = parts[10]
                except Exception:
                    pass
        else:
            try:
                body = await request.json()
                text_input = body.get("text") or body.get("text_log") or text_input
                hr = float(body.get("heart_rate_bpm") or hr)
                spo2 = float(body.get("spo2_percent") or spo2)
                temp = float(body.get("skin_temp_celsius") or temp)
                features_13d = body.get("features_13d")
                face_res = body.get("face_result")
                voice_res = body.get("voice_result")
            except Exception:
                pass

        # 1. Text Stream (PAC-Net / ModernBERT)
        text_res = await predict_text_emotion(text_input)

        # 2. Physiological Stream (WESAD 13-D Calibrated LOSO Estimator)
        stress_res = await predict_wearable_stress(
            hr=hr,
            spo2=spo2,
            temp=temp,
            features_13d=features_13d
        )

        # 3. Face Stream (SwinV2 NAD-Net default/result)
        if face_res is None:
            face_res = {
                "dominant_emotion": "Neutral",
                "confidence": 0.35,
                "scores": {"Surprise": 0.15, "Fear": 0.20, "Disgust": 0.10, "Happiness": 0.05, "Sadness": 0.05, "Anger": 0.15, "Neutral": 0.30},
                "details": {"fsi_suppression_index": 0.129}
            }

        # 4. Voice Stream (PAN Wav2Vec2 default/result)
        if voice_res is None:
            voice_res = {
                "dominant_emotion": "calm",
                "confidence": 0.84,
                "pitch_hz": 118.2,
                "scores": {"calm": 0.84, "neutral": 0.08, "sadness": 0.05, "fear": 0.02, "anger": 0.01},
                "details": {"acoustic_strain_index": 0.14}
            }

        # 5. Multimodal Entropy-Gated Fusion & CSI
        fusion = fuse_multimodal_streams(face_res, voice_res, text_res, stress_res)

        # 6. Clinical Flight Surgeon Summary & Operational Protocols
        csi = fusion["composite_suppression_index"]
        discordant = fusion["discordance_alert"]
        overall_score = round(max(3.5, min(9.5, 9.8 - (csi * 6.0))), 1)

        if discordant:
            status = "You look and sound calm, but your body is showing subtle signs of tiredness from today's work. Overall, you are doing well, but taking a short breather will keep your energy steady."
            recommendations = [
                "Take a 10-minute quiet break before starting your next scheduled checklist.",
                "Dim the bright overhead cabin lights to help your eyes relax.",
                "Drink a cup of water and do 5 deep, slow breaths."
            ]
        elif csi > 0.40:
            status = "Your check-in shows moderate tiredness after a demanding shift. A short rest window will help you recharge comfortably."
            recommendations = [
                "Take a 20-minute rest or stretch break in the crew quarters.",
                "Drink some water or warm tea and have a light snack.",
                "Check in with your crewmates or listen to relaxing audio before sleep."
            ]
        else:
            status = "You are doing great today! Your voice, expressions, and vitals are steady and balanced."
            recommendations = [
                "Continue with your normal planned activities and tasks.",
                "Keep staying hydrated throughout the rest of your shift.",
                "Take short routine breaks between task blocks as planned."
            ]

        summary_block = {
            "overall_score": overall_score,
            "status": status,
            "recommendations": recommendations
        }

        return {
            "predicted_emotion": fusion["predicted_emotion"],
            "confidence": fusion["confidence"],
            "fused_emotion_distribution": fusion["fused_emotion_distribution"],
            "modality_certainty_weights": fusion["modality_certainty_weights"],
            "composite_suppression_index": fusion["composite_suppression_index"],
            "discordance_alert": fusion["discordance_alert"],
            "component_indices": fusion["component_indices"],
            "summary": summary_block,
            "unimodal_summaries": {
                "face": face_res.get("dominant_emotion"),
                "speech": voice_res.get("dominant_emotion"),
                "text": text_res.get("dominant_emotion"),
                "stress_level": stress_res.get("stress_level"),
                "stress_score": stress_res.get("stress_score")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal inference error: {str(e)}")
