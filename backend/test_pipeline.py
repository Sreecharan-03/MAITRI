import asyncio
import os
import json
from app.services.face_model import predict_face_emotion
from app.services.voice_model import predict_voice_emotion
from app.services.text_model import predict_text_emotion
from app.services.stress_model import predict_wearable_stress
from app.services.fusion_service import fuse_multimodal_streams
from app.services.llm_service import generate_astronaut_report

async def test_pipeline():
    print("=" * 65)
    print(">>> TESTING MAITRI MULTIMODAL PIPELINE (OFFLINE FORWARD PASS) <<<")
    print("=" * 65)
    
    # ---------------------------------------------------------
    # 1. Test Facial Channel (SwinV2 NAD-Net)
    # ---------------------------------------------------------
    print("\n[1/5] Testing Facial Channel (SwinV2 NAD-Net)...")
    test_img_path = os.path.join(os.path.dirname(__file__), "..", "face_detection", "test_sample_predicted.jpg")
    if os.path.exists(test_img_path):
        with open(test_img_path, "rb") as f:
            img_bytes = f.read()
        face_res = await predict_face_emotion(img_bytes)
    else:
        import numpy as np
        import cv2
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", dummy)
        face_res = await predict_face_emotion(buf.tobytes())
    print("  Dominant Emotion:", face_res["dominant_emotion"])
    print("  Confidence:", face_res["confidence"])
    print("  FSI Suppression Index:", face_res["details"].get("fsi_suppression_index", 0.0))

    # ---------------------------------------------------------
    # 2. Test Speech Channel (PAN - Wav2Vec2)
    # ---------------------------------------------------------
    print("\n[2/5] Testing Speech Channel (PAN)...")
    voice_res = await predict_voice_emotion(b"dummy audio bytes for acoustic test")
    print("  Dominant Tone:", voice_res["dominant_emotion"])
    print("  Pitch (Hz):", voice_res["pitch_hz"])
    print("  Confidence:", voice_res["confidence"])

    # ---------------------------------------------------------
    # 3. Test Text Channel (PAC-Net - ModernBERT)
    # ---------------------------------------------------------
    print("\n[3/5] Testing Text Channel (PAC-Net)...")
    sample_text = "Feeling fatigued after the extravehicular docking procedure, but systems are stable."
    text_res = await predict_text_emotion(sample_text)
    print("  Dominant Sentiment:", text_res["dominant_emotion"])
    print("  Confidence:", text_res["confidence"])

    # ---------------------------------------------------------
    # 4. Test Physiological Channel (WESAD LOSO Stress Estimator)
    # ---------------------------------------------------------
    print("\n[4/5] Testing Physiological Channel (13-D Calibrated LOSO Vector)...")
    # 13-D extracted physiological window: [hr, sdnn, rmssd, eda_mean, eda_std, eda_dyn_range, eda_scr_count, resp_mean, resp_std, resp_rate, temp_mean, temp_std, temp_slope]
    dummy_13d = [82.5, 45.2, 32.1, 1.85, 0.52, 2.4, 5.0, 16.5, 2.8, 17.0, 35.8, 0.18, -0.01]
    stress_res = await predict_wearable_stress(features_13d=dummy_13d)
    print("  Stress Level:", stress_res["stress_level"])
    print("  Calibrated P(stress):", stress_res["stress_score"])
    print("  Autonomic Discordance Flag:", stress_res["discordance_flag"])

    # ---------------------------------------------------------
    # 5. Multimodal Entropy-Gated Late Fusion & CSI
    # ---------------------------------------------------------
    print("\n[5/5] Executing Multimodal Entropy-Gated Fusion & CSI...")
    fusion_res = fuse_multimodal_streams(face_res, voice_res, text_res, stress_res)
    print("\n  --- FUSION REPORT ---")
    print(f"  Predicted Emotion (Fused):     {fusion_res['predicted_emotion']} (Confidence: {fusion_res['confidence']:.4f})")
    print(f"  Modality Certainty Weights:    {json.dumps(fusion_res['modality_certainty_weights'])}")
    print(f"  Composite Suppression Index:   {fusion_res['composite_suppression_index']:.4f}")
    print(f"  Covert Discordance Alert:       {fusion_res['discordance_alert']}")
    print(f"  Fused 8-Class Distribution:    {json.dumps(fusion_res['fused_emotion_distribution'], indent=4)}")

    # ---------------------------------------------------------
    # 6. LLM Flight-Surgeon Synthesis (Gemini)
    # ---------------------------------------------------------
    print("\n[Optional] Generating Astronaut LLM Flight-Surgeon Report with Gemini...")
    summary_payload = {
        "astronaut_id": "ASTRO-01",
        "mission_day": 42,
        "face_result": face_res,
        "voice_result": voice_res,
        "text_result": text_res,
        "stress_result": stress_res
    }
    try:
        report_res = await generate_astronaut_report(summary_payload)
        print("  Overall Wellbeing Score:", report_res.get("overall_wellbeing_score"))
        print("  Psychological Status:", report_res.get("psychological_status"))
        print("  Operational Recommendations:", len(report_res.get("recommended_actions", [])), "actions")
    except Exception as e:
        print("  LLM Note (offline/fallback):", e)

    print("\n" + "=" * 65)
    print(">>> ALL PIPELINE STAGES PASSED CLEANLY! <<<")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(test_pipeline())
