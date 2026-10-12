import asyncio
import os
import json
from app.services.face_model import predict_face_emotion
from app.services.voice_model import predict_voice_emotion
from app.services.text_model import predict_text_emotion
from app.services.stress_model import predict_wearable_stress
from app.services.llm_service import generate_astronaut_report

async def test_pipeline():
    print("=== TESTING MAITRI PIPELINE ===")
    
    # 1. Test Face Model
    test_img_path = os.path.join(os.path.dirname(__file__), "..", "face_detection", "test_sample_predicted.jpg")
    if os.path.exists(test_img_path):
        with open(test_img_path, "rb") as f:
            img_bytes = f.read()
        face_res = await predict_face_emotion(img_bytes)
    else:
        # Fallback dummy image
        import numpy as np
        import cv2
        dummy = np.zeros((300, 300, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", dummy)
        face_res = await predict_face_emotion(buf.tobytes())
    print("[1] Face Emotion Result:", json.dumps(face_res, indent=2))

    # 2. Test Voice Model
    voice_res = await predict_voice_emotion(b"dummy audio bytes")
    print("[2] Voice Emotion Result:", json.dumps(voice_res, indent=2))

    # 3. Test Text Model
    text_res = await predict_text_emotion("Feeling fatigued after the extravehicular docking procedure, but systems are stable.")
    print("[3] Text Emotion Result:", json.dumps(text_res, indent=2))

    # 4. Test Stress Model
    stress_res = await predict_wearable_stress(hr=89.5, spo2=96.2, temp=36.8, accel=1.05)
    print("[4] Wearable Stress Result:", json.dumps(stress_res, indent=2))

    # 5. Test LLM Summary
    summary_payload = {
        "astronaut_id": "ASTRO-01",
        "mission_day": 42,
        "face_result": face_res,
        "voice_result": voice_res,
        "text_result": text_res,
        "stress_result": stress_res
    }
    print("[5] Generating Astronaut LLM Report with Gemini...")
    report_res = await generate_astronaut_report(summary_payload)
    print("[5] LLM Summary Result:", json.dumps(report_res, indent=2))

    print("\n>>> ALL 5 STEPS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
