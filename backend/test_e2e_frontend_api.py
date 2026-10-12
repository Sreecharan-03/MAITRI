import urllib.request
import json
import numpy as np
import cv2

def run_tests():
    print("=" * 60)
    print(">>> MAITRI FULL FRONTEND-BACKEND INTEGRATION TEST <<<")
    print("=" * 60)

    # 1. Health check
    print("\n[1/6] Checking Backend Health (/health)...")
    with urllib.request.urlopen("http://localhost:8000/health") as r:
        health = json.loads(r.read().decode())
        print("  Status:", health)
        assert health["status"] == "MAITRI Backend Online"
        assert health["models"]["face_dan_loaded"] is True

    # 2. Step 1: Face Emotion Prediction
    print("\n[2/6] Testing Step 1 Face Model Prediction (/api/v1/step1/face)...")
    img = np.full((300, 300, 3), 160, dtype=np.uint8)
    cv2.circle(img, (150, 150), 90, (220, 200, 180), -1)
    cv2.circle(img, (120, 130), 12, (50, 50, 50), -1)
    cv2.circle(img, (180, 130), 12, (50, 50, 50), -1)
    cv2.ellipse(img, (150, 190), (35, 18), 0, 0, 180, (50, 50, 50), 4)
    _, enc = cv2.imencode(".jpg", img)
    face_bytes = enc.tobytes()

    boundary = "WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="astronaut_scan.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + face_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req1 = urllib.request.Request(
        "http://localhost:8000/api/v1/step1/face",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req1) as r:
        step1_res = json.loads(r.read().decode())
        print("  Emotion:", step1_res["dominant_emotion"])
        print("  Confidence:", step1_res["confidence"])
        print("  Scores:", step1_res["scores"])

    # 3. Step 2: Voice Acoustic Prediction
    print("\n[3/6] Testing Step 2 Voice Model Prediction (/api/v1/step2/voice)...")
    body2 = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="voice.webm"\r\n'
        f"Content-Type: audio/webm\r\n\r\n"
    ).encode("utf-8") + b"dummy_audio_bytes" + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req2 = urllib.request.Request(
        "http://localhost:8000/api/v1/step2/voice",
        data=body2,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req2) as r:
        step2_res = json.loads(r.read().decode())
        print("  Tone:", step2_res["dominant_emotion"])
        print("  Pitch:", step2_res["pitch_hz"], "Hz")

    # 4. Step 3: Text Sentiment Prediction
    print("\n[4/6] Testing Step 3 Text NLP Prediction (/api/v1/step3/text)...")
    req3 = urllib.request.Request(
        "http://localhost:8000/api/v1/step3/text",
        data=json.dumps({"text": "Completed solar array EVA. Slight shoulder fatigue, but mental clarity is solid."}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req3) as r:
        step3_res = json.loads(r.read().decode())
        print("  Dominant Sentiment:", step3_res["dominant_emotion"])
        print("  Confidence:", step3_res["confidence"])

    # 5. Step 4: Wearable Biosensor Stress Prediction
    print("\n[5/6] Testing Step 4 Stress Model Prediction (/api/v1/step4/stress)...")
    req4 = urllib.request.Request(
        "http://localhost:8000/api/v1/step4/stress",
        data=json.dumps({"heart_rate_bpm": 84.0, "spo2_percent": 97.2, "skin_temp_celsius": 36.6, "accelerometer_magnitude": 1.03}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req4) as r:
        step4_res = json.loads(r.read().decode())
        print("  Stress Level:", step4_res["stress_level"])
        print("  Stress Index:", step4_res["stress_score"])

    # 6. Step 5: Gemini LLM Synthesis
    print("\n[6/6] Testing Step 5 Gemini LLM Synthesis (/api/v1/step5/summary)...")
    req5 = urllib.request.Request(
        "http://localhost:8000/api/v1/step5/summary",
        data=json.dumps({
            "astronaut_id": "ASTRO-01",
            "mission_day": 42,
            "face_result": step1_res,
            "voice_result": step2_res,
            "text_result": step3_res,
            "stress_result": step4_res
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req5) as r:
        step5_res = json.loads(r.read().decode())
        print("  Overall Score:", step5_res["overall_wellbeing_score"])
        print("  Status:", step5_res["psychological_status"])
        print("  Recommendations:", len(step5_res["recommended_actions"]), "actions generated")

    print("\n" + "=" * 60)
    print(">>> SUCCESS: ALL FRONTEND-TO-BACKEND API CALLS TESTED AND FUNCTIONAL <<<")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
