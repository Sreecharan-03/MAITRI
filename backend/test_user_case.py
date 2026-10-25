import urllib.request
import json

payload_summary = {
    "astronaut_id": "ASTRO-01",
    "mission_day": 42,
    "stress_result": {
        "stress_level": "High Stress / Autonomic Strain",
        "stress_score": 0.99,
        "discordance_flag": True,
        "details": {
            "hr_bpm": 140,
            "spo2_percent": 97.0,
            "eda_us": 4.5,
            "skin_temp_c": 37.1
        }
    }
}

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/v1/step5/summary",
    data=json.dumps(payload_summary).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req) as resp:
    res_summary = json.loads(resp.read().decode("utf-8"))

print("\n--- STEP 5 SUMMARY RESULT ---")
print("Readiness Score:", res_summary.get("readiness_score"))
print("Verdict:", res_summary.get("simple_verdict"))
print("Status:", res_summary.get("status"))

# Test Multimodal
physio_vec = [140.0, 38.0 * 1.15, 38.0, 4.5, 4.5 * 0.15, 4.5 * 1.6, 10.0, 15.0, 1.2, 0.25, 37.1, 0.2, 0.01]
payload_multimodal = {
    "text_log": "Shift completed, routine station tasks completed nominal.",
    "wearable_features": physio_vec,
    "heart_rate_bpm": 140.0,
    "spo2_percent": 97.0
}

req_mm = urllib.request.Request(
    "http://127.0.0.1:8000/api/v1/predict_multimodal",
    data=json.dumps(payload_multimodal).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req_mm) as resp:
    res_mm = json.loads(resp.read().decode("utf-8"))

print("\n--- PREDICT MULTIMODAL RESULT ---")
print("Predicted Emotion:", res_mm.get("predicted_emotion"))
print("Discordance Alert:", res_mm.get("discordance_alert"))
print("Fused Emotion Dist:", res_mm.get("fused_emotion_distribution"))
print("CSI:", res_mm.get("composite_suppression_index"))
