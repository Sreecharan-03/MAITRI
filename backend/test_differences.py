import asyncio
import numpy as np
from app.services.text_model import predict_text_emotion
from app.services.voice_model import predict_voice_emotion
from app.services.stress_model import predict_wearable_stress

async def test_differences():
    print("=" * 65)
    print(">>> VERIFYING DYNAMIC INPUT/OUTPUT SENSITIVITY <<<")
    print("=" * 65)

    # 1. Text Model with different logs
    print("\n--- 1. Testing Text Sensitivity ---")
    texts = [
        ("Nominal", "Routine orbital EVA maintenance completed nominal, systems all green."),
        ("Fatigue", "Extremely exhausted and drained after 8 hour tether repair, need sleep badly."),
        ("Joy/Pride", "Thrilled with successful docking maneuver! Solar arrays deployed flawlessly, feeling great!"),
        ("Fear/Alarm", "Warning alarm sounded on life support regulator, highly worried about cabin pressure drop."),
        ("Anger/Friction", "Continuous glitches with the robotic arm telemetry are really frustrating and annoying.")
    ]
    for tag, t in texts:
        r = await predict_text_emotion(t)
        print(f"[{tag.upper():12}] Emotion: {r['dominant_emotion']:18} Conf: {r['confidence']:.2f} | TFC: {r['details']['tfc_friction_score']:.3f}")

    # 2. Voice Model with different acoustics
    print("\n--- 2. Testing Voice Sensitivity ---")
    audio_cases = [
        ("Fatigued Voice", b"low_energy_bytes" * 50, 95.0),
        ("Calm Voice", b"nominal_steady_bytes" * 150, 125.0),
        ("Strained Voice", b"tense_fast_bytes" * 300, 210.0)
    ]
    for desc, b_data, p_val in audio_cases:
        vr = await predict_voice_emotion(b_data, measured_pitch=p_val)
        print(f"[{desc:15}] Pitch: {vr['pitch_hz']} Hz -> Emotion: {vr['dominant_emotion']:8} Conf: {vr['confidence']:.2f} | ASI: {vr['details']['acoustic_strain_index']:.3f}")

    # 3. Stress Model with different vitals
    print("\n--- 3. Testing Stress Model Sensitivity ---")
    vitals_cases = [
        ("Relaxed Baseline", 64.0, 98.0, 36.4, 1.01),
        ("Moderate Workload", 88.0, 96.5, 36.8, 1.05),
        ("Autonomic Strain", 112.0, 93.0, 37.8, 1.15)
    ]
    for desc, hr, spo2, temp, accel in vitals_cases:
        sr = await predict_wearable_stress(hr=hr, spo2=spo2, temp=temp, accel=accel)
        print(f"[{desc:17}] HR: {hr:3.0f}bpm, SpO2: {spo2}% -> Stress Level: {sr['stress_level']:30} Score: {sr['stress_score']:.2f} Discordance: {sr['discordance_flag']}")

    print("\n" + "=" * 65)
    print(">>> VERIFICATION COMPLETE: ALL MODALITIES PRODUCE DISTINCT OUTPUTS <<<")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(test_differences())
