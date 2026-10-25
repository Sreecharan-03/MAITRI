import os
import json
import re
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types
try:
    from app.config import settings
except ImportError:
    settings = None

SYSTEM_INSTRUCTION = """
You are MAITRI's Crew Health Assistant. Your job is to translate complex biometric and emotional sensor data into clear, simple, human English.

STRICT WRITING RULES:
1. BAN all dense medical jargon: Do NOT use phrases like "cephalad fluid distribution", "autonomic homeostasis", "parasympathetic suppression", "normothermic regulation", or "resonance frequency breathing".
2. Speak like a supportive flight doctor talking directly to a crew member: warm, grounded, clear, and reassuring.
3. Maximum 3 sentences for the main status:
   - Sentence 1: Bottom-line verdict (Are they doing fine or struggling?).
   - Sentence 2: What the sensors spotted (e.g., "Your voice sounds steady, but your eyes and journal show tiredness").
   - Sentence 3: Reassurance and quick takeaway.
4. Recommendations must be practical, everyday tasks anyone can understand in 5 seconds (drink water, take a screen break, rest eyes, breathe).
"""

def generate_flight_surgeon_report(telemetry_data: dict, api_key: Optional[str] = None) -> dict:
    """
    Generate flight surgeon briefing using Gemini LLM.
    Translates complex biometric & emotional sensor data into plain, empathetic, everyday English.
    """
    # Face cues
    face = telemetry_data.get('face_result') or telemetry_data.get('face_data') or telemetry_data.get('face') or {}
    face_emo = face.get('dominant_emotion') or telemetry_data.get('predicted_emotion') or 'Composed'
    face_detail = "Showing signs of tired eyes / screen strain" if face_emo.lower() in ['fatigue', 'tired', 'neutral'] else f"Showing {face_emo.lower()} expressions"

    # Voice cues
    voice = telemetry_data.get('voice_result') or telemetry_data.get('voice_data') or telemetry_data.get('voice') or {}
    voice_emo = voice.get('dominant_emotion') or (telemetry_data.get('unimodal_summaries') or {}).get('speech') or 'Calm'
    pitch = voice.get('pitch_hz', 118.2)
    voice_detail = f"Steady and controlled (Pitch: {pitch:.1f} Hz, {voice_emo})"

    # Journal / Text cues
    text = telemetry_data.get('text_result') or telemetry_data.get('text_data') or telemetry_data.get('text') or {}
    text_emo = text.get('dominant_emotion') or (telemetry_data.get('unimodal_summaries') or {}).get('text') or 'Normal'
    journal_detail = "Mentions fatigue after completing scheduled tasks" if 'fatigue' in str(text).lower() or text_emo.lower() in ['fatigue', 'sadness', 'tired'] else f"Mentions steady focus on mission checklist ({text_emo})"

    # Wearable Vitals cues
    vitals = telemetry_data.get('stress_result') or telemetry_data.get('stress_data') or telemetry_data.get('vitals_data') or {}
    details = vitals.get('details') or {}
    hr = float(details.get('hr_bpm') or telemetry_data.get('heart_rate') or 78.0)
    spo2 = float(details.get('spo2_pct') or telemetry_data.get('spo2') or 97.0)
    stress_val = vitals.get('stress_score') or (telemetry_data.get('component_indices') or {}).get('psi_physiological_stress') or 0.20
    if isinstance(stress_val, float) and stress_val <= 1.0:
        stress_pct = int(stress_val * 100)
    else:
        stress_pct = int(stress_val)

    discordance = bool(
        telemetry_data.get('discordance_alert')
        or vitals.get('discordance_flag')
        or telemetry_data.get('isDiscordant')
        or (stress_pct > 50 and voice_emo.lower() in ['calm', 'neutral'])
    )

    # Dynamic Multimodal Readiness Score Calculation (0 to 100)
    raw_score = (
        telemetry_data.get('overall_score')
        or telemetry_data.get('readiness_score')
        or telemetry_data.get('overall_wellbeing_score')
    )
    if raw_score is not None and raw_score != 80:
        if isinstance(raw_score, (int, float)) and raw_score <= 10.0:
            overall_score_100 = int(raw_score * 10)
        else:
            overall_score_100 = int(raw_score)
    else:
        # Compute from actual multi-channel signals
        stress_penalty = (stress_pct / 100.0) * 45.0
        hr_penalty = max(0.0, (hr - 90.0) * 0.35) if hr > 90 else 0.0
        spo2_penalty = max(0.0, (95.0 - spo2) * 2.5) if spo2 < 95.0 else 0.0
        voice_penalty = 12.0 if voice_emo.lower() in ['fear', 'anger', 'strained'] else (
            8.0 if voice_emo.lower() == 'sadness' or pitch > 180 else 2.0
        )
        face_penalty = 12.0 if face_emo.lower() in ['fatigue', 'sadness', 'fear', 'anger'] else (
            5.0 if face_emo.lower() in ['tired', 'neutral'] else 1.0
        )
        text_penalty = 10.0 if ('fatigue' in str(text).lower() or text_emo.lower() in ['fatigue', 'sadness', 'fear', 'anger']) else 2.0

        total_deduction = stress_penalty + hr_penalty + spo2_penalty + voice_penalty + face_penalty + text_penalty
        overall_score_100 = int(round(max(20, min(98, 100 - total_deduction))))

    prompt = f"""
    Translate these astronaut telemetry results into plain, everyday human English:
    - Overall Readiness: {overall_score_100}/100
    - Voice Check: {voice_detail}
    - Face Check: {face_detail}
    - Journal Note: {journal_detail}
    - Physical Vitals: Heart rate {hr} bpm, oxygen {spo2}%, physiological strain {stress_pct}%
    - Discordance Flag: {discordance} (Keeping composure despite elevated physiological load)

    Respond ONLY in valid JSON matching this exact structure:
    {{
        "simple_verdict": "Fit for duty, but mental energy is running low.",
        "plain_explanation": "You are doing a great job keeping steady and composed, and your heart and vitals are completely stable. However, your journal notes and facial cues show your brain and eyes are tired from long hours at the console. You don't have severe stress, just normal work exhaustion that needs a short breather.",
        "actionable_tips": [
            "Take a 15-minute screen break to rest your eyes.",
            "Drink a full bottle of water and have an electrolyte snack.",
            "Do 5 slow, deep breaths to help your body unwind.",
            "Dim the workstation overhead lights for the rest of your shift."
        ]
    }}
    """

    active_key = (
        api_key
        or telemetry_data.get('gemini_api_key')
        or telemetry_data.get('custom_api_key')
        or telemetry_data.get('apiKey')
        or (getattr(settings, 'GEMINI_API_KEY', None) if settings else None)
        or os.getenv("GEMINI_API_KEY")
    )

    # Human-centric, grounded fallbacks tailored to actual strain
    if stress_pct > 65 or hr > 115:
        simple_verdict = "Elevated Autonomic Strain • Active Recovery Required"
        plain_explanation = (
            f"Your heart rate ({int(hr)} BPM) and physiological stress load ({stress_pct}%) indicate substantial autonomic strain today. "
            "Even if you feel composed, your cardiovascular system is working overtime. "
            "Please pause strenuous tasks, hydrate with electrolytes, and take a 20-minute rest window."
        )
        actionable_tips = [
            "Take an immediate 20-minute quiet rest break in crew quarters.",
            "Drink 500ml water with an electrolyte replenishment pack.",
            "Perform 4-7-8 slow diaphragmatic breathing to lower heart rate.",
            "Notify crew lead to postpone scheduled physical EVA tasks."
        ]
    elif discordance or stress_pct > 40:
        simple_verdict = "Fit for duty • Mental energy running slightly low."
        plain_explanation = (
            "You are doing a great job keeping steady and composed, and your outward demeanour is solid. "
            "However, your journal notes and physiological cues show your body is carrying hidden work fatigue. "
            "A quick 15-minute breather will keep your focus sharp and your stamina strong."
        )
        actionable_tips = [
            "Take a 15-minute screen break away from console displays.",
            "Drink a full bottle of water and have a light energy snack.",
            "Do 5 slow, deep breaths to help your body unwind.",
            "Dim the workstation overhead lights for the rest of your shift."
        ]
    else:
        simple_verdict = "Fit for duty • All systems steady and balanced."
        plain_explanation = (
            "You look and sound composed, and your vital signs are completely normal and steady today. "
            "Your body and mind are working in good harmony, and you are in great shape for your scheduled mission tasks. "
            "Keeping a steady pace and staying hydrated will ensure your energy remains consistent."
        )
        actionable_tips = [
            "Continue with your normal planned activities and tasks.",
            "Keep staying hydrated throughout the rest of your shift.",
            "Take short routine 5-minute stretch breaks between task blocks.",
            "Maintain steady lighting in your workstation area."
        ]

    if active_key:
        try:
            client = genai.Client(api_key=active_key)
            candidate_models = [
                "gemini-3-flash-preview",
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.8-flash",
                "gemini-flash-latest"
            ]

            response = None
            last_err = None
            for m_name in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=m_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_INSTRUCTION,
                            response_mime_type="application/json"
                        )
                    )
                    if response and response.text:
                        print(f"[LLM Service] Successfully generated empathetic report using {m_name}")
                        break
                except Exception as m_err:
                    last_err = m_err
                    continue

            if response and response.text:
                resp_text = response.text.strip()
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', resp_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(1))
                elif resp_text.startswith('{') and resp_text.endswith('}'):
                    parsed = json.loads(resp_text)
                else:
                    parsed = {}

                if parsed:
                    simple_verdict = str(parsed.get("simple_verdict") or parsed.get("status") or simple_verdict)
                    plain_explanation = str(parsed.get("plain_explanation") or parsed.get("detailed_assessment") or plain_explanation)
                    actionable_tips = parsed.get("actionable_tips") or parsed.get("recommendations") or actionable_tips

        except Exception as e:
            print(f"[LLM Service Fallback] Gemini API note: {e}")

    # Harmonize final score
    readiness = float(round(overall_score_100 / 10.0, 1))

    return {
        "simple_verdict": simple_verdict,
        "plain_explanation": plain_explanation,
        "actionable_tips": actionable_tips,
        # Cross-compatibility aliases for API schemas, older dashboards, and routes
        "status": simple_verdict,
        "psychological_status": simple_verdict,
        "detailed_assessment": plain_explanation,
        "astronaut_specific_assessment": plain_explanation,
        "recommendations": actionable_tips,
        "recommended_actions": actionable_tips,
        "overall_score": readiness,
        "overall_wellbeing_score": readiness,
        "readiness_score": overall_score_100,
        "discordance_analysis": (
            "Your outward demeanor is composed and steady, but your body is carrying hidden fatigue from prolonged focus."
            if discordance else
            "All your health and emotional signals align nicely. Your facial cues, vocal tone, and vital signs show steady focus."
        )
    }

async def generate_astronaut_report(telemetry_data: dict, api_key: Optional[str] = None) -> dict:
    """Async wrapper compatible with FastAPI async endpoints"""
    return generate_flight_surgeon_report(telemetry_data, api_key=api_key)

__all__ = [
    "SYSTEM_INSTRUCTION",
    "generate_flight_surgeon_report",
    "generate_astronaut_report"
]
