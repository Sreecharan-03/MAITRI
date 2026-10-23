import os
import json
import re
from typing import Dict, Any, List, Optional
from google import genai
try:
    from app.config import settings
except ImportError:
    settings = None

SYSTEM_INSTRUCTION = """
You are a caring, supportive crew health assistant for space missions.
Your job is to explain the astronaut's health and emotional check-in using simple, clear, everyday language.

Rules for your response:
1. Avoid dense medical jargon (do NOT use words like "autonomic arousal", "compensatory composure", or "psychological homeostasis").
2. Write like an encouraging human doctor speaking directly to the crew member.
3. Be clear, calm, and reassuring.
4. Keep the operational tips simple, practical, and easy to do right now.
"""

def generate_flight_surgeon_report(telemetry_data: dict, api_key: Optional[str] = None) -> dict:
    """
    Generate flight surgeon briefing using Gemini LLM.
    Explains the astronaut's health and emotional check-in using simple, warm, everyday English.
    """
    # Normalize input fields to support both flat telemetry and nested pipeline inputs
    predicted_emotion = (
        telemetry_data.get('predicted_emotion')
        or (telemetry_data.get('face_result') or {}).get('dominant_emotion')
        or (telemetry_data.get('face_data') or {}).get('dominant_emotion')
        or (telemetry_data.get('face') or {}).get('dominant_emotion')
        or 'Neutral'
    )

    unimodal_summaries = telemetry_data.get('unimodal_summaries') or {}
    speech_summary = (
        unimodal_summaries.get('speech')
        or (telemetry_data.get('voice_result') or {}).get('dominant_emotion')
        or (telemetry_data.get('voice_data') or {}).get('dominant_emotion')
        or (telemetry_data.get('voice') or {}).get('dominant_emotion')
        or 'Calm'
    )

    text_summary = (
        unimodal_summaries.get('text')
        or (telemetry_data.get('text_result') or {}).get('dominant_emotion')
        or (telemetry_data.get('text_data') or {}).get('dominant_emotion')
        or (telemetry_data.get('text') or {}).get('dominant_emotion')
        or 'Normal'
    )

    comp_indices = telemetry_data.get('component_indices') or {}
    raw_stress = (
        comp_indices.get('psi_physiological_stress')
        or (telemetry_data.get('stress_result') or {}).get('stress_score')
        or (telemetry_data.get('stress_data') or {}).get('stress_score')
        or (telemetry_data.get('vitals_data') or {}).get('stress_score')
        or 20
    )
    if isinstance(raw_stress, float) and raw_stress <= 1.0:
        stress_level = int(raw_stress * 100)
    else:
        stress_level = int(raw_stress)

    hidden_strain = bool(
        telemetry_data.get('discordance_alert')
        or (telemetry_data.get('stress_result') or {}).get('discordance_flag')
        or (telemetry_data.get('stress_data') or {}).get('discordance_flag')
        or (stress_level > 50 and speech_summary.lower() in ['calm', 'neutral'])
    )

    prompt = f"""
    Here are the astronaut's latest check-in results:
    - Primary Detected Emotion: {predicted_emotion}
    - Outward Demeanor: {speech_summary} voice, {text_summary} notes
    - Physical Vitals Stress Level: {stress_level}%
    - Hidden Strain Detected: {'Yes' if hidden_strain else 'No'}

    Please provide a response in valid JSON matching this exact structure:
    {{
        "overall_score": 7.8,
        "status": "<A 1-2 sentence summary in simple, warm, everyday English explaining how they are doing>",
        "recommendations": [
            "<Practical, simple action step 1>",
            "<Practical, simple action step 2>",
            "<Practical, simple action step 3>"
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

    overall_score = round(max(5.0, min(9.5, 9.8 - (stress_level * 0.04))), 1)
    status = (
        "You look and sound calm, but your body is showing subtle signs of tiredness from today's work. "
        "Overall, you are doing well, but taking a short breather will keep your energy steady."
        if hidden_strain else
        "You are looking great and your check-in signals are steady and balanced today."
    )
    recommendations = [
        "Take a 10-minute quiet break before starting your next scheduled checklist.",
        "Dim the bright overhead cabin lights to help your eyes relax.",
        "Drink a cup of water and do 5 deep, slow breaths."
    ]

    if active_key:
        try:
            client = genai.Client(api_key=active_key)
            from google.genai import types
            # Responsive active Gemini candidate models
            candidate_models = [
                "gemini-3-flash-preview",
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.8-flash",
                "gemini-flash-latest",
                "gemini-2.5-flash"
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
                        print(f"[LLM Service] Successfully generated report using {m_name}")
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
                    overall_score = float(parsed.get("overall_score") or parsed.get("overall_wellbeing_score") or overall_score)
                    status = str(parsed.get("status") or parsed.get("psychological_status") or status)
                    recommendations = parsed.get("recommendations") or parsed.get("recommended_actions") or recommendations
        except Exception as e:
            print(f"[LLM Service Fallback] Gemini API note: {e}")

    return {
        "overall_score": overall_score,
        "status": status,
        "recommendations": recommendations,
        # Cross-compatibility aliases for API schemas and frontend
        "overall_wellbeing_score": overall_score,
        "psychological_status": status,
        "astronaut_specific_assessment": status,
        "detailed_assessment": status,
        "recommended_actions": recommendations,
        "discordance_analysis": (
            "Your verbal logs and expressions are calm and focused, but your vitals show minor physical strain. A brief resting interval is recommended."
            if hidden_strain else
            "All your signals match up nicely. Your facial expression, voice, and vitals show steady composure."
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
