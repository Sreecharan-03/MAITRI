import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

PLUTCHIK_8 = ['Fear', 'Anger', 'Sadness', 'Joy', 'Trust', 'Anticipation', 'Surprise', 'Disgust']

def shannon_entropy(probs: List[float], eps: float = 1e-9) -> float:
    """Calculate Shannon entropy H(X) = -sum(p * log2(p))."""
    h = 0.0
    for p in probs:
        p_clamped = max(eps, min(1.0, float(p)))
        h -= p_clamped * math.log2(p_clamped)
    return max(0.0, h)

def map_face_to_plutchik(scores: Dict[str, float]) -> Dict[str, float]:
    """Map 7-class RAF-DB SwinV2 face predictions to 8 Plutchik axes."""
    p_map = {
        'Fear': scores.get('Fear', 0.05),
        'Anger': scores.get('Anger', 0.05),
        'Sadness': scores.get('Sadness', 0.05),
        'Joy': scores.get('Happiness', 0.05),
        'Surprise': scores.get('Surprise', 0.05),
        'Disgust': scores.get('Disgust', 0.05),
        'Trust': scores.get('Neutral', 0.10) * 0.5,
        'Anticipation': scores.get('Neutral', 0.10) * 0.5
    }
    total = sum(p_map.values()) or 1.0
    return {k: round(v / total, 4) for k, v in p_map.items()}

def map_voice_to_plutchik(scores: Dict[str, float]) -> Dict[str, float]:
    """Map RAVDESS vocal prosody predictions to 8 Plutchik axes."""
    p_map = {
        'Fear': scores.get('fearful', scores.get('fear', 0.02)),
        'Anger': scores.get('angry', scores.get('anger', 0.02)),
        'Sadness': scores.get('sad', scores.get('sadness', 0.05)),
        'Joy': scores.get('happy', scores.get('joy', 0.05)),
        'Surprise': scores.get('surprised', scores.get('surprise', 0.02)),
        'Disgust': scores.get('disgust', 0.02),
        'Trust': scores.get('calm', 0.70),
        'Anticipation': scores.get('neutral', 0.12)
    }
    total = sum(p_map.values()) or 1.0
    return {k: round(v / total, 4) for k, v in p_map.items()}

def map_text_to_plutchik(scores: Dict[str, float]) -> Dict[str, float]:
    """Map PAC-Net / GoEmotions linguistic sentiment to 8 Plutchik axes."""
    p_map = {
        'Fear': scores.get('fear', 0.03),
        'Anger': scores.get('anger', 0.02),
        'Sadness': scores.get('sadness', 0.05),
        'Joy': scores.get('joy', scores.get('positive_sentiment', 0.10)),
        'Trust': scores.get('trust', scores.get('neutral', 0.10) * 0.5),
        'Anticipation': scores.get('anticipation', scores.get('neutral', 0.10) * 0.5),
        'Surprise': scores.get('surprise', 0.02),
        'Disgust': scores.get('disgust', 0.02)
    }
    total = sum(p_map.values()) or 1.0
    return {k: round(v / total, 4) for k, v in p_map.items()}

def map_stress_to_plutchik(stress_score: float) -> Dict[str, float]:
    """Map continuous physiological stress index (WESAD LOSO) to 8 Plutchik axes."""
    s = max(0.0, min(1.0, float(stress_score)))
    p_map = {
        'Fear': round(0.02 + 0.48 * s, 4),
        'Anger': round(0.02 + 0.26 * s, 4),
        'Sadness': round(0.04 + 0.14 * s, 4),
        'Joy': round(max(0.01, 0.20 * (1.0 - s)), 4),
        'Trust': round(max(0.02, 0.45 * (1.0 - s)), 4),
        'Anticipation': round(max(0.02, 0.25 * (1.0 - s)), 4),
        'Surprise': round(0.02 + 0.08 * s, 4),
        'Disgust': round(0.02 + 0.06 * s, 4)
    }
    total = sum(p_map.values()) or 1.0
    return {k: round(v / total, 4) for k, v in p_map.items()}

def fuse_multimodal_streams(
    face_res: Dict[str, Any],
    voice_res: Dict[str, Any],
    text_res: Dict[str, Any],
    stress_res: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Entropy-Gated Multimodal Fusion Engine & Composite Covert Stress Index (CSI).
    Combines:
      - Visual Affect Stream (SwinV2 NAD-Net + FSI)
      - Acoustic Prosody Stream (PAN Wav2Vec2 + ASI)
      - Linguistic Context Stream (PAC-Net ModernBERT + TFC)
      - Physiological Biometrics Stream (WESAD LOSO Stress Estimator + PSI)
    """
    # 1. Align unimodal probability vectors to 8 Plutchik classes
    face_scores = face_res.get('scores', {})
    voice_scores = voice_res.get('scores', {})
    text_scores = text_res.get('detected_8_classes', text_res.get('scores', {}))
    stress_score = float(stress_res.get('stress_score', 0.20))

    p_face = map_face_to_plutchik(face_scores)
    p_voice = map_voice_to_plutchik(voice_scores)
    p_text = map_text_to_plutchik(text_scores)
    p_physio = map_stress_to_plutchik(stress_score)

    modalities = {
        'face': [p_face[c] for c in PLUTCHIK_8],
        'speech': [p_voice[c] for c in PLUTCHIK_8],
        'text': [p_text[c] for c in PLUTCHIK_8],
        'physio': [p_physio[c] for c in PLUTCHIK_8]
    }

    # 2. Compute Inverse-Predictive Entropy Weights (Certainty Gating)
    # H_max for 8 uniform classes is log2(8) = 3.0 bits
    h_max = math.log2(len(PLUTCHIK_8))
    certainty_raw = {}
    for m_name, probs in modalities.items():
        h = shannon_entropy(probs)
        # Normalized certainty C = max(0.02, 1 - H/H_max)
        c = max(0.02, 1.0 - (h / h_max))
        certainty_raw[m_name] = c

    total_certainty = sum(certainty_raw.values()) or 1.0
    weights = {m_name: round(c / total_certainty, 4) for m_name, c in certainty_raw.items()}

    # 3. Dynamic Late Fusion of 8-class distributions
    fused_dist = {c: 0.0 for c in PLUTCHIK_8}
    for m_name, probs in modalities.items():
        w = weights[m_name]
        for idx, c in enumerate(PLUTCHIK_8):
            fused_dist[c] += w * probs[idx]

    # Normalize fused probabilities
    dist_sum = sum(fused_dist.values()) or 1.0
    fused_dist = {c: round(v / dist_sum, 4) for c, v in fused_dist.items()}
    pred_emotion = max(fused_dist, key=fused_dist.get)
    pred_confidence = float(fused_dist[pred_emotion])

    # 4. Composite Covert Stress Index (CSI) Calculation
    # Extract sub-modality strain metrics:
    fsi = float(face_res.get('details', {}).get('fsi_suppression_index', 0.12) if isinstance(face_res.get('details'), dict) else 0.12)
    tfc = float(text_res.get('details', {}).get('tfc_friction_score', 0.18) if isinstance(text_res.get('details'), dict) else 0.18)
    asi = float(voice_res.get('details', {}).get('acoustic_strain_index', 0.14) if isinstance(voice_res.get('details'), dict) else 0.14)
    
    # PSI: Physiological Stress Index (0 to 100)
    p_stress = float(stress_res.get('stress_score', 0.25))
    psi = p_stress * 100.0

    # CSI formula from MAITRI paper: CSI = 0.25*TFC + 0.35*FSI + 0.15*ASI + 0.25*(PSI/100)
    csi_raw = (0.25 * tfc) + (0.35 * fsi) + (0.15 * asi) + (0.25 * (psi / 100.0))
    csi = round(min(1.0, max(0.0, csi_raw)), 4)

    # 5. Discordance Alert (Covert Strain Detection)
    # Triggers when overt channels (calm voice / smiling face / neutral text) mask high physiological/facial strain
    overt_calm = (
        voice_res.get('dominant_emotion', '').lower() in ['calm', 'neutral', 'happy'] or
        face_res.get('dominant_emotion', '').lower() in ['happiness', 'neutral', 'joy'] or
        text_res.get('dominant_emotion', '').lower() in ['reflective', 'neutral', 'joy', 'trust', 'positive_sentiment']
    )
    covert_strain = (psi > 45.0) or (csi > 0.38) or (fsi > 0.28)
    discordance_alert = bool(overt_calm and covert_strain)

    return {
        "predicted_emotion": pred_emotion,
        "confidence": pred_confidence,
        "fused_emotion_distribution": fused_dist,
        "modality_certainty_weights": weights,
        "composite_suppression_index": csi,
        "discordance_alert": discordance_alert,
        "component_indices": {
            "fsi_facial_suppression": round(fsi, 4),
            "tfc_textual_friction": round(tfc, 4),
            "asi_acoustic_strain": round(asi, 4),
            "psi_physiological_stress": round(psi, 2)
        }
    }
