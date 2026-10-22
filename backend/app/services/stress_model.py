import os
import joblib
import numpy as np
from typing import Dict, Any, List, Optional
from app.config import settings

# Paths to trained physiological artifacts
BUNDLE_DIR = settings.BASE_DIR / "models" / "Physiological" / "MAITRI_Physiological_Channel_Bundle"
LOSO_MODEL_PATH = BUNDLE_DIR / "maitri_calibrated_stress_estimator_loso.pkl"
BASELINE_REF_PATH = BUNDLE_DIR / "maitri_baseline_reference.pkl"

feature_names = [
    'hr', 'sdnn', 'rmssd',
    'eda_mean', 'eda_std', 'eda_dyn_range', 'eda_scr_count',
    'resp_mean', 'resp_std', 'resp_rate',
    'temp_mean', 'temp_std', 'temp_slope'
]

# Lazy-loaded model cache
_physio_model = None
_baseline_ref = None

def get_physio_artifacts():
    global _physio_model, _baseline_ref
    if _physio_model is None and os.path.exists(str(LOSO_MODEL_PATH)):
        try:
            _physio_model = joblib.load(str(LOSO_MODEL_PATH))
            if os.path.exists(str(BASELINE_REF_PATH)):
                _baseline_ref = joblib.load(str(BASELINE_REF_PATH))
        except Exception as e:
            print(f"[StressModel] Warning: Error loading physiological model: {e}")
    return _physio_model, _baseline_ref

async def predict_wearable_stress(
    hr: Optional[float] = None,
    spo2: Optional[float] = None,
    temp: Optional[float] = None,
    accel: Optional[float] = None,
    features_13d: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Step 4: Wearable Biosensor Stress Model Service (Calibrated LOSO Estimator on WESAD Telemetry).
    Processes raw telemetry or 13-dimensional extracted biometric vector.
    """
    model, baseline_ref = get_physio_artifacts()

    # Case 1: 13-feature biometric vector provided
    if features_13d is not None and len(features_13d) == 13 and model is not None:
        raw_vec = np.array(features_13d, dtype=float)
        norm_vec = np.zeros_like(raw_vec)
        if baseline_ref is not None:
            for i, col in enumerate(feature_names):
                mean_val = baseline_ref.get(col, {}).get('mean', 0.0)
                std_val = baseline_ref.get(col, {}).get('std', 1.0)
                norm_vec[i] = (raw_vec[i] - mean_val) / (std_val if std_val != 0 else 1.0)
        else:
            norm_vec = raw_vec

        p_stress = float(model.predict_proba(norm_vec.reshape(1, -1))[0, 1])
        stress_score = round(p_stress, 4)
        is_stressed = p_stress >= 0.50
        level = "High Stress / Autonomic Strain" if p_stress > 0.65 else (
            "Mild Stress / Workload Elevation" if p_stress > 0.35 else "Baseline / Relaxed Homeostasis"
        )
        return {
            "stress_level": level,
            "stress_score": stress_score,
            "discordance_flag": is_stressed,
            "details": {
                "p_stress_calibrated": stress_score,
                "feature_count": 13,
                "hr_bpm": float(raw_vec[0]),
                "skin_temp_celsius": float(raw_vec[10]),
                "feature_set": "WESAD-13-LOSO-Calibrated"
            }
        }

    # Case 2: Individual vitals provided
    hr_val = hr if hr is not None else 78.0
    spo2_val = spo2 if spo2 is not None else 97.0
    temp_val = temp if temp is not None else 36.5
    accel_val = accel if accel is not None else 1.02

    is_motion_high = accel_val > 1.8
    hr_elevated = hr_val > 85.0
    spo2_compromised = spo2_val < 95.0
    temp_deviant = temp_val > 37.5 or temp_val < 35.5

    base_score = 0.20
    if hr_elevated and not is_motion_high:
        base_score += 0.45
    if spo2_compromised:
        base_score += 0.20
    if temp_deviant:
        base_score += 0.15

    stress_score = round(min(max(base_score, 0.05), 0.98), 2)
    discordance = stress_score > 0.50
    level = "High Stress / Autonomic Strain" if stress_score > 0.65 else (
        "Mild Stress / Workload Elevation" if stress_score > 0.40 else "Baseline / Relaxed Homeostasis"
    )

    return {
        "stress_level": level,
        "stress_score": stress_score,
        "discordance_flag": discordance,
        "details": {
            "hr_bpm": hr_val,
            "spo2_percent": spo2_val,
            "skin_temp_celsius": temp_val,
            "accel_magnitude_g": accel_val,
            "feature_set": "WESAD-Empatica-E4-Resampled"
        }
    }
