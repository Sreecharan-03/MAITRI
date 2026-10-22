import joblib
import numpy as np

# Load trained model and baseline references
model = joblib.load("maitri_calibrated_stress_estimator_loso.pkl")
baseline_ref = joblib.load("maitri_baseline_reference.pkl")

feature_names = [
    'hr', 'sdnn', 'rmssd',
    'eda_mean', 'eda_std', 'eda_dyn_range', 'eda_scr_count',
    'resp_mean', 'resp_std', 'resp_rate',
    'temp_mean', 'temp_std', 'temp_slope'
]

def predict_stress(features_12d, threshold=0.50):
    """
    Takes 12 raw physiological window features and returns:
    - p_stress: Calibrated stress probability [0, 1]
    - is_stressed: True if stress detected
    """
    raw_vec = np.array(features_12d, dtype=float)
    
    # Standardize using baseline references
    norm_vec = np.zeros_like(raw_vec)
    for i, col in enumerate(feature_names):
        mean_val = baseline_ref[col]['mean']
        std_val = baseline_ref[col]['std'] if baseline_ref[col]['std'] != 0 else 1.0
        norm_vec[i] = (raw_vec[i] - mean_val) / std_val
        
    p_stress = float(model.predict_proba(norm_vec.reshape(1, -1))[0, 1])
    return p_stress, p_stress >= threshold

if __name__ == "__main__":
    print("MAITRI Physiological Estimator ready.")
