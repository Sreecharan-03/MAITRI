# D:\maitri\backend\verify_models.py
import os
import torch
import joblib
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Running Verification on Device: {device} ---")

# -------------------------------------------------------------------------
# 1. Verify Physiological Stress Estimators (XGBoost / Random Forest)
# -------------------------------------------------------------------------
print("\n[1/4] Testing Physiological Models...")
physio_loso_path = "models/Physiological/MAITRI_Physiological_Channel_Bundle/maitri_calibrated_stress_estimator_loso.pkl"
physio_ref_path = "models/Physiological/MAITRI_Physiological_Channel_Bundle/maitri_baseline_reference.pkl"

if os.path.exists(physio_loso_path) and os.path.exists(physio_ref_path):
    physio_model = joblib.load(physio_loso_path)
    baseline_ref = joblib.load(physio_ref_path)
    
    # Test on dummy 13-dimensional normalized vector
    dummy_input = np.zeros((1, 13))
    p_stress = physio_model.predict_proba(dummy_input)[0, 1]
    print(f" -> Physiological Model Loaded! Test P(stress): {p_stress:.4f}")
else:
    print(f" -> ERROR: Physiological artifacts missing at {physio_loso_path}")

# -------------------------------------------------------------------------
# 2. Verify Speech Channel Model (PAN - Wav2Vec 2.0)
# -------------------------------------------------------------------------
print("\n[2/4] Testing Speech Channel (PAN)...")
speech_model_path = "models/Speech/MAITRI_Speech_Channel_Bundle/maitri_pan_best_model.pt"

if os.path.exists(speech_model_path):
    speech_checkpoint = torch.load(speech_model_path, map_location=device)
    print(f" -> Speech Checkpoint Loaded! Epoch: {speech_checkpoint.get('epoch', 'N/A')}, Acc: {speech_checkpoint.get('best_accuracy', 0)*100:.2f}%")
else:
    print(f" -> ERROR: Speech model missing at {speech_model_path}")

# -------------------------------------------------------------------------
# 3. Verify Facial Channel Model (NAD-Net - SwinV2)
# -------------------------------------------------------------------------
print("\n[3/4] Testing Facial Channel (NAD-Net)...")
face_model_path = "models/Face/swinv2_export/swinv2_nadnet_best.pth"

if os.path.exists(face_model_path):
    face_checkpoint = torch.load(face_model_path, map_location=device)
    keys = list(face_checkpoint.keys()) if isinstance(face_checkpoint, dict) else "raw model weights"
    print(f" -> Facial Checkpoint Loaded! State dict keys sample count: {len(keys)}")
else:
    print(f" -> ERROR: Facial model missing at {face_model_path}")

# -------------------------------------------------------------------------
# 4. Verify Text Channel Model (PAC-Net - ModernBERT)
# -------------------------------------------------------------------------
print("\n[4/4] Testing Text Channel (PAC-Net)...")
text_model_path = "models/Text/pacnet_extracted/pacnet_modernbert_best.pth"

if os.path.exists(text_model_path):
    text_checkpoint = torch.load(text_model_path, map_location=device)
    sd = text_checkpoint.get("model_state_dict", text_checkpoint) if isinstance(text_checkpoint, dict) else text_checkpoint
    keys = list(sd.keys()) if isinstance(sd, dict) else "raw model weights"
    print(f" -> Text Checkpoint Loaded! State dict keys sample count: {len(keys)}")
else:
    print(f" -> ERROR: Text model missing at {text_model_path}")

print("\n" + "=" * 55)
print("Model check routine complete.")
print("=" * 55)
