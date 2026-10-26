import os
import gdown

# Map your Drive IDs to the exact folders visible in your workspace
DOWNLOAD_MAP = {
    # 1. Face Channel (NAD-Net)
    "models/Face/swinv2_export/swinv2_nadnet_best.pth": "1Y_OtECi2WRBwbiCsg0sAd3-1E1xSNmkc",

    # 2. Speech Channel (PAN)
    "models/Speech/MAITRI_Speech_Channel_Bundle/maitri_pan_best_model.pt": "1xOA1_JBx0GM8wKauj0vvYvhIqCiqynni",

    # 3. Text Channel (PAC-Net)
    "models/Text/pacnet_extracted/pacnet_modernbert_best.pth": "1gZMoxMQDiXs_FGIaXFFeHu4Twgkk9hsD",

    # 4. Physiological LOSO Model
    "models/Physiological/MAITRI_Physiological_Channel_Bundle/maitri_calibrated_stress_estimator_loso.pkl": "1sO3cW7cQUcJ3ZQBJjZS5ErpN80j9vzSO",

    # 5. Physiological Random Forest Model
    "models/Physiological/MAITRI_Physiological_Channel_Bundle/maitri_rf_stress_estimator.pkl": "1fuqE5dcxiaGBSVeSyMnDPuzf-Cb9CjYC"
}

def run_download():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for relative_path, file_id in DOWNLOAD_MAP.items():
        full_path = os.path.join(base_dir, relative_path)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 1024:
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"[FOUND] {full_path} ({size_mb:.2f} MB)")
        else:
            print(f"[DOWNLOADING] {full_path} ...")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            try:
                gdown.download(id=file_id, output=full_path, quiet=False)
            except Exception as e:
                print(f"[DOWNLOAD NOTICE]: Could not download {relative_path}: {e}")

if __name__ == "__main__":
    run_download()
