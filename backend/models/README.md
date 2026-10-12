# MAITRI Model Repositories

This folder stores model architectures, weights, checkpoints, and preprocessors for each modality in the MAITRI pipeline:

1. **`face_detection/`** (ACTIVE)
   - Real Deep Attention Network (DAN) emotion classification model trained on RAF-DB (7 classes).
   - Checkpoint: `rafdb_epoch21_acc0.897_bacc0.8275.pth`
   - Supports single frame inference as well as 10-second multi-frame temporal focused aggregation.

2. **`voice_analysis/`** (PLANNED)
   - Wav2Vec2 / Whisper emotion & prosody extraction models (RAVDESS / SAVEE datasets).

3. **`text_sentiment/`** (PLANNED)
   - RoBERTa / DistilBERT GoEmotions fine-tuned multi-label classification models.

4. **`stress_detection/`** (PLANNED)
   - XGBoost / Random Forest physiological classifier trained on WESAD multimodal wearable biosignals.
