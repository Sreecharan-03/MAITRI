# MAITRI Multimodal Model Checkpoints

This directory contains the trained weights, configurations, and helper pipelines for the four MAITRI telemetry channels:

## Channels

1. **Facial Channel (NAD-Net / SwinV2)**:
   - Path: `Face/swinv2_export/swinv2_nadnet_best.pth`
   - Architecture: Swin Transformer V2 Tiny with upper/lower facial decoupling and Bidirectional Cross-Attention.
   - Outputs: 7 RAF-DB emotion logits, Facial Suppression Index (FSI).

2. **Speech Channel (PAN / Wav2Vec 2.0)**:
   - Path: `Speech/MAITRI_Speech_Channel_Bundle/maitri_pan_best_model.pt`
   - Architecture: Wav2Vec 2.0 + prosodic acoustic feature branch with cross-attention.
   - Outputs: Vocal emotion probabilities, Acoustic Strain Index (ASI).

3. **Text Channel (PAC-Net / ModernBERT)**:
   - Path: `Text/pacnet_extracted/pacnet_modernbert_best.pth`
   - Tokenizer: `Text/pacnet_extracted/tokenizer.json`
   - Architecture: ModernBERT-large backbone with Gated Orthogonal Projection (PAC-Net).
   - Outputs: 8 Plutchik emotion probabilities, Tension-Flow Conflict (TFC) score.

4. **Physiological Channel (WESAD LOSO Stress Estimator)**:
   - Path: `Physiological/MAITRI_Physiological_Channel_Bundle/maitri_calibrated_stress_estimator_loso.pkl`
   - Reference: `Physiological/MAITRI_Physiological_Channel_Bundle/maitri_baseline_reference.pkl`
   - Architecture: CalibratedClassifierCV on 13-dimensional WESAD biometric vectors.
   - Outputs: Calibrated stress probability $P(\text{stress})$ and physiological strain index (PSI).

---

## Downloading Model Weights

Because weights files exceed GitHub's standard size limit (> 100MB), run the automated download script:

```bash
cd backend
python download_models.py
```

Then verify model integrity:

```bash
python verify_models.py
```
