import torch
import librosa
import numpy as np

def run_speech_pipeline(audio_path, model, device):
    # Standard 16 kHz processing
    wav, _ = librosa.load(audio_path, sr=16000, mono=True)
    wav_trimmed, _ = librosa.effects.trim(wav, top_db=25)
    wav_fixed = np.pad(wav_trimmed, (0, max(0, 48000 - len(wav_trimmed))), mode='constant')[:48000]

    mel_spec = librosa.feature.melspectrogram(y=wav_fixed, sr=16000, n_mels=128, fmax=8000)
    log_mel = np.mean(librosa.power_to_db(mel_spec, ref=np.max), axis=1)
    rms = librosa.feature.rms(y=wav_fixed)
    zcr = librosa.feature.zero_crossing_rate(y=wav_fixed)
    spec_cent = librosa.feature.spectral_centroid(y=wav_fixed, sr=16000)
    
    pros = np.concatenate([log_mel, [np.mean(rms), np.std(rms), np.mean(zcr), np.std(zcr), np.mean(spec_cent), np.std(spec_cent)]])
    
    with torch.no_grad():
        w_t = torch.tensor(wav_fixed, dtype=torch.float32).unsqueeze(0).to(device)
        p_t = torch.tensor(pros, dtype=torch.float32).unsqueeze(0).to(device)
        logits, asi = model(w_t, p_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    return probs, float(asi.cpu().numpy()[0][0])
