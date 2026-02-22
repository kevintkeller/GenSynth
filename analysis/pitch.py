import librosa
import numpy as np

"""

What this determines:
-Stable pitch = synth lead/pad
-Unstable pitch = FX/noise
-Low pitch = bass

"""
def estimate_pitch(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        frame_length=2048,
        hop_length=512
    )

    valid_f0 = f0[~np.isnan(f0)]

    if len(valid_f0) == 0:
        return {
            "mean_pitch": 0,
            "pitch_std": 0,
            "pitch_availability": 0
        }
    mean_pitch = np.mean(valid_f0)
    pitch_std = np.std(valid_f0)
    pitch_stability = 1 - (pitch_std / mean_pitch)

    return {
        "mean_pitch": float(mean_pitch),
        "pitch_std": float(pitch_std),
        "pitch_stability": float(pitch_stability)
    }