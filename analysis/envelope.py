import numpy as np
import librosa

"""

What this determines:
-Fast attack = pluck
-Long attack = pad
-High sustain = pad/lead
-Short duration = stab/pluck

"""
def estimate_envelope(y, sr):
    # RMS energy
    rms = librosa.feature.rms(y=y)[0]

    times = librosa.times_like(rms, sr=sr)

    peak_index = np.argmax(rms)
    peak_time = times[peak_index]

    total_duration = len(y) / sr

    attack_time = peak_time

    # Decay estimation
    post_peak = rms[peak_index:]
    if len(post_peak) > 0:
        decay_index = np.argmin(np.abs(post_peak - (0.7 * np.max(rms))))
        decay_time = times[peak_index + decay_index] - peak_time
    else:
        decay_time = 0

    sustain_level = np.mean(rms[int(len(rms)*0.5):])

    return {
        "attack_time": float(attack_time),
        "decay_time": float(decay_time),
        "sustain_level": float(sustain_level),
        "duration": float(total_duration)
    }