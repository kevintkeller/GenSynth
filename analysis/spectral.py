import librosa
import numpy as np

"""

What this determines:
-High centroid = bright -> Saw wave
-High ZCR = noisy -> Sound oscillator
-Low centroid = sine/triangle wave

"""
def extract_spectral_features(y, sr):
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing = librosa.feature.zero_crossing_rate(y)[0]

    return {
        "brightness": float(np.mean(spectral_centroid)),
        "rolloff": float(np.mean(spectral_rolloff)),
        "noisiness": float(np.mean(zero_crossing))
    }