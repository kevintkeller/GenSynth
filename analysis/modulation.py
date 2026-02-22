import librosa
import numpy as np

"""

What this determines:
-High modulation strength = LFO likely
-Detect preiodidc amplitude changes

TODO: Later let's FFT the amplitude to detect LFO rate more pricesely

"""
def detect_modulation(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    modulation_strength = np.std(onset_env)

    return {
        "estimated_tempo": float(tempo),
        "modulation_strength": float(modulation_strength)
    }