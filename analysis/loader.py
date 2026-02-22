import librosa
import numpy as np

def load_audio(path, target_sr=44100):
    y, sr = librosa.load(path, sr=target_sr, mono=True)

    # Normalize
    y = librosa.util.normalize(y)

    return y, sr