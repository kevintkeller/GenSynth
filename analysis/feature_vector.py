from .pitch import estimate_pitch
from .envelope import estimate_envelope
from .spectral import extract_spectral_features
from .modulation import detect_modulation

def analyze_audio(y, sr):
    pitch = estimate_pitch(y, sr)
    env = estimate_envelope(y, sr)
    spectral = extract_spectral_features(y, sr)
    modulation = detect_modulation(y, sr)

    features = {**pitch, **env, **spectral, **modulation}

    return features