# analysis/feature_vector.py

import numpy as np


def _normalize(value, min_val, max_val):
    return float(np.clip((value - min_val) / (max_val - min_val + 1e-9), 0.0, 1.0))


def build_feature_vector(raw_features: dict) -> dict:
    sr = raw_features.get("sample_rate", 44100.0)

    # Brightness using both centroid and rolloff
    brightness_centroid = _normalize(raw_features["spectral_centroid"], 200.0, sr / 2.0)
    brightness_rolloff = _normalize(raw_features["spectral_rolloff"], 400.0, sr / 2.0)
    brightness = 0.6 * brightness_centroid + 0.4 * brightness_rolloff

    # Attack speed: short attack_time -> high attack_speed (for classification/fallback)
    attack_speed = 1.0 - _normalize(raw_features["attack_time"], 0.001, 1.5)

    # Sustain amount from RMS (for classification/fallback)
    sustain_amount = _normalize(raw_features["sustain_energy"], 0.0, 0.5)

    # Measured envelope in seconds and 0-1 level (used for direct Vital mapping)
    attack_sec = float(np.clip(raw_features["attack_time"], 0.001, 4.0))
    decay_sec = float(np.clip(raw_features.get("decay_time", 0.2), 0.01, 4.0))
    sustain_level = float(np.clip(raw_features.get("sustain_ratio", 0.5), 0.0, 1.0))
    release_sec = float(np.clip(raw_features.get("release_time", 0.3), 0.02, 4.0))

    # Movement from onset strength
    movement = _normalize(raw_features["spectral_flux"], 0.0, 5.0)

    # Noisiness from spectral flatness + ZCR
    noisiness_flatness = _normalize(raw_features["spectral_flatness"], 0.0, 0.5)
    noisiness_zcr = _normalize(raw_features["zero_crossing_rate"], 0.0, 0.3)
    noisiness = 0.6 * noisiness_flatness + 0.4 * noisiness_zcr

    # Tonal vs percussive (keep as-is, already 0–1ish)
    tonal_vs_perc = _normalize(raw_features["harmonic_ratio"], 0.0, 1.0)

    return {
        # Core “EQ / spectrum” features
        "brightness": brightness,
        "noisiness": noisiness,
        "tonal_vs_perc": tonal_vs_perc,

        # Envelope / dynamics (classification)
        "attack_speed": attack_speed,
        "sustain_amount": sustain_amount,

        # Measured envelope (seconds and 0-1) for direct Vital mapping
        "attack_sec": attack_sec,
        "decay_sec": decay_sec,
        "sustain_level": sustain_level,
        "release_sec": release_sec,

        # Motion
        "movement": movement,

        # Pitch (kept in Hz for now)
        "fundamental_freq": float(raw_features["fundamental_freq"]),
    }