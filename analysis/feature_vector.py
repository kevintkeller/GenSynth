# analysis/feature_vector.py

import numpy as np


def normalize(value, min_val, max_val):
    return float(np.clip((value - min_val) / (max_val - min_val), 0, 1))


def build_feature_vector(raw_features: dict) -> dict:

    return {
        "brightness": normalize(raw_features["spectral_centroid"], 200, 8000),
        "attack_speed": 1 - normalize(raw_features["attack_time"], 0.001, 1.5),
        "sustain_amount": normalize(raw_features["sustain_energy"], 0.0, 0.5),
        "movement": normalize(raw_features["spectral_flux"], 0.0, 5.0),
        "fundamental_freq": raw_features["fundamental_freq"],
        "harmonic_ratio": normalize(raw_features["harmonic_ratio"], 0.0, 1.0)
    }