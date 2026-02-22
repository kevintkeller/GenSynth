import numpy as np

def features_to_vector(features):
    return np.array([
        features["mean_pitch"],
        features["pitch_std"],
        features["pitch_stability"],
        features["attack_time"],
        features["decay_time"],
        features["sustain_level"],
        features["duration"],
        features["brightness"],
        features["rolloff"],
        features["noisiness"],
        features["estimated_tempo"],
        features["modulation_strength"],
    ], dtype=np.float32)