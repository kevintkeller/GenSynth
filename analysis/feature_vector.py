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

    # For percussive/pluck sounds, cap decay/release and lower sustain floor so envelope is punchy
    if attack_speed > 0.6 and sustain_amount < 0.4:
        decay_sec = min(decay_sec, 0.7)
        release_sec = min(release_sec, 1.2)
        sustain_level = min(sustain_level, 0.22)  # piano/pluck: low sustain floor

    # Harmonic timbre (0-1): odd vs even balance, and richness
    harmonic_odd_ratio = float(np.clip(raw_features.get("harmonic_odd_ratio", 0.5), 0.0, 1.0))
    harmonic_richness = float(np.clip(raw_features.get("harmonic_richness", 0.5), 0.0, 1.0))

    # Movement from onset strength
    movement = _normalize(raw_features["spectral_flux"], 0.0, 5.0)

    # Noisiness from spectral flatness + ZCR
    noisiness_flatness = _normalize(raw_features["spectral_flatness"], 0.0, 0.5)
    noisiness_zcr = _normalize(raw_features["zero_crossing_rate"], 0.0, 0.3)
    noisiness = 0.6 * noisiness_flatness + 0.4 * noisiness_zcr

    # Tonal vs percussive (keep as-is, already 0–1ish)
    tonal_vs_perc = _normalize(raw_features["harmonic_ratio"], 0.0, 1.0)

    # Transient vs body brightness (piano: bright at hit, duller in tail)
    transient_centroid = raw_features.get("transient_centroid", raw_features["spectral_centroid"])
    body_centroid = raw_features.get("body_centroid", raw_features["spectral_centroid"])
    transient_brightness = _normalize(transient_centroid, 200.0, sr / 2.0)
    body_brightness = _normalize(body_centroid, 200.0, sr / 2.0)

    # Sound class from detection: drives which Vital blocks are on/off and timbre
    fundamental_freq = float(raw_features["fundamental_freq"])
    sound_class = _classify_sound(
        attack_speed=attack_speed,
        sustain_amount=sustain_amount,
        tonal_vs_perc=tonal_vs_perc,
        noisiness=noisiness,
        harmonic_richness=harmonic_richness,
        brightness=brightness,
        fundamental_freq=fundamental_freq,
    )

    return {
        # Core “EQ / spectrum” features
        "brightness": brightness,
        "noisiness": noisiness,
        "tonal_vs_perc": tonal_vs_perc,

        "transient_brightness": transient_brightness,
        "body_brightness": body_brightness,
        "sound_class": sound_class,

        # Envelope / dynamics (classification)
        "attack_speed": attack_speed,
        "sustain_amount": sustain_amount,

        # Measured envelope (seconds and 0-1) for direct Vital mapping
        "attack_sec": attack_sec,
        "decay_sec": decay_sec,
        "sustain_level": sustain_level,
        "release_sec": release_sec,

        # Harmonic timbre
        "harmonic_odd_ratio": harmonic_odd_ratio,
        "harmonic_richness": harmonic_richness,

        # Motion
        "movement": movement,

        # Pitch (kept in Hz for now)
        "fundamental_freq": float(raw_features["fundamental_freq"]),
    }


def _classify_sound(
    attack_speed: float,
    sustain_amount: float,
    tonal_vs_perc: float,
    noisiness: float,
    harmonic_richness: float,
    brightness: float,
    fundamental_freq: float = 440.0,
) -> str:
    """
    Classify into piano_pluck, bass_pluck, pad, pluck, lead, or standard from analyzed features.
    Used by rules to turn Vital blocks on/off and set timbre. Bass = low f0 gets OSC 2 + Filter 2.
    """
    is_plucky = attack_speed >= 0.6 and sustain_amount <= 0.45
    is_bass_range = fundamental_freq > 0 and fundamental_freq < 220.0  # ~A2 and below

    if (
        attack_speed >= 0.65
        and sustain_amount <= 0.4
        and tonal_vs_perc >= 0.72
        and noisiness <= 0.42
        and harmonic_richness >= 0.2
        and not is_bass_range
    ):
        return "piano_pluck"
    if is_bass_range and is_plucky:
        return "bass_pluck"
    if attack_speed <= 0.45 and sustain_amount >= 0.55:
        return "pad"
    if is_plucky:
        return "pluck"
    if 0.35 <= attack_speed <= 0.75 and sustain_amount >= 0.3 and brightness >= 0.45 and harmonic_richness >= 0.4:
        return "lead"
    return "standard"