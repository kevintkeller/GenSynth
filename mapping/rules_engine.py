# mapping/rules_engine.py

def choose_waveform(harmonic_ratio):
    if harmonic_ratio > 0.75:
        return 132.5  # saw-ish
    elif harmonic_ratio > 0.4:
        return 40     # triangle-ish
    else:
        return 0      # sine


def build_vital_parameters(features: dict) -> dict:

    wave_frame = choose_waveform(features["harmonic_ratio"])

    env_attack = 0.01 + (1 - features["attack_speed"]) * 1.5
    env_decay = 0.2 + features["sustain_amount"] * 2.0
    env_sustain = features["sustain_amount"]
    env_release = 0.2 + features["sustain_amount"] * 1.2

    filter_cutoff = 20 + features["brightness"] * 100
    lfo_freq = -3 + features["movement"] * 6

    return {
        "osc_1_on": 1,
        "osc_1_wave_frame": wave_frame,
        "osc_1_unison_voices": 4 if features["brightness"] > 0.6 else 1,
        "osc_1_unison_detune": features["brightness"] * 0.3,

        "env_1_attack": env_attack,
        "env_1_decay": env_decay,
        "env_1_sustain": env_sustain,
        "env_1_release": env_release,

        "filter_1_on": 1,
        "filter_1_cutoff": filter_cutoff,
        "filter_1_resonance": features["brightness"] * 0.7,

        "lfo_1_frequency": lfo_freq,

        "reverb_on": 1,
        "reverb_dry_wet": features["sustain_amount"] * 0.6,

        "chorus_on": 1,
        "chorus_dry_wet": features["movement"] * 0.5,

        "distortion_on": 1,
        "distortion_drive": features["brightness"] * 40
    }