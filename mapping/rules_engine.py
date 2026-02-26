# mapping/rules_engine.py
"""
Heuristic mapping: audio feature vector -> Vital parameter dict.
Designed for ~70% accuracy and easy tuning; same param set will feed ML later.

Tweak RULES_CONFIG to adjust behavior without changing logic.
"""

from math import log2
from typing import Optional

from mapping.vital_params import CONTROLLED_PARAMS, get_controlled_params_set


# -----------------------------------------------------------------------------
# Tunable constants: change these to adjust behavior (no code changes needed).
# -----------------------------------------------------------------------------
RULES_CONFIG = {
    # Envelope classification (pluck vs pad vs standard)
    "envelope": {
        "pluck_attack_speed_min": 0.7,
        "pluck_sustain_max": 0.35,
        "pad_attack_speed_max": 0.4,
        "pad_sustain_min": 0.6,
    },
    # Envelope shape (times in seconds; powers soften curves, reduce clicks)
    "envelope_pluck": {
        "delay": 0.0,
        "hold": 0.005,
        "attack": 0.004,
        "attack_power": -0.2,
        "decay": 0.25,
        "decay_power": -0.3,
        "sustain": 0.05,
        "release": 0.22,
        "release_power": -0.5,
    },
    "envelope_pad": {
        "delay": 0.0,
        "hold_min": 0.02,
        "hold_max_extra": 0.05,
        "attack_min": 0.35,
        "attack_max_extra": 0.4,
        "decay_min": 1.0,
        "decay_extra": 1.0,
        "sustain_min": 0.7,
        "sustain_extra": 0.3,
        "release_min": 0.9,
        "release_extra": 1.0,
        "attack_power": -0.2,
        "decay_power": -0.3,
        "release_power": -0.5,
    },
    "envelope_standard": {
        "delay": 0.0,
        "hold_min": 0.005,
        "hold_extra": 0.03,
        "attack_min": 0.015,
        "attack_extra": 0.25,
        "decay_min": 0.25,
        "decay_extra": 0.7,
        "sustain_min": 0.35,
        "sustain_extra": 0.5,
        "release_min": 0.22,
        "release_extra": 0.7,
        "attack_power": -0.2,
        "decay_power": -0.3,
        "release_power": -0.5,
    },
    "envelope_global": {
        "min_attack": 0.003,
        "min_release": 0.05,
        "max_attack": 4.0,
        "max_decay": 4.0,
        "max_release": 4.0,
        "delay": 0.0,
        "hold": 0.002,
        "attack_power": -0.2,
        "decay_power": -0.3,
        "release_power": -0.5,
    },
    # Oscillator: wavetable frame (Vital: low=sine, mid=triangle/round, high=saw)
    "osc": {
        "wave_frame_max": 255.0,
        "sine_zone_max": 35.0,
        "unison_voices_bright_threshold": 0.6,
        "unison_detune_scale": 0.35,
        "level": 1.0,
    },
    # Acoustic-like (piano, etc.): warm timbre, gentle filter, no pitch shift
    "acoustic_like": {
        "tonal_vs_perc_min": 0.8,
        "noisiness_max": 0.35,
        "unison_detune_scale": 0.05,
        "chorus_dry_wet_min": 0.05,
        "chorus_dry_wet_movement_scale": 0.15,
        "distortion_drive_min": 5.0,
        "distortion_drive_brightness_scale": 20.0,
        "wave_frame_min": 12.0,
        "wave_frame_max": 55.0,
        "filter_cutoff_min": 32.0,
        "filter_cutoff_max": 72.0,
    },
    # Filter (Vital filter_1_cutoff ~0–120 Hz-style units)
    "filter": {
        "cutoff_min": 20.0,
        "cutoff_range": 90.0,
        "resonance_min": 0.2,
        "resonance_scale": 0.8,
    },
    # LFO (Vital lfo_1_frequency roughly -4..+4 or similar)
    "lfo": {
        "freq_min": -4.0,
        "freq_range": 8.0,
    },
    # Effects chain: map features to all effect params (tweak ranges here)
    "fx": {
        "reverb_dry_wet_min": 0.08,
        "reverb_dry_wet_scale": 0.6,
        "reverb_decay_min": 0.3,
        "reverb_decay_scale": 0.8,
        "reverb_size_min": 0.3,
        "reverb_size_scale": 0.6,
        "chorus_dry_wet_min": 0.05,
        "chorus_dry_wet_scale": 0.5,
        "chorus_mod_depth_scale": 0.5,
        "chorus_feedback_scale": 0.4,
        "distortion_drive_min": 8.0,
        "distortion_drive_scale": 45.0,
        "distortion_mix_default": 0.35,
        "distortion_filter_cutoff_scale": 80.0,
        "delay_dry_wet_scale": 0.25,
        "delay_feedback_scale": 0.5,
        "compressor_mix_scale": 0.5,
        "phaser_dry_wet_scale": 0.4,
        "flanger_dry_wet_scale": 0.35,
        "eq_low_gain_min": -12.0,
        "eq_low_gain_scale": 10.0,
        "eq_high_gain_scale": 8.0,
        "eq_band_gain_scale": 6.0,
    },
    # Pitch: map detected f0 to osc_1_transpose (semitones) + osc_1_tune (fine -1..1)
    "pitch": {
        "reference_hz": 440.0,
    },
}


def _classify_envelope(features: dict, config: dict) -> str:
    """Classify into pluck / pad / standard from attack_speed and sustain_amount."""
    ec = config["envelope"]
    attack_speed = features["attack_speed"]
    sustain = features["sustain_amount"]
    if attack_speed >= ec["pluck_attack_speed_min"] and sustain <= ec["pluck_sustain_max"]:
        return "pluck"
    if attack_speed <= ec["pad_attack_speed_max"] and sustain >= ec["pad_sustain_min"]:
        return "pad"
    return "standard"


def _env_params(features: dict, config: dict) -> dict:
    """
    Build envelope 1 from analyzed audio: direct mapping of measured attack/decay/sustain/release
    so the Vital envelope matches the source sound for any input.
    """
    g = config["envelope_global"]
    # Use measured envelope (seconds and 0-1 level); fallbacks for older feature vectors
    attack_sec = float(features.get("attack_sec", 0.01))
    decay_sec = float(features.get("decay_sec", 0.2))
    sustain_level = float(features.get("sustain_level", 0.3))
    release_sec = float(features.get("release_sec", 0.25))

    attack = max(g["min_attack"], min(g["max_attack"], attack_sec))
    decay = max(0.01, min(g["max_decay"], decay_sec))
    release = max(g["min_release"], min(g["max_release"], release_sec))
    sustain = max(0.0, min(1.0, sustain_level))

    return {
        "env_1_delay": g["delay"],
        "env_1_hold": g["hold"],
        "env_1_attack": attack,
        "env_1_attack_power": g["attack_power"],
        "env_1_decay": decay,
        "env_1_decay_power": g["decay_power"],
        "env_1_sustain": sustain,
        "env_1_release": release,
        "env_1_release_power": g["release_power"],
    }


def _osc_params(features: dict, config: dict) -> dict:
    """Oscillator 1: wave frame from brightness + harmonic character (odd/even, richness)."""
    brightness = features["brightness"]
    noisiness = features["noisiness"]
    richness = features.get("harmonic_richness", 0.5)
    odd_ratio = features.get("harmonic_odd_ratio", 0.5)
    oc = config["osc"]
    # Low richness -> sine zone (0–sine_zone_max); high richness -> full range by brightness + even-harmonic tilt
    if richness < 0.25:
        wave_frame = 5.0 + brightness * (oc["sine_zone_max"] - 5.0)
    else:
        # More even harmonics (low odd_ratio) = brighter/saw-ish; more odd = triangle/square
        tilt = (1.0 - odd_ratio) * 0.4 + brightness * 0.6
        wave_frame = oc["sine_zone_max"] + tilt * (oc["wave_frame_max"] - oc["sine_zone_max"])
    wave_frame = float(max(0.0, min(oc["wave_frame_max"], wave_frame)))
    unison_voices = 4.0 if brightness > oc["unison_voices_bright_threshold"] else 1.0
    unison_detune = brightness * oc["unison_detune_scale"]
    return {
        "osc_1_on": 1.0,
        "osc_1_wave_frame": wave_frame,
        "osc_1_unison_voices": unison_voices,
        "osc_1_unison_detune": unison_detune,
        "osc_1_level": oc["level"],
    }


def _filter_params(features: dict, config: dict) -> dict:
    """Filter 1: on, cutoff, resonance, mix. Template has filter off by default."""
    brightness = features["brightness"]
    noisiness = features["noisiness"]
    fc = config["filter"]
    cutoff = fc["cutoff_min"] + brightness * fc["cutoff_range"]
    resonance = fc["resonance_min"] + fc["resonance_scale"] * max(
        0.0, min(1.0, brightness * (1.0 - 0.5 * noisiness))
    )
    return {
        "filter_1_on": 1.0,
        "filter_1_cutoff": cutoff,
        "filter_1_resonance": resonance,
        "filter_1_mix": 1.0,
    }


def _lfo_params(features: dict, config: dict) -> dict:
    """LFO 1 frequency from movement."""
    movement = features["movement"]
    lc = config["lfo"]
    freq = lc["freq_min"] + movement * lc["freq_range"]
    return {"lfo_1_frequency": freq}


def _pitch_params(features: dict, config: dict) -> dict:
    """Map fundamental_freq to osc_1_transpose (semitones) and osc_1_tune (fine -1..1)."""
    f0 = features.get("fundamental_freq") or 0.0
    if f0 <= 0:
        return {}
    ref = config["pitch"]["reference_hz"]
    semitones = 12.0 * log2(f0 / ref)
    coarse = round(semitones)
    fine = max(-1.0, min(1.0, semitones - coarse))
    return {
        "osc_1_transpose": float(coarse),
        "osc_1_tune": float(fine),
    }


def _fx_params(features: dict, config: dict) -> dict:
    """Full effects chain: distortion, chorus, reverb, delay, compressor, phaser, flanger, EQ from features."""
    brightness = features["brightness"]
    noisiness = features["noisiness"]
    sustain = features["sustain_amount"]
    tonal = features["tonal_vs_perc"]
    movement = features["movement"]
    fx = config["fx"]

    # Reverb: more sustain/tonal -> more reverb and longer decay
    reverb_wet = fx["reverb_dry_wet_min"] + fx["reverb_dry_wet_scale"] * sustain * tonal
    reverb_decay = fx["reverb_decay_min"] + fx["reverb_decay_scale"] * sustain
    reverb_size = fx["reverb_size_min"] + fx["reverb_size_scale"] * (0.5 + 0.5 * tonal)

    # Chorus: movement and noisiness -> depth and feedback
    chorus_wet = fx["chorus_dry_wet_min"] + fx["chorus_dry_wet_scale"] * movement * (0.5 + 0.5 * noisiness)
    chorus_mod = fx["chorus_mod_depth_scale"] * movement
    chorus_fb = 0.2 + fx["chorus_feedback_scale"] * movement

    # Distortion: brightness and noisiness -> drive; filter cutoff follows brightness
    drive = fx["distortion_drive_min"] + fx["distortion_drive_scale"] * (0.6 * brightness + 0.4 * noisiness)
    dist_cutoff = 15.0 + fx["distortion_filter_cutoff_scale"] * brightness

    # Delay: movement, less for very tonal
    delay_wet = fx["delay_dry_wet_scale"] * movement * (1.0 - tonal * 0.5)
    delay_fb = 0.3 + fx["delay_feedback_scale"] * movement

    # Compressor: always on, mix from sustain (glue)
    comp_mix = 0.5 + fx["compressor_mix_scale"] * (sustain * 0.5 + movement * 0.3)

    # Phaser / Flanger: subtle for movement; more for synthetic sounds
    phaser_wet = fx["phaser_dry_wet_scale"] * movement * (0.3 + 0.7 * noisiness)
    flanger_wet = fx["flanger_dry_wet_scale"] * movement * noisiness

    # EQ: brightness -> high shelf; low end from tonal; mid from richness
    eq_low = fx["eq_low_gain_min"] + fx["eq_low_gain_scale"] * (1.0 - noisiness)
    eq_high = fx["eq_high_gain_scale"] * brightness
    eq_band = fx["eq_band_gain_scale"] * (0.5 + 0.5 * tonal)

    return {
        "distortion_on": 1.0,
        "distortion_drive": max(5.0, drive),
        "distortion_mix": fx["distortion_mix_default"],
        "distortion_type": 3.0,
        "distortion_filter_cutoff": min(100.0, dist_cutoff),
        "distortion_filter_blend": 0.7,
        "chorus_on": 1.0,
        "chorus_dry_wet": min(1.0, chorus_wet),
        "chorus_feedback": min(1.0, chorus_fb),
        "chorus_mod_depth": min(1.0, chorus_mod),
        "chorus_frequency": -3.0,
        "chorus_cutoff": 50.0 + 30.0 * brightness,
        "chorus_spread": 0.5 + 0.4 * movement,
        "chorus_voices": 8.0,
        "reverb_on": 1.0,
        "reverb_dry_wet": min(1.0, reverb_wet),
        "reverb_decay_time": min(1.5, reverb_decay),
        "reverb_size": min(1.0, reverb_size),
        "reverb_pre_low_cutoff": 20.0,
        "reverb_pre_high_cutoff": 100.0,
        "reverb_chorus_amount": 0.15 + 0.2 * movement,
        "delay_on": 1.0 if delay_wet > 0.04 else 0.0,
        "delay_dry_wet": min(1.0, delay_wet),
        "delay_feedback": min(0.75, delay_fb),
        "delay_filter_cutoff": 50.0 + 25.0 * brightness,
        "delay_frequency": 1.5 + movement * 1.5,
        "compressor_on": 1.0,
        "compressor_mix": min(1.0, comp_mix),
        "compressor_attack": 0.4,
        "compressor_release": 0.5,
        "phaser_on": 1.0 if phaser_wet > 0.06 else 0.0,
        "phaser_dry_wet": min(1.0, phaser_wet),
        "phaser_mod_depth": 15.0 + 15.0 * movement,
        "phaser_center": 60.0 + 40.0 * brightness,
        "phaser_feedback": 0.3 + 0.2 * movement,
        "flanger_on": 1.0 if flanger_wet > 0.05 else 0.0,
        "flanger_dry_wet": min(1.0, flanger_wet),
        "flanger_mod_depth": 0.3 + 0.3 * movement,
        "flanger_feedback": 0.3,
        "eq_on": 1.0,
        "eq_low_gain": max(-18.0, min(6.0, eq_low)),
        "eq_low_cutoff": 80.0,
        "eq_high_gain": max(-6.0, min(9.0, eq_high)),
        "eq_high_cutoff": 8000.0,
        "eq_band_gain": max(-6.0, min(9.0, eq_band)),
        "eq_band_cutoff": 800.0 + 2000.0 * brightness,
    }


def _acoustic_like_overrides(params: dict, features: dict, config: dict) -> None:
    """In-place: for very tonal, low-noise pluck-like sounds (e.g. piano), warm timbre, gentle effects, keep template pitch."""
    ac = config["acoustic_like"]
    if features["tonal_vs_perc"] < ac["tonal_vs_perc_min"]:
        return
    if features["noisiness"] > ac["noisiness_max"]:
        return
    if _classify_envelope(features, config) != "pluck":
        return
    params["osc_1_unison_voices"] = 1.0
    params["osc_1_unison_detune"] = features["brightness"] * ac["unison_detune_scale"]
    b = features["brightness"]
    params["osc_1_wave_frame"] = ac["wave_frame_min"] + b * (ac["wave_frame_max"] - ac["wave_frame_min"])
    params["filter_1_cutoff"] = ac["filter_cutoff_min"] + b * (ac["filter_cutoff_max"] - ac["filter_cutoff_min"])
    params["filter_1_resonance"] = 0.15 + 0.25 * b
    # Gentle effects chain for acoustic (piano, etc.)
    params["chorus_dry_wet"] = ac["chorus_dry_wet_min"] + features["movement"] * ac["chorus_dry_wet_movement_scale"]
    params["chorus_mod_depth"] = min(params.get("chorus_mod_depth", 0.5) * 0.4, 0.3)
    params["distortion_drive"] = ac["distortion_drive_min"] + features["brightness"] * ac["distortion_drive_brightness_scale"]
    params["distortion_mix"] = 0.2
    params["reverb_dry_wet"] = min(0.22, params.get("reverb_dry_wet", 0.3))
    params["reverb_decay_time"] = min(0.5, params.get("reverb_decay_time", 0.6))
    params["reverb_size"] = min(0.5, params.get("reverb_size", 0.6))
    params["delay_dry_wet"] = min(0.08, params.get("delay_dry_wet", 0.2))
    params["phaser_on"] = 0.0
    params["flanger_on"] = 0.0
    params["compressor_mix"] = min(0.65, params.get("compressor_mix", 0.8))
    params["eq_low_gain"] = max(-6.0, min(3.0, params.get("eq_low_gain", 0)))
    params["eq_high_gain"] = max(-3.0, min(5.0, params.get("eq_high_gain", 0)))
    params.pop("osc_1_transpose", None)
    params.pop("osc_1_tune", None)


def build_vital_parameters(features: dict, config: Optional[dict] = None) -> dict:
    """
    Map feature vector to Vital parameter dict.
    Only includes keys in CONTROLLED_PARAMS so schema/ML stay in sync.
    """
    cfg = config if config is not None else RULES_CONFIG

    env_class = _classify_envelope(features, cfg)  # used only for acoustic_like overrides
    params = {}
    params.update(_env_params(features, cfg))
    params.update(_osc_params(features, cfg))
    params.update(_filter_params(features, cfg))
    params.update(_lfo_params(features, cfg))
    params.update(_fx_params(features, cfg))

    params.update(_pitch_params(features, cfg))
    _acoustic_like_overrides(params, features, cfg)

    # Only return params we officially control (for apply_parameters and ML)
    allowed = get_controlled_params_set()
    return {k: v for k, v in params.items() if k in allowed}
