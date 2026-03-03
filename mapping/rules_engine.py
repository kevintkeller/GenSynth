# mapping/rules_engine.py
"""
Rule-based mapping: audio feature vector -> Vital parameter dict.
- sound_class (piano_pluck, bass_pluck, pluck, pad, lead, standard) drives distinct wave ranges and FX.
- Per-class wave_frame (osc_1) and FX caps ensure piano/bass/pluck sound different.
- Optional: mapping.preset_bank.get_per_type_defaults() can seed or tune from Vital preset examples.
Tune RULES_CONFIG["sound_class"] and ["acoustic_like"] without changing code.
"""

import math
from math import log2
from typing import Optional

from mapping.vital_params import get_controlled_params_set


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
    # Pluck/piano/bass: minimum decay and release so tail is audible; scale by wavetable (richer = longer).
    "envelope_pluck_floor": {
        "decay_min": 0.9,
        "decay_max": 2.2,
        "release_min": 0.6,
        "release_max": 1.5,
        "decay_scale_by_wave_tilt": True,
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
        "min_attack": 0.002,
        "min_release": 0.05,
        "max_attack": 4.0,
        "max_decay": 4.0,
        "max_release": 4.0,
        "delay": 0.0,
        "hold": 0.002,
        "attack_power": -0.25,
        "decay_power": -0.35,
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
    # Piano/acoustic-like: clearer wave (more harmonics), open filter so not muffled
    "acoustic_like": {
        "unison_detune_scale": 0.05,
        "chorus_dry_wet_min": 0.05,
        "chorus_dry_wet_movement_scale": 0.15,
        "distortion_drive_min": 5.0,
        "distortion_drive_brightness_scale": 20.0,
        "wave_frame_min": 32.0,
        "wave_frame_max": 52.0,
        "filter_cutoff_min": 58.0,
        "filter_cutoff_max": 95.0,
        "filter_cutoff_floor": 62.0,
        "env_sustain_max": 0.15,
        "env_attack_max": 0.012,
        "env_release_min": 0.35,
        "env_release_max": 1.0,
    },
    # Per instrument: blocks, wave range (rounded = less buzz), resonance/distortion caps, FX. Tuned for clear timbre.
    "sound_class": {
        "piano_pluck": {"osc_2": False, "osc_3": False, "filter_2": False, "lfo": False, "unison": 1,
                        "wave_min": 32.0, "wave_max": 48.0,
                        "resonance_max": 0.18, "distortion_mix_max": 0.08, "filter_cutoff_floor": 58.0,
                        "reverb_wet_max": 0.18, "delay_wet_max": 0.0, "chorus_wet_max": 0.06,
                        "phaser_off": True, "flanger_off": True},
        "bass_pluck": {"osc_2": True, "osc_3": False, "filter_2": True, "lfo": False, "unison": 1,
                       "wave_min": 8.0, "wave_max": 32.0,
                       "resonance_max": 0.22, "distortion_mix_max": 0.1,
                       "reverb_wet_max": 0.14, "delay_wet_max": 0.06, "chorus_wet_max": 0.12,
                       "phaser_off": True, "flanger_off": True},
        "guitar": {"osc_2": True, "osc_3": False, "filter_2": True, "lfo": False, "unison": 1,
                   "wave_min": 22.0, "wave_max": 55.0, "filter_cutoff_floor": 45.0,
                   "resonance_max": 0.26, "distortion_mix_max": 0.12, "filter_2_mix_max": 0.35,
                   "reverb_wet_max": 0.22, "delay_wet_max": 0.12, "chorus_wet_max": 0.1,
                   "phaser_off": True, "flanger_off": True},
        "pad": {"osc_2": True, "osc_3": True, "filter_2": True, "lfo": True, "unison": 2,
                "wave_min": 15.0, "wave_max": 100.0,
                "resonance_max": 0.5, "distortion_mix_max": 0.3,
                "reverb_wet_max": 0.6, "delay_wet_max": 0.35, "chorus_wet_max": 0.5,
                "phaser_off": False, "flanger_off": False},
        "pluck": {"osc_2": True, "osc_3": False, "filter_2": False, "lfo": True, "unison": 1,
                  "wave_min": 28.0, "wave_max": 62.0, "filter_cutoff_floor": 50.0,
                  "resonance_max": 0.28, "distortion_mix_max": 0.12,
                  "reverb_wet_max": 0.2, "delay_wet_max": 0.15, "chorus_wet_max": 0.15,
                  "phaser_off": True, "flanger_off": True},
        "lead": {"osc_2": True, "osc_3": False, "filter_2": True, "lfo": True, "unison": 2,
                 "wave_min": 65.0, "wave_max": 150.0,
                 "resonance_max": 0.42, "distortion_mix_max": 0.25,
                 "reverb_wet_max": 0.4, "delay_wet_max": 0.25, "chorus_wet_max": 0.4,
                 "phaser_off": False, "flanger_off": False},
        "standard": {"osc_2": True, "osc_3": False, "filter_2": True, "lfo": True, "unison": 1,
                     "wave_min": 25.0, "wave_max": 100.0,
                     "resonance_max": 0.38, "distortion_mix_max": 0.2,
                     "reverb_wet_max": 0.35, "delay_wet_max": 0.2, "chorus_wet_max": 0.3,
                     "phaser_off": False, "flanger_off": False},
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
    Build envelope 1 from analyzed audio. Pluck/piano/bass get a higher decay/release floor
    and optional scaling by wavetable tilt (richer waves = longer decay so tail is audible).
    """
    g = config["envelope_global"]
    attack_sec = float(features.get("attack_sec", 0.01))
    decay_sec = float(features.get("decay_sec", 0.2))
    sustain_level = float(features.get("sustain_level", 0.3))
    release_sec = float(features.get("release_sec", 0.25))
    sc = features.get("sound_class", "standard")
    is_pluck_class = sc in ("pluck", "piano_pluck", "bass_pluck", "guitar")

    attack = max(g["min_attack"], min(g["max_attack"], attack_sec))
    decay = max(0.01, min(g["max_decay"], decay_sec))
    release = max(g["min_release"], min(g["max_release"], release_sec))
    sustain = max(0.0, min(1.0, sustain_level))

    if is_pluck_class:
        pf = config.get("envelope_pluck_floor", {})
        decay_min = pf.get("decay_min", 0.9)
        decay_max = pf.get("decay_max", 2.2)
        release_min = pf.get("release_min", 0.6)
        release_max = pf.get("release_max", 1.5)
        decay = max(decay_min, min(decay_max, decay))
        release = max(release_min, min(release_max, release))
        if pf.get("decay_scale_by_wave_tilt", True):
            brightness = features.get("brightness", 0.5)
            odd_ratio = features.get("harmonic_odd_ratio", 0.5)
            tilt = (1.0 - odd_ratio) * 0.3 + brightness * 0.7
            decay = min(decay_max, decay * (0.88 + 0.28 * tilt))
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
    """Oscillator 1: wave frame from sound_class range + brightness so piano/bass/pluck differ."""
    brightness = features["brightness"]
    richness = features.get("harmonic_richness", 0.5)
    odd_ratio = features.get("harmonic_odd_ratio", 0.5)
    oc = config["osc"]
    sc = features.get("sound_class", "standard")
    sc_cfg = config.get("sound_class", {}).get(sc, {})
    w_min = sc_cfg.get("wave_min", oc["sine_zone_max"])
    w_max = sc_cfg.get("wave_max", oc["wave_frame_max"])
    tilt = (1.0 - odd_ratio) * 0.3 + brightness * 0.7
    wave_frame = w_min + tilt * (w_max - w_min)
    wave_frame = float(max(0.0, min(255.0, wave_frame)))
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


def _osc2_osc3_params(features: dict, config: dict) -> dict:
    """OSC 2 and OSC 3: optional layers from richness/brightness (sub or detuned body)."""
    brightness = features["brightness"]
    richness = features.get("harmonic_richness", 0.5)
    # OSC 2: turn on for richer sounds as a subtle second layer (slightly different wave, lower level)
    osc2_on = 1.0 if richness > 0.4 and brightness > 0.25 else 0.0
    # OSC 3: turn on for very rich/bright (e.g. pads, leads)
    osc3_on = 1.0 if richness > 0.6 and brightness > 0.5 else 0.0
    oc = config["osc"]
    # OSC 2: slightly offset wave frame and lower level for thickness
    osc2_frame = min(oc["wave_frame_max"], 20.0 + brightness * 80.0) if osc2_on else 0.0
    osc3_frame = min(oc["wave_frame_max"], 40.0 + brightness * 100.0) if osc3_on else 0.0
    return {
        "osc_2_on": osc2_on,
        "osc_2_level": 0.35 + 0.25 * brightness if osc2_on else 0.0,
        "osc_2_wave_frame": osc2_frame,
        "osc_2_unison_voices": 1.0,
        "osc_2_unison_detune": 0.01 * brightness,
        "osc_2_transpose": 0.0,
        "osc_2_tune": 0.0,
        "osc_3_on": osc3_on,
        "osc_3_level": 0.25 + 0.2 * brightness if osc3_on else 0.0,
        "osc_3_wave_frame": osc3_frame,
        "osc_3_unison_voices": 1.0,
        "osc_3_unison_detune": 0.01 * brightness,
        "osc_3_transpose": 0.0,
        "osc_3_tune": 0.0,
    }


def _sample_params(features: dict, config: dict) -> dict:
    """Sampler (SMP): off by default (no sample content); level/transpose/tune for when used."""
    return {
        "sample_on": 0.0,
        "sample_level": 0.5,
        "sample_transpose": 0.0,
        "sample_tune": 0.0,
    }


def _macro_params(features: dict, config: dict) -> dict:
    """Macro controls 0-1 from features; assign in Vital Matrix to filter/osc/etc."""
    b = features.get("brightness", 0.5)
    m = features.get("movement", 0.3)
    r = features.get("harmonic_richness", 0.5)
    n = features.get("noisiness", 0.2)
    return {
        "macro_control_1": max(0.0, min(1.0, b)),
        "macro_control_2": max(0.0, min(1.0, m)),
        "macro_control_3": max(0.0, min(1.0, r)),
        "macro_control_4": max(0.0, min(1.0, n)),
    }


def _env2_env3_env4_params(features: dict, config: dict) -> dict:
    """ENV 2, 3, 4: filter/modulation envelopes from same analyzed envelope or variants.
    ENV 2 decay/release synced with ENV 1 (same pluck floor + wavetable scaling)."""
    g = config["envelope_global"]
    attack_sec = float(features.get("attack_sec", 0.01))
    decay_sec = float(features.get("decay_sec", 0.2))
    sustain_level = float(features.get("sustain_level", 0.3))
    release_sec = float(features.get("release_sec", 0.25))
    sc = features.get("sound_class", "standard")
    is_pluck_class = sc in ("pluck", "piano_pluck", "bass_pluck", "guitar")

    attack = max(g["min_attack"], min(g["max_attack"], attack_sec))
    decay = max(0.01, min(g["max_decay"], decay_sec))
    release = max(g["min_release"], min(g["max_release"], release_sec))
    sustain = max(0.0, min(1.0, sustain_level))

    if is_pluck_class:
        pf = config.get("envelope_pluck_floor", {})
        decay_min = pf.get("decay_min", 0.9)
        decay_max = pf.get("decay_max", 2.2)
        release_min = pf.get("release_min", 0.6)
        release_max = pf.get("release_max", 1.5)
        decay = max(decay_min, min(decay_max, decay))
        release = max(release_min, min(release_max, release))
        if pf.get("decay_scale_by_wave_tilt", True):
            brightness = features.get("brightness", 0.5)
            odd_ratio = features.get("harmonic_odd_ratio", 0.5)
            tilt = (1.0 - odd_ratio) * 0.3 + brightness * 0.7
            decay = min(decay_max, decay * (0.88 + 0.28 * tilt))
    # ENV 2: same shape as ENV 1 (for filter modulation routing in Vital)
    # ENV 3: faster attack/decay for mod
    # ENV 4: slower, more sustain for pads
    return {
        "env_2_delay": 0.0,
        "env_2_hold": 0.002,
        "env_2_attack": attack,
        "env_2_attack_power": -0.2,
        "env_2_decay": decay,
        "env_2_decay_power": -0.3,
        "env_2_sustain": sustain,
        "env_2_release": release,
        "env_2_release_power": -0.5,
        "env_3_delay": 0.0,
        "env_3_hold": 0.0,
        "env_3_attack": max(0.005, attack * 0.5),
        "env_3_decay": max(0.01, decay * 0.3),
        "env_3_sustain": 0.0,
        "env_3_release": max(0.05, release * 0.4),
        "env_4_delay": 0.0,
        "env_4_hold": 0.01,
        "env_4_attack": min(1.0, attack * 2.0),
        "env_4_decay": min(2.0, decay * 1.5),
        "env_4_sustain": min(0.8, sustain + 0.2),
        "env_4_release": min(2.0, release * 1.2),
    }


def _lfo2_lfo3_lfo4_params(features: dict, config: dict) -> dict:
    """LFO 2, 3, 4: rates from movement for modulation variety."""
    movement = features["movement"]
    lc = config["lfo"]
    base = lc["freq_min"] + movement * lc["freq_range"]
    return {
        "lfo_2_frequency": base * 0.5,
        "lfo_3_frequency": base * 0.3 + 0.5,
        "lfo_4_frequency": base * 0.25 + 0.3,
    }


def _filter2_params(features: dict, config: dict) -> dict:
    """Filter 2: optional second filter (e.g. serial) from brightness."""
    brightness = features["brightness"]
    # Turn on for brighter sounds to add extra tone shaping
    f2_on = 1.0 if brightness > 0.5 else 0.0
    fc = config["filter"]
    cutoff = fc["cutoff_min"] + 30.0 + brightness * 50.0 if f2_on else 80.0
    return {
        "filter_2_on": f2_on,
        "filter_2_cutoff": cutoff,
        "filter_2_resonance": 0.1 + 0.2 * brightness if f2_on else 0.1,
        "filter_2_mix": 0.7 if f2_on else 0.0,
    }


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

    sc = features.get("sound_class", "standard")
    sc_cfg = config.get("sound_class", {}).get(sc, {})
    reverb_cap = sc_cfg.get("reverb_wet_max", 0.5)
    delay_cap = sc_cfg.get("delay_wet_max", 0.3)
    chorus_cap = sc_cfg.get("chorus_wet_max", 0.4)
    reverb_wet = min(reverb_cap, reverb_wet) if reverb_cap else min(1.0, reverb_wet)
    if reverb_cap and reverb_wet < 0.04:
        reverb_wet = 0.06 + (reverb_cap - 0.06) * 0.3
    delay_wet = min(delay_cap, delay_wet) if delay_cap else min(1.0, delay_wet)
    chorus_wet = min(chorus_cap, chorus_wet) if chorus_cap else min(1.0, chorus_wet)
    return {
        "distortion_on": 1.0,
        "distortion_drive": max(5.0, drive),
        "distortion_mix": fx["distortion_mix_default"],
        "distortion_type": 3.0,
        "distortion_filter_cutoff": min(100.0, dist_cutoff),
        "distortion_filter_blend": 0.7,
        "chorus_on": 1.0 if chorus_cap and chorus_wet > 0.02 else 0.0,
        "chorus_dry_wet": min(1.0, chorus_wet),
        "chorus_feedback": min(1.0, chorus_fb),
        "chorus_mod_depth": min(1.0, chorus_mod),
        "chorus_frequency": -3.0,
        "chorus_cutoff": 50.0 + 30.0 * brightness,
        "chorus_spread": 0.5 + 0.4 * movement,
        "chorus_voices": 8.0,
        "reverb_on": 1.0 if reverb_cap and reverb_wet > 0.02 else 0.0,
        "reverb_dry_wet": min(1.0, reverb_wet),
        "reverb_decay_time": min(1.5, reverb_decay),
        "reverb_size": min(1.0, reverb_size),
        "reverb_pre_low_cutoff": 20.0,
        "reverb_pre_high_cutoff": 100.0,
        "reverb_chorus_amount": 0.15 + 0.2 * movement,
        "delay_on": 1.0 if delay_cap and delay_wet > 0.03 else 0.0,
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
    """In-place: when sound_class is piano_pluck, apply piano timbre (triangle-zone wave, filter from transient), keep template pitch."""
    if features.get("sound_class") != "piano_pluck":
        return
    ac = config["acoustic_like"]
    params["osc_1_unison_voices"] = 1.0
    params["osc_1_unison_detune"] = features["brightness"] * ac["unison_detune_scale"]
    params["osc_1_level"] = 1.0
    b = features["brightness"]
    params["osc_1_wave_frame"] = ac["wave_frame_min"] + b * (ac["wave_frame_max"] - ac["wave_frame_min"])
    transient_b = features.get("transient_brightness", b)
    body_b = features.get("body_brightness", b * 0.7)
    cutoff_t = 0.8 * transient_b + 0.2 * body_b
    raw_cutoff = ac["filter_cutoff_min"] + cutoff_t * (ac["filter_cutoff_max"] - ac["filter_cutoff_min"])
    params["filter_1_cutoff"] = max(ac.get("filter_cutoff_floor", 50), raw_cutoff)
    params["filter_1_resonance"] = 0.06 + 0.1 * b
    # Piano-like envelope: snappy attack, low sustain, natural release
    params["env_1_attack"] = min(ac.get("env_attack_max", 0.02), params.get("env_1_attack", 0.01))
    params["env_1_sustain"] = min(ac.get("env_sustain_max", 0.2), params.get("env_1_sustain", 0.1))
    r = params.get("env_1_release", 0.4)
    params["env_1_release"] = max(ac.get("env_release_min", 0.3), min(ac.get("env_release_max", 1.2), r))
    # Keep env 2 in sync for consistency
    params["env_2_attack"] = params["env_1_attack"]
    params["env_2_sustain"] = params["env_1_sustain"]
    params["env_2_release"] = params["env_1_release"]
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
    # Acoustic: single osc, no second filter, no LFO modulation (dry = like input)
    params["osc_2_on"] = 0.0
    params["osc_2_level"] = 0.0
    params["osc_2_wave_frame"] = 5.0
    params["osc_3_on"] = 0.0
    params["filter_2_on"] = 0.0
    params["lfo_1_frequency"] = 0.0
    params["lfo_2_frequency"] = 0.0
    params["lfo_3_frequency"] = 0.0
    params["lfo_4_frequency"] = 0.0
    params.pop("osc_1_transpose", None)
    params.pop("osc_1_tune", None)


def _sound_class_overrides(params: dict, features: dict, config: dict) -> None:
    """In-place: turn Vital blocks on/off, set OSC 2 wave from features, pluck filter floor, LFO."""
    sc = features.get("sound_class", "standard")
    sc_config = config.get("sound_class", {}).get(sc)
    if not sc_config:
        return
    if sc_config.get("filter_cutoff_floor") is not None:
        params["filter_1_cutoff"] = max(params.get("filter_1_cutoff", 50), sc_config["filter_cutoff_floor"])
    if sc_config.get("osc_2") is False:
        params["osc_2_on"] = 0.0
        params["osc_2_level"] = 0.0
    elif sc_config.get("osc_2") is True:
        params["osc_2_on"] = 1.0
        params["osc_2_level"] = max(params.get("osc_2_level", 0), 0.35)
    if sc_config.get("osc_3") is False:
        params["osc_3_on"] = 0.0
        params["osc_3_level"] = 0.0
    elif sc_config.get("osc_3") is True:
        params["osc_3_on"] = 1.0
        params["osc_3_level"] = max(params.get("osc_3_level", 0), 0.25)
    if sc_config.get("filter_2") is False:
        params["filter_2_on"] = 0.0
        params["filter_2_mix"] = 0.0
    elif sc_config.get("filter_2") is True:
        params["filter_2_on"] = 1.0
        params["filter_2_mix"] = max(params.get("filter_2_mix", 0), 0.5)
    if sc_config.get("lfo") is False:
        params["lfo_1_frequency"] = 0.0
        params["lfo_2_frequency"] = 0.0
        params["lfo_3_frequency"] = 0.0
        params["lfo_4_frequency"] = 0.0
    unison = sc_config.get("unison")
    if unison is not None:
        params["osc_1_unison_voices"] = float(unison)

    if sc == "bass_pluck":
        params["osc_2_transpose"] = -12.0
        params["osc_2_wave_frame"] = 8.0
        params["osc_2_level"] = 0.5
    elif sc == "pluck":
        b = features.get("brightness", 0.5)
        r = features.get("harmonic_richness", 0.5)
        params["osc_2_wave_frame"] = 28.0 + 25.0 * b
        params["osc_2_level"] = 0.28 + 0.25 * r
        params["lfo_1_frequency"] = 0.2
    elif sc == "guitar":
        b = features.get("brightness", 0.5)
        params["osc_2_wave_frame"] = 22.0 + 25.0 * b
        params["osc_2_level"] = 0.3
    if sc_config.get("filter_2_mix_max") is not None:
        params["filter_2_mix"] = min(params.get("filter_2_mix", 0.5), sc_config["filter_2_mix_max"])


def _instrument_timbre_overrides(params: dict, features: dict, config: dict) -> None:
    """Apply per-instrument de-buzz: resonance cap, distortion mix cap, phaser/flanger off, wave clamp."""
    sc = features.get("sound_class", "standard")
    sc_config = config.get("sound_class", {}).get(sc, {})
    if not sc_config:
        return
    if sc_config.get("resonance_max") is not None:
        params["filter_1_resonance"] = min(params.get("filter_1_resonance", 0.3), sc_config["resonance_max"])
    if sc_config.get("distortion_mix_max") is not None:
        params["distortion_mix"] = min(params.get("distortion_mix", 0.35), sc_config["distortion_mix_max"])
    if sc_config.get("phaser_off"):
        params["phaser_on"] = 0.0
        params["phaser_dry_wet"] = 0.0
    if sc_config.get("flanger_off"):
        params["flanger_on"] = 0.0
        params["flanger_dry_wet"] = 0.0
    w_min = sc_config.get("wave_min")
    w_max = sc_config.get("wave_max")
    if w_min is not None and w_max is not None:
        w = params.get("osc_1_wave_frame", 50.0)
        params["osc_1_wave_frame"] = float(max(w_min, min(w_max, w)))


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
    params.update(_osc2_osc3_params(features, cfg))
    params.update(_sample_params(features, cfg))
    params.update(_macro_params(features, cfg))
    params.update(_env2_env3_env4_params(features, cfg))
    params.update(_filter_params(features, cfg))
    params.update(_filter2_params(features, cfg))
    params.update(_lfo_params(features, cfg))
    params.update(_lfo2_lfo3_lfo4_params(features, cfg))
    params.update(_fx_params(features, cfg))

    params.update(_pitch_params(features, cfg))
    _acoustic_like_overrides(params, features, cfg)
    _sound_class_overrides(params, features, cfg)
    _instrument_timbre_overrides(params, features, cfg)

    # Only return params we officially control; ensure no NaN/Inf (Vital can crash)
    allowed = get_controlled_params_set()
    out = {}
    for k, v in params.items():
        if k not in allowed:
            continue
        if isinstance(v, (int, float)) and not math.isfinite(v):
            v = 0.0
        elif isinstance(v, float):
            v = max(-1e6, min(1e6, v))
        elif isinstance(v, int):
            v = max(-1000000, min(1000000, v))
        out[k] = v
    return out
