# mapping/vital_params.py
"""
Single source of truth for Vital parameters controlled by GenSynth.
Use this for: applying params to template, ML target vector ordering, validation.

Template reference: blank preset with only OSC 1 sine on; filter/LFO/FX exist but off/default.
"""

# All parameter names we may write. Order is stable for ML (e.g. dataset columns).
CONTROLLED_PARAMS = (
    # Oscillator 1
    "osc_1_on",
    "osc_1_wave_frame",
    "osc_1_unison_voices",
    "osc_1_unison_detune",
    "osc_1_transpose",
    "osc_1_tune",
    "osc_1_level",
    # Envelope 1 (amp)
    "env_1_delay",
    "env_1_hold",
    "env_1_attack",
    "env_1_attack_power",
    "env_1_decay",
    "env_1_decay_power",
    "env_1_sustain",
    "env_1_release",
    "env_1_release_power",
    # Filter 1
    "filter_1_on",
    "filter_1_cutoff",
    "filter_1_resonance",
    "filter_1_mix",
    # LFO 1
    "lfo_1_frequency",
    # --- Effects chain (match source sound) ---
    "distortion_on",
    "distortion_drive",
    "distortion_mix",
    "distortion_type",
    "distortion_filter_cutoff",
    "distortion_filter_blend",
    "chorus_on",
    "chorus_dry_wet",
    "chorus_feedback",
    "chorus_mod_depth",
    "chorus_frequency",
    "chorus_cutoff",
    "chorus_spread",
    "chorus_voices",
    "reverb_on",
    "reverb_dry_wet",
    "reverb_decay_time",
    "reverb_size",
    "reverb_pre_low_cutoff",
    "reverb_pre_high_cutoff",
    "reverb_chorus_amount",
    "delay_on",
    "delay_dry_wet",
    "delay_feedback",
    "delay_filter_cutoff",
    "delay_frequency",
    "compressor_on",
    "compressor_mix",
    "compressor_attack",
    "compressor_release",
    "phaser_on",
    "phaser_dry_wet",
    "phaser_mod_depth",
    "phaser_center",
    "phaser_feedback",
    "flanger_on",
    "flanger_dry_wet",
    "flanger_mod_depth",
    "flanger_feedback",
    "eq_on",
    "eq_low_gain",
    "eq_low_cutoff",
    "eq_high_gain",
    "eq_high_cutoff",
    "eq_band_gain",
    "eq_band_cutoff",
)


def get_controlled_params_set():
    """Set of controlled param names for fast lookup."""
    return set(CONTROLLED_PARAMS)


def filter_to_controlled(params: dict, template_settings: dict) -> dict:
    """
    Return only params that are in CONTROLLED_PARAMS and exist in the template.
    Keeps output safe for apply_parameters and avoids writing unknown keys.
    """
    allowed = get_controlled_params_set()
    return {
        k: v
        for k, v in params.items()
        if k in allowed and k in template_settings
    }
