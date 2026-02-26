# mapping/vital_params.py
"""
Single source of truth for Vital parameters controlled by GenSynth.
Use this for: applying params to template, ML target vector ordering, validation.

Template reference: blank preset with only OSC 1 sine on; filter/LFO/FX exist but off/default.
"""
import math

# Vital expects finite floats; some params are 0/1 (on/off). Never write NaN/Inf.
SAFE_FLOAT_MIN = -1e6
SAFE_FLOAT_MAX = 1e6

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
    # Oscillator 2
    "osc_2_on",
    "osc_2_level",
    "osc_2_wave_frame",
    "osc_2_unison_voices",
    "osc_2_unison_detune",
    "osc_2_transpose",
    "osc_2_tune",
    # Oscillator 3
    "osc_3_on",
    "osc_3_level",
    "osc_3_wave_frame",
    "osc_3_unison_voices",
    "osc_3_unison_detune",
    "osc_3_transpose",
    "osc_3_tune",
    # Sampler (SMP)
    "sample_on",
    "sample_level",
    "sample_transpose",
    "sample_tune",
    # Envelope 2
    "env_2_delay",
    "env_2_hold",
    "env_2_attack",
    "env_2_attack_power",
    "env_2_decay",
    "env_2_decay_power",
    "env_2_sustain",
    "env_2_release",
    "env_2_release_power",
    # Envelope 3
    "env_3_delay",
    "env_3_hold",
    "env_3_attack",
    "env_3_decay",
    "env_3_sustain",
    "env_3_release",
    # Envelope 4
    "env_4_delay",
    "env_4_hold",
    "env_4_attack",
    "env_4_decay",
    "env_4_sustain",
    "env_4_release",
    # LFO 1
    "lfo_1_frequency",
    # LFO 2, 3, 4
    "lfo_2_frequency",
    "lfo_3_frequency",
    "lfo_4_frequency",
    # Filter 2
    "filter_2_on",
    "filter_2_cutoff",
    "filter_2_resonance",
    "filter_2_mix",
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


def _sanitize_value(value, template_value):
    """
    Return a value safe for Vital: no NaN/Inf, same type as template (int/float).
    Clamp to finite range so the plugin never crashes on load.
    """
    if template_value is None:
        return 0.0
    if not isinstance(template_value, (int, float)):
        return None  # do not overwrite non-numeric keys (e.g. sample, modulations)
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            out = float(value)
        else:
            out = float(template_value) if math.isfinite(template_value) else 0.0
    else:
        out = float(template_value) if isinstance(template_value, (int, float)) and math.isfinite(template_value) else 0.0
    out = max(SAFE_FLOAT_MIN, min(SAFE_FLOAT_MAX, out))
    if isinstance(template_value, int):
        return int(round(out))
    return float(out)


def filter_to_controlled(params: dict, template_settings: dict) -> dict:
    """
    Return only params that are in CONTROLLED_PARAMS and exist in the template.
    Only includes keys where the template value is numeric (so we never overwrite
    list/dict like 'sample', 'modulations'). Values are sanitized (no NaN/Inf).
    """
    allowed = get_controlled_params_set()
    result = {}
    for k, v in params.items():
        if k not in allowed or k not in template_settings:
            continue
        template_val = template_settings[k]
        if not isinstance(template_val, (int, float)):
            continue
        sanitized = _sanitize_value(v, template_val)
        if sanitized is not None:
            result[k] = sanitized
    return result
