# mapping/vital_writer.py

import json
import math
import os
import re

# Clamp to range Vital/doc expect (e.g. envelope 0-32s) so the plugin never chokes.
SANITIZE_MIN = -1000.0
SANITIZE_MAX = 1000.0

# Per-param ranges so we never write values that crash Vital (e.g. EQ cutoffs are 0-100 in preset).
# Template uses ~0-100 for many cutoffs; writing 893/1000 caused crashes.
# Envelope times: Vital allows delay/hold 0-4s, attack/decay/release 0-32s (see Vital user guide).
# Wave frame: wavetable position 0-255 (low=sine-like, high=saw-like).
PATCH_SAFE_RANGES = {
    "eq_low_cutoff": (0.0, 100.0),
    "eq_high_cutoff": (0.0, 100.0),
    "eq_band_cutoff": (0.0, 100.0),
    "filter_1_cutoff": (0.0, 100.0),
    "filter_2_cutoff": (0.0, 100.0),
    "distortion_filter_cutoff": (0.0, 100.0),
    "chorus_cutoff": (0.0, 100.0),
    "delay_filter_cutoff": (0.0, 100.0),
    "reverb_pre_low_cutoff": (0.0, 100.0),
    "reverb_pre_high_cutoff": (0.0, 100.0),
    "chorus_voices": (1, 16),
    "osc_1_unison_voices": (1, 16),
    "osc_2_unison_voices": (1, 16),
    "osc_3_unison_voices": (1, 16),
    "osc_1_wave_frame": (0.0, 255.0),
    "osc_2_wave_frame": (0.0, 255.0),
    "osc_3_wave_frame": (0.0, 255.0),
    "macro_control_1": (0.0, 1.0),
    "macro_control_2": (0.0, 1.0),
    "macro_control_3": (0.0, 1.0),
    "macro_control_4": (0.0, 1.0),
}
# Vital envelope: delay/hold 0-4s, attack/decay/release 0-32s, sustain 0-1, curve powers ~-2..2.
for i in range(1, 5):
    PATCH_SAFE_RANGES[f"env_{i}_delay"] = (0.0, 4.0)
    PATCH_SAFE_RANGES[f"env_{i}_hold"] = (0.0, 4.0)
    PATCH_SAFE_RANGES[f"env_{i}_attack"] = (0.0, 32.0)
    PATCH_SAFE_RANGES[f"env_{i}_decay"] = (0.0, 32.0)
    PATCH_SAFE_RANGES[f"env_{i}_release"] = (0.0, 32.0)
    PATCH_SAFE_RANGES[f"env_{i}_sustain"] = (0.0, 1.0)
    PATCH_SAFE_RANGES[f"env_{i}_attack_power"] = (-2.0, 2.0)
    PATCH_SAFE_RANGES[f"env_{i}_decay_power"] = (-2.0, 2.0)
    PATCH_SAFE_RANGES[f"env_{i}_release_power"] = (-2.0, 2.0)

# When PATCH_CORE_ONLY=1, only these params are patched (no FX). Use to avoid Vital crash from effect params.
PATCH_CORE_ONLY_PARAMS = frozenset(
    k for k in (
        "osc_1_on", "osc_1_wave_frame", "osc_1_unison_voices", "osc_1_unison_detune",
        "osc_1_transpose", "osc_1_tune", "osc_1_level",
        "env_1_delay", "env_1_hold", "env_1_attack", "env_1_attack_power",
        "env_1_decay", "env_1_decay_power", "env_1_sustain", "env_1_release", "env_1_release_power",
        "filter_1_on", "filter_1_cutoff", "filter_1_resonance", "filter_1_mix",
        "osc_2_on", "osc_2_level", "osc_2_wave_frame", "osc_2_unison_voices", "osc_2_unison_detune",
        "osc_2_transpose", "osc_2_tune",
        "osc_3_on", "osc_3_level", "osc_3_wave_frame", "osc_3_unison_voices", "osc_3_unison_detune",
        "osc_3_transpose", "osc_3_tune",
        "sample_on", "sample_level", "sample_transpose", "sample_tune",
        "env_2_delay", "env_2_hold", "env_2_attack", "env_2_attack_power",
        "env_2_decay", "env_2_decay_power", "env_2_sustain", "env_2_release", "env_2_release_power",
        "env_3_delay", "env_3_hold", "env_3_attack", "env_3_decay", "env_3_sustain", "env_3_release",
        "env_4_delay", "env_4_hold", "env_4_attack", "env_4_decay", "env_4_sustain", "env_4_release",
        "lfo_1_frequency", "lfo_2_frequency", "lfo_3_frequency", "lfo_4_frequency",
        "filter_2_on", "filter_2_cutoff", "filter_2_resonance", "filter_2_mix",
        "macro_control_1", "macro_control_2", "macro_control_3", "macro_control_4",
    )
)

# When core-only, patch these to turn effects off for piano/pluck/bass so preset is dry.
PATCH_FX_OFF_WHEN_CORE_ONLY = {
    "distortion_on": 0,
    "distortion_mix": 0.0,
    "chorus_on": 0,
    "chorus_dry_wet": 0.0,
    "reverb_on": 0,
    "reverb_dry_wet": 0.0,
    "delay_on": 0,
    "delay_dry_wet": 0.0,
    "compressor_on": 0,
    "compressor_mix": 0.0,
    "phaser_on": 0,
    "phaser_dry_wet": 0.0,
    "flanger_on": 0,
    "flanger_dry_wet": 0.0,
    "eq_on": 0,
}

# When sound_class is pad or lead, we patch these FX from rules (not force-off). Safe ranges only.
PATCH_FX_FOR_PAD_LEAD = frozenset([
    "reverb_on", "reverb_dry_wet", "reverb_decay_time", "reverb_size",
    "delay_on", "delay_dry_wet", "delay_feedback", "delay_filter_cutoff", "delay_frequency",
    "chorus_on", "chorus_dry_wet", "chorus_feedback", "chorus_mod_depth", "chorus_cutoff", "chorus_spread", "chorus_voices",
    "compressor_on", "compressor_mix", "compressor_attack", "compressor_release",
])
# Safe 0-1 for wet/mix; reverb_decay 0-1.5; reverb_size 0-1; delay_feedback 0-0.9
FX_SAFE_RANGES = {
    "reverb_dry_wet": (0.0, 1.0), "reverb_decay_time": (0.0, 1.5), "reverb_size": (0.0, 1.0),
    "delay_dry_wet": (0.0, 1.0), "delay_feedback": (0.0, 0.85),
    "chorus_dry_wet": (0.0, 1.0), "chorus_feedback": (0.0, 1.0), "chorus_mod_depth": (0.0, 1.0),
    "compressor_mix": (0.0, 1.0),
}


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf and clamp numbers. Preserve dict/list structure."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return 0.0
        return max(SANITIZE_MIN, min(SANITIZE_MAX, obj))
    if isinstance(obj, int):
        if not math.isfinite(obj):
            return 0
        return max(int(SANITIZE_MIN), min(int(SANITIZE_MAX), obj))
    return obj


def _validate_preset_structure(data: dict) -> None:
    """Raise if structure is invalid so we don't write a corrupt file."""
    if not isinstance(data.get("settings"), dict):
        raise ValueError("Preset must have 'settings' dict")
    for k, v in data["settings"].items():
        if isinstance(v, (int, float)) and not math.isfinite(v):
            raise ValueError(f"Non-finite value in settings.{k}")


def _format_json_number(v):
    """Format a number for JSON so Vital's parser accepts it. No scientific notation; round to avoid float noise."""
    if isinstance(v, int):
        return str(v)
    if not math.isfinite(v):
        return "0.0"
    v = max(SANITIZE_MIN, min(SANITIZE_MAX, float(v)))
    # Round to 10 decimal places to avoid 1.0499999999999998-style values that can upset C++ parsers
    v = round(v, 10)
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    # Always use decimal form, never scientific (e.g. 0.00001 not 1e-5)
    s = format(v, ".10f").rstrip("0").rstrip(".")
    return s if s else "0.0"


# JSON number: integer or float, optional exponent
_JSON_NUM = r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def export_zero_change(template_path: str, output_path: str):
    """Copy template to output unchanged. Use to verify Vital can open the template after a copy."""
    import shutil
    shutil.copy2(template_path, output_path)


def export_vital_preset_by_patch(template_path: str, param_updates: dict, output_path: str):
    """
    Export by patching only our parameter values into the template file.
    The rest of the file stays byte-identical so Vital's loader never sees
    re-serialized JSON (which may trigger the crash).
    """
    with open(template_path, "r", encoding="utf-8") as f:
        text = f.read()
    for key, value in param_updates.items():
        safe_val = _format_json_number(value)
        pattern = '"' + re.escape(key) + r'":\s*' + _JSON_NUM
        match = re.search(pattern, text)
        if match:
            start, end = match.span()
            text = text[:start] + '"' + key + '":' + safe_val + text[end:]
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def export_vital_preset(preset_data: dict, output_path: str, template_path: str = None):
    """
    Write a Vital preset to disk. If template_path is given, uses patch-based
    export (only our param values change; rest of file unchanged) so Vital can load it.
    """
    from mapping.vital_params import CONTROLLED_PARAMS

    if template_path:
        # Default to core-only (no FX params) so Vital doesn't crash; set PATCH_CORE_ONLY=0 to patch all params.
        core_only = os.environ.get("PATCH_CORE_ONLY", "1").strip().lower() in ("1", "true", "yes")
        updates = {}
        settings = preset_data["settings"]
        for k in CONTROLLED_PARAMS:
            if core_only and k not in PATCH_CORE_ONLY_PARAMS:
                continue
            if k not in settings:
                continue
            v = settings[k]
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            v = float(v) if isinstance(v, (int, float)) else v
            if k in PATCH_SAFE_RANGES:
                lo, hi = PATCH_SAFE_RANGES[k]
                v = max(lo, min(hi, v))
                if isinstance(lo, int) and isinstance(hi, int):
                    v = int(round(v))
            else:
                v = max(SANITIZE_MIN, min(SANITIZE_MAX, v))
            updates[k] = v
        if core_only:
            sound_class = preset_data.get("_sound_class", "standard")
            if sound_class in ("pad", "lead"):
                for k in PATCH_FX_FOR_PAD_LEAD:
                    if k not in settings or k in updates:
                        continue
                    v = settings[k]
                    if not isinstance(v, (int, float)) or not math.isfinite(v):
                        continue
                    v = float(v)
                    if k in FX_SAFE_RANGES:
                        lo, hi = FX_SAFE_RANGES[k]
                        v = max(lo, min(hi, v))
                    elif k in PATCH_SAFE_RANGES:
                        lo, hi = PATCH_SAFE_RANGES[k]
                        v = max(lo, min(hi, v))
                    else:
                        v = max(0.0, min(1.0, v)) if "dry_wet" in k or "mix" in k or "feedback" in k else max(SANITIZE_MIN, min(SANITIZE_MAX, v))
                    updates[k] = v
            else:
                updates.update(PATCH_FX_OFF_WHEN_CORE_ONLY)
        export_vital_preset_by_patch(template_path, updates, output_path)
        return
    safe = _sanitize_for_json(preset_data)
    _validate_preset_structure(safe)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(safe, f, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    with open(output_path, "r", encoding="utf-8") as f:
        json.load(f)
