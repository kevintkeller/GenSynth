# mapping/preset_bank.py
"""
Derive default ranges from Vital preset examples in vital_templates/Presets.
Use full-format .vital files only; infer type from filename (bass, pluck, piano, lead, etc.).
Returns per-type defaults to seed or blend with rule-based mapping.
"""
import json
import os
from typing import Dict, List, Any, Optional

# Keys we care about for per-type stats (must exist in full Vital preset settings).
PARAM_KEYS = (
    "osc_1_wave_frame", "filter_1_cutoff", "filter_1_resonance",
    "reverb_on", "reverb_dry_wet", "delay_on", "delay_dry_wet",
    "chorus_on", "chorus_dry_wet", "env_1_attack", "env_1_decay", "env_1_sustain", "env_1_release",
)

# Filename substring -> sound_class (order matters: first match wins).
NAME_TO_TYPE = [
    ("bass", "bass_pluck"),
    ("pluck", "pluck"),
    ("plucking", "pluck"),
    ("piano", "piano_pluck"),
    ("lead", "lead"),
    ("sub", "bass_pluck"),
    ("pad", "pad"),
    ("tech", "pluck"),
    ("house", "bass_pluck"),
]


def _infer_type_from_filename(name: str) -> Optional[str]:
    n = name.lower()
    for substring, sound_class in NAME_TO_TYPE:
        if substring in n:
            return sound_class
    return None


def _is_full_preset(data: dict) -> bool:
    return isinstance(data.get("settings"), dict) and len(data.get("settings", {})) > 100


def load_preset_bank(dir_path: str = "vital_templates/Presets") -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all full-format .vital files from dir_path, infer type from filename,
    return {sound_class: [{"path": ..., "params": {key: value}, ...}, ...]}.
    """
    bank: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.isdir(dir_path):
        return bank
    for fname in os.listdir(dir_path):
        if not fname.lower().endswith(".vital"):
            continue
        path = os.path.join(dir_path, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not _is_full_preset(data):
            continue
        t = _infer_type_from_filename(fname)
        if t is None:
            t = "standard"
        settings = data.get("settings", {})
        params = {k: settings[k] for k in PARAM_KEYS if k in settings and isinstance(settings[k], (int, float))}
        if t not in bank:
            bank[t] = []
        bank[t].append({"path": path, "params": params})
    return bank


def get_per_type_defaults(dir_path: str = "vital_templates/Presets") -> Dict[str, Dict[str, float]]:
    """
    Return per sound_class default values (mean of numeric params across that type's presets).
    Use to seed RULES_CONFIG or blend with feature-driven values. Keys in PARAM_KEYS only.
    """
    bank = load_preset_bank(dir_path)
    out: Dict[str, Dict[str, float]] = {}
    for sound_class, presets in bank.items():
        if not presets:
            continue
        by_key: Dict[str, List[float]] = {}
        for entry in presets:
            for k, v in entry["params"].items():
                if isinstance(v, (int, float)):
                    by_key.setdefault(k, []).append(float(v))
        out[sound_class] = {k: sum(v) / len(v) for k, v in by_key.items() if v}
    return out
