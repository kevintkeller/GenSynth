import json
import math

def build_vital_preset(param_vector, template_path, output_path):
    with open(template_path, "r") as f:
        preset = json.load(f)

    s = preset["settings"]
    p = param_vector.to_dict()

    # Oscillator
    s["osc_1_level"] = 1.0
    s["osc_1_unison_voices"] = int(1 + p["osc_unison"] * 7)
    s["osc_1_unison_detune"] = p["osc_detune"]

    # Envelope
    s["env_1_attack"] = p["env_attack"] * 4
    s["env_1_decay"] = p["env_decay"] * 4
    s["env_1_sustain"] = p["env_sustain"]
    s["env_1_release"] = p["env_release"] * 4

    # Filter
    cutoff = 20 * (20000/20) ** p["filter_cutoff"]
    s["filter_1_cutoff"] = cutoff
    s["filter_1_resonance"] = p["filter_resonance"] * 8

    # LFO
    s["lfo_1_frequency"] = p["lfo_rate"] * 20
    s["lfo_1_amplitude"] = p["lfo_amount"]

    # FX
    s["distortion_drive"] = p["dist_amount"]
    s["reverb_dry_wet"] = p["reverb_amount"]
    s["delay_dry_wet"] = p["delay_amount"]

    with open(output_path, "w") as f:
        json.dump(preset, f, indent=2)