# main.py

import os

from analysis.audio_analysis import analyze_audio
from analysis.feature_vector import build_feature_vector
from mapping.rules_engine import build_vital_parameters
from mapping.vital_schema import load_template, apply_parameters, get_default_template_path
from mapping.vital_writer import export_vital_preset


def generate_from_audio(input_audio_path: str, output_preset_path: str):

    print("Analyzing audio...")
    raw_features = analyze_audio(input_audio_path)

    print("Building feature vector...")
    features = build_feature_vector(raw_features)

    print("Mapping to Vital parameters...")
    vital_params = build_vital_parameters(features)

    print("Loading Vital template...")
    template_path = get_default_template_path()
    template = load_template(template_path)

    print("Applying parameters...")
    final_preset = apply_parameters(template, vital_params)

    core_only = os.environ.get("PATCH_CORE_ONLY", "1").strip().lower() in ("1", "true", "yes")
    print("Exporting preset (patch mode, core params only, no FX)..." if core_only else "Exporting preset (patch mode)...")
    export_vital_preset(final_preset, output_preset_path, template_path=template_path)

    print("Done! Preset saved.")


if __name__ == "__main__":
    generate_from_audio(
        "test-piano.wav",         # your audio file
        "generated-piano-2.vital"    # output preset
    )