# main.py

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

    print("Exporting preset (patch mode so Vital can open it)...")
    export_vital_preset(final_preset, output_preset_path, template_path=template_path)

    print("Done! Preset saved.")


if __name__ == "__main__":
    generate_from_audio(
        "test-piano.wav",         # your audio file
        "generated-piano-2.vital"    # output preset
    )