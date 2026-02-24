# main.py

from analysis.audio_analysis import analyze_audio
from analysis.feature_vector import build_feature_vector
from mapping.rules_engine import build_vital_parameters
from mapping.vital_schema import load_template, apply_parameters
from mapping.vital_writer import export_vital_preset


def generate_from_audio(input_audio_path: str, output_preset_path: str):

    print("Analyzing audio...")
    raw_features = analyze_audio(input_audio_path)

    print("Building feature vector...")
    features = build_feature_vector(raw_features)

    print("Mapping to Vital parameters...")
    vital_params = build_vital_parameters(features)

    print("Loading Vital template...")
    template = load_template()

    print("Applying parameters...")
    final_preset = apply_parameters(template, vital_params)

    print("Exporting preset...")
    export_vital_preset(final_preset, output_preset_path)

    print("Done! Preset saved.")


if __name__ == "__main__":
    generate_from_audio(
        "test-bass.wav",         # your audio file
        "generated.vital"    # output preset
    )