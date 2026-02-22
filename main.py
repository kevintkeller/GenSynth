from analysis.loader import load_audio
from analysis.feature_vector import analyze_audio, features_to_vector
from ml.infer import load_model, predict_parameters
from mapping.vital_builder import build_vital_preset

TEMPLATE_PATH = "base_template.vital"

def main(audio_path):
    y, sr = load_audio(audio_path)
    features = analyze_audio(y, sr)

    features_vector = features_to_vector(features)
    
    model = load_model()
    param_vector = predict_parameters(features)

    build_vital_preset(param_vector, TEMPLATE_PATH, "GenSynth_Output.vital")

if __name__ == "__main__":
    main("test-bass.wav")
    