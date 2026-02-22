from analysis.loader import load_audio
from analysis.feature_vector import analyze_audio
from analysis.classifier import classify_sound

def run_analysis(path):
    y, sr = load_audio(path)
    features = analyze_audio(y, sr)
    sound_type = classify_sound(features)

    features["sound_type"] = sound_type

    return features

if __name__ == "__main__":
    features = run_analysis("test.wav")
    print(features)