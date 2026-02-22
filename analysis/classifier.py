def classify_sound(features):
    attack = features["attack_time"]
    sustain = features["sustain_level"]
    pitch = features["mean_pitch"]

    if attack < 0.05 and sustain < 0.3:
        return "pluck"
    if attack > 0.3:
        return "pad"
    if pitch < 150:
        return "bass"
    
    return "lead"