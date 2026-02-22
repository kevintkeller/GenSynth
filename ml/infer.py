import torch
from .model import GenSynthModel
from mapping.param_schema import OUTPUT_SIZE, SynthParameterVector

INPUT_SIZE = 12

def load_model(path="gensynth_model.pt"):
    model = GenSynthModel(INPUT_SIZE, OUTPUT_SIZE)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict_parameters(model, feature_vector):
    with torch.no_grad():
        tensor = torch.tensor(feature_vector).unsqueeze(0)
        output = model(tensor).squeeze(0).numpy()
    return SynthParameterVector(output)