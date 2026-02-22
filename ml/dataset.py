import torch
from torch.utils.data import Dataset
import json

class PresetDataset(Dataset):
    def __init__(self, path):
        with open(path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item["features"], dtype=torch.float32),
            torch.tensor(item["parameters"], dtype=torch.float32)
        )