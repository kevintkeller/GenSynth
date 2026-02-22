import torch
from torch.utils.data import DataLoader
from .dataset import PresetDataset
from .model import GenSynthModel
from mapping.param_schema import OUTPUT_SIZE

INPUT_SIZE = 12

def train_model(dataset_path, epochs=50):
    dataset = PresetDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GenSynthModel(INPUT_SIZE, OUTPUT_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0

        for features, targets in loader:
            preds = model(features)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: {total_loss:.4f}")

    torch.save(model.state_dict(), "gensynth_model.pt")
    print("Model saved.")