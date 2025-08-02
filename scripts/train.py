import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from models.multimodal_model import MultimodalFusionModel
from utils.dataloader import MultimodalAudioImageDataset

def contrastive_loss(a, b, temperature=0.07):
    logits = torch.matmul(a, b.T) / temperature
    labels = torch.arange(len(a)).to(a.device)
    return (nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss()(logits.T, labels)) / 2

def train():
    with open("config.json") as f:
        config = json.load(f)

    dataset = MultimodalAudioImageDataset("data/sample_dataset", image_size=config["image_size"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = MultimodalFusionModel(config).to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            image = batch["image"].to(model.device)
            audio = batch["audio"].to(model.device)

            image = image.float()
            audio = audio.unsqueeze(1).float()  # Add channel dimension

            img_embeds, aud_embeds = model(image, audio)
            loss = contrastive_loss(img_embeds, aud_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    train()
