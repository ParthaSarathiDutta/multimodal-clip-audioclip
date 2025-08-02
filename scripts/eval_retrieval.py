import torch
from torch.utils.data import DataLoader
import json
from models.multimodal_model import MultimodalFusionModel
from utils.dataloader import MultimodalAudioImageDataset
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate():
    with open("config.json") as f:
        config = json.load(f)

    dataset = MultimodalAudioImageDataset("data/sample_dataset", image_size=config["image_size"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = MultimodalFusionModel(config).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    image_embeddings = []
    audio_embeddings = []
    image_paths = []
    audio_paths = []

    with torch.no_grad():
        for batch in dataloader:
            image = batch["image"].to(model.device).float()
            audio = batch["audio"].unsqueeze(1).to(model.device).float()

            img_embed, aud_embed = model(image, audio)
            image_embeddings.append(img_embed.cpu().numpy())
            audio_embeddings.append(aud_embed.cpu().numpy())
            image_paths.append(batch["image_path"][0])
            audio_paths.append(batch["audio_path"][0])

    image_embeddings = np.vstack(image_embeddings)
    audio_embeddings = np.vstack(audio_embeddings)

    similarity = np.matmul(image_embeddings, audio_embeddings.T)

    # Visualize similarity heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity, cmap="viridis", aspect="auto")
    plt.colorbar(label="Cosine Similarity")
    plt.xticks(ticks=np.arange(len(audio_paths)), labels=[os.path.basename(p) for p in audio_paths], rotation=90)
    plt.yticks(ticks=np.arange(len(image_paths)), labels=[os.path.basename(p) for p in image_paths])
    plt.xlabel("Audio")
    plt.ylabel("Image")
    plt.title("Cross-Modal Similarity: Image ↔ Audio")
    plt.tight_layout()
    plt.savefig("plots/similarity_heatmap.png")
    plt.close()

if __name__ == "__main__":
    evaluate()
