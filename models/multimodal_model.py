import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel

class MultimodalFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP for images
        self.clip_model = CLIPModel.from_pretrained(config["clip_model_name"]).vision_model
        self.clip_proj = nn.Linear(768, config["embedding_dim"])  # CLIP ViT-B/32 output dim = 768

        # Load AudioCLIP (wrapper around CLAP)
        self.audio_model = AutoModel.from_pretrained(config["audioclip_model_name"])
        self.audio_proj = nn.Linear(self.audio_model.config.hidden_size, config["embedding_dim"])

        self.normalize = nn.functional.normalize

    def forward(self, image_inputs, audio_inputs):
        # Extract image features
        image_outputs = self.clip_model(pixel_values=image_inputs).last_hidden_state[:, 0]
        image_embeds = self.normalize(self.clip_proj(image_outputs), dim=-1)

        # Extract audio features
        audio_outputs = self.audio_model(inputs=audio_inputs).last_hidden_state[:, 0]
        audio_embeds = self.normalize(self.audio_proj(audio_outputs), dim=-1)

        return image_embeds, audio_embeds
