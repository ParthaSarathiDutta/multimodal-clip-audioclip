import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils.audio_utils import load_audio, get_log_mel_spectrogram

class MultimodalAudioImageDataset(Dataset):
    def __init__(self, root_dir, image_size=224, sample_rate=16000):
        self.root_dir = root_dir
        self.image_paths = sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".jpg") or f.endswith(".png")
        ])
        self.audio_paths = sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".wav")
        ])
        assert len(self.image_paths) == len(self.audio_paths), "Mismatch between image and audio samples"

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image_tensor = self.transform(image)

        audio = load_audio(self.audio_paths[idx], sample_rate=self.sample_rate)
        audio_tensor = get_log_mel_spectrogram(audio, sample_rate=self.sample_rate)

        return {
            "image": image_tensor,
            "audio": audio_tensor,
            "image_path": self.image_paths[idx],
            "audio_path": self.audio_paths[idx]
        }
